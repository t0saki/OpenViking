//! HTTP routing and forwarding (proxy.py) plus the hidden tool loop (tool_loop.py) for
//! Anthropic Messages streaming.

use std::collections::HashSet;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;

use crate::anthropic::{add_usage, encode, Anthropic, Call, SseDecoder, PREFIX};
use crate::canon::{digest, prefix_chain};
use crate::kernel::{self, blocking, normalized_usage, write, Credential, Ctx, Prepared};
use crate::store::now;
use crate::text::Headers;

const HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];
const DROP_REQUEST: &[&str] = &[
    "host",
    "content-length",
    "authorization",
    "x-api-key",
    "cookie",
    "accept-encoding",
];

fn error(status: StatusCode, reason: &str) -> Response {
    (
        status,
        axum::Json(json!({"reason": reason, "detail": reason})),
    )
        .into_response()
}

fn credential<'a>(ctx: &'a Ctx, headers: &HeaderMap) -> Option<&'a Credential> {
    let key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .or_else(|| {
            headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .map(|v| v.trim_start_matches("Bearer ").to_string())
        })?;
    ctx.credentials.get(&key)
}

fn upstream_headers(ctx: &Ctx, incoming: &HeaderMap) -> reqwest::header::HeaderMap {
    let mut out = reqwest::header::HeaderMap::new();
    for (k, v) in incoming {
        let name = k.as_str();
        if HOP.contains(&name) || DROP_REQUEST.contains(&name) || name.starts_with("x-openviking-")
        {
            continue;
        }
        out.insert(k.clone(), v.clone());
    }
    let key = &ctx.upstream_key;
    out.insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {key}")).unwrap(),
    );
    out.insert("x-api-key", HeaderValue::from_str(key).unwrap());
    out.insert("accept-encoding", HeaderValue::from_static("identity"));
    out.insert("content-type", HeaderValue::from_static("application/json"));
    out
}

pub async fn messages(State(ctx): State<Arc<Ctx>>, headers: HeaderMap, raw: Bytes) -> Response {
    let started = Instant::now();
    let Some(cred) = credential(&ctx, &headers).cloned() else {
        return error(StatusCode::UNAUTHORIZED, "invalid_gateway_key");
    };
    // Revocation check against OV identity (cached for 30 s, as in Python).
    let _ = ctx.viking.identify(&cred.openviking_key).await;
    let Ok(original) = serde_json::from_slice::<Value>(&raw) else {
        return error(StatusCode::BAD_REQUEST, "unsafe_json");
    };
    let raw_len = raw.len();
    drop(raw);
    let hdrs: Headers = headers
        .iter()
        .filter_map(|(k, v)| {
            v.to_str()
                .ok()
                .map(|v| (k.as_str().to_string(), v.to_string()))
        })
        .collect();
    let stream = original
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let model = original
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let policy = {
        let store = ctx.store.clone();
        let (a, p) = (cred.account.clone(), cred.policy_id.clone());
        blocking(move || store.get_object(&a, "policies", &p)).await
    };
    let Some(policy) = policy else {
        return error(StatusCode::FORBIDDEN, "Context policy is unavailable");
    };
    let mut prepared = match kernel::prepare(&ctx, original, &hdrs, &cred, policy).await {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("preparation failed: {e}");
            None
        }
    };
    let Some(mut prepared) = prepared.take() else {
        return error(StatusCode::BAD_GATEWAY, "memory_store_failure");
    };
    // Not kept after sending: a hidden-tool continuation (rare) rebuilds it from the client body.
    let body = std::mem::take(&mut prepared.out);
    prepared.metrics.insert(
        "request_id".into(),
        json!(uuid::Uuid::new_v4().simple().to_string()),
    );
    prepared.metrics.insert("model".into(), json!(model));
    prepared.metrics.insert("body_bytes".into(), json!(raw_len));
    let up_headers = upstream_headers(&ctx, &headers);
    let target = format!("{}/v1/messages", ctx.upstream_base);
    let upstream_started = Instant::now();
    let resp = match ctx
        .http
        .post(&target)
        .headers(up_headers.clone())
        .body(body)
        .timeout(Duration::from_secs(600))
        .send()
        .await
    {
        Ok(r) => r,
        Err(_) => return error(StatusCode::BAD_GATEWAY, "Model upstream is unavailable"),
    };
    let status = resp.status().as_u16();
    prepared.metrics.insert("status".into(), json!(status));
    prepared
        .metrics
        .insert("upstream_id".into(), json!(cred.upstream_id));
    let mut out_headers = HeaderMap::new();
    for (k, v) in resp.headers() {
        let name = k.as_str();
        if HOP.contains(&name) || matches!(name, "content-length" | "etag" | "content-encoding") {
            continue;
        }
        if let Ok(n) = HeaderName::from_bytes(name.as_bytes()) {
            out_headers.insert(n, v.clone());
        }
    }
    let (tx, rx) = mpsc::channel::<Bytes>(256);
    let body =
        Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx).map(Ok::<_, Infallible>));
    if status < 300 && stream && (prepared.tools_active) {
        out_headers.insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        tokio::spawn(tool_response(
            ctx.clone(),
            prepared,
            cred,
            resp,
            tx,
            up_headers,
            target,
            started,
            upstream_started,
        ));
    } else {
        tokio::spawn(observe_response(
            ctx.clone(),
            prepared,
            cred,
            resp,
            tx,
            status < 300,
            upstream_started,
        ));
    }
    let mut response = Response::new(body);
    *response.status_mut() = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
    *response.headers_mut() = out_headers;
    response
}

struct ToolExec<'a> {
    ctx: &'a Arc<Ctx>,
    prepared: &'a Prepared,
    key: String,
}

impl ToolExec<'_> {
    /// `ToolExecutor.execute` for MCP tools: validate, claim a receipt, call, record it.
    async fn execute(&self, call: &Call) -> (String, String, bool) {
        let schema = self
            .prepared
            .tools
            .iter()
            .find(|t| t["function"]["name"] == call.name.as_str())
            .map(|t| t["function"]["parameters"].clone())
            .unwrap_or(json!({}));
        let args: Value = match serde_json::from_str(&call.arguments) {
            Ok(Value::Object(a)) => {
                let props = schema.get("properties").and_then(Value::as_object);
                let unknown = a
                    .keys()
                    .any(|k| props.map(|p| !p.contains_key(k)).unwrap_or(true));
                let missing = schema
                    .get("required")
                    .and_then(Value::as_array)
                    .map(|r| {
                        r.iter()
                            .filter_map(Value::as_str)
                            .any(|k| !a.contains_key(k))
                    })
                    .unwrap_or(false);
                if unknown || missing {
                    return (
                        call.id.clone(),
                        "Invalid tool arguments or OpenViking operation failed".into(),
                        true,
                    );
                }
                Value::Object(a)
            }
            _ => return (call.id.clone(), "Invalid tool arguments".into(), true),
        };
        let call_json = json!({"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}});
        let anchor = digest(&format!(
            "{}{}",
            self.prepared.chain.last().cloned().unwrap_or_default(),
            call_json
        ));
        let receipt = format!("tool:{}:{anchor}", self.prepared.session);
        let claim = json!({"owner": uuid::Uuid::new_v4().simple().to_string(), "time": now()});
        let store = self.ctx.store.clone();
        let (sc, rk, c) = (self.prepared.scope.clone(), receipt.clone(), claim.clone());
        write(move || store.state_swap(&sc, &rk, 0, &c)).await;
        let result = self
            .ctx
            .viking
            .mcp(
                "tools/call",
                &self.key,
                json!({"name": call.name.trim_start_matches(PREFIX), "arguments": args}),
                30.0,
            )
            .await;
        let (content, failed) = match result {
            Ok(v) => {
                let text = v["content"]
                    .as_array()
                    .map(|c| {
                        c.iter()
                            .filter_map(|b| b.get("text").and_then(Value::as_str))
                            .collect::<Vec<_>>()
                            .join("\n")
                    })
                    .unwrap_or_default();
                let limit = self.prepared.policy["tool_result_bytes"]
                    .as_u64()
                    .unwrap_or(65536) as usize;
                let mut text = text;
                if text.len() > limit {
                    let mut cut = limit;
                    while !text.is_char_boundary(cut) {
                        cut -= 1;
                    }
                    text.truncate(cut);
                    text.push('…');
                }
                (
                    text,
                    v.get("isError").and_then(Value::as_bool).unwrap_or(false),
                )
            }
            Err(_) => (
                "Invalid tool arguments or OpenViking operation failed".into(),
                true,
            ),
        };
        let mut done = claim;
        done["content"] = json!(content);
        done["failed"] = json!(failed);
        let store = self.ctx.store.clone();
        let sc = self.prepared.scope.clone();
        write(move || store.state_swap(&sc, &receipt, 1, &done)).await;
        (call.id.clone(), content, failed)
    }
}

fn notice_head(call: &Call) -> String {
    let short = call.name.trim_start_matches(PREFIX);
    let args: Value = serde_json::from_str(&call.arguments).unwrap_or(json!({}));
    let target = ["query", "uri", "pattern", "path"]
        .iter()
        .find_map(|k| {
            args.get(*k)
                .and_then(Value::as_str)
                .filter(|s| !s.trim().is_empty())
        })
        .map(|s| {
            let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
            let s: String = if s.chars().count() > 80 {
                s.chars().take(79).collect::<String>() + "…"
            } else {
                s
            };
            format!(": \"{s}\"")
        })
        .unwrap_or_default();
    format!("> OpenViking {short}{target}")
}

async fn send(tx: &mpsc::Sender<Bytes>, events: Vec<Value>) -> bool {
    for e in events {
        let waited = Instant::now();
        if tx.send(encode(&e)).await.is_err() {
            return false;
        }
        if waited.elapsed().as_secs_f64() > 5.0 {
            eprintln!("slow client send: {:.1}s", waited.elapsed().as_secs_f64());
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
async fn tool_response(
    ctx: Arc<Ctx>,
    mut prepared: Prepared,
    cred: Credential,
    first: reqwest::Response,
    tx: mpsc::Sender<Bytes>,
    up_headers: reqwest::header::HeaderMap,
    target: String,
    started: Instant,
    upstream_started: Instant,
) {
    let mut adapter = Anthropic::new();
    let allowed: HashSet<String> = prepared
        .tools
        .iter()
        .filter_map(|t| t["function"]["name"].as_str().map(str::to_string))
        .collect();
    let mut transcript: Vec<Value> = vec![];
    let mut usage_total = Map::new();
    let mut rounds = 0;
    let mut hidden = false;
    let mut continuation: Option<Value> = None;
    let mut response = first;
    let mut output: Vec<Value> = vec![];
    let mut complete = false;
    let mut failure: Option<String> = None;
    let mut frames = Vec::new();
    'rounds: loop {
        adapter.begin();
        let mut decoder = SseDecoder::default();
        loop {
            let waited = Instant::now();
            let next = response.chunk().await;
            if waited.elapsed().as_secs_f64() > 20.0 {
                eprintln!(
                    "slow upstream chunk: {:.1}s",
                    waited.elapsed().as_secs_f64()
                );
            }
            match next {
                Ok(Some(chunk)) => {
                    decoder.feed(&chunk, &mut frames);
                    for f in frames.drain(..) {
                        let Some(value) = SseDecoder::data(&f) else {
                            continue;
                        };
                        if value.get("error").is_some()
                            || value.get("type").and_then(Value::as_str) == Some("error")
                        {
                            failure = Some("Model stream failed".into());
                            break 'rounds;
                        }
                        match adapter.event(value) {
                            Ok(events) => {
                                if !send(&tx, events).await {
                                    failure = Some("client_disconnected".into());
                                    break 'rounds;
                                }
                            }
                            Err(e) => {
                                failure = Some(e);
                                break 'rounds;
                            }
                        }
                    }
                }
                Ok(None) => break,
                Err(_) => {
                    failure = Some("upstream read failed".into());
                    break 'rounds;
                }
            }
        }
        if !decoder.buffer.iter().all(u8::is_ascii_whitespace) {
            failure = Some("Incomplete model event stream".into());
            break;
        }
        let round = match adapter.end() {
            Ok(r) => r,
            Err(e) => {
                failure = Some(e);
                break;
            }
        };
        // observe
        transcript.extend(round.output.iter().cloned());
        add_usage(&mut usage_total, &round.usage);
        let normalized = normalized_usage(&round.usage);
        if normalized["input_tokens"].as_i64().unwrap_or(0) > 0 {
            prepared.context_usage = Some(normalized.clone());
        }
        let prefix = if rounds > 0 {
            "hidden_upstream_"
        } else {
            "first_upstream_"
        };
        for (k, v) in normalized.as_object().unwrap() {
            let key = format!("{prefix}{k}");
            let prev = prepared
                .metrics
                .get(&key)
                .and_then(Value::as_i64)
                .unwrap_or(0);
            prepared
                .metrics
                .insert(key, json!(prev + v.as_i64().unwrap_or(0)));
        }
        let (mut owned, mut client): (Vec<Call>, Vec<Call>) = (vec![], vec![]);
        for call in &round.calls {
            if !prepared.tools_active {
                client.push(call.clone());
            } else if call.name.starts_with(PREFIX) && !allowed.contains(&call.name) {
                failure = Some("Model called an unavailable gateway tool".into());
                break 'rounds;
            } else if allowed.contains(&call.name) {
                owned.push(call.clone());
            } else {
                client.push(call.clone());
            }
        }
        if !owned.is_empty() {
            if !round.tool_stop {
                failure = Some("Gateway tool call has an invalid stop reason".into());
                break;
            }
            let exec = ToolExec {
                ctx: &ctx,
                prepared: &prepared,
                key: cred.openviking_key.clone(),
            };
            let mut events = adapter.open_notice();
            let mut results = vec![];
            for call in &owned {
                events.extend(adapter.notice(&format!("\n\n{}", notice_head(call))));
                if !send(&tx, std::mem::take(&mut events)).await {
                    failure = Some("client_disconnected".into());
                    break 'rounds;
                }
                let result = if prepared.tools_closed {
                    (call.id.clone(), "OpenViking tools are unavailable for the rest of this request; continue without them.".to_string(), false)
                } else {
                    exec.execute(call).await
                };
                events = adapter.notice(if result.2 { " — failed" } else { " — done" });
                results.push(result);
            }
            events.extend(adapter.notice("\n\n"));
            events.extend(adapter.close_notice());
            if !send(&tx, events).await {
                failure = Some("client_disconnected".into());
                break;
            }
            let results = Anthropic::results(&results);
            transcript.extend(results.iter().cloned());
            let body = continuation.get_or_insert_with(|| continuation_body(&prepared));
            if let Some(ms) = body["messages"].as_array_mut() {
                ms.extend(round.output.iter().cloned());
                ms.extend(results);
            }
            rounds += 1;
            hidden = true;
            prepared
                .metrics
                .insert("hidden_rounds".into(), json!(rounds));
        }
        if !send(&tx, adapter.publish_calls(&client)).await {
            failure = Some("client_disconnected".into());
            break;
        }
        if !client.is_empty() || owned.is_empty() {
            // persist
            let visible = vec![json!({"role": "assistant", "content": adapter.visible})];
            let anchor = prefix_chain(prepared.messages().iter().chain(visible.iter()))
                .last()
                .cloned()
                .unwrap_or_default();
            if hidden && !anchor.is_empty() {
                let value = json!({"messages": transcript, "visible_count": 1, "upstream_id": prepared.root["upstream_id"]});
                let store = ctx.store.clone();
                let (sc, s) = (prepared.scope.clone(), prepared.session.clone());
                write(move || store.replay_put(&sc, &s, "hidden", &anchor, &value)).await;
            }
            let final_message = adapter.final_message(&usage_total);
            // capture.nonstream(final): the reply as the next request resends it.
            output = vec![json!({"role": "assistant", "content": final_message["content"]})];
            let ended = matches!(
                adapter.finish.as_deref(),
                Some("end_turn") | Some("stop_sequence") | Some("tool_use")
            );
            let handoff = ended
                && final_message["content"]
                    .as_array()
                    .map(|c| c.iter().any(|b| b["type"] == "tool_use"))
                    .unwrap_or(false);
            complete = ended && !handoff;
            kernel::relayed(&ctx, &mut prepared, &output, ended).await;
            if !send(&tx, adapter.terminal(&final_message)).await {
                failure = Some("client_disconnected".into());
            }
            break;
        }
        let body = continuation.as_ref().unwrap();
        let bytes = crate::canon::to_vec(body);
        response = match ctx
            .http
            .post(&target)
            .headers(up_headers.clone())
            .body(bytes)
            .timeout(Duration::from_secs(600))
            .send()
            .await
        {
            Ok(r) if r.status().as_u16() < 300 => r,
            _ => {
                failure = Some("Model upstream rejected a tool continuation".into());
                break;
            }
        };
    }
    if let Some(f) = &failure {
        prepared
            .metrics
            .insert("degradation".into(), json!("hidden_tool_loop_failed"));
        if f != "client_disconnected" {
            let _ = tx.send(encode(&Anthropic::error(f))).await;
        }
        complete = false;
    }
    drop(tx);
    // finish(): metrics, completed(), request log.
    for (k, v) in normalized_usage(&usage_total).as_object().unwrap() {
        prepared.metrics.insert(k.clone(), v.clone());
    }
    prepared.metrics.insert(
        "duration_ms".into(),
        json!(upstream_started.elapsed().as_secs_f64() * 1000.0),
    );
    prepared.metrics.insert(
        "total_ms".into(),
        json!(started.elapsed().as_secs_f64() * 1000.0),
    );
    if failure.is_none() {
        kernel::completed(&ctx, &mut prepared, &cred, &output, complete).await;
    }
}

async fn observe_response(
    ctx: Arc<Ctx>,
    mut prepared: Prepared,
    cred: Credential,
    mut response: reqwest::Response,
    tx: mpsc::Sender<Bytes>,
    ok: bool,
    upstream_started: Instant,
) {
    let mut adapter = Anthropic::new();
    adapter.begin();
    let mut decoder = SseDecoder::default();
    let mut frames = Vec::new();
    while let Ok(Some(chunk)) = response.chunk().await {
        if ok {
            decoder.feed(&chunk, &mut frames);
            for f in frames.drain(..) {
                if let Some(v) = SseDecoder::data(&f) {
                    let _ = adapter.event(v);
                }
            }
        }
        if tx.send(chunk).await.is_err() {
            return;
        }
    }
    drop(tx);
    prepared.metrics.insert(
        "duration_ms".into(),
        json!(upstream_started.elapsed().as_secs_f64() * 1000.0),
    );
    if ok {
        let output = vec![json!({"role": "assistant", "content": adapter.visible})];
        kernel::relayed(&ctx, &mut prepared, &output, true).await;
        kernel::completed(
            &ctx,
            &mut prepared,
            &cred,
            &output,
            adapter.finish.as_deref() == Some("end_turn"),
        )
        .await;
    }
}

/// The upstream body for a hidden-tool continuation: the client body plus gateway tools.
/// (Python keeps its prepared body; this rebuild costs about the same serialization.)
fn continuation_body(prepared: &Prepared) -> Value {
    let mut body = prepared.original.clone();
    let extra = kernel::wire_tools(&prepared.tools);
    match body.get_mut("tools").and_then(Value::as_array_mut) {
        Some(tools) => tools.extend(extra),
        None => body["tools"] = Value::Array(extra),
    }
    body
}

pub async fn health() -> axum::Json<Value> {
    axum::Json(json!({"status": "ok", "service": "ov-gateway-rs"}))
}
