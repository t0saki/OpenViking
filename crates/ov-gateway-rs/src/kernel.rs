//! `MemoryKernel` subset for the Anthropic Messages path: request preparation (identity,
//! replay records, classify, plugin/tool detection, session root and tools, recall with
//! profile, capture confirm, measure, hidden replay) and post-reply bookkeeping
//! (`relayed`, `completed`, capture `stage`). See BENCH.md for what is simplified.

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use regex::Regex;
use serde_json::{json, Map, Value};

use crate::anthropic::{merge_tool_results, PREFIX};
use crate::canon::{digest, prefix_chain, write_list, write_sorted, write_str, write_value};
use crate::ov::Viking;
use crate::store::{now, rec, rec_insert, Document, Records, Store};
use crate::text::{
    self, block, clean_text, client_openviking_tools, is_user, starts_with_notice, text_content,
    token_estimate, unwrap_client, Headers, ToolNames,
};

pub const WRITE_TOOLS: &[&str] = &[
    "remember",
    "write",
    "edit",
    "add_resource",
    "add_skill",
    "forget",
    "set_acl",
    "cancel_watch",
];
const SESSION_HEADERS: &[&str] = &[
    "x-openviking-session",
    "thread-id",
    "x-claude-code-session-id",
    "x-opencode-session-id",
    "x-session-id",
    "session-id",
];
const DEFAULT_WINDOW: i64 = 1_000_000;

pub fn default_policy() -> Value {
    json!({
        "name": "default", "recall": true, "profile": true, "profile_max_tokens": 4000, "capture": true,
        "context_types": ["memory", "resource", "skill"], "quotas": {}, "max_tokens": 1600,
        "session_max_tokens": 30000, "score_threshold": 0.35, "recall_timeout": 2.0, "query_max_chars": 8000,
        "show_recall": false, "commit_tokens": 20000, "keep_recent_messages": 10, "idle_seconds": 600.0,
        "compaction": true, "compaction_threshold": 0.9, "summary_max_tokens": 8000, "context_window": null,
        "gateway_tools": true, "disabled_tools": WRITE_TOOLS, "tool_max_rounds": null, "tool_timeout_seconds": 30.0,
        "tool_result_bytes": 65536, "tool_total_seconds": null, "tool_total_tokens": null, "show_tool_calls": true,
        "agent_windows": false, "window_soft_ratio": 0.7, "window_hard_ratio": 0.85
    })
}

/// Stand-in for `Policy.model_validate`: check the fields this path reads.
pub fn validate_policy(p: &Value) -> Result<(), String> {
    let o = p.as_object().ok_or("policy")?;
    for k in [
        "recall",
        "profile",
        "capture",
        "gateway_tools",
        "show_recall",
        "show_tool_calls",
        "compaction",
    ] {
        o.get(k)
            .and_then(Value::as_bool)
            .ok_or(format!("policy.{k}"))?;
    }
    for k in [
        "max_tokens",
        "session_max_tokens",
        "query_max_chars",
        "profile_max_tokens",
        "commit_tokens",
    ] {
        o.get(k)
            .and_then(Value::as_i64)
            .ok_or(format!("policy.{k}"))?;
    }
    Ok(())
}

#[derive(Clone)]
pub struct Credential {
    pub id: String,
    pub account: String,
    pub user_id: String,
    pub openviking_key: String,
    pub policy_id: String,
    pub upstream_id: String,
}

pub struct Ctx {
    pub store: Arc<Store>,
    pub viking: Viking,
    pub http: reqwest::Client,
    pub upstream_base: String,
    pub upstream_key: String,
    pub credentials: HashMap<String, Credential>,
}

type Job = Box<dyn FnOnce() + Send>;

/// One dedicated writer thread, like the Python store's single-thread write executor,
/// so writes queue among themselves and never occupy the read pool.
static WRITER: Lazy<std::sync::mpsc::Sender<Job>> = Lazy::new(|| {
    let (tx, rx) = std::sync::mpsc::channel::<Job>();
    std::thread::Builder::new()
        .name("gateway-write".into())
        .spawn(move || {
            for job in rx {
                job();
            }
        })
        .unwrap();
    tx
});

pub async fn write<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let queued = Instant::now();
    let (tx, rx) = tokio::sync::oneshot::channel();
    WRITER
        .send(Box::new(move || {
            let ran = Instant::now();
            let r = f();
            let _ = tx.send((r, ran.elapsed().as_secs_f64()));
        }))
        .unwrap();
    let (value, ran) = rx.await.unwrap();
    let total = queued.elapsed().as_secs_f64();
    if total > 1.0 {
        eprintln!("slow write op: total {:.2}s, ran {:.2}s", total, ran);
    }
    value
}

pub async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let queued = Instant::now();
    let r = tokio::task::spawn_blocking(move || {
        let ran = Instant::now();
        let r = f();
        (r, ran.elapsed().as_secs_f64())
    })
    .await
    .unwrap();
    let (value, ran) = r;
    let total = queued.elapsed().as_secs_f64();
    if total > 1.0 {
        eprintln!("slow blocking op: total {:.2}s, ran {:.2}s", total, ran);
    }
    value
}

pub struct Prepared {
    /// The client's body as parsed (messages as the gateway relayed them).
    pub original: Value,
    /// The upstream body, serialized.
    pub out: bytes::Bytes,
    pub scope: String,
    pub session: String,
    pub chain: Vec<String>,
    pub kind: &'static str,
    pub anchor: i64,
    pub root: Value,
    pub records: Records,
    pub disabled: bool,
    pub observation: Document,
    pub capture: Document,
    pub capture_target: String,
    pub tools_active: bool,
    pub tools_closed: bool,
    pub tools: Vec<Value>,
    pub policy: Value,
    pub metrics: Map<String, Value>,
    pub reply_anchor: String,
    pub relayed: bool,
    pub context_usage: Option<Value>,
    pub ancestors: Vec<String>,
}

impl Prepared {
    pub fn messages(&self) -> &[Value] {
        self.original["messages"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

fn header_session(headers: &Headers) -> Option<String> {
    SESSION_HEADERS
        .iter()
        .map(|k| text::header(headers, k))
        .find(|v| !v.is_empty())
        .map(digest)
}

fn role(m: &Value) -> Option<&str> {
    m.get("role").and_then(Value::as_str)
}

fn parameter_fingerprint(body: &Map<String, Value>) -> String {
    // STABLE_PARAMETERS in sorted key order.
    const KEYS: &[&str] = &[
        "instructions",
        "model",
        "reasoning_effort",
        "seed",
        "system",
        "temperature",
        "thinking",
        "tools",
        "top_p",
    ];
    let mut out = Vec::with_capacity(256 * 1024);
    out.push(b'{');
    let mut first = true;
    for k in KEYS {
        if let Some(v) = body.get(*k) {
            if !first {
                out.push(b',');
            }
            first = false;
            write_str(&mut out, k);
            out.push(b':');
            write_sorted(&mut out, v);
        }
    }
    out.push(b'}');
    use sha2::Digest;
    hex::encode(sha2::Sha256::digest(&out))
}

pub fn select_tools(catalog: &Value, policy: &Value) -> Vec<Value> {
    let disabled: HashSet<&str> = policy["disabled_tools"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    catalog
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .filter(|t| !disabled.contains(t["name"].as_str().unwrap_or("")))
                .map(|t| {
                    json!({"type": "function", "function": {
                        "name": format!("{PREFIX}{}", t["name"].as_str().unwrap_or("")),
                        "description": t.get("description").cloned().unwrap_or(json!("")),
                        "parameters": t["inputSchema"].clone(),
                    }})
                })
                .collect()
        })
        .unwrap_or_default()
}

pub fn wire_tools(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| json!({"name": t["function"]["name"], "description": t["function"]["description"], "input_schema": t["function"]["parameters"]}))
        .collect()
}

static TOOL_NAMES: Lazy<Mutex<HashMap<String, Arc<ToolNames>>>> = Lazy::new(Default::default);

fn tool_names(names: &[String]) -> Arc<ToolNames> {
    let key = names.join("\0");
    TOOL_NAMES
        .lock()
        .entry(key)
        .or_insert_with(|| Arc::new(ToolNames::new(names)))
        .clone()
}

/// Append `text` to a user message or tool result as a trailing text block.
fn append_context(message: &mut Value, text: &str) {
    let content = message.get("content").cloned().unwrap_or(json!(""));
    let mut parts = match content {
        Value::String(s) => vec![json!({"type": "text", "text": s})],
        Value::Array(a) => a,
        _ => vec![],
    };
    parts.push(json!({"type": "text", "text": text}));
    message["content"] = Value::Array(parts);
}

fn usage_of(usage: &Map<String, Value>) -> Value {
    let get = |k: &str| usage.get(k).and_then(Value::as_i64).unwrap_or(0);
    let cached = get("cache_read_input_tokens");
    let written = get("cache_creation_input_tokens");
    let mut input = get("input_tokens");
    if usage.contains_key("cache_read_input_tokens") {
        input += cached + written;
    }
    json!({"input_tokens": input, "output_tokens": get("output_tokens"), "cached_tokens": cached, "cache_write_tokens": written})
}

pub fn normalized_usage(usage: &Map<String, Value>) -> Value {
    usage_of(usage)
}

/// Preparation result before serialization: outgoing messages share unchanged ones.
pub async fn prepare(
    ctx: &Arc<Ctx>,
    original: Value,
    headers: &Headers,
    cred: &Credential,
    policy: Value,
) -> Result<Prepared, String> {
    let started = Instant::now();
    let body = original.as_object().ok_or("body")?;
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("invalid message list")?;
    if !messages.iter().all(Value::is_object) {
        return Err("invalid message list".into());
    }
    // strip_lead: only replies starting with a recall notice are edited.
    let _leads = messages
        .iter()
        .filter(|m| role(m) == Some("assistant"))
        .filter(|m| starts_with_notice(&text_content(m)))
        .count();
    let scope = digest(&format!("{}\0{}\0anthropic", cred.account, cred.user_id));

    // sent_history: thinking a client resent as text, keyed by its digest.
    let mut keys: Vec<String> = messages
        .iter()
        .filter(|m| role(m) == Some("assistant"))
        .filter_map(|m| m.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|b| b.get("text").and_then(Value::as_str))
        .filter(|t| !t.is_empty())
        .map(|t| format!("thinking:{}", digest(t)))
        .collect();
    keys.sort_unstable();
    keys.dedup();
    let mut thinking: Vec<String> = vec![];
    if !keys.is_empty() {
        let store = ctx.store.clone();
        let sc = scope.clone();
        let known = blocking(move || store.state_read(&sc, &keys)).await;
        thinking = known.into_keys().collect();
    }

    // identity
    let sid_header = header_session(headers);
    let chain = prefix_chain(messages.iter());
    let endpoints: Vec<String> = chain
        .iter()
        .zip(messages)
        .filter(|(a, m)| !a.is_empty() && role(m) == Some("assistant"))
        .map(|(a, _)| a.clone())
        .collect();
    let owners = if endpoints.is_empty() {
        HashMap::new()
    } else {
        let store = ctx.store.clone();
        let sc = scope.clone();
        let ep = endpoints.clone();
        blocking(move || store.replies(&sc, &ep)).await
    };
    let sid = match sid_header {
        Some(s) => s,
        None => {
            let latest = endpoints.iter().rev().find_map(|a| owners.get(a));
            match latest {
                Some(set) if set.len() == 1 => set.iter().next().unwrap().clone(),
                _ => format!("anonymous-{}", uuid::Uuid::new_v4().simple()),
            }
        }
    };
    let mut ancestors: Vec<String> = owners
        .values()
        .filter(|o| o.len() == 1)
        .map(|o| o.iter().next().unwrap().clone())
        .filter(|s| *s != sid)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    ancestors.sort();

    // load
    let mut anchors: Vec<&str> = Vec::with_capacity(chain.len() + 1);
    let mut seen = HashSet::new();
    for a in std::iter::once("").chain(chain.iter().map(String::as_str)) {
        if seen.insert(a) {
            anchors.push(a);
        }
    }
    let anchors: Vec<String> = anchors.into_iter().map(str::to_string).collect();
    let (mut records, observation, capture) = {
        let store = ctx.store.clone();
        let (sc, s, an) = (scope.clone(), sid.clone(), ancestors.clone());
        blocking(move || {
            let refs: Vec<&str> = anchors.iter().map(String::as_str).collect();
            store.load(&sc, &s, &refs, &an)
        })
        .await
    };

    let plugin = text::plugin_present(&original, headers);
    // session_root
    let root = match rec(&records, "root", "") {
        Some(r) => r.clone(),
        None => {
            let value = json!({
                "upstream_id": cred.upstream_id, "policy": policy, "credential_id": cred.id,
                "vendor": {"prompt_cache_key": format!("ovgw-{}", &digest(&format!("{scope}{sid}"))[..40]),
                           "parameters": parameter_fingerprint(body)},
            });
            let store = ctx.store.clone();
            let (sc, s) = (scope.clone(), sid.clone());
            write(move || store.replay_put(&sc, &s, "root", "", &value)).await
        }
    };
    let policy = root["policy"].clone();
    validate_policy(&policy)?;
    let (kind, anchor) = text::classify(&original, headers, messages);
    let mut disabled = rec(&records, "disabled", "").is_some();
    if plugin && !disabled {
        let store = ctx.store.clone();
        let (sc, s) = (scope.clone(), sid.clone());
        write(move || {
            store.replay_put(
                &sc,
                &s,
                "disabled",
                "",
                &json!({"reason": "plugin_present"}),
            )
        })
        .await;
        disabled = true;
    }
    let owner = !disabled && matches!(kind, "user" | "continuation");

    // session_tools
    let used = chain.iter().any(|a| rec(&records, "hidden", a).is_some());
    let mut decision = rec(&records, "tools", "").cloned();
    let mut tools: Vec<Value> = vec![];
    if decision.is_none() && (owner || used) && policy["gateway_tools"].as_bool() == Some(true) {
        let (selected, reason) = match ctx.viking.tools(&cred.openviking_key).await {
            Ok(catalog) => (select_tools(&catalog, &policy), ""),
            Err(_) => (vec![], "tools_unavailable"),
        };
        let value = json!({"tools": selected, "tool_skip_reason": reason});
        let store = ctx.store.clone();
        let (sc, s) = (scope.clone(), sid.clone());
        let stored = write(move || store.replay_put(&sc, &s, "tools", "", &value)).await;
        rec_insert(&mut records, "tools", "", stored.clone());
        decision = Some(stored);
    }
    if let Some(d) = &decision {
        tools = d["tools"].as_array().cloned().unwrap_or_default();
    }
    let mut closed = rec(&records, "tools_closed", "").is_some();
    if !tools.is_empty() && !closed && owner && !used {
        let names: Vec<String> = tools
            .iter()
            .filter_map(|t| t["function"]["name"].as_str())
            .map(|n| n.trim_start_matches(PREFIX).to_string())
            .collect();
        if client_openviking_tools(&original, &tool_names(&names)) {
            let store = ctx.store.clone();
            let (sc, s) = (scope.clone(), sid.clone());
            write(move || {
                store.replay_put(
                    &sc,
                    &s,
                    "tools_closed",
                    "",
                    &json!({"reason": "tool_name_collision"}),
                )
            })
            .await;
            closed = true;
        }
    }
    if closed && !used {
        tools.clear();
    }

    let mut metrics = Map::new();
    metrics.insert("kind".into(), json!(kind));
    metrics.insert("session".into(), json!(sid));
    metrics.insert("protocol".into(), json!("anthropic"));
    metrics.insert("credential_id".into(), json!(cred.id));
    metrics.insert("recall_count".into(), json!(0));
    metrics.insert("recall_ms".into(), json!(0));

    // Outgoing messages: shared unless changed.
    let mut out_messages: Vec<Cow<Value>> = messages.iter().map(Cow::Borrowed).collect();

    // replay(): injections and missing-record check.
    let sent: HashSet<&str> = observation
        .value
        .get("sent")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut replay_hits = 0;
    let mut missing = false;
    for (index, a) in chain.iter().enumerate() {
        if let Some(d) = rec(&records, "injection", a) {
            if let Some(t) = d
                .get("text")
                .and_then(Value::as_str)
                .filter(|t| !t.is_empty())
            {
                append_context(out_messages[index].to_mut(), t);
            }
            replay_hits += 1;
        } else if sent.contains(a.as_str()) {
            missing = true;
        }
    }
    metrics.insert("replay_hits".into(), json!(replay_hits));
    if missing {
        metrics.insert("degradation".into(), json!("missing_injection_record"));
    }

    // configure_tools
    let mut tools_active = false;
    let mut tools_closed = false;
    let mut extra_tools: Vec<Value> = vec![];
    if !tools.is_empty() {
        let names: HashSet<&str> = tools
            .iter()
            .filter_map(|t| t["function"]["name"].as_str())
            .collect();
        let collision = body
            .get("tools")
            .and_then(Value::as_array)
            .map(|ts| {
                ts.iter().any(|t| {
                    names.contains(
                        t.get("function")
                            .unwrap_or(t)
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or(""),
                    )
                })
            })
            .unwrap_or(false);
        let forced = body
            .get("tool_choice")
            .and_then(|c| c.get("type"))
            .and_then(Value::as_str)
            .map(|t| t != "auto" && t != "none")
            .unwrap_or(false);
        let hidden = chain.iter().any(|a| rec(&records, "hidden", a).is_some());
        if collision || forced {
            metrics.insert("tool_skip_reason".into(), json!("tool_name_collision"));
        } else if owner || hidden {
            extra_tools = wire_tools(&tools);
            tools_active = true;
            tools_closed = !owner;
            metrics.insert(
                "tools_tokens".into(),
                json!(token_estimate(
                    &serde_json::to_string(&extra_tools).unwrap()
                )),
            );
        }
    }

    let mut prepared = Prepared {
        original: Value::Null,
        out: bytes::Bytes::new(),
        scope,
        session: sid,
        chain,
        kind,
        anchor,
        root,
        records,
        disabled,
        observation,
        capture,
        capture_target: String::new(),
        tools_active,
        tools_closed,
        tools,
        policy,
        metrics,
        reply_anchor: String::new(),
        relayed: false,
        context_usage: None,
        ancestors,
    };
    prepared
        .metrics
        .insert("thinking_known".into(), json!(thinking.len()));

    // capture confirm on user turns
    if !disabled
        && prepared.policy["capture"].as_bool() == Some(true)
        && kind == "user"
        && anchor >= 0
    {
        confirm_capture(ctx, &mut prepared, messages, cred).await;
    }

    // measure
    if matches!(kind, "user" | "continuation") {
        let usage = prepared
            .observation
            .value
            .get("usage")
            .cloned()
            .unwrap_or(json!({}));
        let index = usage
            .get("anchor")
            .and_then(Value::as_str)
            .and_then(|a| prepared.chain.iter().position(|c| c == a));
        let tokens = match index {
            Some(i) => {
                usage["input_tokens"].as_i64().unwrap_or(0) as usize
                    + usage["output_tokens"].as_i64().unwrap_or(0) as usize
                    + text::estimate_list(&messages[i + 1..])
            }
            None => {
                // The whole body the upstream would get.
                text::estimate(&original)
            }
        };
        prepared
            .metrics
            .insert("context_tokens".into(), json!(tokens));
        prepared
            .metrics
            .insert("context_window".into(), json!(DEFAULT_WINDOW));
    }

    // recall
    if !disabled
        && kind == "user"
        && anchor >= 0
        && rec(
            &prepared.records,
            "injection",
            &prepared.chain[anchor as usize],
        )
        .is_none()
    {
        if let Some(text) = recall(ctx, &mut prepared, messages, cred).await {
            append_context(out_messages[anchor as usize].to_mut(), &text);
        }
    }

    // assemble: hidden replay
    let out_messages = if prepared.tools_active {
        let mut result: Vec<Cow<Value>> = Vec::with_capacity(out_messages.len());
        let mut offsets = Vec::with_capacity(out_messages.len());
        let mut expanded = false;
        for (index, (message, a)) in out_messages
            .into_iter()
            .zip(prepared.chain.iter())
            .enumerate()
        {
            offsets.push(result.len());
            result.push(message);
            if let Some(record) = rec(&prepared.records, "hidden", a) {
                let count = record["visible_count"].as_u64().unwrap_or(0) as usize;
                if count > 0 && count <= index + 1 {
                    let start = offsets[index + 1 - count];
                    result.truncate(start);
                    if let Some(ms) = record["messages"].as_array() {
                        result.extend(ms.iter().map(|m| Cow::Owned(m.clone())));
                    }
                    expanded = true;
                }
            }
        }
        if expanded {
            let owned: Vec<Value> = result.into_iter().map(Cow::into_owned).collect();
            merge_tool_results(owned)
                .into_iter()
                .map(Cow::Owned)
                .collect()
        } else {
            // merge_tool_results walks the history either way.
            let _ = result
                .windows(2)
                .filter(|w| role(&w[0]) == Some("user") && role(&w[1]) == Some("user"))
                .count();
            result
        }
    } else {
        out_messages
    };

    // Serialize the upstream body: unchanged fields as sent, messages and tools edited.
    let mut out = Vec::with_capacity(body_size_hint(&original));
    out.push(b'{');
    let mut first = true;
    let mut wrote_tools = false;
    for (k, v) in body {
        if !first {
            out.push(b',');
        }
        first = false;
        write_str(&mut out, k);
        out.push(b':');
        match k.as_str() {
            "messages" => write_list(&mut out, out_messages.iter().map(|m| m.as_ref())),
            "tools" if !extra_tools.is_empty() => {
                wrote_tools = true;
                let mut all: Vec<&Value> =
                    v.as_array().map(|a| a.iter().collect()).unwrap_or_default();
                all.extend(extra_tools.iter());
                write_list(&mut out, all.into_iter());
            }
            _ => write_value(&mut out, v),
        }
    }
    if !wrote_tools && !extra_tools.is_empty() {
        out.extend_from_slice(b",\"tools\":");
        write_list(&mut out, extra_tools.iter());
    }
    out.push(b'}');
    drop(out_messages);
    prepared.out = bytes::Bytes::from(out);
    prepared.original = original;
    prepared.metrics.insert(
        "prepare_ms".into(),
        json!(started.elapsed().as_secs_f64() * 1000.0),
    );
    Ok(prepared)
}

fn body_size_hint(v: &Value) -> usize {
    v["messages"]
        .as_array()
        .map(|m| m.len() * 2048)
        .unwrap_or(0)
        + 256 * 1024
}

// ---------------------------------------------------------------- recall + profile

async fn recall(
    ctx: &Arc<Ctx>,
    request: &mut Prepared,
    messages: &[Value],
    cred: &Credential,
) -> Option<String> {
    let started = Instant::now();
    let policy = request.policy.clone();
    let existing: Vec<&Value> = request
        .chain
        .iter()
        .filter_map(|a| rec(&request.records, "injection", a))
        .collect();
    let used: i64 = existing
        .iter()
        .map(|v| v["tokens"].as_i64().unwrap_or(0))
        .sum();
    let exclude: Vec<String> = {
        let mut seen = HashSet::new();
        existing
            .iter()
            .flat_map(|v| v["uris"].as_array().cloned().unwrap_or_default())
            .filter_map(|u| u.as_str().map(str::to_string))
            .filter(|u| seen.insert(u.clone()))
            .collect()
    };
    let has_existing = !existing.is_empty();
    let budget = policy["max_tokens"]
        .as_i64()
        .unwrap()
        .min(policy["session_max_tokens"].as_i64().unwrap() - used);
    let anchor = request.anchor as usize;
    let query: String = clean_text(unwrap_client(&text_content(&messages[anchor])))
        .chars()
        .take(policy["query_max_chars"].as_u64().unwrap_or(8000) as usize)
        .collect();
    let mut reserved = 0;
    if policy["recall"].as_bool() == Some(true) && budget >= 64 && query.chars().count() >= 3 {
        // The window starts at the history's first user message.
        let window = messages
            .iter()
            .zip(&request.chain)
            .find(|(m, _)| is_user(m))
            .map(|(_, a)| a.clone())
            .unwrap_or_default();
        reserved = reserve_recall(ctx, request, &policy, used, &window).await;
    }
    let tools = request.tools.clone();
    let mut lead = "Relevant memory from OpenViking.".to_string();
    if tools
        .iter()
        .any(|t| t["function"]["name"] == "openviking_read")
    {
        lead.push_str(" Use the openviking_read tool to expand URIs.");
    }
    let overhead = token_estimate(&block("gateway-recall", &format!("{lead}\n"))) as i64;
    let key = cred.openviking_key.clone();
    let retrieve = async {
        if reserved <= 0 {
            return json!({"text": "", "uris": [], "tokens": 0, "reason": "disabled"});
        }
        if reserved - overhead < 64 {
            return json!({"text": "", "uris": [], "tokens": 0, "reason": "budget"});
        }
        match ctx
            .viking
            .recall(&key, &query, &policy, &exclude, reserved - overhead)
            .await
        {
            Ok(response) => {
                let rendered = response
                    .get("rendered")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let text = if rendered.is_empty() {
                    String::new()
                } else {
                    format!("{lead}\n{rendered}")
                };
                let uris: Vec<Value> = if text.is_empty() {
                    vec![]
                } else {
                    response["entries"]
                        .as_array()
                        .map(|e| e.iter().filter_map(|x| x.get("uri").cloned()).collect())
                        .unwrap_or_default()
                };
                let tokens = token_estimate(&block("gateway-recall", &text));
                let reason = if text.is_empty() { "empty" } else { "recalled" };
                json!({"text": text, "uris": uris, "tokens": tokens, "reason": reason})
            }
            Err(reason) => json!({"text": "", "uris": [], "tokens": 0, "reason": reason}),
        }
    };
    let opening = async {
        if has_existing {
            return String::new();
        }
        let profile = build_profile(ctx, &key, &policy, &tools).await;
        let note = gateway_note(&policy, &tools);
        block(
            "gateway-session-start",
            &[note, profile]
                .into_iter()
                .filter(|p| !p.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n"),
        )
    };
    let (mut decision, start) = tokio::join!(retrieve, opening);
    let recalled = block("gateway-recall", decision["text"].as_str().unwrap_or(""));
    let text = [start, recalled]
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    decision["text"] = json!(text);
    let anchor_id = request.chain[anchor].clone();
    let stored = {
        let store = ctx.store.clone();
        let (sc, s, a) = (
            request.scope.clone(),
            request.session.clone(),
            anchor_id.clone(),
        );
        write(move || store.replay_put(&sc, &s, "injection", &a, &decision)).await
    };
    if reserved > 0 {
        settle_recall(
            ctx,
            request,
            &anchor_id,
            stored["tokens"].as_i64().unwrap_or(0),
        )
        .await;
    }
    let text = stored["text"].as_str().unwrap_or("").to_string();
    request.metrics.insert(
        "recall_count".into(),
        json!(stored["uris"].as_array().map(Vec::len).unwrap_or(0)),
    );
    request.metrics.insert(
        "recall_ms".into(),
        json!(started.elapsed().as_secs_f64() * 1000.0),
    );
    request
        .metrics
        .insert("recall_reason".into(), stored["reason"].clone());
    rec_insert(&mut request.records, "injection", &anchor_id, stored);
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

async fn reserve_recall(
    ctx: &Arc<Ctx>,
    request: &mut Prepared,
    policy: &Value,
    used: i64,
    window: &str,
) -> i64 {
    let anchor = request.chain[request.anchor as usize].clone();
    let mut old = request.observation.clone();
    for _ in 0..8 {
        let mut ledger = old.value.get("recall").cloned().unwrap_or(json!({}));
        if ledger.get("window").and_then(Value::as_str) != Some(window) {
            ledger = json!({"window": window, "spent": used, "pending": {}});
        }
        if let Some(p) = ledger["pending"].get(&anchor) {
            request.observation = old;
            return p.as_i64().unwrap_or(0);
        }
        let spent = ledger["spent"].as_i64().unwrap_or(0);
        let budget = policy["max_tokens"]
            .as_i64()
            .unwrap()
            .min(policy["session_max_tokens"].as_i64().unwrap() - spent);
        if budget < 64 {
            return 0;
        }
        let mut value = old.value.clone();
        let mut pending = ledger["pending"].as_object().cloned().unwrap_or_default();
        pending.insert(anchor.clone(), json!(budget));
        value["recall"] = json!({"window": window, "spent": spent + budget, "pending": pending});
        let store = ctx.store.clone();
        let (sc, s, v, ver) = (
            request.scope.clone(),
            request.session.clone(),
            value.clone(),
            old.version,
        );
        if write(move || store.state_swap(&sc, &s, ver, &v)).await {
            request.observation = Document {
                value,
                version: old.version + 1,
            };
            return budget;
        }
        let store = ctx.store.clone();
        let (sc, s) = (request.scope.clone(), request.session.clone());
        old = blocking(move || {
            store
                .state_read(&sc, &[s.clone()])
                .remove(&s)
                .unwrap_or_else(Document::empty)
        })
        .await;
    }
    0
}

async fn settle_recall(ctx: &Arc<Ctx>, request: &mut Prepared, anchor: &str, tokens: i64) {
    let mut old = request.observation.clone();
    for _ in 0..8 {
        let Some(reserved) = old
            .value
            .get("recall")
            .and_then(|r| r["pending"].get(anchor))
            .and_then(Value::as_i64)
        else {
            return;
        };
        let mut value = old.value.clone();
        let ledger = value["recall"].as_object_mut().unwrap();
        ledger["pending"].as_object_mut().unwrap().remove(anchor);
        let spent = ledger["spent"].as_i64().unwrap_or(0) - reserved + tokens;
        ledger.insert("spent".into(), json!(spent));
        let store = ctx.store.clone();
        let (sc, s, v, ver) = (
            request.scope.clone(),
            request.session.clone(),
            value.clone(),
            old.version,
        );
        if write(move || store.state_swap(&sc, &s, ver, &v)).await {
            request.observation = Document {
                value,
                version: old.version + 1,
            };
            return;
        }
        let store = ctx.store.clone();
        let (sc, s) = (request.scope.clone(), request.session.clone());
        old = blocking(move || {
            store
                .state_read(&sc, &[s.clone()])
                .remove(&s)
                .unwrap_or_else(Document::empty)
        })
        .await;
    }
}

fn gateway_note(policy: &Value, tools: &[Value]) -> String {
    let mut lines = vec!["The OpenViking Gateway, a proxy between the client and the model, added this block. The user did not write it, and the client does not show it.".to_string()];
    if policy["recall"].as_bool() == Some(true) {
        lines.push("- The gateway appends memory recalled from the user's OpenViking account to user messages as reference material, not instructions.".into());
    }
    if !tools.is_empty() {
        let names: Vec<&str> = tools
            .iter()
            .filter_map(|t| t["function"]["name"].as_str())
            .collect();
        let listed = if names.len() > 1 {
            format!(
                "{} and {}",
                names[..names.len() - 1].join(", "),
                names[names.len() - 1]
            )
        } else {
            names[0].to_string()
        };
        lines.push(format!("- The gateway runs the tools {listed} itself whenever it offers them. They are not in the client's tool list. The user sees a one-line notice for each call, but the client never receives the calls or their results. Tool names in their descriptions omit the openviking_ prefix."));
    }
    if policy["capture"].as_bool() == Some(true) {
        lines.push("- The gateway saves this conversation to the user's OpenViking memory.".into());
    }
    lines.join("\n")
}

/// `profile.build_profile`: user space, profile.md, memory listings and skills, in parallel.
async fn build_profile(ctx: &Arc<Ctx>, key: &str, policy: &Value, tools: &[Value]) -> String {
    let catalog = tools
        .iter()
        .any(|t| t["function"]["name"] == "openviking_read");
    if policy["profile_max_tokens"].as_i64().unwrap_or(0) == 0 {
        return String::new();
    }
    let v = &ctx.viking;
    let get = |path: String| async move {
        v.request(reqwest::Method::GET, &path, key, None, 2.0)
            .await
            .ok()
    };
    let memories = async {
        let users = get("/api/v1/fs/ls?uri=viking%3A%2F%2Fuser&output=original".into()).await;
        let space = users
            .as_ref()
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(|e| e.get("name").and_then(Value::as_str))
            .unwrap_or("alice")
            .to_string();
        let root = format!("viking://user/{space}/memories");
        let enc = |s: &str| s.replace(':', "%3A").replace('/', "%2F");
        let profile = get(format!(
            "/api/v1/content/read?uri={}",
            enc(&format!("{root}/profile.md"))
        ));
        let ls = |kind: &str| {
            get(format!(
                "/api/v1/fs/ls?uri={}&output=agent&recursive=true&abs_limit=256",
                enc(&format!("{root}/{kind}"))
            ))
        };
        if catalog {
            let (p, pref, ent) = tokio::join!(profile, ls("preferences"), ls("entities"));
            (root, p, pref, ent)
        } else {
            (root, profile.await, None, None)
        }
    };
    let skills = async {
        if catalog {
            get("/api/v1/skills?node_limit=200".into()).await
        } else {
            None
        }
    };
    let ((root, profile, pref, ent), skills) = tokio::join!(memories, skills);
    let mut parts = vec![];
    if let Some(Value::String(p)) = profile {
        parts.push(format!(
            "<user-profile uri=\"{root}/profile.md\">\n{p}\n</user-profile>"
        ));
    }
    let mut listing = vec![];
    for (kind, value) in [("preferences", pref), ("entities", ent)] {
        if let Some(Value::Array(entries)) = value {
            let mut entries: Vec<(String, String)> = entries
                .iter()
                .filter(|e| e.get("isDir") != Some(&json!(true)))
                .filter_map(|e| {
                    Some((
                        e.get("name")?.as_str()?.to_string(),
                        e.get("abstract")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .trim()
                            .to_string(),
                    ))
                })
                .collect();
            entries.sort();
            if !entries.is_empty() {
                listing.push(format!("{root}/{kind}/"));
                listing.extend(entries.into_iter().map(|(n, a)| format!("- {n}: {a}")));
            }
        }
    }
    if !listing.is_empty() {
        parts.push(format!(
            "<available-memories>\n{}\n</available-memories>",
            listing.join("\n")
        ));
    }
    if let Some(s) = skills {
        let text = serde_json::to_string(&s).unwrap();
        if text.len() > 2 {
            parts.push(format!("<available-skills>\n{text}\n</available-skills>"));
        }
    }
    parts.join("\n")
}

// ---------------------------------------------------------------- capture

static NOTICE_LINES: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?m)^(?:> OpenViking \w+(?:: [^\n]+)? — (?:done|failed|skipped)|> OpenViking (?:context|recall|recall failed): [^\n]+)$\n*").unwrap()
});

fn new_capture() -> Value {
    json!({"ov_session": format!("gateway-{}", uuid::Uuid::new_v4().simple()), "delivered": "", "pending": [],
           "retained": [], "tokens": 0, "archive": null, "error": {}, "idle": false, "reason": "", "previous": []})
}

/// `capture_messages`: complete tool pairs in the server's ToolPart form.
fn capture_messages(messages: &[Value], chain: &[String]) -> Vec<Value> {
    let mut results: HashMap<&str, &Value> = HashMap::new();
    for m in messages {
        if let Some(Value::Array(blocks)) = m.get("content") {
            for b in blocks {
                if b.get("type").and_then(Value::as_str) == Some("tool_result") {
                    results.insert(
                        b.get("tool_use_id").and_then(Value::as_str).unwrap_or(""),
                        b.get("content").unwrap_or(&Value::Null),
                    );
                }
            }
        }
    }
    let mut output = vec![];
    for (m, a) in messages.iter().zip(chain) {
        let r = role(m);
        if !matches!(r, Some("user") | Some("assistant")) {
            continue;
        }
        let mut text: Cow<str> = text_content(m);
        if r == Some("assistant") {
            if let Cow::Owned(o) = NOTICE_LINES.replace_all(&text, "") {
                text = Cow::Owned(o);
            }
        }
        let text = clean_text(unwrap_client(&text));
        let mut parts = vec![];
        if !text.is_empty() {
            parts.push(json!({"type": "text", "text": text}));
        }
        if let Some(Value::Array(blocks)) = m.get("content") {
            for b in blocks
                .iter()
                .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
            {
                let id = b.get("id").and_then(Value::as_str).unwrap_or("");
                let Some(result) = results.get(id) else {
                    continue;
                };
                let output_text = match result {
                    Value::String(s) => clean_text(s),
                    other => clean_text(&serde_json::to_string(other).unwrap()),
                };
                parts.push(json!({"type": "tool", "tool_id": id, "tool_name": b.get("name").cloned().unwrap_or(json!("tool")),
                    "tool_input": b.get("input").cloned().unwrap_or(json!({})), "tool_status": "completed", "tool_output": output_text}));
            }
        }
        if !parts.is_empty() {
            output.push(
                json!({"role": r, "parts": parts, "source_message_ids": [format!("gateway:{a}")]}),
            );
        }
    }
    output
}

pub fn capture_turns(
    messages: &[Value],
    chain: &[String],
    start: usize,
    stop: usize,
    confirmed: bool,
    idle: f64,
) -> Vec<Value> {
    if start >= stop {
        return vec![];
    }
    let mut bounds = vec![start];
    bounds.extend((start + 1..stop).filter(|&i| is_user(&messages[i])));
    bounds.push(stop);
    let mut out = vec![];
    for w in bounds.windows(2) {
        let captured = capture_messages(&messages[w[0]..w[1]], &chain[w[0]..w[1]]);
        if let Some(last) = captured.last() {
            let anchor = last["source_message_ids"][0]
                .as_str()
                .unwrap_or("")
                .trim_start_matches("gateway:")
                .to_string();
            out.push(json!({"anchor": anchor, "messages": captured, "continued": !is_user(&messages[w[0]]),
                "confirmed": confirmed, "ready": if confirmed { 0.0 } else { now() + idle }}));
        }
    }
    out
}

/// `CapturePipeline.advance`: queue messages[:stop] after what is delivered or confirmed.
fn advance(
    value: &Value,
    messages: &[Value],
    chain: &[String],
    user: usize,
    stop: usize,
    confirmed: bool,
    idle: f64,
) -> (Value, usize) {
    let positions: HashMap<&str, usize> = chain
        .iter()
        .enumerate()
        .filter(|(_, a)| !a.is_empty())
        .map(|(i, a)| (a.as_str(), i))
        .collect();
    let mut value = value.clone();
    let held: Vec<Value> = value["pending"]
        .as_array()
        .map(|p| {
            p.iter()
                .filter(|t| t["confirmed"] == json!(true))
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    let mut ends: Vec<String> = vec![];
    if let Some(d) = value["delivered"].as_str().filter(|d| !d.is_empty()) {
        ends.push(d.to_string());
    }
    ends.extend(
        held.iter()
            .filter_map(|t| t["anchor"].as_str().map(str::to_string)),
    );
    let mut held = held;
    if ends.iter().any(|a| !positions.contains_key(a.as_str())) {
        let fresh = new_capture();
        for (k, v) in fresh.as_object().unwrap() {
            value[k] = v.clone();
        }
        value["reason"] = json!("history_changed");
        held.clear();
        ends.clear();
    }
    let start = ends
        .iter()
        .map(|a| positions[a.as_str()] as i64)
        .max()
        .unwrap_or(-1)
        + 1;
    let start = start as usize;
    let split = start.max(user).min(stop);
    let mut added = capture_turns(messages, chain, start.min(split), split, true, 0.0);
    added.extend(capture_turns(messages, chain, split, stop, confirmed, idle));
    let n = added.len();
    held.extend(added);
    value["pending"] = Value::Array(held);
    (value, n)
}

fn ready_at(value: &Value) -> Option<f64> {
    value["pending"]
        .as_array()
        .and_then(|p| p.first())
        .and_then(|t| t["ready"].as_f64())
}

async fn confirm_capture(
    ctx: &Arc<Ctx>,
    request: &mut Prepared,
    messages: &[Value],
    cred: &Credential,
) {
    let mut old = request.capture.clone();
    let user = request.anchor as usize;
    for _ in 0..4 {
        let base = if old.version == 0 {
            new_capture()
        } else {
            old.value.clone()
        };
        let (mut value, _) = advance(&base, messages, &request.chain, user, user, true, 0.0);
        value["request_anchor"] = json!(request.chain[user]);
        value["credential_id"] = json!(cred.id);
        value["account"] = json!(cred.account);
        value["user_id"] = json!(cred.user_id);
        value["protocol"] = json!("anthropic");
        value["policy"] = request.policy.clone();
        if value == old.value {
            break;
        }
        let ready = ready_at(&value);
        let store = ctx.store.clone();
        let (sc, s, v, ver) = (
            request.scope.clone(),
            request.session.clone(),
            value.clone(),
            old.version,
        );
        if write(move || store.capture_swap(&sc, &s, ver, &v, ready)).await {
            request.capture_target = value["ov_session"].as_str().unwrap_or("").to_string();
            request.capture = Document {
                value,
                version: old.version + 1,
            };
            return;
        }
        let store = ctx.store.clone();
        let (sc, s) = (request.scope.clone(), request.session.clone());
        old = blocking(move || store.capture_get(&sc, &s)).await;
    }
    request.capture_target = request.capture.value["ov_session"]
        .as_str()
        .unwrap_or("")
        .to_string();
}

/// `CapturePipeline.stage`: queue the finished reply as an unconfirmed, idle-ready tail.
async fn stage_capture(ctx: &Arc<Ctx>, request: &Prepared, output: &[Value]) {
    let mut messages: Vec<Value> = request.messages().to_vec();
    messages.extend(output.iter().cloned());
    let chain = prefix_chain(messages.iter());
    let idle = request.policy["idle_seconds"].as_f64().unwrap_or(600.0);
    for _ in 0..4 {
        let store = ctx.store.clone();
        let (sc, s) = (request.scope.clone(), request.session.clone());
        let old = blocking(move || store.capture_get(&sc, &s)).await;
        if old.value.get("ov_session").and_then(Value::as_str)
            != Some(request.capture_target.as_str())
            || old.value.get("request_anchor").and_then(Value::as_str)
                != Some(request.chain[request.anchor as usize].as_str())
        {
            return;
        }
        let (value, added) = advance(
            &old.value,
            &messages,
            &chain,
            request.anchor as usize,
            messages.len(),
            false,
            idle,
        );
        if added == 0 {
            return;
        }
        let ready = ready_at(&value);
        let store = ctx.store.clone();
        let (sc, s) = (request.scope.clone(), request.session.clone());
        if write(move || store.capture_swap(&sc, &s, old.version, &value, ready)).await {
            return;
        }
    }
}

/// `CaptureWorker.once`, simplified: deliver due turns to OV, then check pending tokens.
pub async fn capture_worker(ctx: Arc<Ctx>) {
    loop {
        let store = ctx.store.clone();
        let Some((scope, session, mut doc)) = write(move || store.capture_claim()).await else {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            continue;
        };
        let cred = ctx
            .credentials
            .values()
            .find(|c| Some(c.id.as_str()) == doc.value["credential_id"].as_str())
            .cloned();
        if let Some(cred) = cred {
            let key = cred.openviking_key.clone();
            let ov_session = doc.value["ov_session"].as_str().unwrap_or("").to_string();
            let _ = ctx
                .viking
                .request(reqwest::Method::POST, "/api/v1/sessions", &key, Some(&json!({"session_id": ov_session,
                    "auto_commit_policy": {"pending_token_threshold": 0, "message_count_threshold": 0, "idle_timeout_seconds": 0},
                    "memory_policy": {"working_memory": {"enabled": false}}})), 30.0)
                .await;
            loop {
                let Some(turn) = doc.value["pending"]
                    .as_array()
                    .and_then(|p| p.first())
                    .cloned()
                else {
                    break;
                };
                if turn["ready"].as_f64().unwrap_or(0.0) > now() {
                    break;
                }
                let msgs = turn["messages"].as_array().cloned().unwrap_or_default();
                for chunk in msgs.chunks(100) {
                    let _ = ctx
                        .viking
                        .request(
                            reqwest::Method::POST,
                            &format!("/api/v1/sessions/{ov_session}/messages/batch"),
                            &key,
                            Some(&json!({"messages": chunk})),
                            30.0,
                        )
                        .await;
                }
                let tokens: usize = msgs
                    .iter()
                    .map(|m| (serde_json::to_string(&m["parts"]).unwrap().len() + 2) / 3)
                    .sum();
                let mut value = doc.value.clone();
                value["retained"]
                    .as_array_mut()
                    .map(|r| r.push(json!({"anchor": turn["anchor"], "count": msgs.len()})));
                value["tokens"] = json!(value["tokens"].as_u64().unwrap_or(0) as usize + tokens);
                value["delivered"] = turn["anchor"].clone();
                value["pending"].as_array_mut().map(|p| p.remove(0));
                value["idle"] = json!(turn["confirmed"] != json!(true));
                let ready = ready_at(&value);
                let store = ctx.store.clone();
                let (sc, s, v, ver) = (scope.clone(), session.clone(), value.clone(), doc.version);
                if !write(move || store.capture_swap(&sc, &s, ver, &v, ready)).await {
                    break;
                }
                doc = Document {
                    value,
                    version: doc.version + 1,
                };
            }
            let _ = ctx
                .viking
                .request(
                    reqwest::Method::GET,
                    &format!("/api/v1/sessions/{ov_session}"),
                    &key,
                    None,
                    30.0,
                )
                .await;
        }
        let store = ctx.store.clone();
        write(move || store.capture_release(&scope, &session)).await;
    }
}

// ---------------------------------------------------------------- after the reply

/// `MemoryKernel.relayed`: record the reply anchor before the client sees the end.
pub async fn relayed(ctx: &Arc<Ctx>, request: &mut Prepared, output: &[Value], finished: bool) {
    if request.relayed {
        return;
    }
    if !output.is_empty() && matches!(request.kind, "user" | "continuation") {
        let chain = prefix_chain(request.messages().iter().chain(output.iter()));
        request.reply_anchor = chain.last().cloned().unwrap_or_default();
    }
    if finished && !request.reply_anchor.is_empty() {
        let store = ctx.store.clone();
        let (sc, s, a) = (
            request.scope.clone(),
            request.session.clone(),
            request.reply_anchor.clone(),
        );
        write(move || store.replay_put(&sc, &s, "reply", &a, &json!({}))).await;
    }
    request.relayed = true;
}

/// `MemoryKernel.completed`: observation update, capture stage, and the request log.
pub async fn completed(
    ctx: &Arc<Ctx>,
    request: &mut Prepared,
    cred: &Credential,
    output: &[Value],
    complete: bool,
) {
    let chain: HashSet<&str> = request.chain.iter().map(String::as_str).collect();
    let sent: Vec<String> = request
        .records
        .get("injection")
        .map(|m| {
            m.iter()
                .filter(|(a, v)| {
                    chain.contains(a.as_str())
                        && v["text"].as_str().map(|t| !t.is_empty()).unwrap_or(false)
                })
                .map(|(a, _)| a.clone())
                .collect()
        })
        .unwrap_or_default();
    let t = now();
    let mut usage = request.context_usage.clone().unwrap_or(json!({}));
    usage["anchor"] = json!(request.reply_anchor);
    usage["time"] = json!(t);
    let mut old = request.observation.clone();
    for _ in 0..4 {
        let mut value = old.value.clone();
        if !value.is_object() {
            value = json!({});
        }
        if usage["input_tokens"].as_i64().unwrap_or(0) > 0
            && !request.reply_anchor.is_empty()
            && t >= value
                .get("usage")
                .and_then(|u| u["time"].as_f64())
                .unwrap_or(0.0)
        {
            value["usage"] = usage.clone();
        }
        if !sent.is_empty() {
            let mut all: Vec<Value> = value
                .get("sent")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            for s in &sent {
                if !all.iter().any(|x| x == s) {
                    all.push(json!(s));
                }
            }
            value["sent"] = Value::Array(all);
        }
        if request.kind == "user"
            && t >= value.get("user_at").and_then(Value::as_f64).unwrap_or(0.0)
        {
            value["user_at"] = json!(t);
        }
        if value == old.value {
            break;
        }
        let store = ctx.store.clone();
        let (sc, s, v, ver) = (
            request.scope.clone(),
            request.session.clone(),
            value.clone(),
            old.version,
        );
        if write(move || store.state_swap(&sc, &s, ver, &v)).await {
            request.observation = Document {
                value,
                version: old.version + 1,
            };
            break;
        }
        let store = ctx.store.clone();
        let (sc, s) = (request.scope.clone(), request.session.clone());
        old = blocking(move || {
            store
                .state_read(&sc, &[s.clone()])
                .remove(&s)
                .unwrap_or_else(Document::empty)
        })
        .await;
    }
    if !request.disabled
        && matches!(request.kind, "user" | "continuation")
        && complete
        && request.anchor >= 0
        && request.policy["capture"].as_bool() == Some(true)
    {
        stage_capture(ctx, request, output).await;
    }
    let store = ctx.store.clone();
    let (account, user, metrics) = (
        cred.account.clone(),
        cred.user_id.clone(),
        Value::Object(request.metrics.clone()),
    );
    write(move || store.log(&account, &metrics, &user)).await;
}
