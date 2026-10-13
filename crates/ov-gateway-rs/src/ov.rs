//! OpenViking HTTP client (client.py subset): identify (cached 30 s), MCP tools/list
//! (cached 300 s), tools/call, context search, profile reads and capture writes.

use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::anthropic::SseDecoder;

pub struct Viking {
    pub http: reqwest::Client,
    pub base: String,
    identities: Mutex<std::collections::HashMap<String, (Instant, Value)>>,
    tools: Mutex<Option<(Instant, Value)>>,
}

pub type OvResult<T> = Result<T, String>;

impl Viking {
    pub fn new(http: reqwest::Client, base: &str) -> Self {
        Viking {
            http,
            base: base.trim_end_matches('/').to_string(),
            identities: Mutex::new(Default::default()),
            tools: Mutex::new(None),
        }
    }

    pub async fn request(
        &self,
        method: reqwest::Method,
        path: &str,
        key: &str,
        body: Option<&Value>,
        timeout: f64,
    ) -> OvResult<Value> {
        let mut req = self
            .http
            .request(method, format!("{}{}", self.base, path))
            .header("X-API-Key", key)
            .header("Accept-Encoding", "identity")
            .timeout(Duration::from_secs_f64(timeout));
        if let Some(b) = body {
            req = req
                .header("content-type", "application/json")
                .body(serde_json::to_vec(b).unwrap());
        }
        let resp = req
            .send()
            .await
            .map_err(|_| "openviking_unavailable".to_string())?;
        let status = resp.status().as_u16();
        let bytes = resp
            .bytes()
            .await
            .map_err(|_| "openviking_unavailable".to_string())?;
        if status >= 300 {
            return Err(format!("openviking_http_{status}"));
        }
        let data: Value =
            serde_json::from_slice(&bytes).map_err(|_| "openviking_unavailable".to_string())?;
        if data.get("status").and_then(Value::as_str) == Some("error") {
            return Err("openviking_error".into());
        }
        Ok(data.get("result").cloned().unwrap_or(data))
    }

    pub async fn identify(&self, key: &str) -> OvResult<Value> {
        let digest = crate::canon::digest(key);
        if let Some((at, v)) = self.identities.lock().get(&digest) {
            if at.elapsed() < Duration::from_secs(30) {
                return Ok(v.clone());
            }
        }
        let v = self
            .request(reqwest::Method::GET, "/health", key, None, 5.0)
            .await?;
        self.identities
            .lock()
            .insert(digest, (Instant::now(), v.clone()));
        Ok(v)
    }

    pub async fn mcp(
        &self,
        method: &str,
        key: &str,
        params: Value,
        timeout: f64,
    ) -> OvResult<Value> {
        let resp = self
            .http
            .post(format!("{}/mcp", self.base))
            .header("X-API-Key", key)
            .header("Accept", "application/json, text/event-stream")
            .header("MCP-Protocol-Version", "2025-06-18")
            .header("content-type", "application/json")
            .body(
                serde_json::to_vec(
                    &json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}),
                )
                .unwrap(),
            )
            .timeout(Duration::from_secs_f64(timeout))
            .send()
            .await
            .map_err(|_| "openviking_unavailable".to_string())?;
        if resp.status().as_u16() >= 300 {
            return Err(format!("openviking_http_{}", resp.status().as_u16()));
        }
        let sse = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|v| v.contains("text/event-stream"))
            .unwrap_or(false);
        let bytes = resp
            .bytes()
            .await
            .map_err(|_| "openviking_unavailable".to_string())?;
        if sse {
            let mut decoder = SseDecoder::default();
            let mut frames = Vec::new();
            decoder.feed(&bytes, &mut frames);
            for f in frames {
                if let Some(v) = SseDecoder::data(&f) {
                    if v.get("id") == Some(&json!(1)) {
                        return v
                            .get("result")
                            .cloned()
                            .ok_or_else(|| "openviking_mcp_error".to_string());
                    }
                }
            }
            return Err("openviking_mcp_incomplete".into());
        }
        let v: Value =
            serde_json::from_slice(&bytes).map_err(|_| "openviking_unavailable".to_string())?;
        v.get("result")
            .cloned()
            .ok_or_else(|| "openviking_mcp_error".to_string())
    }

    pub async fn tools(&self, key: &str) -> OvResult<Value> {
        if let Some((at, v)) = self.tools.lock().as_ref() {
            if at.elapsed() < Duration::from_secs(300) {
                return Ok(v.clone());
            }
        }
        let result = self.mcp("tools/list", key, json!({}), 5.0).await?;
        let tools = result
            .get("tools")
            .cloned()
            .ok_or("openviking_mcp_invalid_catalog")?;
        *self.tools.lock() = Some((Instant::now(), tools.clone()));
        Ok(tools)
    }

    pub async fn recall(
        &self,
        key: &str,
        query: &str,
        policy: &Value,
        exclude: &[String],
        budget: i64,
    ) -> OvResult<Value> {
        let start = exclude.len().saturating_sub(200);
        self.request(
            reqwest::Method::POST,
            "/api/v1/search/search",
            key,
            Some(&json!({
                "query": query, "mode": "context", "query_expansion": "off",
                "context_type": policy["context_types"], "max_tokens": budget,
                "quotas": Value::Null, "score_threshold": policy["score_threshold"],
                "exclude_uris": &exclude[start..], "rewrite": false,
            })),
            policy["recall_timeout"].as_f64().unwrap_or(2.0),
        )
        .await
    }
}
