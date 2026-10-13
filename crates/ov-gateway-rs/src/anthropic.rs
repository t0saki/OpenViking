//! `SSEDecoder` + `AnthropicProtocol` (tool_protocols/anthropic.py): assemble the upstream
//! reply from its events, project client-visible events (tool blocks buffered until
//! validated, indices remapped), and encode them back to SSE bytes.

use std::collections::{BTreeMap, HashMap, HashSet};

use bytes::Bytes;
use memchr::memmem;
use serde_json::{json, Map, Value};

pub const PREFIX: &str = "openviking_";

#[derive(Default)]
pub struct SseDecoder {
    pub buffer: Vec<u8>,
}

impl SseDecoder {
    /// Complete frames in `chunk`, split at a blank line (LF or CRLF).
    pub fn feed(&mut self, chunk: &[u8], frames: &mut Vec<Vec<u8>>) {
        self.buffer.extend_from_slice(chunk);
        let mut start = 0;
        loop {
            let rest = &self.buffer[start..];
            let lf = memmem::find(rest, b"\n\n").map(|i| (i, i + 2));
            let crlf = memmem::find(rest, b"\n\r\n").map(|i| (i, i + 3));
            let end = match (lf, crlf) {
                (Some(a), Some(b)) => {
                    if a.0 <= b.0 {
                        a.1
                    } else {
                        b.1
                    }
                }
                (Some(a), None) => a.1,
                (None, Some(b)) => b.1,
                (None, None) => break,
            };
            frames.push(rest[..end].to_vec());
            start += end;
        }
        if start > 0 {
            self.buffer.drain(..start);
        }
    }

    pub fn data(frame: &[u8]) -> Option<Value> {
        let mut data: Vec<u8> = Vec::new();
        let mut first = true;
        for line in frame.split(|&b| b == b'\n') {
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            if let Some(rest) = line.strip_prefix(b"data:") {
                if !first {
                    data.push(b'\n');
                }
                first = false;
                let rest = match rest.iter().position(|&b| b != b' ') {
                    Some(i) => &rest[i..],
                    None => &rest[rest.len()..],
                };
                data.extend_from_slice(rest);
            }
        }
        match serde_json::from_slice::<Value>(&data) {
            Ok(v @ Value::Object(_)) => Some(v),
            _ => None,
        }
    }
}

pub fn encode(value: &Value) -> Bytes {
    let kind = value.get("type").and_then(Value::as_str).unwrap_or("");
    let mut out = Vec::with_capacity(64 + kind.len());
    out.extend_from_slice(b"event: ");
    out.extend_from_slice(kind.as_bytes());
    out.extend_from_slice(b"\ndata: ");
    crate::canon::write_value(&mut out, value);
    out.extend_from_slice(b"\n\n");
    Bytes::from(out)
}

#[derive(Clone, Debug)]
pub struct Call {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

pub struct Round {
    pub output: Vec<Value>,
    pub calls: Vec<Call>,
    pub usage: Map<String, Value>,
    pub tool_stop: bool,
}

pub fn add_usage(total: &mut Map<String, Value>, usage: &Map<String, Value>) {
    for (k, v) in usage {
        match v {
            Value::Object(inner) => {
                let entry = total
                    .entry(k.clone())
                    .or_insert_with(|| Value::Object(Map::new()));
                if let Value::Object(t) = entry {
                    add_usage(t, inner);
                }
            }
            Value::Number(n) => {
                let prev = total.get(k).and_then(Value::as_f64).unwrap_or(0.0);
                let sum = prev + n.as_f64().unwrap_or(0.0);
                total.insert(
                    k.clone(),
                    if n.is_f64() {
                        json!(sum)
                    } else {
                        json!(sum as i64)
                    },
                );
            }
            other => {
                total.insert(k.clone(), other.clone());
            }
        }
    }
}

pub struct Anthropic {
    pub identifier: String,
    /// Client-visible content of the single visible assistant message.
    pub visible: Vec<Value>,
    started: bool,
    pub lead: String,
    blocks: BTreeMap<usize, Value>,
    arguments: HashMap<usize, String>,
    indices: HashMap<usize, usize>,
    opened: bool,
    stopped: bool,
    closed: HashSet<usize>,
    pub envelope: Map<String, Value>,
    pub usage: Map<String, Value>,
    pub finish: Option<String>,
}

impl Anthropic {
    pub fn new() -> Self {
        Anthropic {
            identifier: format!("msg_ovgw-{}", uuid::Uuid::new_v4().simple()),
            visible: Vec::new(),
            started: false,
            lead: String::new(),
            blocks: BTreeMap::new(),
            arguments: HashMap::new(),
            indices: HashMap::new(),
            opened: false,
            stopped: false,
            closed: HashSet::new(),
            envelope: Map::new(),
            usage: Map::new(),
            finish: None,
        }
    }

    pub fn begin(&mut self) {
        self.blocks.clear();
        self.arguments.clear();
        self.indices.clear();
        self.opened = false;
        self.stopped = false;
        self.closed.clear();
        self.envelope = Map::new();
        self.usage = Map::new();
        self.finish = None;
    }

    fn update_usage(&mut self, usage: Option<&Value>) {
        if let Some(Value::Object(u)) = usage {
            for (k, v) in u {
                self.usage.insert(k.clone(), v.clone());
            }
        }
    }

    /// `accumulate` + `event`: consume one upstream event, return client-visible events.
    pub fn event(&mut self, mut value: Value) -> Result<Vec<Value>, String> {
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        match kind.as_str() {
            "message_start" => {
                let message = value
                    .get("message")
                    .and_then(Value::as_object)
                    .cloned()
                    .ok_or("bad message_start")?;
                self.envelope = message;
                let u = self.envelope.get("usage").cloned();
                self.update_usage(u.as_ref());
                self.opened = true;
                if !self.started {
                    self.started = true;
                    let mut message = self.envelope.clone();
                    message.insert("id".into(), Value::String(self.identifier.clone()));
                    message.insert("content".into(), Value::Array(vec![]));
                    value["message"] = Value::Object(message);
                    let mut events = vec![value];
                    events.extend(self.lead_notice());
                    return Ok(events);
                }
                Ok(vec![])
            }
            "content_block_start" => {
                let index = value
                    .get("index")
                    .and_then(Value::as_u64)
                    .ok_or("bad index")? as usize;
                let block = value.get("content_block").cloned().ok_or("bad block")?;
                let tool = block.get("type").and_then(Value::as_str) == Some("tool_use");
                if !tool {
                    self.indices.insert(index, self.visible.len());
                    self.visible.push(block.clone());
                }
                self.blocks.insert(index, block);
                if !tool {
                    value["index"] = json!(self.indices[&index]);
                    return Ok(vec![value]);
                }
                Ok(vec![])
            }
            "content_block_delta" => {
                let index = value
                    .get("index")
                    .and_then(Value::as_u64)
                    .ok_or("bad index")? as usize;
                {
                    let delta = value
                        .get("delta")
                        .and_then(Value::as_object)
                        .ok_or("bad delta")?;
                    let block = self.blocks.get_mut(&index).ok_or("unknown block")?;
                    if let Some(p) = delta.get("partial_json").and_then(Value::as_str) {
                        self.arguments.entry(index).or_default().push_str(p);
                    } else {
                        for key in ["text", "thinking", "signature"] {
                            if let Some(add) = delta.get(key).and_then(Value::as_str) {
                                match block.get_mut(key) {
                                    Some(Value::String(s)) => s.push_str(add),
                                    _ => {
                                        block[key] = Value::String(add.to_string());
                                    }
                                }
                            }
                        }
                        if delta.get("type").and_then(Value::as_str) == Some("citations_delta") {
                            if let Some(c) = delta.get("citation") {
                                let entry = block
                                    .as_object_mut()
                                    .unwrap()
                                    .entry("citations")
                                    .or_insert_with(|| json!([]));
                                entry.as_array_mut().unwrap().push(c.clone());
                            }
                        }
                    }
                }
                if let Some(&mapped) = self.indices.get(&index) {
                    value["index"] = json!(mapped);
                    return Ok(vec![value]);
                }
                Ok(vec![])
            }
            "content_block_stop" => {
                let index = value
                    .get("index")
                    .and_then(Value::as_u64)
                    .ok_or("bad index")? as usize;
                self.closed.insert(index);
                if let Some(args) = self.arguments.get(&index) {
                    if !args.is_empty() {
                        let input: Value = serde_json::from_str(args).map_err(|e| e.to_string())?;
                        self.blocks.get_mut(&index).ok_or("unknown block")?["input"] = input;
                    }
                }
                if let Some(&mapped) = self.indices.get(&index) {
                    // Python's visible list shares the block object; copy it once complete.
                    self.visible[mapped] = self.blocks[&index].clone();
                    value["index"] = json!(mapped);
                    return Ok(vec![value]);
                }
                Ok(vec![])
            }
            "message_delta" => {
                if let Some(Value::Object(delta)) = value.get("delta") {
                    if let Some(stop) = delta.get("stop_reason").and_then(Value::as_str) {
                        self.finish = Some(stop.to_string());
                    }
                    for (k, v) in delta {
                        self.envelope.insert(k.clone(), v.clone());
                    }
                }
                let u = value.get("usage").cloned();
                self.update_usage(u.as_ref());
                Ok(vec![])
            }
            "message_stop" => {
                self.stopped = true;
                Ok(vec![])
            }
            _ => Ok(vec![value]),
        }
    }

    pub fn end(&mut self) -> Result<Round, String> {
        let closed_all = self.blocks.keys().all(|k| self.closed.contains(k))
            && self.closed.len() == self.blocks.len();
        if !self.stopped || self.finish.is_none() || !closed_all {
            return Err("Incomplete Anthropic message".into());
        }
        let blocks: Vec<Value> = self.blocks.values().cloned().collect();
        let calls = blocks
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
            .map(|b| Call {
                id: b["id"].as_str().unwrap_or("").to_string(),
                name: b["name"].as_str().unwrap_or("").to_string(),
                arguments: serde_json::to_string(b.get("input").unwrap_or(&json!({}))).unwrap(),
            })
            .collect();
        Ok(Round {
            output: vec![json!({"role": "assistant", "content": blocks})],
            calls,
            usage: self.usage.clone(),
            tool_stop: self.finish.as_deref() == Some("tool_use"),
        })
    }

    pub fn publish_calls(&mut self, client: &[Call]) -> Vec<Value> {
        let ids: HashSet<&str> = client.iter().map(|c| c.id.as_str()).collect();
        let mut events = Vec::new();
        let tools: Vec<Value> = self
            .blocks
            .values()
            .filter(|b| {
                b.get("type").and_then(Value::as_str) == Some("tool_use")
                    && ids.contains(b.get("id").and_then(Value::as_str).unwrap_or(""))
            })
            .cloned()
            .collect();
        for block in tools {
            let index = self.visible.len();
            let input = serde_json::to_string(block.get("input").unwrap_or(&json!({}))).unwrap();
            let mut start = block.clone();
            start["input"] = json!({});
            self.visible.push(block);
            events.push(
                json!({"type": "content_block_start", "index": index, "content_block": start}),
            );
            events.push(json!({"type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta", "partial_json": input}}));
            events.push(json!({"type": "content_block_stop", "index": index}));
        }
        events
    }

    pub fn open_notice(&mut self) -> Vec<Value> {
        self.visible.push(json!({"type": "text", "text": ""}));
        vec![
            json!({"type": "content_block_start", "index": self.visible.len() - 1,
            "content_block": {"type": "text", "text": ""}}),
        ]
    }

    pub fn notice(&mut self, text: &str) -> Vec<Value> {
        let last = self.visible.len() - 1;
        if let Some(Value::String(s)) = self.visible[last].get_mut("text") {
            s.push_str(text);
        }
        vec![
            json!({"type": "content_block_delta", "index": last, "delta": {"type": "text_delta", "text": text}}),
        ]
    }

    pub fn close_notice(&mut self) -> Vec<Value> {
        vec![json!({"type": "content_block_stop", "index": self.visible.len() - 1})]
    }

    pub fn lead_notice(&mut self) -> Vec<Value> {
        let text = std::mem::take(&mut self.lead);
        if text.is_empty() {
            return vec![];
        }
        let mut events = self.open_notice();
        events.extend(self.notice(&text));
        events.extend(self.close_notice());
        events
    }

    pub fn results(results: &[(String, String, bool)]) -> Vec<Value> {
        let content: Vec<Value> = results
            .iter()
            .map(|(id, content, failed)| {
                let mut b = json!({"type": "tool_result", "tool_use_id": id, "content": content});
                if *failed {
                    b["is_error"] = json!(true);
                }
                b
            })
            .collect();
        vec![json!({"role": "user", "content": content})]
    }

    pub fn final_message(&self, usage: &Map<String, Value>) -> Value {
        let mut out = self.envelope.clone();
        out.insert("id".into(), json!(self.identifier));
        out.insert("type".into(), json!("message"));
        out.insert("role".into(), json!("assistant"));
        out.insert("content".into(), Value::Array(self.visible.clone()));
        out.insert("stop_reason".into(), json!(self.finish));
        out.insert("usage".into(), Value::Object(usage.clone()));
        Value::Object(out)
    }

    pub fn terminal(&self, final_message: &Value) -> Vec<Value> {
        vec![
            json!({"type": "message_delta",
                "delta": {"stop_reason": self.finish, "stop_sequence": final_message.get("stop_sequence")},
                "usage": final_message["usage"]}),
            json!({"type": "message_stop"}),
        ]
    }

    pub fn error(message: &str) -> Value {
        json!({"type": "error", "error": {"type": "gateway_tool_error", "message": message}})
    }
}

impl Default for Anthropic {
    fn default() -> Self {
        Self::new()
    }
}

/// `merge_tool_results`: mixed calls must get all results in the next user message.
pub fn merge_tool_results(messages: Vec<Value>) -> Vec<Value> {
    let mut result: Vec<Value> = Vec::with_capacity(messages.len());
    for message in messages {
        let merge = match result.last() {
            Some(prior) => {
                prior.get("role").and_then(Value::as_str) == Some("user")
                    && message.get("role").and_then(Value::as_str) == Some("user")
                    && matches!((prior.get("content"), message.get("content")), (Some(Value::Array(p)), Some(Value::Array(m)))
                        if p.iter().all(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"))
                        && m.iter().any(|b| b.get("type").and_then(Value::as_str) == Some("tool_result")))
            }
            None => false,
        };
        if merge {
            let prior = result.last_mut().unwrap();
            let extra = message["content"].as_array().cloned().unwrap_or_default();
            prior["content"].as_array_mut().unwrap().extend(extra);
        } else {
            result.push(message);
        }
    }
    result
}
