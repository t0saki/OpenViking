//! `protocols.normalize` + `prefix_chain`: canonical, key-sorted JSON of each message,
//! chained with SHA-256. The output bytes are meant to equal
//! `orjson.dumps(normalize(message), option=OPT_SORT_KEYS)` so anchors match the Python
//! gateway's (checked by the microbench on shared fixtures).
//!
//! Instead of materializing the normalized value the writer streams it straight into a
//! reusable buffer; the work per byte is the same as building and dumping it.

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::text::unwrap_client;

pub const NORMALIZATION_VERSION: &str = "v1";

#[inline]
fn is_text_type(t: Option<&str>) -> bool {
    matches!(t, Some("text") | Some("input_text") | Some("output_text"))
}

#[inline]
fn type_of(v: &Value) -> Option<&str> {
    v.get("type").and_then(Value::as_str)
}

/// Python `str.strip()` on the result of `unwrap_client`.
#[inline]
fn unwrap_strip(s: &str) -> &str {
    unwrap_client(s).trim()
}

/// List elements `normalize` drops: reasoning blocks and empty text blocks.
#[inline]
fn dropped(v: &Value) -> bool {
    match v {
        Value::Object(m) => {
            let t = m.get("type").and_then(Value::as_str);
            matches!(t, Some("thinking") | Some("redacted_thinking"))
                || (is_text_type(t)
                    && m.get("text")
                        .and_then(Value::as_str)
                        .map(|s| s.trim().is_empty())
                        .unwrap_or(true))
        }
        _ => false,
    }
}

/// Whether any of the 8 bytes in `w` needs escaping: '"', '\\' or a C0 control.
#[inline(always)]
fn needs_escape(w: u64) -> bool {
    const ONES: u64 = 0x0101_0101_0101_0101;
    const HIGH: u64 = 0x8080_8080_8080_8080;
    let quote = w ^ (ONES * b'"' as u64);
    let backslash = w ^ (ONES * b'\\' as u64);
    let zero = |x: u64| x.wrapping_sub(ONES) & !x & HIGH;
    // Bytes below 0x20 (ASCII only: high-bit bytes are masked out by !w).
    let control = w.wrapping_sub(ONES * 0x20) & !w & HIGH;
    (zero(quote) | zero(backslash) | control) != 0
}

static CONTROL: [[u8; 6]; 32] = {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut t = [[0u8; 6]; 32];
    let mut i = 0;
    while i < 32 {
        t[i] = [b'\\', b'u', b'0', b'0', HEX[i >> 4], HEX[i & 0xf]];
        i += 1;
    }
    t
};

#[inline(always)]
fn escape_of(b: u8) -> Option<&'static [u8]> {
    match b {
        b'"' => Some(b"\\\""),
        b'\\' => Some(b"\\\\"),
        b'\n' => Some(b"\\n"),
        b'\r' => Some(b"\\r"),
        b'\t' => Some(b"\\t"),
        0x08 => Some(b"\\b"),
        0x0c => Some(b"\\f"),
        0..=0x1f => Some(&CONTROL[b as usize]),
        _ => None,
    }
}

/// A JSON string, escaped like orjson/serde_json compact output: ", \ and C0 controls.
pub fn write_str(out: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let n = bytes.len();
    out.reserve(n + 2);
    out.push(b'"');
    let (mut start, mut i) = (0, 0);
    while i < n {
        // Skip clean 8-byte words; long text rarely needs escaping.
        while i + 8 <= n && !needs_escape(u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap())) {
            i += 8;
        }
        let end = (i + 8).min(n);
        while i < end {
            if let Some(esc) = escape_of(bytes[i]) {
                out.extend_from_slice(&bytes[start..i]);
                out.extend_from_slice(esc);
                start = i + 1;
            }
            i += 1;
        }
    }
    out.extend_from_slice(&bytes[start..]);
    out.push(b'"');
}

/// Insertion-ordered compact dump with the fast escaper (orjson's default layout).
pub fn write_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::Array(items) => {
            out.push(b'[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_value(out, item);
            }
            out.push(b']');
        }
        Value::Object(map) => {
            out.push(b'{');
            for (i, (k, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_str(out, k);
                out.push(b':');
                write_value(out, item);
            }
            out.push(b'}');
        }
        _ => write_scalar(out, v),
    }
}

pub fn write_list<'a>(out: &mut Vec<u8>, items: impl Iterator<Item = &'a Value>) {
    out.push(b'[');
    for (i, item) in items.enumerate() {
        if i > 0 {
            out.push(b',');
        }
        write_value(out, item);
    }
    out.push(b']');
}

pub fn to_string(v: &Value) -> String {
    // write_value only emits valid UTF-8 (input strings are UTF-8, escapes are ASCII).
    unsafe { String::from_utf8_unchecked(to_vec(v)) }
}

pub fn to_vec(v: &Value) -> Vec<u8> {
    let mut out = Vec::with_capacity(1024);
    write_value(&mut out, v);
    out
}

fn write_scalar(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(n) => out.extend_from_slice(n.to_string().as_bytes()),
        Value::String(s) => write_str(out, s),
        _ => unreachable!(),
    }
}

/// Key-sorted dump without normalization (`input`/`arguments` keep their values).
pub fn write_sorted(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::Array(items) => {
            out.push(b'[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_sorted(out, item);
            }
            out.push(b']');
        }
        Value::Object(map) => {
            let mut keys: Vec<(&String, &Value)> = map.iter().collect();
            keys.sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            out.push(b'{');
            for (i, (k, item)) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_str(out, k);
                out.push(b':');
                write_sorted(out, item);
            }
            out.push(b'}');
        }
        _ => write_scalar(out, v),
    }
}

enum Item<'a> {
    Norm(&'a Value),
    Raw(&'a Value),
    Str(&'a str),
}

fn write_norm_list(out: &mut Vec<u8>, items: &[Value]) {
    out.push(b'[');
    let mut first = true;
    for item in items {
        if dropped(item) {
            continue;
        }
        if !first {
            out.push(b',');
        }
        first = false;
        write_norm(out, item);
    }
    out.push(b']');
}

/// The single text a one-text-block list collapses to, after normalization.
fn single_text(items: &[Value]) -> Option<&str> {
    let mut kept = items.iter().filter(|v| !dropped(v));
    let block = kept.next()?;
    if kept.next().is_some() {
        return None;
    }
    let m = block.as_object()?;
    if !is_text_type(m.get("type").and_then(Value::as_str)) {
        return None;
    }
    // set(block) == {"type", "text"} once cache_control is gone.
    if m.keys().filter(|k| k.as_str() != "cache_control").count() != 2 || !m.contains_key("text") {
        return None;
    }
    Some(unwrap_strip(
        m.get("text").and_then(Value::as_str).unwrap_or(""),
    ))
}

fn write_norm_object(out: &mut Vec<u8>, map: &Map<String, Value>) {
    let mut entries: Vec<(&str, Item)> = Vec::with_capacity(map.len() + 1);
    let text_type = is_text_type(map.get("type").and_then(Value::as_str));
    let message = map.get("role").map(truthy).unwrap_or(false)
        && map.get("type").and_then(Value::as_str) == Some("message");
    for (k, v) in map {
        let k = k.as_str();
        match k {
            "cache_control" => continue,
            "input" | "arguments" => entries.push((k, Item::Raw(v))),
            "content" => {
                let item = match v {
                    Value::Array(items) => match single_text(items) {
                        Some(text) => Item::Str(unwrap_strip(text)),
                        None => Item::Norm(v),
                    },
                    Value::String(s) => Item::Str(unwrap_strip(s)),
                    _ => Item::Norm(v),
                };
                entries.push((k, item));
            }
            "text" if text_type => {
                entries.push((k, Item::Str(unwrap_strip(v.as_str().unwrap_or("")))))
            }
            "type" | "status" | "id" if message => continue,
            _ => entries.push((k, Item::Norm(v))),
        }
    }
    if text_type && !map.contains_key("text") {
        entries.push(("text", Item::Str("")));
    }
    entries.sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    out.push(b'{');
    for (i, (k, item)) in entries.into_iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        write_str(out, k);
        out.push(b':');
        match item {
            Item::Norm(v) => write_norm(out, v),
            Item::Raw(v) => write_sorted(out, v),
            Item::Str(s) => write_str(out, s),
        }
    }
    out.push(b'}');
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
    }
}

pub fn write_norm(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::Array(items) => write_norm_list(out, items),
        Value::Object(map) => write_norm_object(out, map),
        _ => write_scalar(out, v),
    }
}

/// Whether `normalize(message)["content"] == []`.
fn normalized_content_empty(message: &Value) -> bool {
    match message.get("content") {
        Some(Value::Array(items)) => items.iter().all(dropped),
        _ => false,
    }
}

pub fn prefix_chain<'a, I>(messages: I) -> Vec<String>
where
    I: IntoIterator<Item = &'a Value>,
{
    let mut previous: [u8; 32] = Sha256::digest(NORMALIZATION_VERSION.as_bytes()).into();
    let mut result = Vec::new();
    let mut buf = Vec::with_capacity(16 * 1024);
    for message in messages {
        let role = message.get("role").and_then(Value::as_str);
        if matches!(role, Some("system") | Some("developer"))
            || type_of(message) == Some("additional_tools")
        {
            result.push(String::new());
            continue;
        }
        if role == Some("assistant")
            && normalized_content_empty(message)
            && !message.get("tool_calls").map(truthy).unwrap_or(false)
        {
            result.push(String::new());
            continue;
        }
        buf.clear();
        write_norm(&mut buf, message);
        let mut h = Sha256::new();
        h.update(previous);
        h.update(&buf);
        previous = h.finalize().into();
        result.push(hex::encode(previous));
    }
    result
}

pub fn digest(value: &str) -> String {
    hex::encode(Sha256::digest(value.as_bytes()))
}
