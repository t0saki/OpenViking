//! Text helpers from `protocols.py`, `blocks.py` and `compaction.py`: classification,
//! plugin detection, client OpenViking tool detection, cleaning and token estimates.
//! Regexes are kept one-for-one with the Python ones (no merging), so the comparison
//! reflects engine/language cost rather than a different algorithm.

use std::borrow::Cow;

use once_cell::sync::Lazy;
use regex::{Regex, RegexBuilder};
use serde_json::Value;

pub const PLUGIN_TAGS: &[&str] = &[
    "openviking-context",
    "relevant-memories",
    "relevant-memory",
    "memory-context",
];
pub const NOISE_TAGS: &[&str] = &[
    "openviking-context",
    "relevant-memories",
    "relevant-memory",
    "memory-context",
    "system-reminder",
    "local-command-caveat",
    "local-command-stdout",
    "command-name",
    "command-message",
    "command-args",
    "environment_context",
    "environment_details",
    "skills_instructions",
    "available_skills",
];
pub const AUXILIARY_PATTERNS: &[&str] = &[
    "Please provide a detailed summary of the conversation",
    "Your task is to create a detailed summary of the conversation",
    "You are a helpful AI assistant tasked with summarizing",
    "Generate a concise.{0,40}title",
    "Create a short.{0,40}title",
    "You are a permission classifier",
];

fn ci(pattern: &str) -> Regex {
    RegexBuilder::new(pattern)
        .case_insensitive(true)
        .build()
        .unwrap()
}

static NOISE: Lazy<Vec<Regex>> = Lazy::new(|| {
    NOISE_TAGS
        .iter()
        .map(|t| ci(&format!(r"(?s)<{t}\b[^>]*>.*?</{t}>")))
        .collect()
});
static SUBAGENT: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^\s*\[Subagent Context\][^\n]*").unwrap());
static USER_QUERY: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)<user_query>\s*(.*?)\s*</user_query>").unwrap());
static AUXILIARY: Lazy<Vec<Regex>> =
    Lazy::new(|| AUXILIARY_PATTERNS.iter().map(|p| ci(p)).collect());
static HINTS: Lazy<Regex> =
    Lazy::new(|| ci("compact|summar|title|classif|heartbeat|suggest|permission"));
static OV_CONTEXT_TAG: Lazy<Regex> = Lazy::new(|| ci(r"<openviking-context\b[^>]*>"));
static ATTRS: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"([\w-]+)\s*=\s*(?:'([^']*)'|"([^"]*)")"#).unwrap());
static PLUGIN: Lazy<Vec<Regex>> = Lazy::new(|| {
    PLUGIN_TAGS
        .iter()
        .map(|t| ci(&format!(r"<{}(?:\s|>)", regex::escape(t))))
        .collect()
});
static MEDIA: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Za-z0-9+/=]{1024,}").unwrap());
static OPENVIKING: Lazy<Regex> = Lazy::new(|| ci("openviking"));

pub fn unwrap_client(text: &str) -> &str {
    // Python runs the regex first; the cheap containment test is equivalent here.
    if let Some(c) = USER_QUERY.captures(text) {
        if text.contains("<context>") {
            return c.get(1).unwrap().as_str();
        }
    }
    text
}

pub fn clean_text(text: &str) -> String {
    let mut s: Cow<str> = Cow::Borrowed(text);
    for re in NOISE.iter() {
        if let Cow::Owned(o) = re.replace_all(&s, " ") {
            s = Cow::Owned(o);
        }
    }
    if let Cow::Owned(o) = SUBAGENT.replace_all(&s, "") {
        s = Cow::Owned(o);
    }
    s.replace('\0', "").trim().to_string()
}

fn is_text_block(b: &Value) -> bool {
    matches!(
        b.get("type").and_then(Value::as_str),
        Some("text") | Some("input_text") | Some("output_text")
    )
}

pub fn text_content(message: &Value) -> Cow<'_, str> {
    match message.get("content") {
        None => Cow::Borrowed(""),
        Some(Value::String(s)) => Cow::Borrowed(s),
        Some(Value::Array(items)) => {
            let mut parts =
                items
                    .iter()
                    .filter(|b| is_text_block(b))
                    .filter_map(|b| match b.get("text") {
                        None => Some(""),
                        Some(Value::String(s)) => Some(s.as_str()),
                        _ => None,
                    });
            let first = match parts.next() {
                None => return Cow::Borrowed(""),
                Some(f) => f,
            };
            match parts.next() {
                None => Cow::Borrowed(first),
                Some(second) => {
                    let mut out = String::with_capacity(first.len() + second.len() + 1);
                    out.push_str(first);
                    out.push('\n');
                    out.push_str(second);
                    for p in parts {
                        out.push('\n');
                        out.push_str(p);
                    }
                    Cow::Owned(out)
                }
            }
        }
        _ => Cow::Borrowed(""),
    }
}

pub fn has_tool_result(message: &Value) -> bool {
    message
        .get("content")
        .and_then(Value::as_array)
        .map(|c| {
            c.iter()
                .any(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"))
        })
        .unwrap_or(false)
}

pub fn is_user(message: &Value) -> bool {
    message.get("role").and_then(Value::as_str) == Some("user")
        && !clean_text(unwrap_client(&text_content(message))).is_empty()
        && !has_tool_result(message)
}

pub type Headers = Vec<(String, String)>;

pub fn header<'a>(headers: &'a Headers, key: &str) -> &'a str {
    headers
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .unwrap_or("")
}

/// Python's `str(body.get("system", ""))`: a repr of the value. The JSON text has the
/// same content and about the same length, which is what the regexes pay for.
fn probe_text(v: Option<&Value>) -> Cow<'_, str> {
    match v {
        None => Cow::Borrowed(""),
        Some(Value::String(s)) => Cow::Borrowed(s),
        Some(other) => Cow::Owned(crate::canon::to_string(other)),
    }
}

pub fn classify(body: &Value, headers: &Headers, messages: &[Value]) -> (&'static str, i64) {
    let anchor = (0..messages.len())
        .rev()
        .find(|&i| is_user(&messages[i]))
        .map(|i| i as i64)
        .unwrap_or(-1);
    let request_class = header(headers, "x-claude-code-request-class");
    if matches!(request_class, "subagent" | "workflow") {
        return ("subagent", anchor);
    }
    if matches!(request_class, "compaction" | "auxiliary") {
        return ("auxiliary", anchor);
    }
    if [
        "x-claude-code-agent-id",
        "x-openai-subagent",
        "x-codex-parent-thread-id",
    ]
    .iter()
    .any(|k| !header(headers, k).is_empty())
    {
        return ("subagent", anchor);
    }
    let hints = [
        "x-openviking-task",
        "x-claude-code-request-type",
        "x-claude-code-request-kind",
        "x-claude-code-agent-type",
        "x-codex-turn-metadata",
    ]
    .iter()
    .map(|k| header(headers, k))
    .collect::<Vec<_>>()
    .join(" ");
    if HINTS.is_match(&hints) {
        return ("auxiliary", anchor);
    }
    let mut probe = String::new();
    probe.push_str(&probe_text(body.get("system")));
    probe.push_str(&probe_text(body.get("instructions")));
    if anchor >= 0 {
        probe.push_str(&text_content(&messages[anchor as usize]));
    }
    if AUXILIARY.iter().any(|re| re.is_match(&probe)) {
        return ("auxiliary", anchor);
    }
    if anchor < 0 {
        return ("continuation", anchor);
    }
    let tail = &messages[anchor as usize + 1..];
    if tail.iter().any(|m| {
        let role = m.get("role").and_then(Value::as_str);
        !matches!(role, Some("system") | Some("developer"))
            && !(role == Some("user") && !is_user(m))
    }) {
        return ("continuation", anchor);
    }
    ("user", anchor)
}

/// The system prompt and client-written messages.
pub fn client_texts(body: &Value) -> Vec<Cow<'_, str>> {
    let mut texts: Vec<Cow<str>> = Vec::new();
    match body.get("system") {
        Some(Value::String(s)) => texts.push(Cow::Borrowed(s)),
        Some(Value::Array(items)) => {
            let joined = items
                .iter()
                .filter(|b| is_text_block(b))
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n");
            texts.push(Cow::Owned(joined));
        }
        _ => texts.push(Cow::Borrowed("")),
    }
    if let Some(Value::String(s)) = body.get("instructions") {
        texts.push(Cow::Borrowed(s));
    }
    for field in ["messages", "input"] {
        match body.get(field) {
            Some(Value::String(s)) => texts.push(Cow::Borrowed(s)),
            Some(Value::Array(items)) => {
                for m in items {
                    if matches!(
                        m.get("role").and_then(Value::as_str),
                        Some("user") | Some("system") | Some("developer")
                    ) {
                        texts.push(text_content(m));
                    }
                }
            }
            _ => {}
        }
    }
    texts
}

pub fn plugin_present(body: &Value, headers: &Headers) -> bool {
    if !header(headers, "x-openviking-plugin").is_empty() {
        return true;
    }
    client_texts(body).iter().any(|text| {
        let text = OV_CONTEXT_TAG.replace_all(text, |c: &regex::Captures| {
            let tag = c.get(0).unwrap().as_str();
            let gateway = ATTRS.captures_iter(tag).any(|a| {
                let value = a
                    .get(2)
                    .or_else(|| a.get(3))
                    .map(|m| m.as_str())
                    .unwrap_or("");
                a[1].eq_ignore_ascii_case("source")
                    && value.to_ascii_lowercase().starts_with("gateway-")
            });
            if gateway {
                String::new()
            } else {
                tag.to_string()
            }
        });
        PLUGIN.iter().any(|re| re.is_match(&text))
    })
}

/// `(?<![a-z0-9])openviking(?![a-z0-9])` with re.I; the regex crate has no lookaround.
pub fn has_openviking_word(label: &str) -> bool {
    let bytes = label.as_bytes();
    OPENVIKING.find_iter(label).any(|m| {
        let before = m.start() == 0 || !bytes[m.start() - 1].is_ascii_alphanumeric();
        let after = m.end() == bytes.len() || !bytes[m.end()].is_ascii_alphanumeric();
        before && after
    })
}

pub struct ToolNames {
    ending: Regex,
    deferred: Regex,
}

impl ToolNames {
    pub fn new(names: &[String]) -> Self {
        let alternatives = names
            .iter()
            .map(|n| regex::escape(n))
            .collect::<Vec<_>>()
            .join("|");
        ToolNames {
            ending: ci(&format!(r"[^a-z0-9](?:{alternatives})$")),
            deferred: ci(&format!(
                r"(?m)^[ \t]*mcp__[\w-]*?openviking[\w-]*__(?:{alternatives})[ \t]*$"
            )),
        }
    }
}

/// `client_openviking_tools`: whether the client already exposes OpenViking tools.
pub fn client_openviking_tools(body: &Value, names: &ToolNames) -> bool {
    let mut pending: Vec<(String, &Value)> = Vec::new();
    for key in ["tools", "additional_tools", "mcp_servers"] {
        if let Some(Value::Array(items)) = body.get(key) {
            pending.extend(items.iter().map(|t| (String::new(), t)));
        }
    }
    while let Some((parent, tool)) = pending.pop() {
        let Some(tool_obj) = tool.as_object() else {
            continue;
        };
        let definition = tool_obj.get("function").unwrap_or(tool);
        let Some(def) = definition.as_object() else {
            continue;
        };
        let labels: Vec<String> = ["name", "namespace", "server_label", "mcp_server_name"]
            .iter()
            .filter_map(|k| def.get(*k).and_then(Value::as_str))
            .map(|v| format!("{parent}{v}"))
            .collect();
        if let Some(Value::Array(children)) = def.get("tools") {
            let prefix = labels.first().map(|l| format!("{l}__")).unwrap_or_default();
            pending.extend(children.iter().map(|c| (prefix.clone(), c)));
            continue;
        }
        let server = matches!(
            def.get("type").and_then(Value::as_str),
            Some("mcp") | Some("mcp_toolset") | Some("url")
        );
        let named_tool = def.get("name").map(Value::is_string).unwrap_or(false) && !server;
        if labels
            .iter()
            .any(|l| has_openviking_word(l) && (!named_tool || names.ending.is_match(l)))
        {
            return true;
        }
        if labels.iter().any(|l| names.ending.is_match(l))
            && serde_json::to_string(definition)
                .unwrap()
                .to_lowercase()
                .contains("viking://")
        {
            return true;
        }
    }
    client_texts(body).iter().any(|t| {
        names
            .deferred
            .find_iter(t)
            .any(|m| has_openviking_word(m.as_str().trim().trim_start_matches("mcp__")))
    })
}

pub fn token_estimate(text: &str) -> usize {
    if text.is_ascii() {
        return (text.len() + 3) / 4;
    }
    let mut units = 0usize;
    for c in text.chars() {
        let p = c as u32;
        units += if (0x4E00..=0x9FFF).contains(&p) {
            6
        } else if p < 0x1100 {
            1
        } else if p <= 0x11FF
            || (0x3000..=0x30FF).contains(&p)
            || (0x3130..=0x318F).contains(&p)
            || (0x31F0..=0x31FF).contains(&p)
            || (0x3400..=0x4DBF).contains(&p)
            || (0xAC00..=0xD7AF).contains(&p)
            || (0xF900..=0xFAFF).contains(&p)
            || (0xFF00..=0xFFEF).contains(&p)
            || (0x20000..=0x2EBEF).contains(&p)
        {
            6
        } else if p > 0xFFFF {
            8
        } else {
            1
        };
    }
    (units + 3) / 4
}

/// `compaction.estimate`: tokens of a JSON value, inline media counted like one image.
pub fn estimate_json(dumped: &str) -> usize {
    let mut blobs = 0;
    let mut kept = String::with_capacity(dumped.len());
    let mut last = 0;
    for m in MEDIA.find_iter(dumped) {
        kept.push_str(&dumped[last..m.start()]);
        last = m.end();
        blobs += 1;
    }
    if blobs == 0 {
        return token_estimate(dumped);
    }
    kept.push_str(&dumped[last..]);
    token_estimate(&kept) + blobs * 1600
}

pub fn estimate(value: &Value) -> usize {
    estimate_json(&crate::canon::to_string(value))
}

pub fn estimate_list(items: &[Value]) -> usize {
    let mut out = Vec::with_capacity(16 * 1024);
    crate::canon::write_list(&mut out, items.iter());
    estimate_json(std::str::from_utf8(&out).unwrap())
}

/// `blocks.block` with `neutralize`.
pub fn block(source: &str, text: &str) -> String {
    static LEGACY: Lazy<Regex> = Lazy::new(|| ci(r"</?relevant-memor(?:y|ies)\b[^>]*>"));
    static MARKER: Lazy<Regex> = Lazy::new(|| ci(r"</?openviking-context\b[^>]*>"));
    if text.is_empty() {
        return String::new();
    }
    let t = LEGACY.replace_all(text, "legacy memory wrapper");
    let t = MARKER.replace_all(&t, "openviking context marker");
    format!("<openviking-context source=\"{source}\">\n{t}\n</openviking-context>")
}

/// `notices.without_recall_notice` guard: only replies starting with a notice are edited.
pub fn starts_with_notice(text: &str) -> bool {
    let head = &text[..text.len().min(64)];
    let head = if text.is_char_boundary(head.len()) {
        head
    } else {
        text
    };
    head.trim_start().starts_with("> OpenViking ")
}
