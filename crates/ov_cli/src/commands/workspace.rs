//! Workspace identity, layered workspace config, and the per-machine registry.
//!
//! This is the Rust half of a reader that also ships as JavaScript, in
//! `examples/memory-plugin-shared/lib/workspace-identity.mjs`,
//! `workspace-config.mjs`, `workspace-peer.mjs` and `workspace-registry.mjs`.
//! The memory plugins pick a peer with that reader and `ov` reports and edits it
//! with this one, so the two have to agree key for key — a divergence surfaces
//! as memories written under a peer the CLI cannot name. Rules that look
//! arbitrary here are byte-compatible with the JS original on purpose; the tests
//! at the bottom pin the same cases its `workspace-*.test.mjs` pin.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Component, Path, PathBuf};

use colored::Colorize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::client::HttpClient;
use crate::error::{Error, Result};
use crate::i18n::{Language, copy};
use crate::output::{OutputFormat, output_success};
use crate::terminal_ui::pad_to_display_width;
use crate::theme;

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// Everything the reader takes from outside the process, threaded explicitly so
/// the tests can run in parallel without mutating the real environment.
#[derive(Debug, Clone)]
pub struct WorkspaceEnv {
    /// `$HOME`. The root walk stops here rather than at it, so a stray `.git` in
    /// the home directory cannot make every unrelated directory one workspace.
    pub home: PathBuf,
    /// `$OPENVIKING_HOME` or `~/.openviking` — the registry's parent.
    pub openviking_home: PathBuf,
    /// The ovcli.conf whose `plugin` section is the lowest config layer.
    pub cli_config_file: PathBuf,
    /// `OPENVIKING_*` overrides, which outrank every file layer.
    pub vars: BTreeMap<String, String>,
}

/// Env vars that override a workspace key, paired with the key they set.
const ENV_OVERRIDES: &[(&str, &str)] = &[
    ("OPENVIKING_PEER_ID", "peer.id"),
    ("OPENVIKING_PEER_SOURCE", "peer.source"),
    ("OPENVIKING_AUTO_RECALL", "recall.enabled"),
    ("OPENVIKING_RECALL_PEER_SCOPE", "recall.peer_scope"),
    ("OPENVIKING_RECALL_DEDUP_TURNS", "recall.dedup_turns"),
    ("OPENVIKING_RECALL_LIMIT", "recall.max_items"),
    ("OPENVIKING_SCORE_THRESHOLD", "recall.score_threshold"),
    ("OPENVIKING_AUTO_CAPTURE", "capture.enabled"),
    (
        "OPENVIKING_COMMIT_TOKEN_THRESHOLD",
        "capture.commit_token_threshold",
    ),
    (
        "OPENVIKING_BYPASS_SESSION_PATTERNS",
        "bypass.session_patterns",
    ),
];

/// Booleans, so the env layer lands the same type a config file would.
const ENV_BOOLEAN_KEYS: &[&str] = &["recall.enabled", "capture.enabled"];

impl WorkspaceEnv {
    pub fn from_process() -> Self {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .or_else(dirs::home_dir)
            .unwrap_or_default();
        let openviking_home = match std::env::var("OPENVIKING_HOME") {
            Ok(value) if !value.trim().is_empty() => expand_home(value.trim(), &home),
            _ => home.join(".openviking"),
        };
        let cli_config_file = match std::env::var("OPENVIKING_CLI_CONFIG_FILE") {
            Ok(value) if !value.trim().is_empty() => expand_home(value.trim(), &home),
            _ => home.join(".openviking").join("ovcli.conf"),
        };
        let mut vars = BTreeMap::new();
        for (name, _) in ENV_OVERRIDES
            .iter()
            .chain(std::iter::once(&("OPENVIKING_WORKSPACE_PEER", "")))
        {
            if let Ok(value) = std::env::var(name)
                && !value.is_empty()
            {
                vars.insert((*name).to_string(), value);
            }
        }
        Self {
            home,
            openviking_home,
            cli_config_file,
            vars,
        }
    }

    pub fn registry_dir(&self) -> PathBuf {
        self.openviking_home.join("workspaces")
    }
}

fn expand_home(value: &str, home: &Path) -> PathBuf {
    match value.strip_prefix('~') {
        Some("") => home.to_path_buf(),
        Some(rest) if rest.starts_with('/') => home.join(rest.trim_start_matches('/')),
        _ => PathBuf::from(value),
    }
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/// 255 is the AGFS path-segment limit; stopping well short leaves room for the
/// hash suffix and for anything that later prefixes a peer id.
const MAX_PEER_ID_LENGTH: usize = 100;

fn short_hash(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    hex::encode(digest)[..12].to_string()
}

/// Legacy peer sanitation: one unit in, one unit out, no collapsing and no
/// trimming, so a leading `/` still becomes a leading `-`. Kept exact because
/// `peer.source: "cwd"` and the legacy id that dual-read recomputes both depend
/// on it — do not "improve" it. UTF-16 units rather than UTF-8 bytes, because
/// the JS original counts what `String.prototype.replace` counts.
pub fn legacy_sanitize(value: &str) -> String {
    value
        .encode_utf16()
        .map(|unit| match u8::try_from(unit) {
            Ok(byte) if byte.is_ascii_alphanumeric() => byte as char,
            _ => '-',
        })
        .collect()
}

/// Readable sanitation for values that were never path-shaped — a normalized
/// remote, a directory name. Mirrors the server's own `_sanitize_component`
/// (`openviking/ingest/peer.py`) so both languages agree on the id, then
/// enforces what `validate_identifier_part` additionally requires.
pub fn sanitize_peer_id(value: &str) -> String {
    let raw = value.trim();
    let mut cleaned = String::with_capacity(raw.len());
    let mut in_run = false;
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '@' | '-') {
            cleaned.push(ch);
            in_run = false;
        } else if !in_run {
            cleaned.push('-');
            in_run = true;
        }
    }
    cleaned = collapse_dashes(&cleaned);
    let cleaned = cleaned
        .trim_matches(|ch| ch == '-' || ch == '.')
        .to_string();
    if cleaned.is_empty() {
        return String::new();
    }

    // The server accepts at most one `@` in an identifier part.
    let mut cleaned = match cleaned.find('@') {
        Some(at) => format!("{}{}", &cleaned[..=at], cleaned[at + 1..].replace('@', "-")),
        None => cleaned,
    };
    // `ext-` is the server's namespace for base64-encoded external identities,
    // and `__self` its operation-target sentinel. Neither is ours to occupy.
    if cleaned.starts_with("ext-") {
        cleaned = format!("x-{cleaned}");
    }
    if cleaned == "__self" {
        cleaned = "self".to_string();
    }
    if cleaned == "." || cleaned == ".." {
        return String::new();
    }
    if cleaned.len() > MAX_PEER_ID_LENGTH {
        let head = cleaned[..MAX_PEER_ID_LENGTH - 13].trim_end_matches(['-', '.']);
        cleaned = format!("{head}-{}", short_hash(raw));
    }
    cleaned
}

fn collapse_dashes(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut previous_dash = false;
    for ch in value.chars() {
        if ch == '-' {
            if !previous_dash {
                out.push(ch);
            }
            previous_dash = true;
        } else {
            out.push(ch);
            previous_dash = false;
        }
    }
    out
}

/// Normalize a remote URL to `host/path`, lowercased.
///
/// Both spellings of the same repo converge, and userinfo is dropped — so a
/// remote with an embedded token cannot leak it into a peer id. The cost of case
/// folding is that two repos differing only in case share one namespace on the
/// rare case-sensitive forge.
pub fn normalize_git_remote(url: &str) -> String {
    let raw = url.trim();
    if raw.is_empty() {
        return String::new();
    }

    // `C:\src\repo` and `C:/src/repo` are one machine's directory, not a shared
    // identity — and the scp pattern would happily read the drive as a host.
    let mut bytes = raw.bytes();
    if let (Some(first), Some(second), Some(third)) = (bytes.next(), bytes.next(), bytes.next())
        && first.is_ascii_alphabetic()
        && second == b':'
        && (third == b'\\' || third == b'/')
    {
        return String::new();
    }

    let (host, path) = match scp_like_parts(raw) {
        Some(parts) if !has_url_scheme(raw) => parts,
        _ => {
            let Ok(parsed) = url::Url::parse(raw) else {
                return String::new();
            };
            // A local clone has no stable shared identity — fall through to the
            // path rules instead of minting one from `file://` or a bare path.
            let host = parsed.host_str().unwrap_or_default().to_string();
            if parsed.scheme() == "file" || host.is_empty() {
                return String::new();
            }
            (host, parsed.path().to_string())
        }
    };

    // `/^\[|\]$/g` — one bracket at each end, the IPv6 literal's own.
    let host = host.to_lowercase();
    let host = host.strip_prefix('[').unwrap_or(&host);
    let host = host.strip_suffix(']').unwrap_or(host).to_string();
    let path = path
        .trim_start_matches('/')
        .trim_end_matches('/')
        .to_string();
    let path = strip_git_suffix(&path).to_lowercase();
    if host.is_empty() || path.is_empty() {
        return String::new();
    }
    format!("{host}/{path}")
}

fn strip_git_suffix(path: &str) -> &str {
    if path.len() >= 4 && path[path.len() - 4..].eq_ignore_ascii_case(".git") {
        &path[..path.len() - 4]
    } else {
        path
    }
}

/// `^[a-zA-Z][a-zA-Z0-9+.-]*://`
fn has_url_scheme(raw: &str) -> bool {
    let Some(rest) = raw.split_once("://").map(|(scheme, _)| scheme) else {
        return false;
    };
    let mut chars = rest.chars();
    matches!(chars.next(), Some(ch) if ch.is_ascii_alphabetic())
        && chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '+' | '.' | '-'))
}

/// `^(?:[^@/\\]+@)?([^:/\\]+):(?!\/)(.+)$` — the `git@host:path` spelling.
fn scp_like_parts(raw: &str) -> Option<(String, String)> {
    let rest = match raw.find('@') {
        Some(at) if at > 0 && !raw[..at].contains(['/', '\\']) => &raw[at + 1..],
        _ => raw,
    };
    let colon = rest.find(':')?;
    let host = &rest[..colon];
    let path = &rest[colon + 1..];
    if host.is_empty() || host.contains(['/', '\\']) || path.is_empty() || path.starts_with('/') {
        return None;
    }
    Some((host.to_string(), path.to_string()))
}

/// Read `[remote "<name>"] url` out of a git config with a minimal INI parse.
///
/// `include` / `includeIf` are deliberately not followed: resolving them means
/// more filesystem walking for a value the fallback chain already covers, so an
/// unreadable remote just falls through to the next template. `git` is never
/// invoked — it may be absent from PATH, or refuse the repo over ownership.
pub fn read_git_remote_url(common_dir: &Path, remote: &str) -> String {
    let Ok(text) = fs::read_to_string(common_dir.join("config")) else {
        return String::new();
    };

    let mut in_section = false;
    for line in text.split('\n') {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with(';') {
            continue;
        }
        // `^\[([^\]]+)\]`: a `[` with no closing `]`, or an empty `[]`, is not a
        // section header at all — it leaves the current section standing rather
        // than opening or closing one.
        if let Some(header) = trimmed
            .strip_prefix('[')
            .and_then(|rest| rest.split_once(']'))
            .map(|(header, _)| header)
            .filter(|header| !header.is_empty())
        {
            let header = header.trim();
            // git folds section and key names to lower case but keeps a quoted
            // subsection exact, so `[Remote "origin"]` is the same section and
            // `[remote "Origin"]` is not.
            in_section = strip_prefix_ignore_ascii_case(header, "remote")
                .filter(|rest| rest.starts_with(char::is_whitespace))
                .map(str::trim)
                .and_then(|rest| rest.strip_prefix('"'))
                .and_then(|rest| rest.strip_suffix('"'))
                == Some(remote);
            continue;
        }
        if !in_section {
            continue;
        }
        if let Some(rest) = strip_prefix_ignore_ascii_case(trimmed, "url")
            .map(|rest| rest.trim_start())
            .and_then(|rest| rest.strip_prefix('='))
        {
            // git treats an unquoted `#` or `;` as starting a comment anywhere
            // on the line, so a trailing note is not part of the URL.
            let value = rest.split(['#', ';']).next().unwrap_or_default().trim();
            // `/^["']|["']$/g` strips one quote at each end, not a run of them.
            let value = value.strip_prefix(['"', '\'']).unwrap_or(value);
            return value.strip_suffix(['"', '\'']).unwrap_or(value).to_string();
        }
    }
    String::new()
}

/// `str::strip_prefix` folding ASCII case, the way a regex with `/i` does.
fn strip_prefix_ignore_ascii_case<'a>(value: &'a str, prefix: &str) -> Option<&'a str> {
    value
        .get(..prefix.len())
        .filter(|head| head.eq_ignore_ascii_case(prefix))
        .map(|head| &value[head.len()..])
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitInfo {
    pub git_dir: PathBuf,
    pub common_dir: PathBuf,
    pub kind: &'static str,
}

fn resolve_git_dir(root: &Path) -> Option<GitInfo> {
    let dot_git = root.join(".git");
    let metadata = fs::metadata(&dot_git).ok()?;
    if metadata.is_dir() {
        return Some(GitInfo {
            git_dir: dot_git.clone(),
            common_dir: dot_git,
            kind: "repo",
        });
    }
    if !metadata.is_file() {
        return None;
    }

    let content = fs::read_to_string(&dot_git).ok()?;
    let pointer = content
        .lines()
        .find_map(|line| line.trim_start().strip_prefix("gitdir:"))?
        .trim();
    if pointer.is_empty() {
        return None;
    }
    let git_dir = lexical_resolve(root, pointer);

    // `commondir` is the worktree signal, so it is read first — a repository
    // that merely lives under a directory called `modules` is not a submodule.
    let common_ref = fs::read_to_string(git_dir.join("commondir"))
        .map(|value| value.trim().to_string())
        .unwrap_or_default();
    let common_dir = if common_ref.is_empty() {
        git_dir.clone()
    } else {
        lexical_resolve(&git_dir, &common_ref)
    };

    // A submodule keeps its own remote under the superproject's
    // `.git/modules/<name>`; converging it onto the superproject would merge two
    // repositories that release, and are reviewed, separately. Only the segment
    // right after a `.git` directory means that, which is also why a worktree of
    // a submodule resolves here through its own commondir.
    let kind = if has_git_modules_segment(&common_dir) {
        "submodule"
    } else if common_ref.is_empty() {
        "repo"
    } else {
        "worktree"
    };
    Some(GitInfo {
        git_dir,
        common_dir,
        kind,
    })
}

fn has_git_modules_segment(common_dir: &Path) -> bool {
    let mut parts = common_dir
        .components()
        .filter_map(|component| match component {
            Component::Normal(part) => part.to_str(),
            _ => None,
        });
    while let Some(part) = parts.next() {
        if part == ".git" && parts.next() == Some("modules") {
            return true;
        }
    }
    false
}

/// Node's `path.resolve`: purely lexical, never touching the filesystem.
fn lexical_resolve(base: &Path, value: &str) -> PathBuf {
    let candidate = Path::new(value);
    let joined = if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        base.join(candidate)
    };

    let mut out = PathBuf::new();
    for component in joined.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkspaceIdentity {
    /// The cwd exactly as given; `vars.cwd` is derived from this, not from the
    /// canonicalized form, so a pre-existing peer keeps resolving.
    pub cwd: String,
    pub root: String,
    pub is_git: bool,
    pub git_kind: String,
    pub git_dir: String,
    pub git_common_dir: String,
    pub remote: String,
    pub vars: BTreeMap<String, String>,
}

/// Walk up from `cwd` to the nearest directory holding a `.git`, falling back to
/// the working directory itself.
///
/// `$HOME` and the filesystem root are never workspace roots, judged on the
/// starting directory rather than on where the walk stops: a walk that reaches
/// `/` without finding a repository has still started somewhere legitimate, and
/// a `.openviking/config.json` there has to apply.
pub fn find_workspace_root(cwd: &str, env: &WorkspaceEnv) -> (Option<PathBuf>, Option<GitInfo>) {
    let start = cwd.trim();
    if start.is_empty() {
        return (None, None);
    }
    let absolute = lexical_resolve(&std::env::current_dir().unwrap_or_default(), start);

    // `current` and `stop_at` are compared as paths, so they must be
    // canonicalized together: a `$HOME` reached through a symlink plus an
    // unresolvable cwd would otherwise walk straight past the guard.
    let (mut current, stop_at) = match (fs::canonicalize(&absolute), fs::canonicalize(&env.home)) {
        (Ok(current), Ok(home)) => (current, home),
        _ => (absolute, env.home.clone()),
    };
    let filesystem_root = current.ancestors().last().map(Path::to_path_buf);
    if current == stop_at || Some(&current) == filesystem_root.as_ref() {
        return (None, None);
    }
    let working_directory = current.clone();

    while !current.as_os_str().is_empty()
        && Some(&current) != filesystem_root.as_ref()
        && current != stop_at
    {
        if let Some(git) = resolve_git_dir(&current) {
            return (Some(current), Some(git));
        }
        match current.parent() {
            Some(parent) if parent != current => current = parent.to_path_buf(),
            _ => break,
        }
    }
    (Some(working_directory), None)
}

/// Everything the peer templates can substitute, for one cwd.
///
/// `git_remote` and `dir` are already sanitized; `git_root` and `cwd` carry the
/// legacy rule, because they are the two that must reproduce a peer minted
/// before any of this existed.
pub fn resolve_workspace_identity(cwd: &str, env: &WorkspaceEnv) -> WorkspaceIdentity {
    let (root, git) = find_workspace_root(cwd, env);
    // Only the normalized form is kept: the raw URL may carry a token.
    let remote = git
        .as_ref()
        .map(|git| normalize_git_remote(&read_git_remote_url(&git.common_dir, "origin")))
        .unwrap_or_default();
    let root_display = root
        .as_ref()
        .map(|root| root.to_string_lossy().to_string())
        .unwrap_or_default();

    let mut vars = BTreeMap::new();
    vars.insert("git_remote".to_string(), sanitize_peer_id(&remote));
    // Empty outside a repository even though `root` is set there, so the `git`
    // preset still falls through to `{cwd}` rather than stopping at a repository
    // root that does not exist.
    vars.insert(
        "git_root".to_string(),
        if git.is_some() {
            legacy_sanitize(&root_display)
        } else {
            String::new()
        },
    );
    vars.insert("cwd".to_string(), legacy_sanitize(cwd));
    vars.insert(
        "dir".to_string(),
        root.as_ref()
            .and_then(|root| root.file_name())
            .map(|name| sanitize_peer_id(&name.to_string_lossy()))
            .unwrap_or_default(),
    );

    WorkspaceIdentity {
        cwd: cwd.to_string(),
        root: root_display,
        is_git: git.is_some(),
        git_kind: git
            .as_ref()
            .map(|git| git.kind.to_string())
            .unwrap_or_default(),
        git_dir: git
            .as_ref()
            .map(|git| git.git_dir.to_string_lossy().to_string())
            .unwrap_or_default(),
        git_common_dir: git
            .as_ref()
            .map(|git| git.common_dir.to_string_lossy().to_string())
            .unwrap_or_default(),
        remote,
        vars,
    }
}

// ---------------------------------------------------------------------------
// Peer resolution
// ---------------------------------------------------------------------------

/// `git` is the default. It prefers the remote, because that is the one name
/// every clone agrees on; falls back to the repository root, which at least
/// stops a subdirectory from forking the identity; and finally to the working
/// directory, which is what a non-repository had all along.
pub fn peer_source_preset(name: &str) -> Option<&'static [&'static str]> {
    match name {
        "git" => Some(&["{git_remote}", "{git_root}", "{cwd}"]),
        "cwd" => Some(&["{cwd}"]),
        "none" => Some(&[]),
        _ => None,
    }
}

/// Normalize `peer.source` — a preset name, a template, or a list — to templates.
pub fn peer_source_templates(source: Option<&Value>) -> Vec<String> {
    if let Some(Value::Array(items)) = source {
        // `source.map(String).filter(Boolean)`: every member is stringified, so
        // a member that is not a string still names a template.
        return items
            .iter()
            .map(js_string)
            .filter(|item| !item.is_empty())
            .collect();
    }
    // `String(source ?? "")` — only an absent or null source is the default.
    let raw = match source {
        None | Some(Value::Null) => String::new(),
        Some(value) => js_string(value),
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return preset_templates("git");
    }
    match peer_source_preset(raw) {
        Some(preset) => preset.iter().map(|item| (*item).to_string()).collect(),
        None => vec![raw.to_string()],
    }
}

fn preset_templates(name: &str) -> Vec<String> {
    peer_source_preset(name)
        .unwrap_or(&[])
        .iter()
        .map(|item| (*item).to_string())
        .collect()
}

/// `String(value)`, for a `peer.source` that is not the string it should be.
///
/// A shape this reader would rather reject still has to be read the way the JS
/// half reads it: `{}` becomes the literal template `[object Object]`, which
/// renders to itself and becomes the peer. Falling back to the default preset
/// instead would have `ov` name one peer while every plugin sent another.
fn js_string(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(flag) => flag.to_string(),
        Value::Number(number) => match number.as_f64() {
            // `String(5.0)` is "5"; `Number::to_string` would say "5.0".
            Some(float) if float.fract() == 0.0 && float.abs() < 1.0e21 => {
                format!("{}", float as i128)
            }
            _ => number.to_string(),
        },
        Value::String(text) => text.clone(),
        // `Array.prototype.join` renders a null member as an empty string.
        Value::Array(items) => items
            .iter()
            .map(|item| {
                if item.is_null() {
                    String::new()
                } else {
                    js_string(item)
                }
            })
            .collect::<Vec<_>>()
            .join(","),
        Value::Object(_) => "[object Object]".to_string(),
    }
}

/// Substitute one template, or return "" when any variable it names is empty.
///
/// All-or-nothing on purpose: a half-resolved template like `git-` would be a
/// silently shared identity, so an empty variable falls through to the next
/// template instead.
pub fn render_peer_template(template: &str, vars: &BTreeMap<String, String>) -> String {
    if template.is_empty() {
        return String::new();
    }
    let mut rendered = String::with_capacity(template.len());
    let mut empty = false;
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        let Some(close) = rest[open + 1..].find('}').map(|index| open + 1 + index) else {
            break;
        };
        let name = &rest[open + 1..close];
        if name.is_empty() || !name.chars().all(|ch| ch.is_ascii_lowercase() || ch == '_') {
            // Resume just past the `{`, not past the `}`: the JS regex scans
            // forward one position at a time, so `{a{b}` still finds `{b}`.
            rendered.push_str(&rest[..=open]);
            rest = &rest[open + 1..];
            continue;
        }
        rendered.push_str(&rest[..open]);
        match vars.get(name) {
            Some(value) if !value.is_empty() => rendered.push_str(value),
            _ => empty = true,
        }
        rest = &rest[close + 1..];
    }
    rendered.push_str(rest);
    if empty { String::new() } else { rendered }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectivePeer {
    pub peer_id: String,
    /// `explicit`, `workspace` or `none` — call sites compare this literal to
    /// decide whether a session pin may be reused.
    pub source: &'static str,
    /// The template that actually produced the id.
    pub origin: String,
    /// The pre-git id, whenever it differs, so recall can still reach memories
    /// written under it.
    pub legacy_peer_id: String,
}

/// The peer this workspace should send, and where it came from.
pub fn resolve_effective_peer_id(
    config: &Value,
    workspace_peer_enabled: bool,
    identity: &WorkspaceIdentity,
) -> EffectivePeer {
    let explicit = config_get(config, "peer.id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    if !explicit.is_empty() {
        return EffectivePeer {
            peer_id: explicit,
            source: "explicit",
            origin: "explicit".to_string(),
            legacy_peer_id: String::new(),
        };
    }

    // `OPENVIKING_WORKSPACE_PEER=0` predates `peer.source` and still means "none".
    if !workspace_peer_enabled {
        return EffectivePeer {
            peer_id: String::new(),
            source: "none",
            origin: "disabled".to_string(),
            legacy_peer_id: String::new(),
        };
    }

    let templates = peer_source_templates(config_get(config, "peer.source"));
    if templates.is_empty() {
        return EffectivePeer {
            peer_id: String::new(),
            source: "none",
            origin: "none".to_string(),
            legacy_peer_id: String::new(),
        };
    }

    let legacy_peer_id = legacy_sanitize(&identity.cwd);
    for template in templates {
        let peer_id = render_peer_template(&template, &identity.vars);
        if peer_id.is_empty() {
            continue;
        }
        let legacy = if peer_id == legacy_peer_id {
            String::new()
        } else {
            legacy_peer_id
        };
        return EffectivePeer {
            peer_id,
            source: "workspace",
            origin: template,
            legacy_peer_id: legacy,
        };
    }
    EffectivePeer {
        peer_id: String::new(),
        source: "none",
        origin: "unresolved".to_string(),
        legacy_peer_id: String::new(),
    }
}

// ---------------------------------------------------------------------------
// Layered configuration
// ---------------------------------------------------------------------------

pub const CONFIG_DIR_NAME: &str = ".openviking";
pub const TEAM_FILE: &str = "config.json";
pub const LOCAL_FILE: &str = "config.local.json";
pub const CONFIG_VERSION: u64 = 1;
pub const MAX_CONFIG_BYTES: u64 = 64 * 1024;

/// Deep enough for any real config; a file can nest far past the stack limit
/// inside the 64 KiB cap, and an overflow here would take out sibling layers.
const MAX_DEPTH: usize = 32;

/// `JSON.parse` keeps `__proto__` as an own property in the JS reader, and
/// assigning it walks into `Object.prototype`. Harmless in Rust, but the key
/// list has to stay identical so the two readers agree on what a file means.
const UNSAFE_KEYS: &[&str] = &["__proto__", "constructor", "prototype"];

/// Keys no workspace file may set, at any depth. Connection and credentials
/// belong to ovcli.conf and the environment, full stop.
pub const FORBIDDEN_KEYS: &[&str] = &[
    "url",
    "base_url",
    "mcp_url",
    "api_key",
    "bearer_token",
    "root_api_key",
    "gateway_token",
    "oidc_token",
    "ldap_username",
    "ldap_password",
    "account",
    "account_id",
    "user",
    "user_id",
    "auth_mode",
    "extra_headers",
    "credential_source",
    "cli_config_file",
    "config_file",
    // The camelCase spellings the harness loaders use. The projection into
    // harness knobs is an allowlist, so these could never take effect anyway —
    // but someone who writes `apiKey` here deserves to be told it was ignored,
    // not to have it vanish.
    "baseUrl",
    "mcpUrl",
    "apiKey",
    "bearerToken",
    "rootApiKey",
    "gatewayToken",
    "accountId",
    "userId",
    "authMode",
    "extraHeaders",
    "credentialSource",
    "credentialPath",
    "configPath",
];

/// Registry-only. Naming an ovcli.conf profile decides which credentials reach
/// which server, so a repository setting it would be `url` tampering by proxy.
pub const REGISTRY_ONLY_KEYS: &[&str] = &["cli_config_profile"];

/// Sections whose keys are the user's own vocabulary, not ours.
pub const FREE_FORM_SECTIONS: &[&str] = &["labels"];

pub fn is_valid_profile_name(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !first.is_ascii_lowercase() && !first.is_ascii_digit() {
        return false;
    }
    value.len() <= 64
        && chars.all(|ch| {
            ch.is_ascii_lowercase() || ch.is_ascii_digit() || matches!(ch, '.' | '_' | '-')
        })
}

/// The version a workspace file's `min_client_version` is measured against.
const CLIENT_VERSION: &str = env!("OPENVIKING_CLI_VERSION");

/// `Number.parseInt(part, 10) || 0`: leading digits win and anything else is 0,
/// so a `1.2.3-rc1` tail compares as `3`.
fn js_parse_int(part: &str) -> i64 {
    let text = part.trim_start();
    let (negative, digits) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    let head: String = digits.chars().take_while(char::is_ascii_digit).collect();
    if head.is_empty() {
        return 0;
    }
    let value = head.parse::<i64>().unwrap_or(i64::MAX);
    if negative { -value } else { value }
}

fn compare_versions(left: &str, right: &str) -> std::cmp::Ordering {
    let left: Vec<i64> = left.split('.').map(js_parse_int).collect();
    let right: Vec<i64> = right.split('.').map(js_parse_int).collect();
    for index in 0..left.len().max(right.len()) {
        let ordering = left
            .get(index)
            .copied()
            .unwrap_or(0)
            .cmp(&right.get(index).copied().unwrap_or(0));
        if ordering != std::cmp::Ordering::Equal {
            return ordering;
        }
    }
    std::cmp::Ordering::Equal
}

/// `min_client_version` warns and never blocks. A committed file that could stop
/// an older client from running would be a denial of service anyone with commit
/// access could mount, so it says "this was written for a newer client" and the
/// settings still apply.
pub fn check_min_client_version(
    declared: &str,
    client_version: &str,
    warnings: &mut Vec<String>,
) -> bool {
    let required = declared.trim();
    let current = client_version.trim();
    if required.is_empty()
        || current.is_empty()
        || compare_versions(current, required) != std::cmp::Ordering::Less
    {
        return true;
    }
    warnings.push(format!(
        "this workspace asks for OpenViking plugin {required} and this one is {current}; \
         settings it introduced will be ignored rather than blocking the session"
    ));
    false
}

#[derive(Debug, Clone)]
pub struct LayerFile {
    pub exists: bool,
    pub data: Option<Value>,
    pub warnings: Vec<String>,
}

/// Strip banned keys wherever they appear, collecting one warning per hit. They
/// are removed rather than rejected: a file is not made unusable by carrying a
/// key this layer refuses to honour.
fn strip_forbidden(
    value: &Value,
    banned: &[&str],
    warnings: &mut Vec<String>,
    path: &str,
    depth: usize,
) -> std::result::Result<Value, String> {
    if depth > MAX_DEPTH {
        return Err(format!("nested more than {MAX_DEPTH} levels at '{path}'"));
    }
    match value {
        Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for (index, item) in items.iter().enumerate() {
                out.push(strip_forbidden(
                    item,
                    banned,
                    warnings,
                    &format!("{path}[{index}]"),
                    depth + 1,
                )?);
            }
            Ok(Value::Array(out))
        }
        Value::Object(object) => {
            let mut out = Map::new();
            for (key, child) in object {
                let here = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                if UNSAFE_KEYS.contains(&key.as_str()) {
                    warnings.push(format!(
                        "ignored '{here}': a config file may not reach the object prototype"
                    ));
                    continue;
                }
                if banned.contains(&key.as_str()) {
                    warnings.push(format!(
                        "ignored '{here}': connection and credential settings belong in ovcli.conf or the environment"
                    ));
                    continue;
                }
                // Matched on the key itself, not on depth: a registry entry keeps
                // the same section under `settings.`, and it must not end up
                // stricter than the committed file it outranks.
                let child_banned: &[&str] = if FREE_FORM_SECTIONS.contains(&key.as_str()) {
                    &[]
                } else {
                    banned
                };
                out.insert(
                    key.clone(),
                    strip_forbidden(child, child_banned, warnings, &here, depth + 1)?,
                );
            }
            Ok(Value::Object(out))
        }
        other => Ok(other.clone()),
    }
}

/// JavaScript truthiness, for the handful of places the JS reader tests a value
/// rather than its presence. An array or an object is always truthy there.
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        _ => true,
    }
}

/// Read one layer file. Every failure is a warning and an empty layer.
pub fn read_workspace_file(path: &Path, root: Option<&Path>, registry: bool) -> LayerFile {
    let mut file = LayerFile {
        exists: false,
        data: None,
        warnings: Vec::new(),
    };
    let display = path.display().to_string();

    let Ok(metadata) = fs::metadata(path) else {
        return file;
    };
    file.exists = true;
    if !metadata.is_file() {
        file.warnings
            .push(format!("{display} is not a regular file"));
        return file;
    }
    if metadata.len() > MAX_CONFIG_BYTES {
        file.warnings
            .push(format!("{display} is larger than {MAX_CONFIG_BYTES} bytes"));
        return file;
    }
    // A symlink out of the workspace would let a repository read a file the user
    // never meant to expose to it.
    if let Some(root) = root {
        match (fs::canonicalize(root), fs::canonicalize(path)) {
            (Ok(root), Ok(resolved)) => {
                if !resolved.starts_with(&root) || resolved == root {
                    file.warnings
                        .push(format!("{display} resolves outside the workspace"));
                    return file;
                }
            }
            _ => {
                file.warnings
                    .push(format!("{display} could not be resolved"));
                return file;
            }
        }
    }

    // Deliberately bare: no `${VAR}` expansion reaches a workspace file.
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            file.warnings
                .push(format!("{display} could not be read ({error})"));
            return file;
        }
    };
    let parsed: Value = match serde_json::from_str(&text) {
        Ok(parsed) => parsed,
        Err(error) => {
            file.warnings
                .push(format!("{display} is not valid JSON ({error})"));
            return file;
        }
    };
    let Some(object) = parsed.as_object() else {
        file.warnings
            .push(format!("{display} must contain a JSON object"));
        return file;
    };
    // `parsed.version !== CONFIG_VERSION` compares JS numbers, where `1.0` and
    // `1` are one value — so the check has to widen rather than demand a u64.
    let declared_version = object.get("version").and_then(Value::as_f64);
    if declared_version != Some(CONFIG_VERSION as f64) {
        let declared = object
            .get("version")
            .map(ToString::to_string)
            .unwrap_or_else(|| "undefined".to_string());
        file.warnings.push(format!(
            "{display} declares version {declared}; this client understands {CONFIG_VERSION}"
        ));
        return file;
    }

    let mut banned: Vec<&str> = FORBIDDEN_KEYS.to_vec();
    if !registry {
        banned.extend_from_slice(REGISTRY_ONLY_KEYS);
    }
    let mut rest = object.clone();
    // `shift_remove`, not `remove`: with `preserve_order` the latter is a swap
    // remove, which would shuffle the key order the JS reader preserves — and
    // the registry file this produces is read back by both.
    rest.shift_remove("version");
    rest.shift_remove("$schema");
    let min_client_version = rest.shift_remove("min_client_version");

    let mut warnings = Vec::new();
    match strip_forbidden(&Value::Object(rest), &banned, &mut warnings, "", 0) {
        Ok(Value::Object(mut data)) => {
            // `if (minClientVersion)` — a falsy declaration (null, `false`, `0`,
            // `""`) carries no version and is dropped rather than stringified.
            if let Some(version) = min_client_version.filter(is_truthy) {
                data.insert(
                    "min_client_version".to_string(),
                    Value::String(js_string(&version)),
                );
            }
            file.warnings.extend(warnings);
            file.data = Some(Value::Object(data));
        }
        Ok(_) => unreachable!("an object strips to an object"),
        Err(error) => {
            file.warnings
                .push(format!("{display} is nested too deeply ({error})"));
        }
    }
    file
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShadowedValue {
    pub value: Value,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Provenance {
    pub key: String,
    pub value: Value,
    pub source: String,
    pub shadowed: Vec<ShadowedValue>,
}

#[derive(Debug, Clone)]
pub struct ConfigLayer {
    pub layer: String,
    pub data: Value,
}

#[derive(Debug, Clone, Default)]
pub struct MergeResult {
    pub value: Value,
    /// Insertion-ordered, matching the JS object it mirrors.
    pub provenance: Vec<Provenance>,
}

impl MergeResult {
    pub fn get(&self, key: &str) -> Option<&Provenance> {
        self.provenance.iter().find(|entry| entry.key == key)
    }
}

fn shadow(provenance: &mut Vec<Provenance>, key: &str, value: Value, source: &str) {
    match provenance.iter().position(|entry| entry.key == key) {
        Some(index) => {
            let previous = &provenance[index];
            let mut shadowed = vec![ShadowedValue {
                value: previous.value.clone(),
                source: previous.source.clone(),
            }];
            shadowed.extend(previous.shadowed.iter().cloned());
            provenance[index] = Provenance {
                key: key.to_string(),
                value,
                source: source.to_string(),
                shadowed,
            };
        }
        None => provenance.push(Provenance {
            key: key.to_string(),
            value,
            source: source.to_string(),
            shadowed: Vec::new(),
        }),
    }
}

fn merge_into(
    target: &mut Map<String, Value>,
    source: &Map<String, Value>,
    layer: &str,
    provenance: &mut Vec<Provenance>,
    path: &str,
    depth: usize,
) -> std::result::Result<(), String> {
    if depth > MAX_DEPTH {
        return Err(format!("nested more than {MAX_DEPTH} levels at '{path}'"));
    }
    for (key, value) in source {
        if UNSAFE_KEYS.contains(&key.as_str()) {
            continue;
        }
        let here = if path.is_empty() {
            key.clone()
        } else {
            format!("{path}.{key}")
        };

        if let Value::Object(section) = value {
            // A section replacing a scalar is still a change of the effective
            // value, so the scalar has to be recorded as shadowed rather than
            // left standing in provenance as if it were still in force.
            if !target.get(key).is_some_and(Value::is_object) {
                if target.contains_key(key) {
                    shadow(provenance, &here, Value::String("(section)".into()), layer);
                }
                target.insert(key.clone(), Value::Object(Map::new()));
            }
            let Some(Value::Object(nested)) = target.get_mut(key) else {
                unreachable!("just inserted an object")
            };
            merge_into(nested, section, layer, provenance, &here, depth + 1)?;
            continue;
        }

        if let Value::Array(items) = value {
            // `"!reset"` drops everything the lower layers contributed, the way
            // EditorConfig's `unset` and git's empty `safe.directory` do.
            let reset = items.first() == Some(&Value::String("!reset".into()));
            let incoming = if reset { &items[1..] } else { &items[..] };
            let inherited = target.get(key).and_then(Value::as_array).cloned();
            let inheritable = !reset && inherited.is_some();
            let mut merged = if inheritable {
                inherited.unwrap_or_default()
            } else {
                Vec::new()
            };
            for item in incoming {
                // Only scalars dedupe: the JS reader compares list members by
                // reference, and two objects parsed from two layers are never
                // the same reference.
                let duplicate = !item.is_object() && !item.is_array() && merged.contains(item);
                if !duplicate {
                    merged.push(item.clone());
                }
            }

            // Only a genuine union credits both layers. A list landing on a
            // scalar, or on nothing, belongs to this layer alone.
            let previous = provenance.iter().position(|entry| entry.key == here);
            match (inheritable, previous) {
                (true, Some(index)) => {
                    let source = if provenance[index].source.is_empty() {
                        layer.to_string()
                    } else {
                        format!("{} + {layer}", provenance[index].source)
                    };
                    provenance[index].value = Value::Array(merged.clone());
                    provenance[index].source = source;
                }
                _ => {
                    let source = if reset {
                        format!("{layer} (reset)")
                    } else {
                        layer.to_string()
                    };
                    shadow(provenance, &here, Value::Array(merged.clone()), &source);
                }
            }
            target.insert(key.clone(), Value::Array(merged));
            continue;
        }

        shadow(provenance, &here, value.clone(), layer);
        target.insert(key.clone(), value.clone());
    }
    Ok(())
}

/// Merge layers given lowest-precedence first, returning the effective value
/// plus, for every key, where it came from and what it covered up — the same
/// question `git config --show-origin --show-scope` answers.
pub fn merge_config_layers(layers: &[ConfigLayer], warnings: &mut Vec<String>) -> MergeResult {
    let mut value = Map::new();
    let mut provenance = Vec::new();
    for layer in layers {
        let Some(data) = layer.data.as_object() else {
            continue;
        };
        // One pathological layer must not take the others down with it.
        if let Err(error) = merge_into(&mut value, data, &layer.layer, &mut provenance, "", 0) {
            warnings.push(format!("skipped {}: {error}", layer.layer));
        }
    }
    MergeResult {
        value: Value::Object(value),
        provenance,
    }
}

pub fn config_get<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    let mut node = value;
    for key in path.split('.') {
        node = node.as_object()?.get(key)?;
    }
    Some(node)
}

fn config_unset(value: &mut Value, path: &str) {
    let mut keys = path.split('.').collect::<Vec<_>>();
    let Some(last) = keys.pop() else { return };
    let mut node = value;
    for key in keys {
        let Some(next) = node.as_object_mut().and_then(|object| object.get_mut(key)) else {
            return;
        };
        node = next;
    }
    if let Some(object) = node.as_object_mut() {
        object.shift_remove(last);
    }
}

fn config_set(value: &mut Value, path: &str, new_value: Value) {
    let mut keys = path.split('.').collect::<Vec<_>>();
    let Some(last) = keys.pop() else { return };
    let mut node = value;
    for key in keys {
        let Value::Object(object) = node else {
            return;
        };
        if !object.get(key).is_some_and(Value::is_object) {
            object.insert(key.to_string(), Value::Object(Map::new()));
        }
        node = object.get_mut(key).expect("just inserted");
    }
    if let Some(object) = node.as_object_mut() {
        object.insert(last.to_string(), new_value);
    }
}

struct Range {
    key: &'static str,
    min: f64,
    max: f64,
    integer: bool,
}

const RANGES: &[Range] = &[
    Range {
        key: "recall.dedup_turns",
        min: 0.0,
        max: 20.0,
        integer: true,
    },
    Range {
        key: "recall.max_items",
        min: 1.0,
        max: 100.0,
        integer: true,
    },
    Range {
        key: "recall.score_threshold",
        min: 0.0,
        max: 1.0,
        integer: false,
    },
    Range {
        key: "capture.commit_token_threshold",
        min: 1000.0,
        max: 1_000_000.0,
        integer: true,
    },
];

const ENUMS: &[(&str, &[&str])] = &[("recall.peer_scope", &["all", "actor"])];

/// Clamp numbers and reject unknown enum values, warning once per key.
///
/// A repository can raise a cost knob, so the ceiling is enforced here rather
/// than trusted; an out-of-range value is clamped instead of rejected so a typo
/// degrades rather than disables.
pub fn normalize_workspace_config(value: &mut Value, warnings: &mut Vec<String>) {
    for range in RANGES {
        let Some(raw) = config_get(value, range.key).cloned() else {
            continue;
        };
        // A null, a bool or a list would coerce to a finite number in JS, which
        // would silently pin a knob to a bound instead of reporting a bad value.
        let number = match &raw {
            Value::Number(number) => number.as_f64(),
            Value::String(text) => js_number(text),
            _ => None,
        }
        .filter(|number| number.is_finite());
        let Some(number) = number else {
            warnings.push(format!("ignored '{}': {raw} is not a number", range.key));
            config_unset(value, range.key);
            continue;
        };
        let candidate = if range.integer {
            number.floor()
        } else {
            number
        };
        let clamped = candidate.clamp(range.min, range.max);
        if clamped != number {
            warnings.push(format!(
                "clamped '{}' from {number} to {clamped} (allowed {}..{})",
                range.key, range.min, range.max
            ));
        }
        config_set(value, range.key, json_number(clamped));
    }

    for (key, allowed) in ENUMS {
        let Some(raw) = config_get(value, key).cloned() else {
            continue;
        };
        if !raw.as_str().is_some_and(|text| allowed.contains(&text)) {
            warnings.push(format!(
                "ignored '{key}': {raw} is not one of {}",
                allowed.join(", ")
            ));
            config_unset(value, key);
        }
    }
}

/// `Number(string)`, which is not `str::parse`: an empty or all-whitespace
/// string is `0`, radix literals are accepted, and `inf` / `nan` are not.
///
/// A clamped knob is the one place the JS reader coerces a string, so the two
/// have to agree on what `"": 0` and `"0x10": 16` mean — otherwise one clamps
/// where the other reports a bad value.
fn js_number(text: &str) -> Option<f64> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Some(0.0);
    }
    let (sign, digits) = match trimmed.strip_prefix(['+', '-']) {
        Some(rest) if trimmed.starts_with('-') => (-1.0, rest),
        Some(rest) => (1.0, rest),
        None => (1.0, trimmed),
    };
    if let Some((radix, body)) = digits
        .strip_prefix("0x")
        .or_else(|| digits.strip_prefix("0X"))
        .map(|body| (16, body))
        .or_else(|| {
            digits
                .strip_prefix("0o")
                .or_else(|| digits.strip_prefix("0O"))
                .map(|body| (8, body))
        })
        .or_else(|| {
            digits
                .strip_prefix("0b")
                .or_else(|| digits.strip_prefix("0B"))
                .map(|body| (2, body))
        })
    {
        // A radix literal may not carry a sign in JS, and `Number` reads it as
        // an unsigned integer.
        if sign < 0.0 || trimmed.starts_with('+') {
            return None;
        }
        return u128::from_str_radix(body, radix)
            .ok()
            .map(|value| value as f64);
    }
    if digits == "Infinity" {
        return Some(sign * f64::INFINITY);
    }
    // `str::parse` also accepts `inf`, `infinity` and `nan` in any case, where
    // `Number` reads every spelling but the exact `Infinity` above as NaN.
    if digits.eq_ignore_ascii_case("inf")
        || digits.eq_ignore_ascii_case("infinity")
        || digits.eq_ignore_ascii_case("nan")
    {
        return None;
    }
    trimmed.parse::<f64>().ok()
}

/// Keep integral knobs integral, so `recall.max_items` prints as `20` rather
/// than `20.0` and compares equal to what a config file spelled.
fn json_number(value: f64) -> Value {
    if value.fract() == 0.0 && value.abs() < 9.0e15 {
        return Value::from(value as i64);
    }
    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

pub fn workspace_config_paths(root: &Path) -> Vec<(String, PathBuf)> {
    let dir = root.join(CONFIG_DIR_NAME);
    vec![
        (format!("{TEAM_FILE} (workspace)"), dir.join(TEAM_FILE)),
        (format!("{LOCAL_FILE} (workspace)"), dir.join(LOCAL_FILE)),
    ]
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

pub const REGISTRY_VERSION: u64 = 1;

/// A readable name plus a hash, keyed on the workspace's identity rather than
/// its path wherever git supplies one.
///
/// Two linked worktrees of one repository are one workspace — the same peer, so
/// the same settings and the same `ov peer link` — and keying on the checkout
/// path would silently split them in two. Outside a repository there is no
/// identity but the path, so two `~/src/api` clones still get separate entries.
pub fn slot_name(root: &str, identity: Option<&WorkspaceIdentity>) -> String {
    let key = identity
        .map(identity_key)
        .unwrap_or_else(|| "path".to_string());
    let source = if key == "path" { root } else { key.as_str() };
    let base = match key.starts_with("remote:") {
        true => key.rsplit('/').next().unwrap_or_default().to_string(),
        false => Path::new(root)
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default(),
    };
    let mut readable = String::with_capacity(base.len());
    let mut in_run = false;
    for ch in base.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
            readable.push(ch);
            in_run = false;
        } else if !in_run {
            readable.push('-');
            in_run = true;
        }
    }
    let readable: String = readable
        .trim_matches('-')
        .chars()
        .take(40)
        .collect::<String>();
    let digest = short_hash(source);
    if readable.is_empty() {
        format!("{digest}.json")
    } else {
        format!("{readable}-{digest}.json")
    }
}

pub fn entry_path(root: &str, identity: Option<&WorkspaceIdentity>, env: &WorkspaceEnv) -> PathBuf {
    env.registry_dir().join(slot_name(root, identity))
}

/// The identity a stored entry is checked against. Path alone is not enough: a
/// directory can be deleted and a different repository cloned in its place, and
/// inheriting the old entry's peer would silently cross two projects.
pub fn identity_key(identity: &WorkspaceIdentity) -> String {
    // The normalized remote, so re-spelling origin (ssh <-> https, or rotating
    // an embedded token) is not mistaken for a different repository.
    let remote = identity.remote.trim();
    if !remote.is_empty() {
        return format!("remote:{remote}");
    }
    if identity.is_git {
        return format!("git:{}", identity.git_common_dir);
    }
    "path".to_string()
}

#[derive(Debug, Clone)]
pub struct RegistryEntry {
    pub path: PathBuf,
    pub entry: Option<Value>,
    pub warnings: Vec<String>,
    pub conflict: bool,
}

/// Read this workspace's entry, or nothing.
///
/// A stored entry whose identity contradicts the current one is treated as a
/// miss — negative evidence. Nothing is inherited from it, and a later write
/// replaces it.
pub fn read_entry(
    root: &str,
    identity: Option<&WorkspaceIdentity>,
    env: &WorkspaceEnv,
) -> RegistryEntry {
    let path = entry_path(root, identity, env);
    let file = read_workspace_file(&path, None, true);
    let Some(mut entry) = file.data else {
        return RegistryEntry {
            path,
            entry: None,
            warnings: file.warnings,
            conflict: false,
        };
    };

    let mut warnings = file.warnings;
    if let Some(identity) = identity {
        let expected = identity_key(identity);
        let stored = entry
            .get("identity")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if !stored.is_empty() && stored != expected {
            warnings.push(format!(
                "{} was recorded for a different repository ({stored}); starting a fresh entry for {expected}",
                path.display()
            ));
            return RegistryEntry {
                path,
                entry: None,
                warnings,
                conflict: true,
            };
        }
    }
    if let Some(profile) = entry.get("cli_config_profile").cloned()
        && !profile.as_str().is_some_and(is_valid_profile_name)
    {
        warnings.push(format!(
            "{}: cli_config_profile {profile} is not a valid profile name",
            path.display()
        ));
        if let Some(object) = entry.as_object_mut() {
            object.shift_remove("cli_config_profile");
        }
    }
    RegistryEntry {
        path,
        entry: Some(entry),
        warnings,
        conflict: false,
    }
}

fn read_raw_entry(path: &Path) -> Option<Value> {
    let text = fs::read_to_string(path).ok()?;
    let parsed: Value = serde_json::from_str(&text).ok()?;
    parsed.is_object().then_some(parsed)
}

/// Write this workspace's entry. Read-modify-write on this one small file only:
/// anything the caller does not mention is preserved, so `ov peer link` does not
/// erase settings and vice versa.
pub fn write_entry(
    root: &str,
    patch: Map<String, Value>,
    identity: Option<&WorkspaceIdentity>,
    env: &WorkspaceEnv,
    now: i64,
) -> Result<(PathBuf, Value)> {
    let path = entry_path(root, identity, env);
    // Several copies of this reader ship independently, so a newer client's
    // entry can be sitting here. Refuse rather than flatten it.
    if let Some(on_disk) = read_raw_entry(&path)
        && on_disk.get("version").and_then(Value::as_u64) != Some(REGISTRY_VERSION)
    {
        let declared = on_disk
            .get("version")
            .map(ToString::to_string)
            .unwrap_or_else(|| "undefined".to_string());
        return Err(Error::Config(format!(
            "{} was written by a newer client (version {declared}); refusing to overwrite it",
            path.display()
        )));
    }

    let existing = read_entry(root, identity, env);
    let previous = existing
        .entry
        .as_ref()
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let mut entry = previous.clone();
    for (key, value) in patch {
        entry.insert(key, value);
    }
    entry.insert("version".to_string(), Value::from(REGISTRY_VERSION));
    entry.insert("root".to_string(), Value::from(root));
    entry.insert(
        "identity".to_string(),
        Value::from(match identity {
            Some(identity) => identity_key(identity),
            None => previous
                .get("identity")
                .and_then(Value::as_str)
                .unwrap_or("path")
                .to_string(),
        }),
    );
    let label = identity
        .map(|identity| identity.remote.clone())
        .filter(|remote| !remote.is_empty())
        .or_else(|| {
            previous
                .get("label")
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
        .unwrap_or_default();
    entry.insert("label".to_string(), Value::from(label));
    // `previous.first_seen_at || now`: a hand-edited `0` or `null` is not a
    // timestamp, so it is restamped rather than carried forward.
    let first_seen = previous
        .get("first_seen_at")
        .filter(|value| is_truthy(value))
        .cloned()
        .unwrap_or_else(|| Value::from(now));
    entry.insert("first_seen_at".to_string(), first_seen);
    entry.insert("last_seen_at".to_string(), Value::from(now));

    if let Some(profile) = entry.get("cli_config_profile")
        && !profile.as_str().is_some_and(is_valid_profile_name)
    {
        return Err(Error::Config(format!(
            "cli_config_profile must match ^[a-z0-9][a-z0-9._-]{{0,63}}$, got {profile}"
        )));
    }

    let entry = Value::Object(entry);
    let dir = env.registry_dir();
    fs::create_dir_all(&dir)?;
    let body = format!("{}\n", serde_json::to_string_pretty(&entry)?);
    write_entry_atomically(&path, body.as_bytes())?;
    Ok((path, entry))
}

fn write_entry_atomically(path: &Path, content: &[u8]) -> Result<()> {
    use std::io::Write;

    let parent = path
        .parent()
        .ok_or_else(|| Error::Config("Could not determine registry directory".to_string()))?;
    let mut temp = tempfile::Builder::new()
        .prefix(".ov-workspace.")
        .suffix(".tmp")
        .tempfile_in(parent)?;
    temp.write_all(content)?;
    temp.flush()?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = temp.as_file().metadata()?.permissions();
        permissions.set_mode(0o600);
        temp.as_file().set_permissions(permissions)?;
    }

    temp.as_file().sync_all()?;
    temp.persist(path).map_err(|error| {
        Error::Config(format!("Failed to replace registry entry: {}", error.error))
    })?;
    Ok(())
}

/// Record a peer this workspace used to write under, so a later `ov peer` run
/// and doctor can point at memories the current peer no longer reaches.
pub fn remember_previous_peer(
    root: &str,
    peer_id: &str,
    identity: Option<&WorkspaceIdentity>,
    env: &WorkspaceEnv,
    now: i64,
) -> Result<bool> {
    let id = peer_id.trim();
    if id.is_empty() {
        return Ok(false);
    }
    let previous = read_entry(root, identity, env);
    let mut seen = previous
        .entry
        .as_ref()
        .and_then(|entry| entry.get("previous_peer_ids"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if seen.iter().any(|item| item.as_str() == Some(id)) {
        return Ok(false);
    }
    seen.push(Value::from(id));
    let keep = seen.len().saturating_sub(20);
    let mut patch = Map::new();
    patch.insert(
        "previous_peer_ids".to_string(),
        Value::Array(seen.split_off(keep)),
    );
    write_entry(root, patch, identity, env, now)?;
    Ok(true)
}

// ---------------------------------------------------------------------------
// The resolved view every command works from
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LayerReport {
    pub layer: String,
    pub path: String,
    pub exists: bool,
    pub applied: bool,
}

#[derive(Debug, Clone)]
pub struct ResolvedWorkspace {
    pub identity: WorkspaceIdentity,
    pub layers: Vec<LayerReport>,
    pub merged: MergeResult,
    pub peer: EffectivePeer,
    pub previous_peer_ids: Vec<String>,
    pub registry_path: PathBuf,
    pub registry_exists: bool,
    /// The stored entry named a different repository, so nothing was inherited
    /// from it — the negative evidence that keeps two projects from crossing.
    pub registry_conflict: bool,
    pub warnings: Vec<String>,
}

impl ResolvedWorkspace {
    pub fn root(&self) -> Option<&str> {
        Some(self.identity.root.as_str()).filter(|root| !root.is_empty())
    }

    pub fn require_root(&self) -> Result<&str> {
        self.root().ok_or_else(|| {
            let language = Language::current();
            Error::Client(
                copy(
                    language,
                    "No workspace root here: the home directory and the filesystem root are never workspaces. Run the command from a project directory.",
                    "此处没有工作区根目录：主目录与文件系统根目录不能作为工作区。请在项目目录中运行该命令。",
                )
                .to_string(),
            )
        })
    }
}

/// Everything `ov workspace show` and `ov peer` need, resolved in one pass.
///
/// `harness` optionally applies the `plugin.<harness>` sub-object of ovcli.conf
/// on top of the shared `plugin` keys, matching what a harness loader sees.
pub fn resolve_workspace(
    cwd: &str,
    harness: Option<&str>,
    env: &WorkspaceEnv,
) -> ResolvedWorkspace {
    let identity = resolve_workspace_identity(cwd, env);
    let mut warnings = Vec::new();
    let mut layers = Vec::new();
    let mut reports = Vec::new();

    // ovcli.conf is the trusted file that owns `url` and `api_key`, so its
    // `plugin` section is read raw rather than through the workspace rules. It
    // is also the reason `KNOWN_CONFIG_KEYS` exists: `plugin` is not a key
    // `Config` models, which is what keeps `ov config edit` from deleting it.
    let cli_config = read_raw_entry(&env.cli_config_file);
    let plugin = cli_config
        .as_ref()
        .and_then(|value| value.get("plugin"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let shared: Map<String, Value> = plugin
        .iter()
        .filter(|(_, value)| !value.is_object())
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    reports.push(LayerReport {
        layer: "ovcli.conf plugin".to_string(),
        path: env.cli_config_file.display().to_string(),
        exists: cli_config.is_some(),
        applied: !shared.is_empty(),
    });
    if !shared.is_empty() {
        layers.push(ConfigLayer {
            layer: "ovcli.conf plugin".to_string(),
            data: plugin_layer(&shared),
        });
    }

    let scoped = harness
        .and_then(|harness| plugin.get(harness))
        .and_then(Value::as_object)
        .cloned();
    reports.push(LayerReport {
        layer: format!("ovcli.conf plugin.{}", harness.unwrap_or("<harness>")),
        path: env.cli_config_file.display().to_string(),
        exists: scoped.is_some(),
        applied: scoped.is_some(),
    });
    if let Some(scoped) = scoped.as_ref() {
        layers.push(ConfigLayer {
            layer: format!("ovcli.conf plugin.{}", harness.unwrap_or_default()),
            data: plugin_layer(scoped),
        });
    }

    if !identity.root.is_empty() {
        let root = Path::new(&identity.root);
        for (layer, path) in workspace_config_paths(root) {
            let file = read_workspace_file(&path, Some(root), false);
            warnings.extend(file.warnings.clone());
            reports.push(LayerReport {
                layer: layer.clone(),
                path: path.display().to_string(),
                exists: file.exists,
                applied: file.data.is_some(),
            });
            if let Some(mut data) = file.data {
                // Metadata, not a setting: it is lifted out before the layer is
                // merged, and it only ever warns.
                let declared = data
                    .as_object_mut()
                    .and_then(|object| object.shift_remove("min_client_version"));
                if let Some(declared) = declared.as_ref().and_then(Value::as_str) {
                    check_min_client_version(declared, CLIENT_VERSION, &mut warnings);
                }
                layers.push(ConfigLayer { layer, data });
            }
        }
    }

    let registry = read_entry(&identity.root, Some(&identity), env);
    warnings.extend(registry.warnings.clone());
    let registry_exists = registry.entry.is_some();
    if let Some(entry) = registry.entry.as_ref() {
        if let Some(settings) = entry.get("settings").filter(|value| value.is_object()) {
            layers.push(ConfigLayer {
                layer: "registry".to_string(),
                data: settings.clone(),
            });
        }
        if let Some(peer) = entry.get("peer").filter(|value| value.is_object()) {
            let mut data = Map::new();
            data.insert("peer".to_string(), peer.clone());
            layers.push(ConfigLayer {
                layer: "registry".to_string(),
                data: Value::Object(data),
            });
        }
    }
    reports.push(LayerReport {
        layer: "registry".to_string(),
        path: registry.path.display().to_string(),
        exists: registry_exists,
        applied: registry_exists,
    });

    let env_layer = env_config_layer(env);
    reports.push(LayerReport {
        layer: "environment".to_string(),
        path: "OPENVIKING_*".to_string(),
        exists: !env.vars.is_empty(),
        applied: env_layer.is_some(),
    });
    if let Some(data) = env_layer {
        layers.push(ConfigLayer {
            layer: "environment".to_string(),
            data,
        });
    }

    let mut merged = merge_config_layers(&layers, &mut warnings);
    normalize_workspace_config(&mut merged.value, &mut warnings);

    // `envBool("OPENVIKING_WORKSPACE_PEER") ?? (cfg.workspacePeer !== false)`.
    // `workspacePeer` is not a workspace-schema key — only ovcli.conf's plugin
    // section carries it — and only an explicit `false` there turns the derived
    // peer off.
    let workspace_peer_enabled = match env
        .vars
        .get("OPENVIKING_WORKSPACE_PEER")
        .map(String::as_str)
    {
        Some("0" | "false" | "no" | "off") => false,
        Some(_) => true,
        None => {
            scoped
                .as_ref()
                .and_then(|scoped| scoped.get("workspacePeer"))
                .or_else(|| plugin.get("workspacePeer"))
                .and_then(Value::as_bool)
                != Some(false)
        }
    };
    let peer = resolve_effective_peer_id(&merged.value, workspace_peer_enabled, &identity);
    let previous_peer_ids = registry
        .entry
        .as_ref()
        .and_then(|entry| entry.get("previous_peer_ids"))
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default();

    ResolvedWorkspace {
        identity,
        layers: reports,
        merged,
        peer,
        previous_peer_ids,
        registry_path: registry.path,
        registry_exists,
        registry_conflict: registry.conflict,
        warnings,
    }
}

/// ovcli.conf's `plugin` section speaks the flat knob names a harness loader
/// reads (`peerId`, `recallLimit`), while a workspace file spells the same
/// settings in the nested schema (`peer.id`, `recall.max_items`) that
/// `projectWorkspaceSettings` flattens onto those knobs. The two meet in one
/// precedence chain, so the flat end has to be lifted into the schema before it
/// can be merged with — and shadowed by — the layers above it.
///
/// This is the inverse of `KNOB_MAP` in `workspace-config.mjs`.
const PLUGIN_KNOBS: &[(&str, &str)] = &[
    // `peer_id` first: the loaders read `cfg.peerId` and fall back to
    // `cfg.peer_id`, so the camelCase spelling has to land last and win.
    ("peer_id", "peer.id"),
    ("peerId", "peer.id"),
    ("peerSource", "peer.source"),
    ("autoRecall", "recall.enabled"),
    ("recallPeerScope", "recall.peer_scope"),
    ("recallDedupTurns", "recall.dedup_turns"),
    ("recallLimit", "recall.max_items"),
    ("scoreThreshold", "recall.score_threshold"),
    ("autoCapture", "capture.enabled"),
    ("commitTokenThreshold", "capture.commit_token_threshold"),
    ("bypassSessionPatterns", "bypass.session_patterns"),
];

/// One `plugin` (or `plugin.<harness>`) object, in the workspace schema.
///
/// A key that names no knob rides along verbatim, the way the JS loader spreads
/// the section over its settings: a harness may add its own without this reader
/// having to learn it first.
fn plugin_layer(section: &Map<String, Value>) -> Value {
    let mut out = Map::new();
    for (key, value) in section {
        // A key that shares a knob's section name is dropped rather than passed
        // through: `plugin: { peer: { id } }` is the workspace-file schema in
        // the wrong file, and the JS loader — which only ever reads `peerId`
        // here — pins nothing from it. Copying it in would let `ov` report a
        // pin no plugin honours.
        let shadows_a_section = PLUGIN_KNOBS
            .iter()
            .any(|(knob, path)| knob == key || path.split('.').next() == Some(key.as_str()));
        if shadows_a_section {
            continue;
        }
        out.insert(key.clone(), value.clone());
    }
    let mut out = Value::Object(out);
    for (knob, path) in PLUGIN_KNOBS {
        if let Some(value) = section.get(*knob) {
            config_set(&mut out, path, value.clone());
        }
    }
    out
}

fn env_config_layer(env: &WorkspaceEnv) -> Option<Value> {
    let mut value = Value::Object(Map::new());
    let mut any = false;
    for (name, key) in ENV_OVERRIDES {
        let Some(raw) = env.vars.get(*name) else {
            continue;
        };
        let parsed = if ENV_BOOLEAN_KEYS.contains(key) {
            match raw.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => Value::Bool(true),
                "0" | "false" | "no" | "off" => Value::Bool(false),
                _ => continue,
            }
        } else if *key == "bypass.session_patterns" {
            Value::Array(
                raw.split(',')
                    .map(str::trim)
                    .filter(|item| !item.is_empty())
                    .map(Value::from)
                    .collect(),
            )
        } else {
            Value::String(raw.clone())
        };
        config_set(&mut value, key, parsed);
        any = true;
    }
    any.then_some(value)
}

// ---------------------------------------------------------------------------
// `ov workspace show`
// ---------------------------------------------------------------------------

pub fn show(cwd: &str, harness: Option<&str>, output_format: OutputFormat, compact: bool) {
    let env = WorkspaceEnv::from_process();
    let resolved = resolve_workspace(cwd, harness, &env);

    if matches!(output_format, OutputFormat::Json) {
        output_success(show_json(&resolved), output_format, compact);
        return;
    }
    print!("{}", render_show(&resolved, Language::current()));
}

fn show_json(resolved: &ResolvedWorkspace) -> Value {
    let identity = &resolved.identity;
    serde_json::json!({
        "cwd": identity.cwd,
        "root": identity.root,
        "found": if identity.is_git { identity.git_kind.clone() } else { "none".to_string() },
        "git_dir": identity.git_dir,
        "git_common_dir": identity.git_common_dir,
        "remote": identity.remote,
        "vars": identity.vars,
        "peer": {
            "id": resolved.peer.peer_id,
            "source": resolved.peer.source,
            "origin": resolved.peer.origin,
            "pinned_by": resolved.merged.get("peer.id").map(|entry| entry.source.clone()),
            "legacy_id": resolved.peer.legacy_peer_id,
            "previous_ids": resolved.previous_peer_ids,
        },
        "registry": {
            "path": resolved.registry_path.display().to_string(),
            "exists": resolved.registry_exists,
            "conflict": resolved.registry_conflict,
        },
        "layers": resolved.layers.iter().map(|layer| serde_json::json!({
            "layer": layer.layer,
            "path": layer.path,
            "exists": layer.exists,
            "applied": layer.applied,
        })).collect::<Vec<_>>(),
        "keys": resolved.merged.provenance.iter().map(|entry| serde_json::json!({
            "key": entry.key,
            "value": effective_value(resolved, entry),
            "written": entry.value,
            "source": entry.source,
            "shadowed": entry.shadowed.iter().map(|shadowed| serde_json::json!({
                "value": shadowed.value,
                "source": shadowed.source,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "warnings": resolved.warnings,
    })
}

fn render_show(resolved: &ResolvedWorkspace, language: Language) -> String {
    let identity = &resolved.identity;
    let mut out = String::new();

    out.push_str(&format!(
        "{}\n",
        theme::heading(copy(language, "Workspace", "工作区")).bold()
    ));
    if identity.root.is_empty() {
        out.push_str(&format!(
            "  {}\n",
            theme::muted(copy(
                language,
                "no workspace root (the home directory and the filesystem root are never workspaces)",
                "没有工作区根目录（主目录与文件系统根目录不作为工作区）",
            ))
        ));
    } else {
        out.push_str(&field(
            copy(language, "root", "根目录"),
            &format!(
                "{} ({})",
                identity.root,
                if identity.git_kind.is_empty() {
                    copy(language, "no repository", "非仓库")
                } else {
                    &identity.git_kind
                }
            ),
        ));
        if !identity.git_common_dir.is_empty() {
            out.push_str(&field(
                copy(language, "git dir", "git 目录"),
                &identity.git_common_dir,
            ));
        }
        out.push_str(&field(
            copy(language, "remote", "远端"),
            if identity.remote.is_empty() {
                copy(language, "(none)", "（无）")
            } else {
                &identity.remote
            },
        ));
    }

    out.push('\n');
    out.push_str(&format!(
        "{}\n",
        theme::heading(copy(language, "Template variables", "模板变量")).bold()
    ));
    for (name, value) in &identity.vars {
        out.push_str(&field(
            &format!("{{{name}}}"),
            if value.is_empty() {
                copy(language, "(empty)", "（空）")
            } else {
                value
            },
        ));
    }

    out.push('\n');
    out.push_str(&format!("{}\n", theme::heading("Peer").bold()));
    out.push_str(&field(
        copy(language, "effective", "生效值"),
        &if resolved.peer.peer_id.is_empty() {
            format!(
                "{} ({})",
                copy(language, "(none)", "（无）"),
                resolved.peer.origin
            )
        } else {
            format!("{} ({})", resolved.peer.peer_id, resolved.peer.origin)
        },
    ));
    if let Some(entry) = resolved.merged.get("peer.id") {
        out.push_str(&field(copy(language, "pinned by", "固定于"), &entry.source));
    }
    if !resolved.peer.legacy_peer_id.is_empty() {
        out.push_str(&field(
            copy(language, "legacy", "旧值"),
            &resolved.peer.legacy_peer_id,
        ));
    }
    if !resolved.previous_peer_ids.is_empty() {
        out.push_str(&field(
            copy(language, "previous", "此前使用"),
            &resolved.previous_peer_ids.join(", "),
        ));
    }

    out.push('\n');
    out.push_str(&format!(
        "{}\n",
        theme::heading(copy(
            language,
            "Config layers (lowest precedence first)",
            "配置层（优先级从低到高）",
        ))
        .bold()
    ));
    if resolved.registry_conflict {
        out.push_str(&format!(
            "  {} {}\n",
            theme::warning("!"),
            copy(
                language,
                "the registry entry was recorded for a different repository; nothing was inherited from it",
                "注册表条目属于另一个仓库，其中的配置未被继承",
            )
        ));
    }
    for layer in &resolved.layers {
        let state = match (layer.exists, layer.applied) {
            (_, true) => copy(language, "applied", "已应用"),
            (true, false) => copy(language, "present, nothing to apply", "存在但无内容"),
            (false, false) => copy(language, "missing", "不存在"),
        };
        out.push_str(&format!(
            "  {}\n      {} {}\n",
            theme::value(&layer.layer),
            theme::muted(&layer.path),
            theme::muted(format!("[{state}]"))
        ));
    }

    out.push('\n');
    out.push_str(&format!(
        "{}\n",
        theme::heading(copy(language, "Effective settings", "生效配置")).bold()
    ));
    if resolved.merged.provenance.is_empty() {
        out.push_str(&format!(
            "  {}\n",
            theme::muted(copy(
                language,
                "no layer sets anything",
                "没有任何层设置了配置"
            ))
        ));
    }
    for entry in &resolved.merged.provenance {
        // Provenance records what a layer wrote; the merged value is what
        // survived clamping and enum checks. Report the latter, and say so when
        // normalization threw the key away entirely.
        let (effective, note) = match effective_value(resolved, entry) {
            Some(value) => (value.to_string(), String::new()),
            None => (
                entry.value.to_string(),
                format!(" {}", copy(language, "(ignored)", "（已忽略）")),
            ),
        };
        out.push_str(&format!(
            "  {} = {}{}\n      {}\n",
            theme::value(&entry.key),
            theme::sky_value(effective),
            theme::warning(note),
            theme::muted(format!("from {}", entry.source))
        ));
        for shadowed in &entry.shadowed {
            out.push_str(&format!(
                "      {}\n",
                theme::muted(format!(
                    "shadowed {} from {}",
                    shadowed.value, shadowed.source
                ))
            ));
        }
    }

    if !resolved.warnings.is_empty() {
        out.push('\n');
        out.push_str(&format!(
            "{}\n",
            theme::heading(copy(language, "Warnings", "警告")).bold()
        ));
        for warning in &resolved.warnings {
            out.push_str(&format!("  {} {}\n", theme::warning("!"), warning));
        }
    }

    out
}

/// The value in force for one provenance key, or nothing when normalization
/// dropped it. A `(section)` marker has no scalar to look up.
fn effective_value<'a>(
    resolved: &'a ResolvedWorkspace,
    entry: &'a Provenance,
) -> Option<&'a Value> {
    if entry.value == Value::String("(section)".to_string()) {
        return Some(&entry.value);
    }
    config_get(&resolved.merged.value, &entry.key)
}

/// Pad on the label's display width, not its byte count: the Chinese labels are
/// double-width and the colour codes are zero-width.
fn field(label: &str, value: &str) -> String {
    format!(
        "  {} {}\n",
        theme::muted(pad_to_display_width(label, 14)),
        theme::value(value)
    )
}

// ---------------------------------------------------------------------------
// Shared helpers for `ov peer`
// ---------------------------------------------------------------------------

/// The server's own rule for an identifier part
/// (`openviking/core/identifiers.py: validate_identifier_part`).
pub fn validate_peer_id(value: &str) -> Result<()> {
    let language = Language::current();
    if value.is_empty() {
        return Err(Error::Client(
            copy(language, "peer id is empty", "peer id 为空").to_string(),
        ));
    }
    if value == "." || value == ".." {
        return Err(Error::Client(
            copy(
                language,
                "peer id must not be '.' or '..'",
                "peer id 不能是 '.' 或 '..'",
            )
            .to_string(),
        ));
    }
    if !value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.' | '@' | '-'))
    {
        return Err(Error::Client(
            copy(
                language,
                "peer id may only contain letters, digits, and _ . @ -",
                "peer id 只能包含字母、数字以及 _ . @ -",
            )
            .to_string(),
        ));
    }
    if value.matches('@').count() > 1 {
        return Err(Error::Client(
            copy(
                language,
                "peer id must have at most one @",
                "peer id 最多只能包含一个 @",
            )
            .to_string(),
        ));
    }
    Ok(())
}

pub fn now_millis() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

/// The authenticated user, which decides the `viking://user/<user>/…` prefix.
pub async fn resolve_user(client: &HttpClient, configured: Option<&str>) -> Result<String> {
    if let Some(user) = configured.map(str::trim).filter(|user| !user.is_empty()) {
        return Ok(user.to_string());
    }
    let status: Value = client.get("/api/v1/system/status", &[]).await?;
    status
        .get("user")
        .and_then(Value::as_str)
        .filter(|user| !user.is_empty())
        .map(ToString::to_string)
        .ok_or_else(|| {
            Error::Client(
                copy(
                    Language::current(),
                    "The server did not report a user id; set `user` in ovcli.conf.",
                    "服务端未返回 user id；请在 ovcli.conf 中设置 user。",
                )
                .to_string(),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of this module is that `ov` and the plugins agree on what
    /// a config file means. The key lists are the part most likely to drift —
    /// they are edited by whoever is fixing a bug in one language — so read them
    /// out of the JS source and compare rather than trusting a comment.
    fn js_string_array(source: &str, name: &str) -> Vec<String> {
        let start = source
            .find(&format!("export const {name} = ["))
            .unwrap_or_else(|| panic!("{name} is missing from the JS module"));
        let body = &source[start..];
        let end = body.find("];").expect("unterminated array");
        // Every quoted string inside the array literal, so a one-line array and
        // a one-per-line array both read the same. Comments in these blocks
        // never contain a double quote.
        body[..end]
            .split('"')
            .skip(1)
            .step_by(2)
            .map(str::to_string)
            .collect()
    }

    fn shared_lib(file: &str) -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/memory-plugin-shared/lib")
            .join(file);
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
    }

    #[test]
    fn the_key_lists_match_the_javascript_reader() {
        let source = shared_lib("workspace-config.mjs");
        assert_eq!(js_string_array(&source, "FORBIDDEN_KEYS"), FORBIDDEN_KEYS);
        assert_eq!(
            js_string_array(&source, "REGISTRY_ONLY_KEYS"),
            REGISTRY_ONLY_KEYS
        );
        assert_eq!(
            js_string_array(&source, "FREE_FORM_SECTIONS"),
            FREE_FORM_SECTIONS
        );
    }
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_dir(name: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be valid")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("ov-workspace-{name}-{suffix}"));
        fs::create_dir_all(&dir).expect("temp dir");
        fs::canonicalize(&dir).expect("temp dir should resolve")
    }

    fn test_env(home: &Path) -> WorkspaceEnv {
        WorkspaceEnv {
            home: home.to_path_buf(),
            openviking_home: home.join(".openviking"),
            cli_config_file: home.join(".openviking").join("ovcli.conf"),
            vars: BTreeMap::new(),
        }
    }

    fn make_repo(root: &Path, remote: Option<&str>) -> PathBuf {
        let git_dir = root.join(".git");
        fs::create_dir_all(&git_dir).expect("git dir");
        let config = match remote {
            Some(remote) => format!(
                "[core]\n\trepositoryformatversion = 0\n[remote \"origin\"]\n\turl = {remote}\n"
            ),
            None => "[core]\n\trepositoryformatversion = 0\n".to_string(),
        };
        fs::write(git_dir.join("config"), config).expect("git config");
        git_dir
    }

    fn layer(name: &str, data: Value) -> ConfigLayer {
        ConfigLayer {
            layer: name.to_string(),
            data,
        }
    }

    #[test]
    fn legacy_sanitize_keeps_the_byte_for_byte_rule() {
        assert_eq!(
            legacy_sanitize("/Users/x/Dev/OpenViking"),
            "-Users-x-Dev-OpenViking"
        );
        assert_eq!(legacy_sanitize("/tmp/a  b/"), "-tmp-a--b-");
        assert_eq!(legacy_sanitize("abc.DEF_123@x-y"), "abc-DEF-123-x-y");
        assert_eq!(legacy_sanitize(""), "");
        assert_eq!(legacy_sanitize("///"), "---");
        // One dash per UTF-16 unit, the way the JS reader counts.
        assert_eq!(legacy_sanitize("项目"), "--");
    }

    #[test]
    fn sanitize_peer_id_produces_a_server_valid_id_and_dodges_reserved_names() {
        assert_eq!(
            sanitize_peer_id("github.com/volcengine/openviking"),
            "github.com-volcengine-openviking"
        );
        assert_eq!(sanitize_peer_id("--weird//name--"), "weird-name");
        assert_eq!(sanitize_peer_id("__self"), "self");
        assert_eq!(sanitize_peer_id("ext-YWJj"), "x-ext-YWJj");
        assert_eq!(sanitize_peer_id("a@b@c"), "a@b-c");
        assert_eq!(sanitize_peer_id(".."), "");
        assert_eq!(sanitize_peer_id("///"), "");
    }

    #[test]
    fn sanitize_peer_id_keeps_long_ids_unique_after_truncation() {
        let a = sanitize_peer_id(&format!("git.example.com/{}/one", "a".repeat(200)));
        let b = sanitize_peer_id(&format!("git.example.com/{}/two", "a".repeat(200)));
        assert!(
            a.len() <= 100 && b.len() <= 100,
            "{} / {}",
            a.len(),
            b.len()
        );
        assert_ne!(a, b, "truncation must not collapse two repos into one peer");
    }

    #[test]
    fn normalize_git_remote_folds_every_spelling_of_one_repo_together() {
        for url in [
            "git@github.com:volcengine/OpenViking.git",
            "https://github.com/volcengine/OpenViking.git",
            "https://github.com/volcengine/OpenViking",
            "https://GitHub.com/Volcengine/OpenViking.git/",
            "ssh://git@github.com:22/volcengine/OpenViking.git",
        ] {
            assert_eq!(
                normalize_git_remote(url),
                "github.com/volcengine/openviking",
                "failed on {url}"
            );
        }
    }

    #[test]
    fn normalize_git_remote_drops_userinfo_so_a_token_never_reaches_the_peer_id() {
        let normalized = normalize_git_remote(
            "https://someone:ghp_averysecrettoken@github.com:8443/volcengine/OpenViking.git",
        );
        assert_eq!(normalized, "github.com/volcengine/openviking");
        assert!(!normalized.contains("ghp_") && !normalized.contains("someone"));
    }

    #[test]
    fn normalize_git_remote_refuses_identities_that_are_only_local() {
        for url in [
            "",
            "   ",
            "/srv/git/bare.git",
            "file:///srv/git/bare.git",
            "../sibling",
            "C:\\src\\repo",
            "C:/src/repo",
            "d:\\work\\api.git",
        ] {
            assert_eq!(normalize_git_remote(url), "", "should be empty for {url:?}");
        }
    }

    #[test]
    fn read_git_remote_url_reads_only_origin_and_does_not_follow_includes() {
        let dir = unique_dir("gitconfig");
        let git_dir = dir.join(".git");
        fs::create_dir_all(&git_dir).expect("git dir");
        fs::write(
            git_dir.join("config"),
            [
                "[include]",
                "\tpath = ./extra",
                "# a comment mentioning url = https://decoy.example/nope.git",
                "[remote \"upstream\"]",
                "\turl = git@github.com:volcengine/OpenViking.git",
                "[remote \"origin\"]",
                "\turl = git@github.com:t0saki/OpenViking.git",
                "",
            ]
            .join("\n"),
        )
        .expect("config");
        fs::write(
            git_dir.join("extra"),
            "[remote \"origin\"]\n\turl = https://included.example/x.git\n",
        )
        .expect("extra");

        assert_eq!(
            read_git_remote_url(&git_dir, "origin"),
            "git@github.com:t0saki/OpenViking.git"
        );
        assert_eq!(
            read_git_remote_url(&git_dir, "upstream"),
            "git@github.com:volcengine/OpenViking.git"
        );
        assert_eq!(read_git_remote_url(&git_dir, "missing"), "");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_trailing_git_comment_is_not_part_of_the_url() {
        let dir = unique_dir("comment");
        let git_dir = dir.join(".git");
        fs::create_dir_all(&git_dir).expect("git dir");
        fs::write(
            git_dir.join("config"),
            "[remote \"origin\"]\n\turl = git@github.com:a/b.git # my fork\n",
        )
        .expect("config");

        assert_eq!(
            read_git_remote_url(&git_dir, "origin"),
            "git@github.com:a/b.git"
        );
        assert_eq!(
            normalize_git_remote(&read_git_remote_url(&git_dir, "origin")),
            "github.com/a/b"
        );
        fs::remove_dir_all(&dir).ok();
    }

    // `^\[([^\]]+)\]` needs both brackets and something between them. A line
    // that only looks like a header is ordinary text: it neither opens a
    // section nor closes the one already open. Each expectation here is what
    // the JS `readGitRemoteUrl` returns for the same file.
    #[test]
    fn a_line_that_is_not_a_section_header_leaves_the_section_alone() {
        let dir = unique_dir("headers");
        let git_dir = dir.join(".git");
        fs::create_dir_all(&git_dir).expect("git dir");

        let read = |text: &str| {
            fs::write(git_dir.join("config"), text).expect("config");
            read_git_remote_url(&git_dir, "origin")
        };

        // An unterminated header never opens `origin`, so nothing is read.
        assert_eq!(read("[remote \"origin\"\n\turl = https://a/b\n"), "");
        // …and it does not close it either.
        assert_eq!(
            read("[remote \"origin\"]\n[\n\turl = https://a/b\n"),
            "https://a/b"
        );
        assert_eq!(
            read("[remote \"origin\"]\n[]\n\turl = https://a/b\n"),
            "https://a/b"
        );
        // One quote at each end, not a run of them.
        assert_eq!(
            read("[remote \"origin\"]\n\turl = \"https://a/b\"\"\n"),
            "https://a/b\""
        );
        assert_eq!(
            read("[remote \"origin\"]\n\turl = 'https://a/b'\n"),
            "https://a/b"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_workspace_root_is_the_repo_from_any_depth_below_it() {
        let root = unique_dir("root");
        make_repo(&root, Some("git@github.com:volcengine/OpenViking.git"));
        let deep = root.join("examples").join("codex-memory-plugin");
        fs::create_dir_all(&deep).expect("deep dir");
        let env = test_env(Path::new("/nonexistent-home"));

        for cwd in [root.clone(), root.join("examples"), deep] {
            let (found, git) = find_workspace_root(&cwd.to_string_lossy(), &env);
            assert_eq!(
                found.as_deref(),
                Some(root.as_path()),
                "wrong root from {cwd:?}"
            );
            assert_eq!(git.expect("git").kind, "repo");
        }
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_linked_worktree_resolves_back_to_the_repository_it_shares() {
        let main = unique_dir("main");
        let main_git = make_repo(&main, Some("git@github.com:volcengine/OpenViking.git"));
        let worktree_git_dir = main_git.join("worktrees").join("feature");
        fs::create_dir_all(&worktree_git_dir).expect("worktree git dir");
        fs::write(worktree_git_dir.join("commondir"), "../..\n").expect("commondir");

        let linked = unique_dir("linked");
        fs::write(
            linked.join(".git"),
            format!("gitdir: {}\n", worktree_git_dir.display()),
        )
        .expect("dot git file");

        let env = test_env(Path::new("/nonexistent-home"));
        let (_, git) = find_workspace_root(&linked.to_string_lossy(), &env);
        let git = git.expect("worktree git");
        assert_eq!(git.kind, "worktree");
        assert_eq!(git.common_dir, main_git);
        assert_eq!(
            read_git_remote_url(&git.common_dir, "origin"),
            "git@github.com:volcengine/OpenViking.git"
        );
        fs::remove_dir_all(&main).ok();
        fs::remove_dir_all(&linked).ok();
    }

    #[test]
    fn a_submodule_keeps_its_own_identity_instead_of_the_superprojects() {
        let parent = unique_dir("parent");
        let parent_git = make_repo(&parent, Some("git@github.com:volcengine/OpenViking.git"));
        let module_git_dir = parent_git.join("modules").join("vendor");
        fs::create_dir_all(&module_git_dir).expect("module git dir");
        fs::write(
            module_git_dir.join("config"),
            "[remote \"origin\"]\n\turl = git@github.com:other/vendor.git\n",
        )
        .expect("module config");

        let sub = parent.join("vendor");
        fs::create_dir_all(&sub).expect("sub dir");
        fs::write(
            sub.join(".git"),
            format!("gitdir: {}\n", module_git_dir.display()),
        )
        .expect("dot git file");

        let env = test_env(Path::new("/nonexistent-home"));
        let (_, git) = find_workspace_root(&sub.to_string_lossy(), &env);
        let git = git.expect("submodule git");
        assert_eq!(git.kind, "submodule");
        assert_eq!(
            normalize_git_remote(&read_git_remote_url(&git.common_dir, "origin")),
            "github.com/other/vendor"
        );
        fs::remove_dir_all(&parent).ok();
    }

    #[test]
    fn a_repository_under_a_directory_named_modules_is_not_a_submodule() {
        let container = unique_dir("modules-container");
        let main = container.join("modules").join("app");
        fs::create_dir_all(&main).expect("main dir");
        let main_git = make_repo(&main, Some("git@github.com:o/app.git"));
        let worktree_git_dir = main_git.join("worktrees").join("feature");
        fs::create_dir_all(&worktree_git_dir).expect("worktree git dir");
        fs::write(worktree_git_dir.join("commondir"), "../..\n").expect("commondir");

        let linked = unique_dir("modules-linked");
        fs::write(
            linked.join(".git"),
            format!("gitdir: {}\n", worktree_git_dir.display()),
        )
        .expect("dot git file");

        let env = test_env(Path::new("/nonexistent-home"));
        let (_, git) = find_workspace_root(&linked.to_string_lossy(), &env);
        let git = git.expect("worktree git");
        assert_eq!(git.kind, "worktree");
        assert_eq!(
            normalize_git_remote(&read_git_remote_url(&git.common_dir, "origin")),
            "github.com/o/app"
        );
        fs::remove_dir_all(&container).ok();
        fs::remove_dir_all(&linked).ok();
    }

    #[test]
    fn home_and_the_filesystem_root_are_never_workspace_roots() {
        let home = unique_dir("home");
        make_repo(&home, Some("git@github.com:someone/dotfiles.git"));
        let inside = home.join("notes");
        fs::create_dir_all(&inside).expect("notes dir");
        let env = test_env(&home);

        assert_eq!(find_workspace_root(&home.to_string_lossy(), &env).0, None);
        assert_eq!(find_workspace_root("/", &env).0, None);
        // A stray repository at `$HOME` must not claim the directories beneath
        // it: `notes` is its own workspace with no git identity.
        let (root, git) = find_workspace_root(&inside.to_string_lossy(), &env);
        assert_eq!(root.as_deref(), Some(inside.as_path()));
        assert!(git.is_none(), "it must not inherit the repository at $HOME");
        fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn a_directory_outside_any_repository_is_still_its_own_workspace() {
        let plain = unique_dir("plain-workspace");
        let env = test_env(Path::new("/nonexistent-home"));

        let (root, git) = find_workspace_root(&plain.to_string_lossy(), &env);
        assert_eq!(
            root.as_deref(),
            Some(plain.as_path()),
            "a config file here has to be readable"
        );
        assert!(git.is_none());

        // …and the config file there really does apply.
        let config_dir = plain.join(CONFIG_DIR_NAME);
        fs::create_dir_all(&config_dir).expect("config dir");
        fs::write(
            config_dir.join(TEAM_FILE),
            serde_json::json!({ "version": 1, "recall": { "max_items": 9 } }).to_string(),
        )
        .expect("team file");
        let resolved = resolve_workspace(&plain.to_string_lossy(), None, &env);
        assert_eq!(
            config_get(&resolved.merged.value, "recall.max_items"),
            Some(&Value::from(9))
        );
        assert_eq!(resolved.identity.vars["git_root"], "");
        fs::remove_dir_all(&plain).ok();
    }

    #[test]
    fn git_folds_section_and_key_case_but_not_a_quoted_subsection() {
        let dir = unique_dir("case");
        let git_dir = dir.join(".git");
        fs::create_dir_all(&git_dir).expect("git dir");
        fs::write(
            git_dir.join("config"),
            "[Remote \"origin\"]\n\tURL = git@github.com:Org/Repo.git\n",
        )
        .expect("config");

        assert_eq!(
            read_git_remote_url(&git_dir, "origin"),
            "git@github.com:Org/Repo.git",
            "`git config` reads this file fine"
        );
        assert_eq!(
            normalize_git_remote(&read_git_remote_url(&git_dir, "origin")),
            "github.com/org/repo"
        );
        assert_eq!(
            read_git_remote_url(&git_dir, "Origin"),
            "",
            "a quoted subsection stays case-sensitive"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn identity_exposes_every_template_variable_git_and_non_git_alike() {
        let root = unique_dir("vars");
        make_repo(&root, Some("git@github.com:volcengine/OpenViking.git"));
        let deep = root.join("examples").join("codex-memory-plugin");
        fs::create_dir_all(&deep).expect("deep dir");
        let env = test_env(Path::new("/nonexistent-home"));

        let deep_text = deep.to_string_lossy().to_string();
        let identity = resolve_workspace_identity(&deep_text, &env);
        assert_eq!(
            identity.vars["git_remote"],
            "github.com-volcengine-openviking"
        );
        assert_eq!(
            identity.vars["git_root"],
            legacy_sanitize(&root.to_string_lossy())
        );
        assert_eq!(identity.vars["cwd"], legacy_sanitize(&deep_text));
        assert_eq!(
            identity.vars["dir"],
            sanitize_peer_id(&root.file_name().unwrap().to_string_lossy())
        );

        let plain = unique_dir("plain");
        let outside = resolve_workspace_identity(&plain.to_string_lossy(), &env);
        assert!(!outside.is_git);
        assert_eq!(outside.root, plain.to_string_lossy());
        assert_eq!(outside.vars["git_remote"], "");
        assert_eq!(
            outside.vars["git_root"], "",
            "no repository means no repository root"
        );
        assert_eq!(
            outside.vars["cwd"],
            legacy_sanitize(&plain.to_string_lossy())
        );
        fs::remove_dir_all(&root).ok();
        fs::remove_dir_all(&plain).ok();
    }

    #[test]
    fn the_git_preset_falls_back_through_the_root_to_the_working_directory() {
        let env = test_env(Path::new("/nonexistent-home"));

        let no_remote = unique_dir("noremote");
        make_repo(&no_remote, None);
        let identity = resolve_workspace_identity(&no_remote.to_string_lossy(), &env);
        let resolved = resolve_effective_peer_id(&Value::Object(Map::new()), true, &identity);
        assert_eq!(
            resolved.peer_id,
            legacy_sanitize(&no_remote.to_string_lossy())
        );
        assert_eq!(resolved.origin, "{git_root}");

        let plain = unique_dir("plain-peer");
        let identity = resolve_workspace_identity(&plain.to_string_lossy(), &env);
        let resolved = resolve_effective_peer_id(&Value::Object(Map::new()), true, &identity);
        assert_eq!(resolved.peer_id, legacy_sanitize(&plain.to_string_lossy()));
        assert_eq!(resolved.origin, "{cwd}");
        assert_eq!(resolved.legacy_peer_id, "");
        fs::remove_dir_all(&no_remote).ok();
        fs::remove_dir_all(&plain).ok();
    }

    #[test]
    fn a_template_list_is_tried_in_order_and_an_empty_variable_falls_through() {
        let root = unique_dir("templates");
        make_repo(&root, None);
        let env = test_env(Path::new("/nonexistent-home"));
        let identity = resolve_workspace_identity(&root.to_string_lossy(), &env);

        let mut config = Value::Object(Map::new());
        config_set(
            &mut config,
            "peer.source",
            serde_json::json!(["{git_remote}", "team-{dir}"]),
        );
        let resolved = resolve_effective_peer_id(&config, true, &identity);
        assert_eq!(
            resolved.peer_id,
            format!("team-{}", root.file_name().unwrap().to_string_lossy())
        );
        assert_eq!(resolved.origin, "team-{dir}");

        let mut only_remote = Value::Object(Map::new());
        config_set(
            &mut only_remote,
            "peer.source",
            serde_json::json!(["{git_remote}"]),
        );
        let unresolved = resolve_effective_peer_id(&only_remote, true, &identity);
        assert_eq!(unresolved.peer_id, "");
        assert_eq!(unresolved.origin, "unresolved");
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn render_peer_template_is_all_or_nothing() {
        let vars: BTreeMap<String, String> = [
            ("git_remote", "github.com-o-r"),
            ("git_root", "-src-r"),
            ("cwd", "-src-r-sub"),
            ("dir", "r"),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();

        assert_eq!(
            render_peer_template("{git_remote}", &vars),
            "github.com-o-r"
        );
        assert_eq!(render_peer_template("a-{dir}-b", &vars), "a-r-b");
        assert_eq!(render_peer_template("{git_remote}-{missing}", &vars), "");
        assert_eq!(render_peer_template("literal", &vars), "literal");
        assert_eq!(render_peer_template("", &vars), "");
        assert_eq!(
            render_peer_template("{git_remote}", &BTreeMap::new()),
            "",
            "an absent variable is as empty as an empty one"
        );
    }

    // `/\{([a-z_]+)\}/g` scans forward one position at a time, so a `{` that
    // opens nothing is literal text the next match can start inside of. These
    // are the outputs the JS reader produces for the same variables.
    #[test]
    fn a_brace_that_opens_no_variable_is_literal_and_does_not_hide_the_next_one() {
        let vars: BTreeMap<String, String> = [("dir", "r")]
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect();

        assert_eq!(render_peer_template("{a{dir}", &vars), "{ar");
        assert_eq!(render_peer_template("{{dir}}", &vars), "{r}");
        assert_eq!(render_peer_template("pre{a{dir}post", &vars), "pre{arpost");
        assert_eq!(render_peer_template("{}", &vars), "{}");
        assert_eq!(render_peer_template("{DIR}", &vars), "{DIR}");
        assert_eq!(render_peer_template("{ dir }", &vars), "{ dir }");
        assert_eq!(render_peer_template("{dir", &vars), "{dir");
    }

    #[test]
    fn peer_source_templates_resolves_presets_and_passes_templates_through() {
        assert_eq!(
            peer_source_templates(None),
            vec!["{git_remote}", "{git_root}", "{cwd}"]
        );
        assert_eq!(
            peer_source_templates(Some(&Value::from(""))),
            vec!["{git_remote}", "{git_root}", "{cwd}"]
        );
        assert_eq!(
            peer_source_templates(Some(&Value::from("cwd"))),
            vec!["{cwd}"]
        );
        assert!(peer_source_templates(Some(&Value::from("none"))).is_empty());
        assert_eq!(
            peer_source_templates(Some(&Value::from("my-{dir}"))),
            vec!["my-{dir}"]
        );
        assert_eq!(
            peer_source_templates(Some(&serde_json::json!(["{git_remote}", "{cwd}"]))),
            vec!["{git_remote}", "{cwd}"]
        );
    }

    // `String(source)` runs before anything else looks at the value, so a
    // `peer.source` that is not a string still names a template — including the
    // one an object stringifies to. Falling back to the default preset here
    // would have `ov` report a git-derived peer while the plugins sent
    // `[object Object]`; these are the templates the JS reader produces.
    #[test]
    fn a_peer_source_that_is_not_a_string_is_still_stringified_into_a_template() {
        assert_eq!(
            peer_source_templates(Some(&serde_json::json!({ "a": 1 }))),
            vec!["[object Object]"]
        );
        assert_eq!(peer_source_templates(Some(&Value::Null)).len(), 3);
        assert_eq!(peer_source_templates(Some(&Value::from(5))), vec!["5"]);
        assert_eq!(peer_source_templates(Some(&Value::from(5.0))), vec!["5"]);
        assert_eq!(peer_source_templates(Some(&Value::from(5.5))), vec!["5.5"]);
        assert_eq!(
            peer_source_templates(Some(&Value::from(true))),
            vec!["true"]
        );
        assert_eq!(
            peer_source_templates(Some(&serde_json::json!([null, "x"]))),
            vec!["null", "x"],
            "a list member is stringified, not skipped"
        );
        assert_eq!(
            peer_source_templates(Some(&serde_json::json!([[1, 2], "y"]))),
            vec!["1,2", "y"]
        );

        // `[object Object]` names no variable, so it renders to itself.
        let identity = WorkspaceIdentity {
            vars: [("git_remote", "github.com-o-r"), ("cwd", "-src-r")]
                .into_iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect(),
            ..WorkspaceIdentity::default()
        };
        let mut config = Value::Object(Map::new());
        config_set(&mut config, "peer.source", serde_json::json!({ "a": 1 }));
        let resolved = resolve_effective_peer_id(&config, true, &identity);
        assert_eq!(resolved.peer_id, "[object Object]");
        assert_eq!(resolved.origin, "[object Object]");
    }

    #[test]
    fn an_explicit_peer_id_still_wins_over_every_rule() {
        let identity = WorkspaceIdentity::default();
        let mut config = Value::Object(Map::new());
        config_set(&mut config, "peer.id", Value::from(" configured "));
        let resolved = resolve_effective_peer_id(&config, true, &identity);
        assert_eq!(resolved.peer_id, "configured");
        assert_eq!(resolved.source, "explicit");
    }

    #[test]
    fn a_committed_file_cannot_say_where_the_data_goes() {
        let root = unique_dir("forbidden");
        let dir = root.join(CONFIG_DIR_NAME);
        fs::create_dir_all(&dir).expect("config dir");
        fs::write(
            dir.join(TEAM_FILE),
            serde_json::json!({
                "version": 1,
                "url": "https://attacker.example",
                "api_key": "sk-stolen",
                "account": "victim",
                "extra_headers": { "Authorization": "Bearer x" },
                "credential_source": "env",
                "cli_config_profile": "attacker",
                "recall": { "max_items": 8, "url": "https://nested.example" },
                "labels": { "user": "alice", "url": "https://wiki.example/project" },
                "peer": { "source": "git", "id": "${HOME}-peer" },
            })
            .to_string(),
        )
        .expect("team file");

        let file = read_workspace_file(&dir.join(TEAM_FILE), Some(&root), false);
        let data = file.data.expect("data");
        for key in [
            "url",
            "api_key",
            "account",
            "extra_headers",
            "credential_source",
            "cli_config_profile",
        ] {
            assert!(data.get(key).is_none(), "{key} must not survive");
        }
        assert!(config_get(&data, "recall.url").is_none());
        assert_eq!(config_get(&data, "recall.max_items"), Some(&Value::from(8)));
        // `labels` keeps the user's own vocabulary; every other section is swept.
        assert_eq!(
            config_get(&data, "labels.url"),
            Some(&Value::from("https://wiki.example/project"))
        );
        // `${VAR}` stays a literal — a workspace file never expands the environment.
        assert_eq!(
            config_get(&data, "peer.id"),
            Some(&Value::from("${HOME}-peer"))
        );
        assert!(file.warnings.iter().any(|w| w.contains("recall.url")));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_file_this_client_does_not_understand_is_skipped_not_obeyed() {
        let root = unique_dir("skipped");
        let dir = root.join(CONFIG_DIR_NAME);
        fs::create_dir_all(&dir).expect("config dir");
        fs::write(
            dir.join("v2.json"),
            r#"{"version":2,"recall":{"enabled":false}}"#,
        )
        .unwrap();
        fs::write(dir.join("bad.json"), "{ not json").unwrap();
        fs::write(dir.join("array.json"), "[1,2,3]").unwrap();
        fs::write(
            dir.join("big.json"),
            serde_json::json!({ "version": 1, "notes": "x".repeat(70_000) }).to_string(),
        )
        .unwrap();
        fs::create_dir_all(dir.join("dir.json")).unwrap();

        let cases = [
            ("v2.json", "version 2"),
            ("bad.json", "is not valid JSON"),
            ("array.json", "must contain a JSON object"),
            ("big.json", "larger than"),
            ("dir.json", "is not a regular file"),
        ];
        for (name, expected) in cases {
            let file = read_workspace_file(&dir.join(name), Some(&root), false);
            assert!(file.data.is_none(), "{name} should yield no layer");
            assert!(
                file.warnings.iter().any(|w| w.contains(expected)),
                "{name}: expected a warning containing {expected}, got {:?}",
                file.warnings
            );
        }
        fs::remove_dir_all(&root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_out_of_the_workspace_is_refused() {
        let outside = unique_dir("outside");
        fs::write(
            outside.join("secrets.json"),
            serde_json::json!({ "version": 1, "labels": { "leak": "yes" } }).to_string(),
        )
        .expect("secrets");
        let root = unique_dir("symlinked");
        let dir = root.join(CONFIG_DIR_NAME);
        fs::create_dir_all(&dir).expect("config dir");
        std::os::unix::fs::symlink(outside.join("secrets.json"), dir.join(TEAM_FILE))
            .expect("symlink");

        let file = read_workspace_file(&dir.join(TEAM_FILE), Some(&root), false);
        assert!(file.data.is_none());
        assert!(
            file.warnings
                .iter()
                .any(|w| w.contains("outside the workspace"))
        );
        fs::remove_dir_all(&outside).ok();
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_file_nested_past_the_limit_costs_only_itself() {
        let root = unique_dir("deep");
        let dir = root.join(CONFIG_DIR_NAME);
        fs::create_dir_all(&dir).expect("config dir");
        let deep = format!(
            "{{\"version\":1,\"deep\":{}{}}}",
            "[".repeat(40),
            "]".repeat(40)
        );
        fs::write(dir.join(TEAM_FILE), deep).expect("deep file");
        fs::write(
            dir.join(LOCAL_FILE),
            serde_json::json!({ "version": 1, "recall": { "max_items": 9 } }).to_string(),
        )
        .expect("local file");

        let team = read_workspace_file(&dir.join(TEAM_FILE), Some(&root), false);
        assert!(team.data.is_none());
        assert!(
            team.warnings
                .iter()
                .any(|w| w.contains("nested too deeply"))
        );

        let local = read_workspace_file(&dir.join(LOCAL_FILE), Some(&root), false);
        assert_eq!(
            config_get(&local.data.expect("local data"), "recall.max_items"),
            Some(&Value::from(9))
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn layers_stack_and_provenance_records_what_was_covered() {
        let mut warnings = Vec::new();
        let merged = merge_config_layers(
            &[
                layer(
                    "ovcli.conf",
                    serde_json::json!({ "recall": { "enabled": true, "max_items": 10 }, "capture": { "enabled": true } }),
                ),
                layer(
                    "config.json (workspace)",
                    serde_json::json!({ "recall": { "max_items": 20 }, "peer": { "source": "git" } }),
                ),
                layer(
                    "config.local.json (workspace)",
                    serde_json::json!({ "recall": { "max_items": 30 } }),
                ),
            ],
            &mut warnings,
        );

        assert_eq!(
            config_get(&merged.value, "recall.max_items"),
            Some(&Value::from(30))
        );
        assert_eq!(
            config_get(&merged.value, "recall.enabled"),
            Some(&Value::from(true)),
            "an untouched key keeps the lower layer's value"
        );
        let entry = merged.get("recall.max_items").expect("provenance");
        assert_eq!(entry.source, "config.local.json (workspace)");
        assert_eq!(
            entry
                .shadowed
                .iter()
                .map(|s| (s.value.clone(), s.source.clone()))
                .collect::<Vec<_>>(),
            vec![
                (Value::from(20), "config.json (workspace)".to_string()),
                (Value::from(10), "ovcli.conf".to_string()),
            ]
        );
        assert!(
            merged
                .get("recall.enabled")
                .expect("entry")
                .shadowed
                .is_empty()
        );
    }

    #[test]
    fn lists_union_across_layers_and_reset_clears_what_was_inherited() {
        let mut warnings = Vec::new();
        let union = merge_config_layers(
            &[
                layer(
                    "low",
                    serde_json::json!({ "bypass": { "session_patterns": ["*-scratch"] } }),
                ),
                layer(
                    "high",
                    serde_json::json!({ "bypass": { "session_patterns": ["**/tmp/**", "*-scratch"] } }),
                ),
            ],
            &mut warnings,
        );
        assert_eq!(
            config_get(&union.value, "bypass.session_patterns"),
            Some(&serde_json::json!(["*-scratch", "**/tmp/**"]))
        );
        assert_eq!(
            union.get("bypass.session_patterns").expect("entry").source,
            "low + high"
        );

        let reset = merge_config_layers(
            &[
                layer(
                    "low",
                    serde_json::json!({ "bypass": { "session_patterns": ["*-scratch", "**/tmp/**"] } }),
                ),
                layer(
                    "high",
                    serde_json::json!({ "bypass": { "session_patterns": ["!reset", "only-this"] } }),
                ),
            ],
            &mut warnings,
        );
        assert_eq!(
            config_get(&reset.value, "bypass.session_patterns"),
            Some(&serde_json::json!(["only-this"]))
        );
        assert!(
            reset
                .get("bypass.session_patterns")
                .expect("entry")
                .source
                .contains("reset")
        );
    }

    #[test]
    fn provenance_stays_honest_when_layers_disagree_about_a_keys_type() {
        let mut warnings = Vec::new();
        let scalar_then_list = merge_config_layers(
            &[
                layer(
                    "ovcli.conf",
                    serde_json::json!({ "bypass": { "session_patterns": "not-a-list" } }),
                ),
                layer(
                    "config.json (workspace)",
                    serde_json::json!({ "bypass": { "session_patterns": ["only-this"] } }),
                ),
            ],
            &mut warnings,
        );
        let entry = scalar_then_list
            .get("bypass.session_patterns")
            .expect("entry");
        assert_eq!(entry.source, "config.json (workspace)");
        assert_eq!(
            entry.shadowed,
            vec![ShadowedValue {
                value: Value::from("not-a-list"),
                source: "ovcli.conf".to_string(),
            }]
        );

        let scalar_then_section = merge_config_layers(
            &[
                layer("ovcli.conf", serde_json::json!({ "capture": false })),
                layer(
                    "config.json (workspace)",
                    serde_json::json!({ "capture": { "enabled": true } }),
                ),
            ],
            &mut warnings,
        );
        assert_eq!(
            config_get(&scalar_then_section.value, "capture.enabled"),
            Some(&Value::from(true))
        );
        let entry = scalar_then_section.get("capture").expect("entry");
        assert_eq!(entry.value, Value::from("(section)"));
        assert_eq!(
            entry.shadowed,
            vec![ShadowedValue {
                value: Value::from(false),
                source: "ovcli.conf".to_string(),
            }]
        );
    }

    #[test]
    fn unknown_keys_ride_along_untouched_so_old_and_new_clients_coexist() {
        let mut warnings = Vec::new();
        let merged = merge_config_layers(
            &[layer(
                "config.json (workspace)",
                serde_json::json!({ "future": { "knob": 1 }, "recall": { "unheard_of": true } }),
            )],
            &mut warnings,
        );
        assert_eq!(
            config_get(&merged.value, "future.knob"),
            Some(&Value::from(1))
        );
        assert_eq!(
            config_get(&merged.value, "recall.unheard_of"),
            Some(&Value::from(true))
        );
    }

    #[test]
    fn cost_knobs_are_clamped_and_bad_enum_values_fall_back() {
        let mut warnings = Vec::new();
        let mut value = serde_json::json!({
            "recall": { "peer_scope": "everything", "dedup_turns": 999, "max_items": 0 },
            "capture": { "commit_token_threshold": -4 },
        });
        normalize_workspace_config(&mut value, &mut warnings);

        assert_eq!(
            config_get(&value, "recall.dedup_turns"),
            Some(&Value::from(20))
        );
        assert_eq!(
            config_get(&value, "recall.max_items"),
            Some(&Value::from(1))
        );
        assert_eq!(
            config_get(&value, "capture.commit_token_threshold"),
            Some(&Value::from(1000))
        );
        assert!(config_get(&value, "recall.peer_scope").is_none());
        assert_eq!(
            warnings.iter().filter(|w| w.starts_with("clamped")).count(),
            3
        );
    }

    #[test]
    fn a_non_numeric_knob_is_reported_rather_than_coerced_to_a_bound() {
        let mut warnings = Vec::new();
        let mut value = serde_json::json!({
            "recall": { "max_items": null, "score_threshold": true },
            "capture": { "commit_token_threshold": [] },
        });
        normalize_workspace_config(&mut value, &mut warnings);

        assert!(config_get(&value, "recall.max_items").is_none());
        assert!(config_get(&value, "recall.score_threshold").is_none());
        assert!(config_get(&value, "capture.commit_token_threshold").is_none());
        assert_eq!(
            warnings
                .iter()
                .filter(|w| w.contains("is not a number"))
                .count(),
            3
        );
    }

    // `Number(string)` is not `str::parse`. These are the values the JS reader
    // lands on for the same strings, and getting them wrong means one reader
    // clamps a knob where the other throws it away.
    #[test]
    fn a_string_knob_is_read_the_way_javascript_reads_it() {
        for (raw, expected) in [
            ("", Some(0.0)),
            ("   ", Some(0.0)),
            (" 12 ", Some(12.0)),
            ("0x10", Some(16.0)),
            ("0o17", Some(15.0)),
            ("0b101", Some(5.0)),
            ("1e2", Some(100.0)),
            ("-5", Some(-5.0)),
            ("-0x10", None),
            ("abc", None),
            ("inf", None),
            ("12abc", None),
        ] {
            assert_eq!(js_number(raw), expected, "js_number({raw:?})");
        }
        // `Infinity` parses but is not finite, so the caller still refuses it.
        assert_eq!(js_number("Infinity"), Some(f64::INFINITY));

        let mut warnings = Vec::new();
        let mut value = serde_json::json!({ "recall": { "dedup_turns": "" } });
        normalize_workspace_config(&mut value, &mut warnings);
        assert_eq!(
            config_get(&value, "recall.dedup_turns"),
            Some(&Value::from(0))
        );
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    // Both of these are read back by the JS half, which compares numbers rather
    // than integer types and drops a falsy `min_client_version` outright.
    #[test]
    fn a_layer_file_is_accepted_and_trimmed_on_javascript_terms() {
        let dir = unique_dir("js-terms");

        let read = |name: &str, body: &str| {
            let path = dir.join(name);
            fs::write(&path, body).expect("layer file");
            read_workspace_file(&path, None, false)
        };

        // `1.0 === 1` in JS, so a float spelling of version 1 is still version 1.
        let float_version = read("float.json", r#"{"version":1.0,"peer":{"id":"x"}}"#);
        assert_eq!(
            float_version
                .data
                .as_ref()
                .and_then(|d| config_get(d, "peer.id")),
            Some(&Value::from("x"))
        );
        assert!(float_version.warnings.is_empty());

        for (name, body) in [
            ("zero.json", r#"{"version":1,"min_client_version":0}"#),
            ("false.json", r#"{"version":1,"min_client_version":false}"#),
            ("empty.json", r#"{"version":1,"min_client_version":""}"#),
            ("null.json", r#"{"version":1,"min_client_version":null}"#),
        ] {
            let file = read(name, body);
            let data = file.data.expect("layer should parse");
            assert!(
                data.get("min_client_version").is_none(),
                "{name} should drop a falsy declaration, got {data}"
            );
        }
        let real = read("real.json", r#"{"version":1,"min_client_version":3}"#);
        assert_eq!(
            real.data.as_ref().and_then(|d| d.get("min_client_version")),
            Some(&Value::from("3"))
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn min_client_version_warns_and_still_applies_the_settings() {
        let mut warnings = Vec::new();
        assert!(!check_min_client_version("9.9.0", "0.8.1", &mut warnings));
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("9.9.0") && warnings[0].contains("0.8.1"));

        warnings.clear();
        assert!(check_min_client_version("9.9.0", "9.9.0", &mut warnings));
        assert!(check_min_client_version("9.9.0", "10.0.0", &mut warnings));
        // Compared numerically, not as text, and a non-numeric tail is ignored.
        assert!(check_min_client_version("0.9", "0.10.0", &mut warnings));
        assert!(check_min_client_version(
            "1.2.3",
            "1.2.3-rc1",
            &mut warnings
        ));
        // Neither side can judge without a version.
        assert!(check_min_client_version("", "0.8.1", &mut warnings));
        assert!(check_min_client_version("9.9.0", "", &mut warnings));
        assert!(warnings.is_empty(), "{warnings:?}");

        // A version note must never disable a workspace, and it is metadata
        // rather than a setting, so it never reaches the merged config.
        let root = unique_dir("min-client-version");
        let config_dir = root.join(CONFIG_DIR_NAME);
        fs::create_dir_all(&config_dir).expect("config dir");
        fs::write(
            config_dir.join(TEAM_FILE),
            serde_json::json!({
                "version": 1,
                "min_client_version": "9999.0.0",
                "recall": { "max_items": 7 },
            })
            .to_string(),
        )
        .expect("team file");
        let env = test_env(Path::new("/nonexistent-home"));
        let resolved = resolve_workspace(&root.to_string_lossy(), None, &env);
        assert_eq!(
            config_get(&resolved.merged.value, "recall.max_items"),
            Some(&Value::from(7))
        );
        assert!(resolved.merged.get("min_client_version").is_none());
        assert!(
            resolved
                .warnings
                .iter()
                .any(|w| w.contains("9999.0.0") && w.contains(CLIENT_VERSION)),
            "{:?}",
            resolved.warnings
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_slot_is_readable_and_still_unique_per_absolute_path() {
        let a = slot_name("/Users/x/src/api", None);
        let b = slot_name("/Users/x/work/api", None);
        assert!(a.starts_with("api-") && a.ends_with(".json"), "{a}");
        assert_eq!(a.len(), "api-".len() + 12 + ".json".len());
        assert_ne!(a, b, "same basename, different path, different slot");
        assert_eq!(a, slot_name("/Users/x/src/api", None), "slots are stable");
        assert_eq!(slot_name("/", None).len(), 12 + ".json".len());
    }

    #[test]
    fn two_worktrees_of_one_repository_share_a_slot_and_two_clones_do_not() {
        let repo = repo_identity("github.com/volcengine/openviking");
        let other = repo_identity("github.com/someone/else");

        // A linked worktree is a second checkout of the same repository — same
        // peer, so the same settings and the same `ov peer link`.
        let main = slot_name("/Users/x/src/api", Some(&repo));
        let worktree = slot_name("/Users/x/wt/api-feature", Some(&repo));
        assert_eq!(
            worktree, main,
            "the checkout path must not split one workspace in two"
        );
        assert!(main.starts_with("openviking-"), "{main}");
        assert_eq!(main.len(), "openviking-".len() + 12 + ".json".len());
        assert_ne!(
            slot_name("/Users/x/src/api", Some(&other)),
            main,
            "a different repository is a different slot"
        );

        // Without a repository there is no identity but the path.
        let plain = WorkspaceIdentity::default();
        assert_ne!(
            slot_name("/Users/x/src/notes", Some(&plain)),
            slot_name("/Users/x/work/notes", Some(&plain))
        );
        assert_eq!(
            slot_name("/Users/x/src/notes", Some(&plain)),
            slot_name("/Users/x/src/notes", None)
        );
    }

    #[test]
    fn identity_key_prefers_the_remote_and_degrades_honestly() {
        let repo = WorkspaceIdentity {
            is_git: true,
            remote: "github.com/volcengine/openviking".to_string(),
            git_common_dir: "/x/.git".to_string(),
            ..WorkspaceIdentity::default()
        };
        assert_eq!(
            identity_key(&repo),
            "remote:github.com/volcengine/openviking"
        );
        assert_eq!(
            identity_key(&WorkspaceIdentity {
                is_git: true,
                git_common_dir: "/x/.git".to_string(),
                ..WorkspaceIdentity::default()
            }),
            "git:/x/.git"
        );
        assert_eq!(identity_key(&WorkspaceIdentity::default()), "path");
    }

    fn repo_identity(remote: &str) -> WorkspaceIdentity {
        WorkspaceIdentity {
            is_git: true,
            remote: remote.to_string(),
            git_common_dir: "/x/.git".to_string(),
            ..WorkspaceIdentity::default()
        }
    }

    #[test]
    fn an_entry_round_trips_and_only_the_callers_keys_change() {
        let home = unique_dir("registry-roundtrip");
        let env = test_env(&home);
        let repo = repo_identity("github.com/volcengine/openviking");
        let root = "/Users/x/src/api";

        let mut settings = Map::new();
        settings.insert(
            "settings".to_string(),
            serde_json::json!({ "recall": { "max_items": 7 } }),
        );
        write_entry(root, settings, Some(&repo), &env, 1000).expect("first write");
        let mut peer = Map::new();
        peer.insert("peer".to_string(), serde_json::json!({ "id": "pinned" }));
        write_entry(root, peer, Some(&repo), &env, 2000).expect("second write");

        let read = read_entry(root, Some(&repo), &env);
        let entry = read.entry.expect("entry");
        assert_eq!(
            config_get(&entry, "settings.recall.max_items"),
            Some(&Value::from(7)),
            "an earlier write is not erased"
        );
        assert_eq!(config_get(&entry, "peer.id"), Some(&Value::from("pinned")));
        assert_eq!(entry["root"], Value::from(root));
        assert_eq!(entry["label"], Value::from(repo.remote.as_str()));
        assert_eq!(entry["first_seen_at"], Value::from(1000));
        assert_eq!(entry["last_seen_at"], Value::from(2000));
        fs::remove_dir_all(&home).ok();
    }

    #[cfg(unix)]
    #[test]
    fn entries_are_written_0600_so_nobody_else_on_the_machine_reads_them() {
        use std::os::unix::fs::PermissionsExt;

        let home = unique_dir("registry-mode");
        let env = test_env(&home);
        let repo = repo_identity("github.com/o/r");
        let (path, _) =
            write_entry("/Users/x/src/api", Map::new(), Some(&repo), &env, 1).expect("write");
        let mode = fs::metadata(&path).expect("metadata").permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "expected 0600, got {mode:o}");
        fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn a_path_reused_by_a_different_repository_does_not_inherit_the_old_peer() {
        let home = unique_dir("registry-negative");
        let env = test_env(&home);
        let repo = repo_identity("github.com/volcengine/openviking");
        let other = repo_identity("github.com/someone/else");
        let root = "/Users/x/src/api";

        let mut patch = Map::new();
        patch.insert(
            "peer".to_string(),
            serde_json::json!({ "id": "from-the-old-repo" }),
        );
        patch.insert(
            "settings".to_string(),
            serde_json::json!({ "recall": { "max_items": 3 } }),
        );
        write_entry(root, patch, Some(&repo), &env, 1).expect("write");

        // Keying the slot on identity makes the crossing physically impossible:
        // the new repository looks in a different file and finds nothing.
        let miss = read_entry(root, Some(&other), &env);
        assert!(miss.entry.is_none(), "a conflicting identity is a miss");
        assert_ne!(
            entry_path(root, Some(&other), &env),
            entry_path(root, Some(&repo), &env)
        );

        let hit = read_entry(root, Some(&repo), &env);
        assert_eq!(
            config_get(&hit.entry.expect("entry"), "peer.id"),
            Some(&Value::from("from-the-old-repo"))
        );

        // Writing after a conflict replaces the entry instead of merging into it.
        let mut replacement = Map::new();
        replacement.insert("peer".to_string(), serde_json::json!({ "id": "new" }));
        write_entry(root, replacement, Some(&other), &env, 2).expect("replace");
        let entry = read_entry(root, Some(&other), &env).entry.expect("entry");
        assert_eq!(config_get(&entry, "peer.id"), Some(&Value::from("new")));
        assert!(
            entry.get("settings").is_none(),
            "nothing carries over from the repository that used to be here"
        );
        fs::remove_dir_all(&home).ok();
    }

    // Slot isolation is the first defence; the recorded identity is the second,
    // for a file that was hand-edited or moved into the slot.
    #[test]
    fn an_entry_whose_recorded_identity_contradicts_the_caller_is_still_refused() {
        let home = unique_dir("registry-hand-edited");
        let env = test_env(&home);
        let repo = repo_identity("github.com/volcengine/openviking");
        let other = repo_identity("github.com/someone/else");
        let root = "/Users/x/src/api";
        fs::create_dir_all(env.registry_dir()).expect("registry dir");
        fs::write(
            entry_path(root, Some(&repo), &env),
            serde_json::json!({
                "version": 1,
                "identity": identity_key(&other),
                "peer": { "id": "someone-elses" },
            })
            .to_string(),
        )
        .expect("seed");

        let miss = read_entry(root, Some(&repo), &env);
        assert!(miss.entry.is_none());
        assert!(miss.conflict);
        assert!(
            miss.warnings
                .iter()
                .any(|w| w.contains("different repository"))
        );
        fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn an_entry_from_a_newer_client_is_refused_not_flattened() {
        let home = unique_dir("registry-newer");
        let env = test_env(&home);
        let repo = repo_identity("github.com/o/r");
        let root = "/Users/x/src/api";
        fs::create_dir_all(env.registry_dir()).expect("registry dir");
        let future = serde_json::json!({ "version": 2, "peer": { "id": "pinned" } });
        fs::write(entry_path(root, Some(&repo), &env), future.to_string()).expect("seed");

        let error = write_entry(root, Map::new(), Some(&repo), &env, 1).unwrap_err();
        assert!(error.to_string().contains("newer client"), "{error}");
        assert_eq!(
            read_raw_entry(&entry_path(root, Some(&repo), &env)).expect("still there"),
            future
        );
        fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn cli_config_profile_is_a_name_never_a_path() {
        let home = unique_dir("registry-profile");
        let env = test_env(&home);
        let repo = repo_identity("github.com/o/r");
        let root = "/Users/x/src/api";

        let mut patch = Map::new();
        patch.insert("cli_config_profile".to_string(), Value::from("work"));
        write_entry(root, patch, Some(&repo), &env, 1).expect("write");
        assert_eq!(
            read_entry(root, Some(&repo), &env).entry.expect("entry")["cli_config_profile"],
            Value::from("work")
        );

        for bad in ["../../etc/ovcli.conf", "/abs/path", "Work", "has space", ""] {
            let mut patch = Map::new();
            patch.insert("cli_config_profile".to_string(), Value::from(bad));
            let error = write_entry(root, patch, Some(&repo), &env, 1).unwrap_err();
            assert!(
                error.to_string().contains("cli_config_profile"),
                "should reject {bad:?}"
            );
        }
        fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn previous_peers_accumulate_without_duplicates() {
        let home = unique_dir("registry-previous");
        let env = test_env(&home);
        let repo = repo_identity("github.com/o/r");
        let root = "/Users/x/src/api";

        assert!(remember_previous_peer(root, "-Users-x-src-api", Some(&repo), &env, 1).unwrap());
        assert!(!remember_previous_peer(root, "-Users-x-src-api", Some(&repo), &env, 2).unwrap());
        assert!(remember_previous_peer(root, "-Users-x-old-path", Some(&repo), &env, 3).unwrap());
        assert!(!remember_previous_peer(root, "  ", Some(&repo), &env, 4).unwrap());

        let entry = read_entry(root, Some(&repo), &env).entry.expect("entry");
        assert_eq!(
            entry["previous_peer_ids"],
            serde_json::json!(["-Users-x-src-api", "-Users-x-old-path"])
        );
        fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn the_registry_outranks_the_workspace_files_it_sits_above() {
        let home = unique_dir("resolve-home");
        let root = unique_dir("resolve-root");
        make_repo(&root, Some("git@github.com:volcengine/OpenViking.git"));
        let config_dir = root.join(CONFIG_DIR_NAME);
        fs::create_dir_all(&config_dir).expect("config dir");
        fs::write(
            config_dir.join(TEAM_FILE),
            serde_json::json!({ "version": 1, "recall": { "max_items": 20 } }).to_string(),
        )
        .expect("team file");
        let env = test_env(&home);

        let identity = resolve_workspace_identity(&root.to_string_lossy(), &env);
        let mut patch = Map::new();
        patch.insert(
            "settings".to_string(),
            serde_json::json!({ "recall": { "max_items": 5 } }),
        );
        write_entry(&root.to_string_lossy(), patch, Some(&identity), &env, 1)
            .expect("registry write");

        let resolved = resolve_workspace(&root.to_string_lossy(), None, &env);
        assert_eq!(
            config_get(&resolved.merged.value, "recall.max_items"),
            Some(&Value::from(5))
        );
        let entry = resolved.merged.get("recall.max_items").expect("provenance");
        assert_eq!(entry.source, "registry");
        assert_eq!(
            entry.shadowed.first().map(|s| s.source.clone()),
            Some("config.json (workspace)".to_string())
        );
        assert_eq!(resolved.peer.peer_id, "github.com-volcengine-openviking");
        fs::remove_dir_all(&home).ok();
        fs::remove_dir_all(&root).ok();
    }

    // ovcli.conf's `plugin` section names the flat knobs a harness loader reads,
    // and `loadPluginSettings` puts them in the same precedence chain as the
    // workspace files, which spell the same settings in the nested schema. A
    // reader that kept them apart would report a derived peer while every plugin
    // sent the pinned one.
    #[test]
    fn the_ovcli_plugin_section_speaks_flat_knobs_and_still_joins_the_chain() {
        let home = unique_dir("plugin-knobs");
        let root = unique_dir("plugin-knobs-root");
        make_repo(&root, Some("git@github.com:volcengine/OpenViking.git"));
        fs::create_dir_all(home.join(".openviking")).expect("ov home");
        fs::write(
            home.join(".openviking").join("ovcli.conf"),
            serde_json::json!({
                "url": "https://ov.example",
                "plugin": {
                    "peerSource": "cwd",
                    "recallLimit": 4,
                    "recallQueryExpansion": "off",
                    "codex": { "peerId": "team-api" },
                },
            })
            .to_string(),
        )
        .expect("ovcli.conf");
        let env = test_env(&home);
        let cwd = root.to_string_lossy().to_string();

        // The shared section alone: `peerSource` is `peer.source`, and it picks
        // the peer even though the repository has an origin.
        let shared = resolve_workspace(&cwd, None, &env);
        assert_eq!(
            config_get(&shared.merged.value, "peer.source"),
            Some(&Value::from("cwd"))
        );
        assert_eq!(
            config_get(&shared.merged.value, "recall.max_items"),
            Some(&Value::from(4))
        );
        assert_eq!(shared.peer.peer_id, legacy_sanitize(&cwd));
        assert_eq!(shared.peer.origin, "{cwd}");
        // A key that names no knob rides along untranslated.
        assert_eq!(
            shared.merged.value.get("recallQueryExpansion"),
            Some(&Value::from("off"))
        );
        assert!(
            shared.merged.value.get("peerSource").is_none(),
            "a translated knob does not also keep its flat spelling"
        );

        // `plugin.<harness>` outranks the shared keys, and an explicit id wins.
        let scoped = resolve_workspace(&cwd, Some("codex"), &env);
        assert_eq!(scoped.peer.peer_id, "team-api");
        assert_eq!(scoped.peer.source, "explicit");
        assert_eq!(
            scoped.merged.get("peer.id").map(|e| e.source.clone()),
            Some("ovcli.conf plugin.codex".to_string())
        );

        // A workspace file still outranks both.
        let config_dir = root.join(CONFIG_DIR_NAME);
        fs::create_dir_all(&config_dir).expect("config dir");
        fs::write(
            config_dir.join(TEAM_FILE),
            serde_json::json!({ "version": 1, "peer": { "source": "git" } }).to_string(),
        )
        .expect("team file");
        let overridden = resolve_workspace(&cwd, Some("codex"), &env);
        assert_eq!(
            config_get(&overridden.merged.value, "peer.source"),
            Some(&Value::from("git"))
        );
        assert_eq!(
            overridden
                .merged
                .get("peer.source")
                .and_then(|e| e.shadowed.first().map(|s| s.source.clone())),
            Some("ovcli.conf plugin".to_string())
        );

        // The nested workspace schema written into the flat file. `settings`
        // would carry a `peer` object no loader reads, so it pins nothing there
        // — and must pin nothing here. (The shared section drops objects
        // outright, so only `plugin.<harness>` can reach this.)
        fs::write(
            home.join(".openviking").join("ovcli.conf"),
            serde_json::json!({ "plugin": { "codex": { "peer": { "id": "wrong-file" } } } })
                .to_string(),
        )
        .expect("ovcli.conf");
        let misspelled = resolve_workspace(&cwd, Some("codex"), &env);
        assert!(config_get(&misspelled.merged.value, "peer.id").is_none());
        assert_eq!(misspelled.peer.peer_id, "github.com-volcengine-openviking");
        fs::remove_dir_all(&home).ok();
        fs::remove_dir_all(&root).ok();
    }

    // `cfg.workspacePeer !== false` — the one knob that lives only in the
    // ovcli.conf plugin section, and only an explicit `false` turns it off.
    #[test]
    fn only_an_explicit_false_workspace_peer_turns_the_derived_peer_off() {
        let home = unique_dir("workspace-peer-off");
        let root = unique_dir("workspace-peer-off-root");
        make_repo(&root, Some("git@github.com:volcengine/OpenViking.git"));
        fs::create_dir_all(home.join(".openviking")).expect("ov home");
        let conf = home.join(".openviking").join("ovcli.conf");
        let env = test_env(&home);
        let cwd = root.to_string_lossy().to_string();

        fs::write(
            &conf,
            serde_json::json!({ "plugin": { "workspacePeer": true } }).to_string(),
        )
        .expect("ovcli.conf");
        assert_eq!(
            resolve_workspace(&cwd, None, &env).peer.peer_id,
            "github.com-volcengine-openviking"
        );

        fs::write(
            &conf,
            serde_json::json!({ "plugin": { "workspacePeer": false } }).to_string(),
        )
        .expect("ovcli.conf");
        let disabled = resolve_workspace(&cwd, None, &env);
        assert_eq!(disabled.peer.peer_id, "");
        assert_eq!(disabled.peer.origin, "disabled");
        fs::remove_dir_all(&home).ok();
        fs::remove_dir_all(&root).ok();
    }

    // The registry file is read back by the JavaScript half of this reader, and
    // a rewrite that reshuffles keys makes every entry look changed in a diff.
    // `Map::remove` is a swap remove under `preserve_order`, which does exactly
    // that; this pins the order two writes and a remember produce.
    #[test]
    fn a_rewritten_entry_keeps_its_key_order() {
        let home = unique_dir("registry-order");
        let env = test_env(&home);
        let repo = repo_identity("github.com/o/r");
        let root = "/Users/x/src/api";

        let mut patch = Map::new();
        patch.insert(
            "settings".to_string(),
            serde_json::json!({ "recall": { "max_items": 7 } }),
        );
        write_entry(root, patch, Some(&repo), &env, 1000).expect("first write");
        let mut patch = Map::new();
        patch.insert("peer".to_string(), serde_json::json!({ "id": "pinned" }));
        write_entry(root, patch, Some(&repo), &env, 2000).expect("second write");
        remember_previous_peer(root, "-Users-x-old", Some(&repo), &env, 3000).expect("remember");

        let entry = read_raw_entry(&entry_path(root, Some(&repo), &env)).expect("entry");
        assert_eq!(
            entry
                .as_object()
                .expect("object")
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec![
                "settings",
                "root",
                "identity",
                "label",
                "first_seen_at",
                "last_seen_at",
                "peer",
                "previous_peer_ids",
                "version",
            ]
        );
        fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn only_the_registry_may_name_a_credential_profile() {
        let root = unique_dir("profile-layer");
        let dir = root.join(CONFIG_DIR_NAME);
        fs::create_dir_all(&dir).expect("config dir");
        let body = serde_json::json!({ "version": 1, "cli_config_profile": "work" }).to_string();
        fs::write(dir.join(TEAM_FILE), &body).expect("team file");

        let workspace = read_workspace_file(&dir.join(TEAM_FILE), Some(&root), false);
        assert!(
            workspace
                .data
                .expect("data")
                .get("cli_config_profile")
                .is_none()
        );

        let registry = read_workspace_file(&dir.join(TEAM_FILE), Some(&root), true);
        assert_eq!(
            registry.data.expect("data")["cli_config_profile"],
            Value::from("work")
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn validate_peer_id_matches_the_servers_rule() {
        for good in ["github.com-o-r", "a@b", "_x.y-z", "-Users-x-Dev"] {
            assert!(validate_peer_id(good).is_ok(), "{good} should be accepted");
        }
        for bad in ["", ".", "..", "a/b", "a b", "a@b@c", "проект"] {
            assert!(validate_peer_id(bad).is_err(), "{bad:?} should be rejected");
        }
    }
}
