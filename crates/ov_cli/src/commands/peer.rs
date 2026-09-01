//! `ov peer` — pin, migrate, and forget the peer a workspace writes under.
//!
//! The peer is derived, not stored, so a repository that moves or gets a new
//! origin silently starts a fresh namespace. These commands are how a user
//! reconciles that: pin the id, move what the old id holds, and drop the
//! bookkeeping once it is no longer interesting. All three read and write the
//! same per-workspace registry entry the memory plugins do — see
//! `commands::workspace` for the reader both languages share.

use colored::Colorize;
use serde_json::{Map, Value};

use crate::client::HttpClient;
use crate::commands::workspace::{
    ResolvedWorkspace, WorkspaceEnv, legacy_sanitize, now_millis, remember_previous_peer,
    resolve_user, resolve_workspace, validate_peer_id, write_entry,
};
use crate::error::{Error, Result};
use crate::i18n::{Language, copy};
use crate::output::{OutputFormat, output_success};
use crate::terminal_ui::pad_to_display_width;
use crate::theme;

/// The two subtrees a peer owns. Everything else under `peers/<id>` is the
/// server's own bookkeeping and is not ours to move.
const PEER_SUBTREES: &[&str] = &["memories", "resources"];

pub fn link(cwd: &str, peer_id: &str, output_format: OutputFormat, compact: bool) -> Result<()> {
    let language = Language::current();
    let env = WorkspaceEnv::from_process();
    let resolved = resolve_workspace(cwd, None, &env);
    let root = resolved.require_root()?.to_string();

    let peer_id = peer_id.trim();
    validate_peer_id(peer_id)?;

    let previous = resolved.peer.peer_id.clone();
    let mut patch = Map::new();
    patch.insert("peer".to_string(), serde_json::json!({ "id": peer_id }));
    let (path, _) = write_entry(&root, patch, Some(&resolved.identity), &env, now_millis())?;

    if matches!(output_format, OutputFormat::Json) {
        output_success(
            serde_json::json!({
                "root": root,
                "entry": path.display().to_string(),
                "previous_peer_id": previous,
                "peer_id": peer_id,
            }),
            output_format,
            compact,
        );
        return Ok(());
    }

    println!(
        "{} {}",
        theme::success(copy(language, "Pinned peer for", "已固定工作区 peer：")),
        theme::value(&root)
    );
    println!(
        "  {} {}",
        theme::muted(pad_to_display_width(copy(language, "was", "原值"), 6)),
        if previous.is_empty() {
            theme::muted(copy(language, "(none)", "（无）")).to_string()
        } else {
            theme::value(&previous).to_string()
        }
    );
    println!(
        "  {} {}",
        theme::muted(pad_to_display_width(copy(language, "now", "新值"), 6)),
        theme::sky_value(peer_id)
    );
    println!("  {}", theme::muted(path.display().to_string()));
    Ok(())
}

pub fn forget_previous(cwd: &str, output_format: OutputFormat, compact: bool) -> Result<()> {
    let language = Language::current();
    let env = WorkspaceEnv::from_process();
    let resolved = resolve_workspace(cwd, None, &env);
    let root = resolved.require_root()?.to_string();

    let dropped = resolved.previous_peer_ids.clone();
    if !dropped.is_empty() {
        let mut patch = Map::new();
        patch.insert("previous_peer_ids".to_string(), Value::Array(Vec::new()));
        write_entry(&root, patch, Some(&resolved.identity), &env, now_millis())?;
    }

    if matches!(output_format, OutputFormat::Json) {
        output_success(
            serde_json::json!({ "root": root, "dropped": dropped }),
            output_format,
            compact,
        );
        return Ok(());
    }

    if dropped.is_empty() {
        println!(
            "{}",
            theme::muted(copy(
                language,
                "No previous peer ids were recorded for this workspace.",
                "该工作区没有记录任何历史 peer id。",
            ))
        );
        return Ok(());
    }
    println!(
        "{} {}",
        theme::success(copy(language, "Dropped", "已清除")),
        theme::value(format!(
            "{} {}",
            dropped.len(),
            copy(language, "previous peer id(s)", "个历史 peer id")
        ))
    );
    for id in &dropped {
        println!("  {}", theme::muted(id));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Move {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Collision {
    pub from: String,
    pub to: String,
}

pub async fn migrate(
    client: &HttpClient,
    cwd: &str,
    configured_user: Option<&str>,
    from: Option<String>,
    to: Option<String>,
    apply: bool,
    output_format: OutputFormat,
    compact: bool,
) -> Result<()> {
    let language = Language::current();
    let env = WorkspaceEnv::from_process();
    let resolved = resolve_workspace(cwd, None, &env);
    let root = resolved.require_root()?.to_string();

    let from = match from {
        Some(from) => from.trim().to_string(),
        None => default_from(&resolved),
    };
    let to = match to {
        Some(to) => to.trim().to_string(),
        None => resolved.peer.peer_id.clone(),
    };
    if from.is_empty() || to.is_empty() {
        return Err(Error::Client(copy(
            language,
            "Nothing to migrate: pass --from and --to explicitly, this workspace has no previous or effective peer.",
            "无法迁移：该工作区没有历史 peer 或生效 peer，请显式传入 --from 与 --to。",
        )
        .to_string()));
    }
    validate_peer_id(&from)?;
    validate_peer_id(&to)?;
    if from == to {
        return Err(Error::Client(
            copy(
                language,
                "--from and --to name the same peer; there is nothing to move.",
                "--from 与 --to 指向同一个 peer，无需迁移。",
            )
            .to_string(),
        ));
    }

    let user = resolve_user(client, configured_user).await?;
    let mut moves = Vec::new();
    let mut collisions = Vec::new();
    for subtree in PEER_SUBTREES {
        let source = peer_uri(&user, &from, subtree);
        let target = peer_uri(&user, &to, subtree);
        if client.stat(&source).await.is_err() {
            continue;
        }
        if client.stat(&target).await.is_err() {
            moves.push(Move {
                from: source,
                to: target,
            });
            continue;
        }
        // The server's `mv` renames; it has no merge. Descend so a partially
        // migrated peer can still be finished, and refuse the moment a name
        // exists on both sides — silently overwriting is not ours to choose.
        plan_merge(client, &source, &target, &mut moves, &mut collisions).await?;
    }

    if !collisions.is_empty() {
        return Err(Error::Client(collision_message(&collisions, language)));
    }

    if moves.is_empty() {
        if matches!(output_format, OutputFormat::Json) {
            output_success(
                serde_json::json!({ "from": from, "to": to, "moves": [], "applied": false }),
                output_format,
                compact,
            );
            return Ok(());
        }
        println!(
            "{}",
            theme::muted(copy(
                language,
                "Nothing to migrate: the source peer holds no memories or resources.",
                "无需迁移：源 peer 下没有记忆或资源。",
            ))
        );
        return Ok(());
    }

    if apply {
        for planned in &moves {
            ensure_parent(client, &planned.to).await?;
            client.mv(&planned.from, &planned.to).await?;
        }
        remember_previous_peer(&root, &from, Some(&resolved.identity), &env, now_millis())?;
    }

    if matches!(output_format, OutputFormat::Json) {
        output_success(
            serde_json::json!({
                "from": from,
                "to": to,
                "applied": apply,
                "moves": moves.iter().map(|planned| serde_json::json!({
                    "from": planned.from,
                    "to": planned.to,
                })).collect::<Vec<_>>(),
            }),
            output_format,
            compact,
        );
        return Ok(());
    }

    println!(
        "{} {} {} {}",
        theme::heading(if apply {
            copy(language, "Migrated", "已迁移")
        } else {
            copy(language, "Would migrate", "将迁移")
        })
        .bold(),
        theme::value(&from),
        theme::muted("->"),
        theme::sky_value(&to)
    );
    for planned in &moves {
        println!("  {}", theme::muted(&planned.from));
        println!("    -> {}", theme::value(&planned.to));
    }
    if !apply {
        println!(
            "\n{}",
            theme::muted(copy(
                language,
                "Dry run. Re-run with --apply to perform the move.",
                "这是预演。加上 --apply 才会真正执行。",
            ))
        );
    }
    Ok(())
}

/// How many colliding paths to name before summarizing. Enough to see the
/// shape of the overlap, few enough to stay readable in an error box.
const MAX_REPORTED_COLLISIONS: usize = 10;

/// The server's `mv` renames rather than merges, so a name that exists on both
/// sides can only be resolved by the user. Name the paths instead of guessing.
fn collision_message(collisions: &[Collision], language: Language) -> String {
    let mut message = copy(
        language,
        "Refusing to migrate: these paths exist under both peers, and moving would overwrite them.",
        "拒绝迁移：以下路径在两个 peer 下都存在，迁移会覆盖它们。",
    )
    .to_string();
    for collision in collisions.iter().take(MAX_REPORTED_COLLISIONS) {
        message.push_str(&format!("\n  {}", collision.to));
    }
    if let Some(rest) = collisions.len().checked_sub(MAX_REPORTED_COLLISIONS)
        && rest > 0
    {
        message.push_str(&format!(
            "\n  {}",
            copy(
                language,
                &format!("... and {rest} more"),
                &format!("……还有 {rest} 个"),
            )
        ));
    }
    message.push_str(copy(
        language,
        "\nMove or delete the colliding paths, then run the migration again.",
        "\n请先移动或删除冲突路径，然后重新执行迁移。",
    ));
    message
}

/// Default `--from`: the most recently recorded previous peer, else the id the
/// pre-git rule would have minted for this directory.
fn default_from(resolved: &ResolvedWorkspace) -> String {
    if let Some(previous) = resolved.previous_peer_ids.last() {
        return previous.clone();
    }
    if !resolved.peer.legacy_peer_id.is_empty() {
        return resolved.peer.legacy_peer_id.clone();
    }
    let legacy = legacy_sanitize(&resolved.identity.cwd);
    if legacy == resolved.peer.peer_id {
        String::new()
    } else {
        legacy
    }
}

fn peer_uri(user: &str, peer: &str, subtree: &str) -> String {
    format!("viking://user/{user}/peers/{peer}/{subtree}")
}

/// Walk two subtrees that both exist, planning the moves that can be made and
/// collecting the names that cannot.
async fn plan_merge(
    client: &HttpClient,
    source: &str,
    target: &str,
    moves: &mut Vec<Move>,
    collisions: &mut Vec<Collision>,
) -> Result<()> {
    let mut pending = vec![(source.to_string(), target.to_string())];
    while let Some((source, target)) = pending.pop() {
        let source_entries = list_children(client, &source).await?;
        let target_entries = list_children(client, &target).await?;

        for (name, is_dir) in source_entries {
            let child_source = format!("{source}/{name}");
            let child_target = format!("{target}/{name}");
            match target_entries.iter().find(|(other, _)| *other == name) {
                None => moves.push(Move {
                    from: child_source,
                    to: child_target,
                }),
                Some((_, target_is_dir)) if is_dir && *target_is_dir => {
                    pending.push((child_source, child_target));
                }
                Some(_) => collisions.push(Collision {
                    from: child_source,
                    to: child_target,
                }),
            }
        }
    }
    Ok(())
}

/// High enough that no real peer subtree reaches it, low enough to stay one
/// request. A listing that fills it is refused rather than trusted: a truncated
/// listing could hide exactly the collision this walk exists to find.
const LIST_LIMIT: i32 = 10_000;

async fn list_children(client: &HttpClient, uri: &str) -> Result<Vec<(String, bool)>> {
    let response = client
        .ls(uri, false, false, "agent", 0, true, LIST_LIMIT, &[])
        .await?;
    let entries = workspace_entries(&response);
    if entries.len() >= LIST_LIMIT as usize {
        return Err(Error::Client(format!(
            "{uri} holds at least {LIST_LIMIT} entries; migrate it by hand rather than on a listing this command cannot verify."
        )));
    }
    Ok(entries)
}

/// `ls` answers with either a bare array or an object holding one; both shapes
/// carry the same `name` / `isDir` entries.
fn workspace_entries(response: &Value) -> Vec<(String, bool)> {
    let items = response
        .as_array()
        .or_else(|| response.get("nodes").and_then(Value::as_array))
        .or_else(|| response.get("matches").and_then(Value::as_array))
        .or_else(|| response.get("result").and_then(Value::as_array));
    let Some(items) = items else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|item| {
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .or_else(|| {
                    item.get("uri")
                        .and_then(Value::as_str)
                        .and_then(|uri| uri.trim_end_matches('/').rsplit('/').next())
                })
                .filter(|name| !name.is_empty())?;
            let is_dir = item.get("isDir").and_then(Value::as_bool).unwrap_or(false);
            Some((name.to_string(), is_dir))
        })
        .collect()
}

/// A move needs its target's parent to exist. Peer directories are created
/// lazily server-side, so the first migration into a fresh peer makes them.
async fn ensure_parent(client: &HttpClient, uri: &str) -> Result<()> {
    let Some((parent, _)) = uri.rsplit_once('/') else {
        return Ok(());
    };
    if client.stat(parent).await.is_ok() {
        return Ok(());
    }
    Box::pin(ensure_parent(client, parent)).await?;
    match client.mkdir(parent, None).await {
        Ok(_) => Ok(()),
        // A concurrent writer, or a directory the server materialized between
        // the stat and the mkdir, is not an error for us.
        Err(error) if error.code() == "CONFLICT" => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::workspace::{EffectivePeer, MergeResult, WorkspaceIdentity};

    fn resolved(previous: &[&str], legacy: &str, effective: &str, cwd: &str) -> ResolvedWorkspace {
        let identity = WorkspaceIdentity {
            cwd: cwd.to_string(),
            root: "/repo".to_string(),
            ..WorkspaceIdentity::default()
        };
        ResolvedWorkspace {
            identity,
            layers: Vec::new(),
            merged: MergeResult::default(),
            peer: EffectivePeer {
                peer_id: effective.to_string(),
                source: "workspace",
                origin: "{git_remote}".to_string(),
                legacy_peer_id: legacy.to_string(),
            },
            previous_peer_ids: previous.iter().map(|id| (*id).to_string()).collect(),
            registry_path: std::path::PathBuf::new(),
            registry_exists: false,
            registry_conflict: false,
            warnings: Vec::new(),
        }
    }

    #[test]
    fn default_from_prefers_the_recorded_previous_peer() {
        let workspace = resolved(&["old-one", "old-two"], "-repo", "github.com-o-r", "/repo");
        assert_eq!(default_from(&workspace), "old-two");
    }

    #[test]
    fn default_from_falls_back_to_the_recomputed_legacy_id() {
        let workspace = resolved(&[], "-repo", "github.com-o-r", "/repo");
        assert_eq!(default_from(&workspace), "-repo");

        // A workspace already using the legacy id has nothing to migrate from.
        let legacy_already = resolved(&[], "", "-repo", "/repo");
        assert_eq!(default_from(&legacy_already), "");
    }

    #[test]
    fn peer_uri_names_the_two_subtrees_a_peer_owns() {
        assert_eq!(
            peer_uri("alice", "github.com-o-r", "memories"),
            "viking://user/alice/peers/github.com-o-r/memories"
        );
        assert_eq!(
            peer_uri("alice", "github.com-o-r", "resources"),
            "viking://user/alice/peers/github.com-o-r/resources"
        );
    }

    #[test]
    fn workspace_entries_reads_both_ls_shapes() {
        let bare = serde_json::json!([
            { "name": "a.md", "isDir": false },
            { "name": "sub", "isDir": true },
        ]);
        assert_eq!(
            workspace_entries(&bare),
            vec![("a.md".to_string(), false), ("sub".to_string(), true)]
        );

        let wrapped = serde_json::json!({
            "nodes": [{ "uri": "viking://user/u/peers/p/memories/sub", "isDir": true }]
        });
        assert_eq!(workspace_entries(&wrapped), vec![("sub".to_string(), true)]);

        assert!(workspace_entries(&serde_json::json!({ "status": "ok" })).is_empty());
    }
}
