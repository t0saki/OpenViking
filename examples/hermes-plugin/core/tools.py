"""The ``viking_*`` tool catalog, tool argument checks and the tool handlers."""

from __future__ import annotations

import json
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from .host import tool_error
from .http import _resolve_user_space
from .log import get_logger

logger = get_logger()


_REMOTE_RESOURCE_PREFIXES = ("http://", "https://", "git@", "ssh://", "git://")
_READ_BATCH_LIMIT = 3
_READ_BATCH_FULL_LIMIT = 2500
_LEVEL_ENDPOINTS = {"abstract": "/api/v1/content/abstract", "overview": "/api/v1/content/overview", "full": "/api/v1/content/read"}
_LEVEL_MAX_CHARS = {"abstract": 1200, "overview": 4000}
# OpenViking-generated summaries; non-.md sidecars are already rejected by the .md check.
_GENERATED_MEMORY_SUMMARY_FILENAMES = {".abstract.md", ".overview.md"}


def _tool_schema(name: str, description: str, properties: dict, required: list) -> dict:
    return {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required}}


def _str(description: str, **extra) -> dict:
    return {"type": "string", **extra, "description": description}


SEARCH_SCHEMA = _tool_schema(
    "viking_search",
    "Semantic search over the OpenViking knowledge base. Returns ranked results with viking:// URIs for deeper reading. "
    "Use mode='deep' for complex queries that need reasoning across multiple sources, 'fast' for simple lookups.",
    {
        "query": _str("Search query."),
        "mode": _str("Search depth (default: auto).", enum=["auto", "fast", "deep"]),
        "scope": _str("Viking URI prefix to scope search (e.g. 'viking://resources/docs/')."),
        "limit": {"type": "integer", "description": "Max results (default: 10)."},
    },
    ["query"],
)

READ_SCHEMA = _tool_schema(
    "viking_read",
    "Read one or a few specific viking:// URIs returned by viking_search or viking_browse. Three detail levels:\n"
    "  abstract — ~100 token summary (L0)\n  overview — ~2k token key points (L1)\n  full — complete content (L2)\n"
    "Start with abstract/overview, only use full when you need details. For multiple strong candidates, pass uris with up to three URIs.",
    {
        "uri": _str("Single viking:// URI to read."),
        "uris": {"type": "array", "items": {"type": "string"}, "description": "Optional batch of up to three viking:// URIs to read."},
        "level": _str("Detail level (default: overview).", enum=["abstract", "overview", "full"]),
    },
    [],
)

BROWSE_SCHEMA = _tool_schema(
    "viking_browse",
    "Browse the OpenViking knowledge store like a filesystem.\n  list — show directory contents\n  tree — show hierarchy\n  stat — show metadata for a URI",
    {
        "action": _str("Browse action.", enum=["tree", "list", "stat"]),
        "path": _str("Viking URI path (default: viking://). Examples: 'viking://resources/', 'viking://~/memories/'."),
    },
    ["action"],
)

REMEMBER_SCHEMA = _tool_schema(
    "viking_remember",
    "Submit important long-term information to OpenViking through session memory extraction. Success means the source was "
    "submitted, not that a distinct memory file was created. OpenViking can add, merge, or skip the final memory. Use this tool "
    "when OpenViking should decide how to retain the information. Do not use it when an exact memory file or URI is required. "
    "If the message is accepted but commit fails, it normally remains live and unextracted because server auto-commit is "
    "disabled by default; follow the returned recovery instructions.",
    {"content": _str("The information to remember.")},
    ["content"],
)

FORGET_SCHEMA = _tool_schema(
    "viking_forget",
    "Delete one OpenViking memory file by exact viking:// URI. Use only when the user explicitly asks to forget or delete a "
    "specific memory and you have the exact memory file URI. Resources, skills, sessions, directories, generated summaries, "
    "and broad deletes are rejected.",
    {"uri": _str("Exact viking:// memory file URI ending in .md.")},
    ["uri"],
)

ADD_RESOURCE_SCHEMA = _tool_schema(
    "viking_add_resource",
    "Add a remote URL or local file/directory to the OpenViking knowledge base. Remote resources must be public http(s), git, "
    "or ssh URLs. Local files are uploaded first using OpenViking temp_upload. The system automatically parses, indexes, and "
    "generates summaries.",
    {
        "url": _str("Remote URL or local file/directory path to add."),
        "reason": _str("Why this resource is relevant (improves search)."),
        "to": _str("Optional target viking:// URI for the resource."),
        "parent": _str("Optional parent viking:// URI. Cannot be used with to."),
        "instruction": _str("Optional processing instruction for semantic extraction."),
        "wait": {"type": "boolean", "description": "Whether to wait for processing to complete."},
        "timeout": {"type": "number", "description": "Timeout in seconds when wait is true."},
    },
    ["url"],
)

_TOOL_SCHEMAS = [SEARCH_SCHEMA, READ_SCHEMA, BROWSE_SCHEMA, REMEMBER_SCHEMA, FORGET_SCHEMA, ADD_RESOURCE_SCHEMA]
# Recall tools (read-only) whose results are never re-ingested — echoing recalled
# memory back into the transcript would re-store it. Write tools are deliberately absent.
_OPENVIKING_RECALL_TOOL_NAMES = {SEARCH_SCHEMA["name"], READ_SCHEMA["name"], BROWSE_SCHEMA["name"]}
# viking_* tool name -> provider method (resolved via getattr so instance patches apply).
_TOOL_HANDLERS = {schema["name"]: "_tool_" + schema["name"].removeprefix("viking_") for schema in _TOOL_SCHEMAS}
# Per-tool system-prompt guidance; system_prompt_block() keeps only the lines whose tool
# is registered, so the prompt never names a tool the model cannot call.
_SYSTEM_PROMPT_TOOL_GUIDANCE = (
    (SEARCH_SCHEMA["name"],
     "Use viking_search for extracted memories, facts, entities, events, and resources. For questions about "
     "remembered people, preferences, projects, events, or prior user context, search OpenViking before asking the "
     "user to repeat context. Prefer one or two focused searches, then read the strongest result URIs. If repeated "
     "searches return the same evidence or no stronger evidence, stop searching, answer from available evidence, and "
     "state uncertainty if needed."),
    (READ_SCHEMA["name"],
     "Use viking_read when you already have a specific viking:// memory or resource URI and need more detail; it can "
     "read up to three URIs at once."),
    (BROWSE_SCHEMA["name"], "Use viking_browse for URI diagnostics only; prefer search and read tools for evidence."),
    (REMEMBER_SCHEMA["name"], "Use viking_remember to store important facts."),
    (FORGET_SCHEMA["name"], "Use viking_forget to delete exact memory file URIs."),
    (ADD_RESOURCE_SCHEMA["name"], "Use viking_add_resource to index URLs/docs."),
)


def _zip_directory(dir_path: Path) -> Path:
    """Zip a directory tree into a temp file, skipping symlinks, escapes, and read-blocked files."""
    from .host import raise_if_read_blocked

    root = dir_path.resolve()
    zip_path = Path(tempfile.gettempdir()) / f"openviking_upload_{uuid.uuid4().hex}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in dir_path.rglob("*"):
            if file_path.is_symlink() or not file_path.is_file():
                continue
            try:
                resolved = file_path.resolve()
                resolved.relative_to(root)
                raise_if_read_blocked(str(resolved))
            except ValueError:
                continue
            zipf.write(file_path, arcname=str(file_path.relative_to(dir_path)).replace("\\", "/"))
    return zip_path


def _is_windows_absolute_path(value: str) -> bool:
    return len(value) >= 3 and value[0].isalpha() and value[1] == ":" and value[2] in {"/", "\\"}


def _validate_forget_memory_uri(raw_uri: Any, *, user_space: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    uri = raw_uri.strip() if isinstance(raw_uri, str) else ""
    if not uri:
        return None, "uri is required"
    parsed = urlparse(uri)
    if parsed.scheme != "viking" or not uri.startswith("viking://"):
        return None, "viking_forget only accepts viking:// memory file URIs"
    if parsed.query or parsed.fragment:
        return None, "viking_forget requires an exact URI without query or fragment"
    if uri.endswith("/") or not uri.endswith(".md"):
        return None, "viking_forget only deletes concrete .md memory files"
    parts = [part for part in uri[len("viking://") :].split("/") if part]
    if any(unquote(part) in {".", ".."} for part in parts):
        return None, "viking_forget does not accept dot path segments"
    # ``memories`` index for ``<scope>/[peers/<agent>/]memories/``; under ``user`` the uid is
    # required, since the uid-less shorthands are deprecated upstream.
    offsets = ((1, None), (3, 1)) if parts[:1] == ["~"] else ((2, None), (4, 2)) if parts[:1] == ["user"] else ()
    memories_idx = next((idx for idx, peer_at in offsets
                         if len(parts) > idx and parts[idx] == "memories" and (peer_at is None or parts[peer_at] == "peers")), None)
    if memories_idx is None or len(parts) < memories_idx + 2:
        return None, "viking_forget only deletes user memory file URIs"
    # An explicit uid can name someone else's space. Do not send a destructive
    # request unless the server has confirmed that this uid belongs to the caller.
    if parts[0] == "user":
        if not user_space:
            return None, "viking_forget could not verify the current OpenViking user identity; retry or use viking://~/..."
        if parts[1] != user_space:
            return None, (f"viking_forget only deletes your own memories; use viking://user/{user_space}/... "
                          "or viking://~/... instead")
    if uri.rsplit("/", 1)[-1] in _GENERATED_MEMORY_SUMMARY_FILENAMES:
        return None, "viking_forget cannot delete generated memory summary files"
    return uri, None


def _is_local_path_reference(value: str) -> bool:
    if not value or "\n" in value or "\r" in value or value.startswith(_REMOTE_RESOURCE_PREFIXES):
        return False
    if _is_windows_absolute_path(value):
        return True
    return value.startswith(("/", "./", "../", "~/", ".\\", "..\\", "~\\")) or "/" in value or "\\" in value


class ToolsMixin:
    """Methods of ``OpenVikingMemoryProvider`` moved here unchanged; mixed into that class."""

    @staticmethod
    def _normalize_summary_uri(uri: str) -> str:
        """Map pseudo summary files to their parent directory URI for L0/L1 reads."""
        for suffix in ("/.abstract.md", "/.overview.md", "/.read.md", "/.full.md"):
            if uri and uri.endswith(suffix):
                return uri[: -len(suffix)] or "viking://"
        return uri

    def _is_directory_uri(self, uri: str) -> bool | None:
        """fs/stat probe: True/False on a clean answer, None when unknown (callers fall back)."""
        try:
            result = self._unwrap_result(self._client.get("/api/v1/fs/stat", params={"uri": uri}))
        except Exception:
            return None
        if not isinstance(result, dict):
            return None
        for key in ("isDir", "is_dir"):
            if key in result:
                return bool(result.get(key))
        return result["type"] == "dir" if result.get("type") in {"dir", "file"} else None

    def _tool_search(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return tool_error("query is required")
        payload: Dict[str, Any] = {"query": query, **({"target_uri": args["scope"]} if args.get("scope") else {}), **({"limit": args["limit"]} if args.get("limit") else {})}
        deep = args.get("mode", "auto") == "deep"
        if deep and self._session_id:
            payload["session_id"] = self._session_id
        result = self._client.post("/api/v1/search/search" if deep else "/api/v1/search/find", payload).get("result", {})

        scored_entries = []
        for ctx_type in ("memories", "resources", "skills"):
            for item in result.get(ctx_type, []):
                raw_score = item.get("score")
                entry = {"uri": item.get("uri", ""), "type": ctx_type.rstrip("s"),
                         "score": round(raw_score, 3) if raw_score is not None else 0.0, "abstract": item.get("abstract", "")}
                if item.get("relations"):
                    entry["related"] = [r.get("uri") for r in item["relations"][:3]]
                scored_entries.append((raw_score if raw_score is not None else 0.0, entry))
        formatted = [entry for _, entry in sorted(scored_entries, key=lambda x: x[0], reverse=True)]
        return json.dumps({"results": formatted, "total": result.get("total", len(formatted))}, ensure_ascii=False)

    def _read_uri_payload(self, uri: str, level: str, *, limit: Optional[int] = None) -> Dict[str, Any]:
        summary_level = level in {"abstract", "overview"}
        # Pseudo summary files (viking://x/.overview.md) are read as their directory.
        resolved_uri = self._normalize_summary_uri(uri) if summary_level else uri
        # abstract/overview are directory-only (v0.3.x returns 500/412 for files):
        # probe fs/stat for non-pseudo URIs and route files straight to content/read.
        used_fallback = summary_level and resolved_uri == uri and self._is_directory_uri(uri) is False
        endpoint = "/api/v1/content/read" if used_fallback else _LEVEL_ENDPOINTS[level if summary_level else "full"]
        try:
            resp = self._client.get(endpoint, params={"uri": resolved_uri})
        except Exception:
            # Servers may still 500 on summary reads of plain files; fall back to a full read.
            if not summary_level or resolved_uri != uri or used_fallback:
                raise
            resp = self._client.get("/api/v1/content/read", params={"uri": uri})
            used_fallback = True

        result = self._unwrap_result(resp)
        content = result if isinstance(result, str) else (result.get("content", "") or result.get("text", "")) if isinstance(result, dict) else ""
        max_len = _LEVEL_MAX_CHARS.get(level, 8000)
        if limit is not None:
            max_len = max(200, min(max_len, limit))
        if len(content) > max_len:
            content = content[:max_len] + "\n\n[... truncated, use a more specific URI or full level]"
        return {"uri": uri, "resolved_uri": resolved_uri, "level": level, "content": content, **({"fallback": "content/read"} if used_fallback else {})}

    def _tool_read(self, args: dict) -> str:
        level = args.get("level", "overview")
        uri_arg = args.get("uri", "")
        uris_arg = args.get("uris", [])
        batch_requested = bool(uris_arg) or isinstance(uri_arg, list)
        raw_uris = uris_arg if isinstance(uris_arg, list) and uris_arg else uri_arg if isinstance(uri_arg, list) else [uri_arg]
        uris = list(dict.fromkeys(u.strip() for u in raw_uris if isinstance(u, str) and u.strip()))
        if not uris:
            return tool_error("uri or uris is required")

        selected = uris[:_READ_BATCH_LIMIT]
        if len(selected) == 1 and not batch_requested:
            return json.dumps(self._read_uri_payload(selected[0], level), ensure_ascii=False)
        per_item_limit = _READ_BATCH_FULL_LIMIT if len(selected) > 1 and level == "full" else None
        results: List[Dict[str, Any]] = []
        for uri in selected:
            try:
                results.append(self._read_uri_payload(uri, level, limit=per_item_limit))
            except Exception as e:
                results.append({"uri": uri, "level": level, "error": str(e)})
        return json.dumps({"level": level, "results": results, "requested": len(uris), "returned": len(results),
                           "truncated": len(uris) > len(selected)}, ensure_ascii=False)

    def _tool_browse(self, args: dict) -> str:
        action = args.get("action", "list")
        path = args.get("path", "viking://")
        result = self._unwrap_result(self._client.get(f"/api/v1/fs/{ {'tree': 'tree', 'stat': 'stat'}.get(action, 'ls') }", params={"uri": path}))

        if action in {"list", "tree"}:
            raw_entries = (result.get("entries") or result.get("items") or result.get("children") or []) if isinstance(result, dict) else result
            if isinstance(raw_entries, list):
                entries = [{"name": e.get("rel_path") or e.get("name") or (e.get("uri") or "").rsplit("/", 1)[-1], "uri": e.get("uri", ""),
                            "type": "dir" if (e.get("isDir") or e.get("is_dir") or e.get("type") == "dir") else "file", "abstract": e.get("abstract", "")}
                           for e in raw_entries[:50]]
                return json.dumps({"path": path, "entries": entries}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)

    def _tool_remember(self, args: dict) -> str:
        """Submit content through a dedicated session so it never touches the live Hermes session."""
        content = args.get("content", "")
        if not content:
            return tool_error("content is required")
        client = self._ensure_client()
        if not client:
            return tool_error("OpenViking server not connected")

        session_id = f"hermes-remember-{uuid.uuid4().hex[:12]}"
        session_uri = f"viking://user/{self._user_space(client)}/sessions/{session_id}"

        def failure(message: str, *, stage: str, message_status: str) -> str:
            return tool_error(
                message, session_id=session_id, session_uri=session_uri, failure_stage=stage, message_status=message_status,
                recovery_command=f"ov session commit {session_id}",
                recovery_note=(
                    "Inspect session_uri before recovery. If history/archive_* exists, do not retry. If messages.jsonl contains "
                    "the fact and no archive exists, run recovery_command with the same OpenViking profile and credentials as "
                    "Hermes. Otherwise, do not resubmit automatically; report the uncertain state to the user."
                ),
            )
        try:
            client.post(f"/api/v1/sessions/{session_id}/messages", {"role": "user", "parts": [{"type": "text", "text": content}]})
        except Exception as e:
            logger.error("OpenViking remember message failed for %s: %s", session_id, e)
            return failure(f"Memory message submission failed for session {session_id}: {e}", stage="message", message_status="unknown")
        try:
            commit = self._unwrap_result(client.post(f"/api/v1/sessions/{session_id}/commit", {"keep_recent_count": 0}))
        except Exception as e:
            logger.error("OpenViking remember commit failed for %s: %s", session_id, e)
            return failure(f"Memory message was accepted, but commit failed for session {session_id}: {e}", stage="commit", message_status="accepted")
        commit = commit if isinstance(commit, dict) else {}
        return json.dumps({
            "status": "submitted", "session_id": session_id, "session_uri": session_uri, "message_status": "accepted",
            "extraction_status": str(commit.get("status") or "accepted"),
            "message": "Memory source submitted to OpenViking session extraction. OpenViking may add, merge, or skip the final memory.",
            **{key: commit[key] for key in ("task_id", "trace_id") if commit.get(key)},
        })

    def _tool_forget(self, args: dict) -> str:
        # _resolve_user_space, not _user_space: its "default" fallback is a guess, not an identity.
        client = self._client
        uri, error = _validate_forget_memory_uri(args.get("uri"), user_space=_resolve_user_space(client))
        if error:
            return tool_error(error)
        result = self._unwrap_result(client.delete("/api/v1/fs", params={"uri": uri, "recursive": False}))
        result = result if isinstance(result, dict) else {}
        payload = {"status": "deleted", "uri": result.get("uri") or uri,
                   **{key: result[key] for key in ("estimated_deleted_count", "memory_cleanup", "semantic_root_uri", "semantic_status", "queue_status") if key in result}}
        return json.dumps(payload, ensure_ascii=False)

    def _tool_add_resource(self, args: dict) -> str:
        from .host import raise_if_read_blocked

        url = args.get("url", "")
        if not url:
            return tool_error("url is required")
        if args.get("to") and args.get("parent"):
            return tool_error("Cannot specify both 'to' and 'parent'")
        payload: Dict[str, Any] = {
            key: args[key] for key in ("reason", "to", "parent", "instruction", "wait", "timeout") if key in args and args[key] not in {None, ""}
        }

        parsed_url = urlparse(url)
        source_path = None
        if url.startswith(_REMOTE_RESOURCE_PREFIXES):
            pass
        elif parsed_url.scheme == "file":
            if parsed_url.netloc not in {"", "localhost"}:
                return tool_error(f"Unsupported non-local file URI: {url}")
            source_path = Path(url2pathname(parsed_url.path)).expanduser()
        elif not parsed_url.scheme or _is_windows_absolute_path(url):
            source_path = Path(url).expanduser()

        cleanup_path: Optional[Path] = None
        try:
            if source_path is None or not source_path.exists():
                if source_path is not None and _is_local_path_reference(url):
                    return tool_error(f"Local resource path does not exist: {url}")
                payload["path"] = url
            elif source_path.is_dir() or source_path.is_file():
                if source_path.is_dir():
                    cleanup_path = _zip_directory(source_path)  # directories upload as a zip
                else:
                    try:
                        raise_if_read_blocked(str(source_path))
                    except ValueError as exc:
                        return tool_error(str(exc))
                payload["source_name"] = source_path.name
                payload["temp_file_id"] = self._client.upload_temp_file(cleanup_path or source_path)
            else:
                return tool_error(f"Unsupported local resource path: {url}")
            result = self._client.post("/api/v1/resources", payload).get("result", {})
        finally:
            if cleanup_path:
                cleanup_path.unlink(missing_ok=True)

        return json.dumps({
            "status": "added",
            "root_uri": result.get("root_uri", ""),
            "message": "Resource queued for processing. Use viking_search after a moment to find it.",
        }, ensure_ascii=False)
