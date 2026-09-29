"""The ``viking_*`` tool catalog, tool argument checks and the tool handlers."""

from __future__ import annotations

import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

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
