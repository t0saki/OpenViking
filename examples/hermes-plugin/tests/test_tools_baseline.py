"""Characterization tests for the six viking_* tools.

These pin the current schemas, REST requests and returned strings so a later rename
can prove it changed names only. Requests go through the real ``_VikingClient`` over an
``httpx.MockTransport``; nothing reaches a network.
"""

import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

ENDPOINT = "http://openviking.test"
NOT_FOUND = (404, {"status": "error", "error": {"code": "NOT_FOUND", "message": "no route"}})
BOOM = (500, {"status": "error", "error": {"code": "INTERNAL", "message": "boom"}})


class Backend:
    """Route table keyed by (method, path); records every request it receives."""

    def __init__(self):
        self.requests = []
        self.uploads = []
        self.routes = {}

    def route(self, method, path, response):
        # response: (status, body) or a callable(request) -> (status, body).
        self.routes[(method, path)] = response

    def __call__(self, request):
        entry = {"method": request.method, "path": request.url.path, "params": dict(request.url.params)}
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("multipart/form-data"):
            entry["upload"] = re.search(rb'filename="([^"]+)"', request.content).group(1).decode()
            self.uploads.append(request.content)
        elif request.content:
            entry["json"] = json.loads(request.content)
        self.requests.append(entry)
        response = self.routes.get((request.method, request.url.path), NOT_FOUND)
        status, body = response(request) if callable(response) else response
        return httpx.Response(status, json=body)


@pytest.fixture
def wired(external_provider, monkeypatch):
    _, provider, module, _ = external_provider("tools-baseline")
    backend = Backend()
    http = httpx.Client(transport=httpx.MockTransport(backend))
    monkeypatch.setattr(module, "_get_httpx", lambda: http)
    provider._client = module._VikingClient(ENDPOINT, account="acme", user="alice", agent="hermes")
    try:
        yield provider, module, backend
    finally:
        http.close()


def dumps(payload):
    return json.dumps(payload, ensure_ascii=False)


def ok(result):
    return 200, {"status": "ok", "result": result}


def get(path, **params):
    return {"method": "GET", "path": path, "params": params}


def post(path, body):
    return {"method": "POST", "path": path, "params": {}, "json": body}


STATUS = get("/api/v1/system/status")


# -- schemas and dispatch -----------------------------------------------------

EXPECTED_SCHEMAS = [
    ("viking_search", ["query"], {
        "query": {"type": "string"},
        "mode": {"type": "string", "enum": ["auto", "fast", "deep"]},
        "scope": {"type": "string"},
        "limit": {"type": "integer"},
    }),
    ("viking_read", [], {
        "uri": {"type": "string"},
        "uris": {"type": "array", "items": {"type": "string"}},
        "level": {"type": "string", "enum": ["abstract", "overview", "full"]},
    }),
    ("viking_browse", ["action"], {
        "action": {"type": "string", "enum": ["tree", "list", "stat"]},
        "path": {"type": "string"},
    }),
    ("viking_remember", ["content"], {
        "content": {"type": "string"},
    }),
    ("viking_forget", ["uri"], {
        "uri": {"type": "string"},
    }),
    ("viking_add_resource", ["url"], {
        "url": {"type": "string"},
        "reason": {"type": "string"},
        "to": {"type": "string"},
        "parent": {"type": "string"},
        "instruction": {"type": "string"},
        "wait": {"type": "boolean"},
        "timeout": {"type": "number"},
    }),
]


def test_tool_schemas_names_and_parameters(wired):
    provider, _, _ = wired
    schemas = provider.get_tool_schemas()

    assert [s["name"] for s in schemas] == [name for name, _, _ in EXPECTED_SCHEMAS]
    for schema, (name, required, properties) in zip(schemas, EXPECTED_SCHEMAS, strict=True):
        assert set(schema) == {"name", "description", "parameters"}, name
        assert schema["description"].strip(), name
        params = schema["parameters"]
        assert set(params) == {"type", "properties", "required"}, name
        assert params["type"] == "object"
        assert params["required"] == required, name
        assert list(params["properties"]) == list(properties), name
        for key, expected in properties.items():
            prop = params["properties"][key]
            assert prop["description"].strip(), (name, key)
            assert {k: v for k, v in prop.items() if k != "description"} == expected, (name, key)


def test_tool_schemas_returns_fresh_list(wired):
    provider, _, _ = wired
    first = provider.get_tool_schemas()
    first.clear()
    assert len(provider.get_tool_schemas()) == 6


def test_unknown_tool_and_missing_client(wired):
    provider, _, backend = wired
    assert provider.handle_tool_call("viking_nope", {}) == dumps({"error": "Unknown tool: viking_nope"})
    provider._client = None
    assert provider.handle_tool_call("viking_search", {"query": "x"}) == dumps(
        {"error": "OpenViking server not connected"}
    )
    assert backend.requests == []


# -- viking_search ------------------------------------------------------------

SEARCH_RESULT = {
    "memories": [
        {"uri": "viking://user/alice/memories/preferences/tea.md", "score": 0.61234, "abstract": "Prefers tea"},
        {"uri": "viking://user/alice/memories/profile.md", "abstract": "No score"},
    ],
    "resources": [
        {"uri": "viking://resources/docs/a.md", "score": 0.9, "abstract": "文档",
         "relations": [{"uri": "viking://resources/docs/r1.md"}, {"uri": "viking://resources/docs/r2.md"},
                       {"uri": "viking://resources/docs/r3.md"}, {"uri": "viking://resources/docs/r4.md"}]},
    ],
    "skills": [{"uri": "viking://agent/skills/deploy", "score": 0.7, "abstract": "Deploy skill"}],
    "total": 42,
}


def test_search_fast_posts_find_and_formats_ranked_results(wired):
    provider, _, backend = wired
    backend.route("POST", "/api/v1/search/find", ok(SEARCH_RESULT))

    result = provider.handle_tool_call(
        "viking_search", {"query": "tea", "mode": "fast", "scope": "viking://user/", "limit": 5}
    )

    assert backend.requests == [
        post("/api/v1/search/find", {"query": "tea", "target_uri": "viking://user/", "limit": 5})
    ]
    # Pinned quirk: ctx_type.rstrip("s") turns "memories" into "memorie".
    assert result == dumps({
        "results": [
            {"uri": "viking://resources/docs/a.md", "type": "resource", "score": 0.9, "abstract": "文档",
             "related": ["viking://resources/docs/r1.md", "viking://resources/docs/r2.md",
                         "viking://resources/docs/r3.md"]},
            {"uri": "viking://agent/skills/deploy", "type": "skill", "score": 0.7, "abstract": "Deploy skill"},
            {"uri": "viking://user/alice/memories/preferences/tea.md", "type": "memorie", "score": 0.612,
             "abstract": "Prefers tea"},
            {"uri": "viking://user/alice/memories/profile.md", "type": "memorie", "score": 0.0,
             "abstract": "No score"},
        ],
        "total": 42,
    })


@pytest.mark.parametrize("session_id,expected_body", [
    ("live-sid", {"query": "tea", "session_id": "live-sid"}),
    (None, {"query": "tea"}),
])
def test_search_deep_posts_search_with_session(wired, session_id, expected_body):
    provider, _, backend = wired
    provider._session_id = session_id
    backend.route("POST", "/api/v1/search/search", ok({}))

    result = provider.handle_tool_call("viking_search", {"query": "tea", "mode": "deep"})

    assert backend.requests == [post("/api/v1/search/search", expected_body)]
    assert result == dumps({"results": [], "total": 0})


def test_search_auto_mode_uses_find_without_session(wired):
    provider, _, backend = wired
    provider._session_id = "live-sid"
    backend.route("POST", "/api/v1/search/find", ok({"memories": []}))

    provider.handle_tool_call("viking_search", {"query": "tea"})

    assert backend.requests == [post("/api/v1/search/find", {"query": "tea"})]


def test_search_failures(wired):
    provider, _, backend = wired
    assert provider.handle_tool_call("viking_search", {"query": ""}) == dumps({"error": "query is required"})
    assert backend.requests == []

    backend.route("POST", "/api/v1/search/find", BOOM)
    assert provider.handle_tool_call("viking_search", {"query": "tea"}) == dumps({"error": "INTERNAL: boom"})


# -- viking_read --------------------------------------------------------------

def stat_route(dirs):
    def respond(request):
        return ok({"isDir": request.url.params["uri"] in dirs})
    return respond


def test_read_overview_of_directory_probes_stat_then_reads_overview(wired):
    provider, _, backend = wired
    backend.route("GET", "/api/v1/fs/stat", stat_route({"viking://resources/docs"}))
    backend.route("GET", "/api/v1/content/overview", ok("Docs overview"))

    result = provider.handle_tool_call("viking_read", {"uri": "viking://resources/docs"})

    assert backend.requests == [
        get("/api/v1/fs/stat", uri="viking://resources/docs"),
        get("/api/v1/content/overview", uri="viking://resources/docs"),
    ]
    assert result == dumps({"uri": "viking://resources/docs", "resolved_uri": "viking://resources/docs",
                            "level": "overview", "content": "Docs overview"})


def test_read_pseudo_summary_file_reads_parent_directory_without_stat(wired):
    provider, _, backend = wired
    backend.route("GET", "/api/v1/content/abstract", ok({"content": "a" * 1300}))

    result = json.loads(provider.handle_tool_call(
        "viking_read", {"uri": "viking://resources/docs/.abstract.md", "level": "abstract"}
    ))

    assert backend.requests == [get("/api/v1/content/abstract", uri="viking://resources/docs")]
    assert result == {
        "uri": "viking://resources/docs/.abstract.md", "resolved_uri": "viking://resources/docs",
        "level": "abstract",
        "content": "a" * 1200 + "\n\n[... truncated, use a more specific URI or full level]",
    }


def test_read_overview_of_file_falls_back_to_content_read(wired):
    provider, _, backend = wired
    backend.route("GET", "/api/v1/fs/stat", stat_route(set()))
    backend.route("GET", "/api/v1/content/read", ok({"text": "File body"}))

    result = provider.handle_tool_call("viking_read", {"uri": "viking://resources/docs/a.md"})

    assert backend.requests == [
        get("/api/v1/fs/stat", uri="viking://resources/docs/a.md"),
        get("/api/v1/content/read", uri="viking://resources/docs/a.md"),
    ]
    assert result == dumps({"uri": "viking://resources/docs/a.md", "resolved_uri": "viking://resources/docs/a.md",
                            "level": "overview", "content": "File body", "fallback": "content/read"})


def test_read_overview_retries_content_read_when_stat_and_overview_fail(wired):
    provider, _, backend = wired
    backend.route("GET", "/api/v1/fs/stat", BOOM)
    backend.route("GET", "/api/v1/content/overview", BOOM)
    backend.route("GET", "/api/v1/content/read", ok("File body"))

    result = json.loads(provider.handle_tool_call("viking_read", {"uri": "viking://resources/x.md"}))

    assert [r["path"] for r in backend.requests] == [
        "/api/v1/fs/stat", "/api/v1/content/overview", "/api/v1/content/read"
    ]
    assert result["content"] == "File body"
    assert result["fallback"] == "content/read"


def test_read_full_reads_content_directly(wired):
    provider, _, backend = wired
    backend.route("GET", "/api/v1/content/read", ok("x" * 9000))

    result = json.loads(provider.handle_tool_call(
        "viking_read", {"uri": "viking://resources/big.md", "level": "full"}
    ))

    assert backend.requests == [get("/api/v1/content/read", uri="viking://resources/big.md")]
    assert result["content"] == "x" * 8000 + "\n\n[... truncated, use a more specific URI or full level]"
    assert "fallback" not in result


def test_read_batch_caps_uris_and_reports_item_errors(wired):
    provider, _, backend = wired

    def content(request):
        uri = request.url.params["uri"]
        return BOOM if uri.endswith("bad.md") else ok("y" * 3000)

    backend.route("GET", "/api/v1/content/read", content)
    uris = ["viking://r/a.md", "viking://r/bad.md", " viking://r/a.md ", "viking://r/c.md", "viking://r/d.md"]

    result = provider.handle_tool_call("viking_read", {"uris": uris, "level": "full"})

    assert backend.requests == [
        get("/api/v1/content/read", uri="viking://r/a.md"),
        get("/api/v1/content/read", uri="viking://r/bad.md"),
        get("/api/v1/content/read", uri="viking://r/c.md"),
    ]
    truncated = "y" * 2500 + "\n\n[... truncated, use a more specific URI or full level]"
    assert result == dumps({
        "level": "full",
        "results": [
            {"uri": "viking://r/a.md", "resolved_uri": "viking://r/a.md", "level": "full", "content": truncated},
            {"uri": "viking://r/bad.md", "level": "full", "error": "INTERNAL: boom"},
            {"uri": "viking://r/c.md", "resolved_uri": "viking://r/c.md", "level": "full", "content": truncated},
        ],
        "requested": 4,
        "returned": 3,
        "truncated": True,
    })


def test_read_single_uri_in_uris_still_returns_batch_shape(wired):
    provider, _, backend = wired
    backend.route("GET", "/api/v1/content/read", ok("body"))

    result = json.loads(provider.handle_tool_call("viking_read", {"uris": ["viking://r/a.md"], "level": "full"}))

    assert result == {
        "level": "full",
        "results": [{"uri": "viking://r/a.md", "resolved_uri": "viking://r/a.md", "level": "full", "content": "body"}],
        "requested": 1, "returned": 1, "truncated": False,
    }


def test_read_failures(wired):
    provider, _, backend = wired
    assert provider.handle_tool_call("viking_read", {}) == dumps({"error": "uri or uris is required"})
    assert provider.handle_tool_call("viking_read", {"uris": ["  "]}) == dumps({"error": "uri or uris is required"})
    assert backend.requests == []

    backend.route("GET", "/api/v1/content/read", BOOM)
    assert provider.handle_tool_call("viking_read", {"uri": "viking://r/a.md", "level": "full"}) == dumps(
        {"error": "INTERNAL: boom"}
    )


# -- viking_browse ------------------------------------------------------------

def test_browse_list_defaults_to_root_and_formats_entries(wired):
    provider, _, backend = wired
    entries = [
        {"uri": "viking://resources", "isDir": True, "abstract": "Resources"},
        {"name": "profile.md", "uri": "viking://user/alice/memories/profile.md"},
        {"rel_path": "docs/a.md", "name": "a.md", "uri": "viking://resources/docs/a.md", "type": "file"},
    ] + [{"uri": f"viking://resources/f{i}.md"} for i in range(60)]
    backend.route("GET", "/api/v1/fs/ls", ok(entries))

    result = json.loads(provider.handle_tool_call("viking_browse", {"action": "list"}))

    assert backend.requests == [get("/api/v1/fs/ls", uri="viking://")]
    assert result["path"] == "viking://"
    assert len(result["entries"]) == 50
    assert result["entries"][:3] == [
        {"name": "resources", "uri": "viking://resources", "type": "dir", "abstract": "Resources"},
        {"name": "profile.md", "uri": "viking://user/alice/memories/profile.md", "type": "file", "abstract": ""},
        {"name": "docs/a.md", "uri": "viking://resources/docs/a.md", "type": "file", "abstract": ""},
    ]


def test_browse_tree_reads_entries_from_dict_result(wired):
    provider, _, backend = wired
    backend.route("GET", "/api/v1/fs/tree", ok({"children": [{"uri": "viking://resources/docs", "type": "dir"}]}))

    result = provider.handle_tool_call("viking_browse", {"action": "tree", "path": "viking://resources/"})

    assert backend.requests == [get("/api/v1/fs/tree", uri="viking://resources/")]
    assert result == dumps({"path": "viking://resources/", "entries": [
        {"name": "docs", "uri": "viking://resources/docs", "type": "dir", "abstract": ""}
    ]})


def test_browse_stat_returns_raw_result(wired):
    provider, _, backend = wired
    backend.route("GET", "/api/v1/fs/stat", ok({"uri": "viking://resources/a.md", "isDir": False, "size": 12}))

    result = provider.handle_tool_call("viking_browse", {"action": "stat", "path": "viking://resources/a.md"})

    assert backend.requests == [get("/api/v1/fs/stat", uri="viking://resources/a.md")]
    assert result == dumps({"uri": "viking://resources/a.md", "isDir": False, "size": 12})


def test_browse_unknown_action_lists_but_returns_raw_result(wired):
    provider, _, backend = wired
    backend.route("GET", "/api/v1/fs/ls", ok([{"uri": "viking://resources"}]))

    result = provider.handle_tool_call("viking_browse", {"action": "walk", "path": "viking://"})

    assert backend.requests == [get("/api/v1/fs/ls", uri="viking://")]
    assert result == dumps([{"uri": "viking://resources"}])


def test_browse_failure(wired):
    provider, _, backend = wired
    assert provider.handle_tool_call("viking_browse", {"action": "list", "path": "viking://nope"}) == dumps(
        {"error": "NOT_FOUND: no route"}
    )
    assert backend.requests == [get("/api/v1/fs/ls", uri="viking://nope")]


# -- viking_remember ----------------------------------------------------------

REMEMBER_SID = "hermes-remember-0123456789ab"
REMEMBER_URI = f"viking://user/alice/sessions/{REMEMBER_SID}"
RECOVERY_NOTE = (
    "Inspect session_uri before recovery. If history/archive_* exists, do not retry. If messages.jsonl contains "
    "the fact and no archive exists, run recovery_command with the same OpenViking profile and credentials as "
    "Hermes. Otherwise, do not resubmit automatically; report the uncertain state to the user."
)


@pytest.fixture
def remember(wired, monkeypatch):
    provider, module, backend = wired
    monkeypatch.setattr(module, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="0123456789abcdef")))
    backend.route("GET", "/api/v1/system/status", ok({"user": "alice"}))
    return provider, backend


def remember_requests():
    return [
        STATUS,
        post(f"/api/v1/sessions/{REMEMBER_SID}/messages",
             {"role": "user", "parts": [{"type": "text", "text": "Alice prefers tea"}]}),
        post(f"/api/v1/sessions/{REMEMBER_SID}/commit", {"keep_recent_count": 0}),
    ]


def test_remember_submits_through_dedicated_session(remember):
    provider, backend = remember
    backend.route("POST", f"/api/v1/sessions/{REMEMBER_SID}/messages", ok({}))
    backend.route("POST", f"/api/v1/sessions/{REMEMBER_SID}/commit",
                  ok({"status": "accepted", "task_id": "task-1", "trace_id": ""}))

    result = provider.handle_tool_call("viking_remember", {"content": "Alice prefers tea"})

    assert backend.requests == remember_requests()
    assert result == json.dumps({
        "status": "submitted", "session_id": REMEMBER_SID, "session_uri": REMEMBER_URI,
        "message_status": "accepted", "extraction_status": "accepted",
        "message": "Memory source submitted to OpenViking session extraction. "
                   "OpenViking may add, merge, or skip the final memory.",
        "task_id": "task-1",
    })


def test_remember_commit_without_status_reports_accepted(remember):
    provider, backend = remember
    backend.route("POST", f"/api/v1/sessions/{REMEMBER_SID}/messages", ok({}))
    backend.route("POST", f"/api/v1/sessions/{REMEMBER_SID}/commit", ok(None))

    result = json.loads(provider.handle_tool_call("viking_remember", {"content": "Alice prefers tea"}))

    assert result["extraction_status"] == "accepted"
    assert "task_id" not in result and "trace_id" not in result


def test_remember_message_failure(remember):
    provider, backend = remember
    backend.route("POST", f"/api/v1/sessions/{REMEMBER_SID}/messages", BOOM)

    result = provider.handle_tool_call("viking_remember", {"content": "Alice prefers tea"})

    assert backend.requests == remember_requests()[:2]
    assert result == dumps({
        "error": f"Memory message submission failed for session {REMEMBER_SID}: INTERNAL: boom",
        "session_id": REMEMBER_SID, "session_uri": REMEMBER_URI, "failure_stage": "message",
        "message_status": "unknown", "recovery_command": f"ov session commit {REMEMBER_SID}",
        "recovery_note": RECOVERY_NOTE,
    })


def test_remember_commit_failure(remember):
    provider, backend = remember
    backend.route("POST", f"/api/v1/sessions/{REMEMBER_SID}/messages", ok({}))
    backend.route("POST", f"/api/v1/sessions/{REMEMBER_SID}/commit", BOOM)

    result = provider.handle_tool_call("viking_remember", {"content": "Alice prefers tea"})

    assert backend.requests == remember_requests()
    assert result == dumps({
        "error": f"Memory message was accepted, but commit failed for session {REMEMBER_SID}: INTERNAL: boom",
        "session_id": REMEMBER_SID, "session_uri": REMEMBER_URI, "failure_stage": "commit",
        "message_status": "accepted", "recovery_command": f"ov session commit {REMEMBER_SID}",
        "recovery_note": RECOVERY_NOTE,
    })


def test_remember_falls_back_to_configured_user_when_identity_probe_fails(remember):
    provider, backend = remember
    backend.route("GET", "/api/v1/system/status", BOOM)
    provider._client._user = "configured"
    backend.route("POST", f"/api/v1/sessions/{REMEMBER_SID}/messages", ok({}))
    backend.route("POST", f"/api/v1/sessions/{REMEMBER_SID}/commit", ok({"status": "queued"}))

    result = json.loads(provider.handle_tool_call("viking_remember", {"content": "Alice prefers tea"}))

    assert result["session_uri"] == f"viking://user/configured/sessions/{REMEMBER_SID}"
    assert result["extraction_status"] == "queued"


def test_remember_requires_content(remember):
    provider, backend = remember
    assert provider.handle_tool_call("viking_remember", {"content": ""}) == dumps({"error": "content is required"})
    assert backend.requests == []


# -- viking_forget ------------------------------------------------------------

@pytest.mark.parametrize("uri", [
    "viking://user/alice/memories/preferences/mem_abc123.md",
    "viking://~/memories/preferences/mem_abc123.md",
])
def test_forget_probes_identity_then_deletes_non_recursively(wired, uri):
    provider, _, backend = wired
    backend.route("GET", "/api/v1/system/status", ok({"user": "alice"}))
    backend.route("DELETE", "/api/v1/fs", ok({
        "uri": uri, "estimated_deleted_count": 1, "semantic_status": "queued", "unrelated": "dropped",
    }))

    result = provider.handle_tool_call("viking_forget", {"uri": f"  {uri}  "})

    assert backend.requests == [
        STATUS,
        {"method": "DELETE", "path": "/api/v1/fs", "params": {"uri": uri, "recursive": "false"}},
    ]
    assert result == dumps({"status": "deleted", "uri": uri, "estimated_deleted_count": 1,
                            "semantic_status": "queued"})


def test_forget_uses_requested_uri_when_server_omits_it(wired):
    provider, _, backend = wired
    uri = "viking://~/memories/profile.md"
    backend.route("GET", "/api/v1/system/status", ok({"user": "alice"}))
    backend.route("DELETE", "/api/v1/fs", ok("deleted"))

    assert provider.handle_tool_call("viking_forget", {"uri": uri}) == dumps({"status": "deleted", "uri": uri})


@pytest.mark.parametrize("uri,error", [
    ("", "uri is required"),
    ("https://example.com/a.md", "viking_forget only accepts viking:// memory file URIs"),
    ("viking://user/alice/memories/a.md?x=1", "viking_forget requires an exact URI without query or fragment"),
    ("viking://user/alice/memories/preferences/", "viking_forget only deletes concrete .md memory files"),
    ("viking://user/alice/memories/preferences", "viking_forget only deletes concrete .md memory files"),
    ("viking://user/alice/memories/../x.md", "viking_forget does not accept dot path segments"),
    ("viking://resources/docs/a.md", "viking_forget only deletes user memory file URIs"),
    ("viking://agent/skills/a.md", "viking_forget only deletes user memory file URIs"),
    ("viking://user/alice/sessions/s1.md", "viking_forget only deletes user memory file URIs"),
    ("viking://user/memories/preferences/a.md", "viking_forget only deletes user memory file URIs"),
    ("viking://user/alice/memories.md", "viking_forget only deletes user memory file URIs"),
    ("viking://user/alice/memories/.overview.md", "viking_forget cannot delete generated memory summary files"),
    ("viking://~/memories/preferences/.abstract.md", "viking_forget cannot delete generated memory summary files"),
    ("viking://user/bob/memories/a.md",
     "viking_forget only deletes your own memories; use viking://user/alice/... or viking://~/... instead"),
])
def test_forget_rejects_invalid_uris_without_deleting(wired, uri, error):
    provider, _, backend = wired
    backend.route("GET", "/api/v1/system/status", ok({"user": "alice"}))

    assert provider.handle_tool_call("viking_forget", {"uri": uri}) == dumps({"error": error})
    # The identity probe runs before validation, even for URIs that fail on shape alone.
    assert backend.requests == [STATUS]


def test_forget_without_confirmed_identity(wired):
    provider, _, backend = wired
    backend.route("GET", "/api/v1/system/status", BOOM)
    backend.route("DELETE", "/api/v1/fs", ok({}))

    explicit = provider.handle_tool_call("viking_forget", {"uri": "viking://user/alice/memories/a.md"})
    assert explicit == dumps({"error": "viking_forget could not verify the current OpenViking user identity; "
                                       "retry or use viking://~/..."})
    assert [r["method"] for r in backend.requests] == ["GET"]

    shorthand = provider.handle_tool_call("viking_forget", {"uri": "viking://~/memories/a.md"})
    assert json.loads(shorthand)["status"] == "deleted"
    assert [r["method"] for r in backend.requests] == ["GET", "GET", "DELETE"]


def test_forget_delete_failure(wired):
    provider, _, backend = wired
    backend.route("GET", "/api/v1/system/status", ok({"user": "alice"}))
    backend.route("DELETE", "/api/v1/fs", BOOM)

    assert provider.handle_tool_call("viking_forget", {"uri": "viking://~/memories/a.md"}) == dumps(
        {"error": "INTERNAL: boom"}
    )


# -- viking_add_resource ------------------------------------------------------

ADDED_MESSAGE = "Resource queued for processing. Use viking_search after a moment to find it."


def added(root_uri):
    return dumps({"status": "added", "root_uri": root_uri, "message": ADDED_MESSAGE})


@pytest.mark.parametrize("url", [
    "https://example.com/doc.md",
    "http://example.com/doc.md",
    "git@github.com:org/repo.git",
    "ssh://git@example.com/repo.git",
    "git://example.com/repo.git",
])
def test_add_resource_remote_url_posts_path(wired, url):
    provider, _, backend = wired
    backend.route("POST", "/api/v1/resources", ok({"root_uri": "viking://resources/doc"}))

    result = provider.handle_tool_call("viking_add_resource", {
        "url": url, "reason": "docs", "to": "", "parent": None, "instruction": "summarize",
        "wait": False, "timeout": 30,
    })

    assert backend.requests == [post("/api/v1/resources", {
        "reason": "docs", "instruction": "summarize", "wait": False, "timeout": 30, "path": url,
    })]
    assert result == added("viking://resources/doc")


def test_add_resource_non_path_string_is_sent_to_server_as_path(wired):
    provider, _, backend = wired
    backend.route("POST", "/api/v1/resources", ok({}))

    result = provider.handle_tool_call("viking_add_resource", {"url": "notes-that-do-not-exist", "to": "viking://resources/n"})

    assert backend.requests == [post("/api/v1/resources", {"to": "viking://resources/n", "path": "notes-that-do-not-exist"})]
    assert result == added("")


def test_add_resource_uploads_local_file_first(wired, tmp_path):
    provider, _, backend = wired
    source = tmp_path / "notes.md"
    source.write_text("# Notes\n", encoding="utf-8")
    backend.route("POST", "/api/v1/resources/temp_upload", ok({"temp_file_id": "tmp-1"}))
    backend.route("POST", "/api/v1/resources", ok({"root_uri": "viking://resources/notes"}))

    for url in (str(source), source.as_uri(), source.as_uri().replace("file://", "file://localhost")):
        backend.requests.clear()
        result = provider.handle_tool_call("viking_add_resource", {"url": url, "parent": "viking://resources/"})

        assert backend.requests == [
            {"method": "POST", "path": "/api/v1/resources/temp_upload", "params": {}, "upload": "notes.md"},
            post("/api/v1/resources", {"parent": "viking://resources/", "source_name": "notes.md",
                                       "temp_file_id": "tmp-1"}),
        ], url
        assert result == added("viking://resources/notes")
    assert b"# Notes\n" in backend.uploads[-1]


def test_add_resource_zips_local_directory_and_cleans_up(wired, tmp_path):
    provider, _, backend = wired
    source = tmp_path / "project"
    (source / "sub").mkdir(parents=True)
    (source / "a.md").write_text("A", encoding="utf-8")
    (source / "sub" / "b.md").write_text("B", encoding="utf-8")
    backend.route("POST", "/api/v1/resources/temp_upload", ok({"temp_file_id": "tmp-zip"}))
    backend.route("POST", "/api/v1/resources", ok({"root_uri": "viking://resources/project"}))

    result = provider.handle_tool_call("viking_add_resource", {"url": str(source)})

    upload, create = backend.requests
    assert re.fullmatch(r"openviking_upload_[0-9a-f]{32}\.zip", upload["upload"])
    assert (upload["method"], upload["path"]) == ("POST", "/api/v1/resources/temp_upload")
    assert create == post("/api/v1/resources", {"source_name": "project", "temp_file_id": "tmp-zip"})
    assert result == added("viking://resources/project")
    assert not (Path(tempfile.gettempdir()) / upload["upload"]).exists()
    zip_bytes = backend.uploads[0]
    start = zip_bytes.index(b"PK\x03\x04")
    archive = tmp_path / "uploaded.zip"
    archive.write_bytes(zip_bytes[start:zip_bytes.rindex(b"\r\n--")])
    with zipfile.ZipFile(archive) as zipf:
        assert sorted(zipf.namelist()) == ["a.md", "sub/b.md"]


def test_add_resource_local_path_failures(wired, tmp_path):
    provider, _, backend = wired
    missing = tmp_path / "missing.md"
    cases = [
        ({}, "url is required"),
        ({"url": ""}, "url is required"),
        ({"url": "https://example.com/a", "to": "viking://a", "parent": "viking://b"},
         "Cannot specify both 'to' and 'parent'"),
        ({"url": str(missing)}, f"Local resource path does not exist: {missing}"),
        ({"url": "./missing/notes.md"}, "Local resource path does not exist: ./missing/notes.md"),
        ({"url": "C:\\docs\\notes.md"}, "Local resource path does not exist: C:\\docs\\notes.md"),
        ({"url": "file://fileserver/share/a.md"}, "Unsupported non-local file URI: file://fileserver/share/a.md"),
    ]
    for args, error in cases:
        assert provider.handle_tool_call("viking_add_resource", args) == dumps({"error": error}), args
    assert backend.requests == []


def test_add_resource_missing_file_uri_is_rejected(wired, tmp_path):
    provider, _, backend = wired
    url = (tmp_path / "missing.md").as_uri()

    assert provider.handle_tool_call("viking_add_resource", {"url": url}) == dumps(
        {"error": f"Local resource path does not exist: {url}"}
    )
    assert backend.requests == []


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs a FIFO")
def test_add_resource_rejects_special_local_path(wired, tmp_path):
    provider, _, backend = wired
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)

    assert provider.handle_tool_call("viking_add_resource", {"url": str(fifo)}) == dumps(
        {"error": f"Unsupported local resource path: {fifo}"}
    )
    assert backend.requests == []


def test_add_resource_honours_read_block(wired, tmp_path, monkeypatch):
    import agent.file_safety as file_safety

    provider, _, backend = wired
    source = tmp_path / "secret.md"
    source.write_text("x", encoding="utf-8")

    def blocked(path):
        raise ValueError(f"Access denied: {path}")

    monkeypatch.setattr(file_safety, "raise_if_read_blocked", blocked)

    assert provider.handle_tool_call("viking_add_resource", {"url": str(source)}) == dumps(
        {"error": f"Access denied: {source}"}
    )
    assert backend.requests == []


def test_add_resource_upload_and_create_failures(wired, tmp_path):
    provider, _, backend = wired
    source = tmp_path / "notes.md"
    source.write_text("x", encoding="utf-8")

    backend.route("POST", "/api/v1/resources/temp_upload", ok({}))
    assert provider.handle_tool_call("viking_add_resource", {"url": str(source)}) == dumps(
        {"error": "OpenViking temp upload did not return temp_file_id"}
    )
    assert [r["path"] for r in backend.requests] == ["/api/v1/resources/temp_upload"]

    backend.requests.clear()
    backend.route("POST", "/api/v1/resources", BOOM)
    assert provider.handle_tool_call("viking_add_resource", {"url": "https://example.com/a"}) == dumps(
        {"error": "INTERNAL: boom"}
    )
    assert backend.requests == [post("/api/v1/resources", {"path": "https://example.com/a"})]
