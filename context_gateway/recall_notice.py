# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""The summary of automatic context a reply starts with when ``show_recall`` is on.

The user sees what OpenViking added to the turn; the model never does. Every line
rendered here must match ``RECALL_LINE`` in tool_protocols/common.py, which strips
the notice from replies before anything else reads them.
"""

from .tool_catalog import clip

# What the session-start block holds, in the order the notice names it.
PARTS = {
    "profile": "user profile",
    "memories": "memory index",
    "skills": "skill list",
    "history": "earlier sessions",
}
CATEGORIES = {
    **dict.fromkeys(("events", "entities", "preferences", "experiences", "memories"), "memory"),
    "resources": "resource",
    "skills": "skill",
}
PLURALS = {"memory": "memories", "resource": "resources", "skill": "skills", "item": "items"}
# Recall outcomes that are not worth a line: nothing found, or recall did not run.
QUIET = {"recalled", "empty", "disabled", "budget", "reminder"}
FAILURES = {
    "openviking_unavailable": "OpenViking unavailable",
    "openviking_http_401": "OpenViking rejected the key (401)",
    "openviking_version_mismatch": "OpenViking version mismatch",
    "recall_timeout": "OpenViking timed out",
}
NAMES = 3


def counted(number, noun):
    return f"{number} {noun if number == 1 else PLURALS[noun]}"


def category(entry):
    """memory, resource or skill, from the server's category or else the URI; else ""."""
    kind = CATEGORIES.get(entry.get("category"))
    if kind:
        return kind
    uri = entry.get("uri") or ""
    return next((k for k in ("memory", "resource", "skill") if f"/{PLURALS[k]}/" in uri), "")


def entry_name(uri):
    return clip(uri.rstrip("/").rsplit("/", 1)[-1].removesuffix(".md"), 40)


def recall_line(entries):
    """``> OpenViking recall: 4 items (3 memories, 1 resource) — a, b, +2 more``."""
    if not entries:
        return "> OpenViking recall: context added"
    kinds = [category(entry) for entry in entries]
    groups = ", ".join(
        counted(kinds.count(kind), kind)
        for kind in ("memory", "resource", "skill")
        if kind in kinds
    )
    line = "> OpenViking recall: " + counted(len(entries), "item")
    line += f" ({groups})" if groups else ""
    names = [entry_name(e["uri"]) for e in entries if isinstance(e.get("uri"), str)]
    names = [name for name in names if name]
    if names:
        more = len(entries) - min(len(names), NAMES)
        line += " — " + ", ".join(names[:NAMES]) + (f", +{more} more" if more else "")
    return line


def failure_line(reason):
    if reason in FAILURES:
        text = FAILURES[reason]
    elif reason.startswith("openviking_http_"):
        text = "OpenViking HTTP " + reason.removeprefix("openviking_http_")
    else:
        text = reason
    # The grammar needs text after the colon.
    return "> OpenViking recall failed: " + (clip(text) or "unknown error")


def render(parts, reason, entries):
    """The notice for one injection, or "" when there is nothing to show.

    ``parts`` are the PARTS keys the session-start block holds (none after the first
    injection), ``reason`` is the recall decision's and ``entries`` the search entries.
    """
    lines = []
    shown = [PARTS[part] for part in PARTS if part in parts]
    if shown:
        lines.append("> OpenViking context: " + ", ".join(shown))
    if reason == "recalled":
        lines.append(recall_line(entries))
    elif reason and reason not in QUIET:
        lines.append(failure_line(reason))
    return "\n".join(lines) + "\n\n" if lines else ""
