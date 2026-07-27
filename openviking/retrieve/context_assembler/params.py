# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Parameter contract and normalization for server-side context assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple

Detail = Literal["auto", "abstract", "overview", "full"]
Tier = Literal["uri", "abstract", "overview", "full"]
Purpose = Literal["chat", "coding"]

MEMORY_CATEGORIES: Tuple[str, ...] = ("events", "entities", "preferences", "experiences")
CATEGORY_KEYS: Tuple[str, ...] = (*MEMORY_CATEGORIES, "resources", "skills")

TIER_ORDER: Tuple[Tier, ...] = ("uri", "abstract", "overview", "full")
TIER_RANK: Dict[str, int] = {tier: rank for rank, tier in enumerate(TIER_ORDER)}

DEFAULT_MAX_TOKENS = 1600
DEFAULT_FULL_SCORE_THRESHOLD = 0.5
DEFAULT_LIMIT = 10
MAX_EXCLUDE_URIS = 200
MAX_PLANNED_QUERIES = 3
READ_CONCURRENCY = 8
OTHER_PEER_OVERFETCH = 4

ORIGIN_ORDER: Tuple[str, ...] = ("actor_peer", "self", "other_peer")

# Signature-level ceiling under detail="auto" keeps a large resource file or a
# sensitive skill body out of the injected block; an explicit detail overrides it.
AUTO_CEILING_BY_CATEGORY: Dict[str, Tier] = {"resources": "overview", "skills": "overview"}

# Purpose presets only supply quota ratios; they are overridden by explicit quotas.
# Defaults await production telemetry, so keep them here as one-line knobs.
PURPOSE_PRESETS: Dict[str, Dict[str, int]] = {
    "coding": {"events": 4, "entities": 6, "preferences": 2, "experiences": 2},
    "chat": {"events": 6, "entities": 6, "preferences": 3, "experiences": 1},
}

DEFAULT_QUOTAS: Dict[str, int] = {
    "events": 10,
    "entities": 10,
    "preferences": 3,
    "experiences": 0,
}

DEFAULT_OTHER_PEER_PENALTIES: Dict[str, float] = {
    "events": 0.1,
    "entities": 0.1,
    "preferences": 0.02,
    "experiences": 0.02,
    "resources": 0.02,
    "skills": 0.02,
}


@dataclass
class AssembleParams:
    """Resolved request contract for one context assembly run."""

    query: str = ""
    image_url: Optional[str] = None
    limit: int = DEFAULT_LIMIT
    score_threshold: Optional[float] = None
    filter: Optional[Dict[str, Any]] = None

    session_id: Optional[str] = None
    query_expansion: Literal["off", "auto"] = "auto"

    max_tokens: int = DEFAULT_MAX_TOKENS
    quotas: Optional[Mapping[str, int]] = None
    purpose: Optional[Purpose] = None
    detail: Detail = "auto"
    full_score_threshold: float = DEFAULT_FULL_SCORE_THRESHOLD
    dedup_turns: int = 0
    exclude_uris: Sequence[str] = field(default_factory=tuple)
    peer_scope: Literal["actor", "all"] = "all"
    other_peer_penalty: Any = None

    rewrite: Any = False
    rewrite_max_bullets: int = 6

    render: bool = True


def normalize_quotas(
    quotas: Optional[Mapping[str, Any]], purpose: Optional[str] = None
) -> Optional[Dict[str, int]]:
    """Resolve the active quota map, or ``None`` when bucketed sampling is off."""
    if quotas is None:
        if not purpose:
            return None
        preset = PURPOSE_PRESETS.get(purpose)
        return dict(preset) if preset else None

    resolved: Dict[str, int] = {}
    for key, value in quotas.items():
        if key not in CATEGORY_KEYS:
            continue
        try:
            resolved[key] = max(0, int(value))
        except (TypeError, ValueError):
            resolved[key] = 0
    return resolved


def _clamp_penalty(value: Any, fallback: float) -> float:
    try:
        penalty = float(value)
    except (TypeError, ValueError):
        penalty = fallback
    return min(1.0, max(0.0, penalty))


def normalize_penalties(value: Any = None) -> Dict[str, float]:
    """Normalize other-peer score penalties per category."""
    if value is None:
        return dict(DEFAULT_OTHER_PEER_PENALTIES)
    if isinstance(value, Mapping):
        merged = dict(DEFAULT_OTHER_PEER_PENALTIES)
        for key, penalty in value.items():
            if key not in DEFAULT_OTHER_PEER_PENALTIES:
                continue
            merged[key] = _clamp_penalty(penalty, merged[key])
        return merged
    penalty = _clamp_penalty(value, 0.0)
    return dict.fromkeys(CATEGORY_KEYS, penalty)


def normalize_exclude_uris(values: Optional[Sequence[str]]) -> set[str]:
    return {str(uri).strip() for uri in (values or ())[:MAX_EXCLUDE_URIS] if str(uri).strip()}


def ceiling_for(category: str, detail: Detail) -> Tier:
    """Highest tier a candidate of ``category`` may reach under ``detail``."""
    if detail == "auto":
        return AUTO_CEILING_BY_CATEGORY.get(category, "full")
    return detail
