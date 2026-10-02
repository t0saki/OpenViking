# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Ledger keys shared by the kernel, capture worker and tool executor."""

from enum import Enum


class RecordKind(str, Enum):
    ROOT = "root"
    DISABLED = "disabled"
    INJECTION = "injection"
    SENT = "sent"
    HIDDEN = "hidden"
    USAGE = "usage"
    TAKEOVER = "takeover"
    ARCHIVE = "archive"
    REPLACEMENT = "replacement"
    RECALL_STATE = "recall_state"
    CAPTURE_CURSOR = "capture_cursor"
    CAPTURE_STATE = "capture_state"
    CAPTURE_FAILED = "capture_failed"
    CREATED = "created"
    WRITTEN = "written"
    CAPTURED = "captured"
    TOOL_CLAIM = "tool_claim"
    TOOL_RESULT = "tool_result"
    VENDOR = "vendor"


SESSION_KINDS = [
    RecordKind.ROOT,
    RecordKind.DISABLED,
    RecordKind.USAGE,
    RecordKind.RECALL_STATE,
    RecordKind.CAPTURE_CURSOR,
    RecordKind.CAPTURE_FAILED,
    RecordKind.TAKEOVER,
]
REPLAY_KINDS = [
    RecordKind.INJECTION,
    RecordKind.HIDDEN,
    RecordKind.SENT,
    RecordKind.ARCHIVE,
    RecordKind.REPLACEMENT,
]
