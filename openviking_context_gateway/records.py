# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""The five immutable decisions that change the upstream conversation."""

from enum import Enum


class RecordKind(str, Enum):
    ROOT = "root"
    INJECTION = "injection"
    DISABLED = "disabled"
    HIDDEN = "hidden"
    REPLACEMENT = "replacement"
