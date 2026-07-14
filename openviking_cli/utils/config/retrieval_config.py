# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    """Configuration for retrieval ranking behavior."""

    hotness_alpha: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for blending hotness into final retrieval scores. "
            "0 disables hotness boost; 1 uses only hotness."
        ),
    )
    score_propagation_alpha: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for each child result's own score when blending with its parent score "
            "during hierarchical retrieval. 0 uses only the parent score; "
            "1 uses only the child score."
        ),
    )
    recall_intent_timeout_s: float = Field(
        default=10.0,
        gt=0.0,
        description=(
            "Timeout in seconds for optional recall query expansion. "
            "On timeout recall silently degrades to the original query; "
            "tune to your model endpoint latency."
        ),
    )
    recall_rewrite_timeout_s: float = Field(
        default=20.0,
        gt=0.0,
        description=(
            "Timeout in seconds for optional recall digest rewriting. "
            "On timeout recall returns the unrewritten rendered context; "
            "sized for a full-scale VLM emitting ~300-500 tokens."
        ),
    )

    model_config = {"extra": "forbid"}
