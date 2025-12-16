"""Evaluation utilities for UVLP."""

from uvlp.evaluation.minimax_eval import (
    run_minimax_eval,
    posthoc_minimax_eval_from_distilled,
    ensure_minimax_deps,
)
from uvlp.evaluation.clip_diagnostics import (
    compute_clip_diagnostics_for_root,
)

__all__ = [
    "run_minimax_eval",
    "posthoc_minimax_eval_from_distilled",
    "ensure_minimax_deps",
    "compute_clip_diagnostics_for_root",
]

