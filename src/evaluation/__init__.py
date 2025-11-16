"""Evaluation metrics for recommendation systems."""

from .metrics import (
    precision_at_k, recall_at_k, ndcg_at_k, map_at_k, hit_rate_at_k,
    coverage, diversity, evaluate_model, compare_models
)