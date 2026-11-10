"""Inspect tree split nodes and held-out permutation importance."""

from time import monotonic

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def split_nodes(booster: lgb.Booster, fold: int) -> list[dict]:
    """Return every internal split, including its feature and training gain."""
    rows = []
    names = booster.feature_name()

    def visit(node: dict, tree_index: int, depth: int) -> None:
        if "split_feature" not in node:
            return
        rows.append(
            {
                "fold": fold,
                "tree": tree_index,
                "node": node["split_index"],
                "depth": depth,
                "feature": names[node["split_feature"]],
                "threshold": str(node["threshold"]),
                "split_gain": float(node["split_gain"]),
                "row_count": node["internal_count"],
            }
        )
        visit(node["left_child"], tree_index, depth + 1)
        visit(node["right_child"], tree_index, depth + 1)

    for tree_index, tree in enumerate(booster.dump_model()["tree_info"]):
        visit(tree["tree_structure"], tree_index, 0)
    return rows


def permutation_importance(
    model: lgb.LGBMClassifier,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    fold: int,
    sample_size: int,
    logger,
) -> list[dict]:
    """Measure ROC AUC loss after shuffling one feature on held-out rows."""
    rng = np.random.default_rng(2026 + fold)
    n = min(sample_size, len(x_valid))
    if n < len(x_valid):
        positions = rng.choice(len(x_valid), size=n, replace=False)
        x_sample = x_valid.iloc[positions].reset_index(drop=True)
        y_sample = y_valid.iloc[positions].reset_index(drop=True)
    else:
        x_sample = x_valid.reset_index(drop=True)
        y_sample = y_valid.reset_index(drop=True)
    if y_sample.nunique() != 2:
        raise ValueError("Permutation sample needs examples from both classes")
    baseline_auc = roc_auc_score(y_sample, model.predict_proba(x_sample)[:, 1])
    shuffled = x_sample.copy()
    rows = []
    start = monotonic()
    for i, feature in enumerate(x_sample.columns, start=1):
        positions = rng.permutation(n)
        shuffled[feature] = x_sample[feature].iloc[positions].reset_index(drop=True)
        shuffled_auc = roc_auc_score(y_sample, model.predict_proba(shuffled)[:, 1])
        shuffled[feature] = x_sample[feature]
        rows.append(
            {
                "fold": fold,
                "feature": feature,
                "baseline_auc": float(baseline_auc),
                "shuffled_auc": float(shuffled_auc),
                "auc_drop": float(baseline_auc - shuffled_auc),
                "sample_rows": n,
            }
        )
        if i % 5 == 0 or i == len(x_sample.columns):
            elapsed = monotonic() - start
            remaining = elapsed / i * (len(x_sample.columns) - i)
            logger.info("Fold %d importance: %d/%d features, ~%.0f s left", fold, i, len(x_sample.columns), remaining)
    return rows
