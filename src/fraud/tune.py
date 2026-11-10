"""Reproducible random search over LightGBM parameters."""

import json
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from fraud.data import load_data
from fraud.feature_engineering import apply_categorical_encoding
from fraud.train import BASE_PARAMS, SEED, evaluate, save_report, setup_logger

SEARCH_SPACE = {
    "learning_rate": [0.04, 0.07, 0.10],
    "max_depth": [5, 7, -1],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [30, 100, 200],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "subsample": [0.8, 1.0],
    "reg_lambda": [0.1, 1.0, 10.0],
    "max_bin": [127, 255],
}


def run_tuning(
    data_dir: Path,
    run_id: str,
    n_jobs: int,
    n_trials: int,
    importance_sample_size: int,
    categorical_encoding: str = "native",
) -> dict:
    logger = setup_logger()
    report_dir = Path("experiments") / run_id
    if report_dir.exists() or (Path("artifacts") / run_id).exists():
        raise FileExistsError(f"Run ID already exists: {run_id}")
    report_dir.mkdir(parents=True)
    logger.info("Tuning run %s: loading data from %s", run_id, data_dir)
    data = load_data(data_dir)
    apply_categorical_encoding(data, categorical_encoding)
    logger.info(
        "Loaded %d train, %d test, %d features; %d trials x 5 folds",
        len(data.x_train), len(data.x_test), len(data.x_train.columns), n_trials,
    )

    candidates = [dict(BASE_PARAMS)]
    if n_trials > 1:
        candidates.extend(ParameterSampler(SEARCH_SPACE, n_iter=n_trials - 1, random_state=SEED))
    trial_rows = []
    start = monotonic()
    best_score = -np.inf
    best_params = None
    best_trial = None

    for number, candidate in enumerate(candidates, start=1):
        params = {**BASE_PARAMS, **candidate, "n_jobs": n_jobs}
        logger.info("Trial %d/%d: %s", number, n_trials, json.dumps({k: params[k] for k in SEARCH_SPACE}))
        trial_start = monotonic()
        result = evaluate(data, params, save_importance=False, logger=logger)
        duration = monotonic() - trial_start
        score = result["oof_roc_auc"]
        row = {
            "trial": number,
            "source": "baseline" if number == 1 else "random_search",
            "oof_roc_auc": score,
            "fold_mean_roc_auc": result["fold_mean_roc_auc"],
            "fold_std_roc_auc": result["fold_std_roc_auc"],
            "mean_best_iteration": float(np.mean([fold["best_iteration"] for fold in result["folds"]])),
            "duration_seconds": round(duration, 1),
            **{key: params[key] for key in SEARCH_SPACE},
        }
        trial_rows.append(row)
        pd.DataFrame(trial_rows).to_csv(report_dir / "trials.csv", index=False)
        if score > best_score:
            best_score, best_params, best_trial = score, params, number
        elapsed = monotonic() - start
        logger.info(
            "Trial %d done: OOF ROC AUC=%.6f; best=%.6f (trial %d); ~%.0f s search remaining",
            number, score, best_score, best_trial, elapsed / number * (n_trials - number),
        )

    logger.info("Search finished. Best trial %d, parameters: %s", best_trial, json.dumps(best_params))
    logger.info("Training final 5-fold model with the best parameters")
    final = evaluate(
        data,
        best_params,
        artifact_dir=Path("artifacts") / run_id,
        importance_sample_size=importance_sample_size,
        logger=logger,
    )
    final.update(
        run_id=run_id,
        kind="tuning",
        n_trials=n_trials,
        best_trial=best_trial,
        search_best_oof_roc_auc=best_score,
        search_seed=SEED,
        categorical_encoding=categorical_encoding,
    )
    save_report(final, report_dir)
    permutation = pd.read_csv(report_dir / "permutation_mean.csv")
    logger.info("Top features by held-out ROC AUC drop:\n%s", permutation.head(10).to_string(index=False))
    logger.info("Final OOF ROC AUC=%.6f; report: %s; submission: artifacts/%s/submission.csv",
                final["oof_roc_auc"], report_dir, run_id)
    return final
