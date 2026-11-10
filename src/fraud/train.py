"""Five-fold LightGBM evaluation and submission generation."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fraud.data import ID, TARGET, CompetitionData, load_data
from fraud.feature_engineering import apply_categorical_encoding
from fraud.importance import permutation_importance, split_nodes

N_SPLITS = 5
SEED = 42
BASE_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 400,
    "learning_rate": 0.07,
    "max_depth": 5,
    "num_leaves": 15,
    "min_child_samples": 100,
    "colsample_bytree": 0.9,
    "subsample": 0.9,
    "subsample_freq": 1,
    "reg_lambda": 1.0,
    "max_bin": 127,
    "verbosity": -1,
    "random_state": SEED,
    "n_jobs": 4,
}


def new_run_id(prefix: str) -> str:
    return f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{prefix}"


def setup_logger() -> logging.Logger:
    report = Path("experiments/latest_report.txt")
    report.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fraud")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(report, mode="w", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def progress_callback(logger: logging.Logger, fold: int, max_trees: int):
    start = monotonic()

    def callback(env) -> None:
        trees = env.iteration + 1
        if trees == 1 or trees % 50 == 0 or trees == max_trees:
            elapsed = monotonic() - start
            upper_bound = elapsed / trees * (max_trees - trees)
            auc = next(
                (value for _, metric, value, _ in env.evaluation_result_list if metric == "auc"),
                None,
            )
            logger.info(
                "Fold %d: tree %d/%d, valid AUC=%s, elapsed %.0f s, ~%.0f s to tree limit",
                fold, trees, max_trees, f"{auc:.6f}" if auc is not None else "n/a", elapsed, upper_bound,
            )

    callback.order = 20
    callback.before_iteration = False
    return callback


def evaluate(
    data: CompetitionData,
    params: dict,
    *,
    artifact_dir: Path | None = None,
    save_importance: bool = True,
    importance_sample_size: int = 20000,
    logger: logging.Logger | None = None,
) -> dict:
    logger = logger or logging.getLogger("fraud")
    if data.y_train.value_counts().min() < N_SPLITS:
        raise ValueError("Each target class needs at least five rows for stratified CV")
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full(len(data.y_train), np.nan)
    test_preds = np.zeros(len(data.x_test)) if artifact_dir is not None else None
    fold_rows = []
    importance_rows = []
    permutation_rows = []
    node_rows = []
    cv_start = monotonic()
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=False)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(data.x_train, data.y_train), start=1):
        fold_start = monotonic()
        logger.info("Fold %d/%d: train=%d, valid=%d", fold, N_SPLITS, len(train_idx), len(valid_idx))
        model = lgb.LGBMClassifier(**params)
        model.fit(
            data.x_train.iloc[train_idx],
            data.y_train.iloc[train_idx],
            eval_set=[(data.x_train.iloc[valid_idx], data.y_train.iloc[valid_idx])],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(40, first_metric_only=True, verbose=False),
                progress_callback(logger, fold, params["n_estimators"]),
            ],
        )
        valid_preds = model.predict_proba(data.x_train.iloc[valid_idx])[:, 1]
        oof[valid_idx] = valid_preds
        fold_auc = roc_auc_score(data.y_train.iloc[valid_idx], valid_preds)
        fold_rows.append({"fold": fold, "roc_auc": float(fold_auc), "best_iteration": model.best_iteration_})
        if artifact_dir is not None:
            test_preds += model.predict_proba(data.x_test)[:, 1] / N_SPLITS
            model.booster_.save_model(str(artifact_dir / f"model_fold_{fold}.txt"))
        if save_importance:
            for kind in ("gain", "split"):
                values = model.booster_.feature_importance(importance_type=kind)
                importance_rows.extend(
                    {"fold": fold, "feature": feature, "type": kind, "importance": float(value)}
                    for feature, value in zip(data.x_train.columns, values)
                )
            node_rows.extend(split_nodes(model.booster_, fold))
            logger.info("Fold %d: checking held-out permutation importance", fold)
            permutation_rows.extend(
                permutation_importance(
                    model,
                    data.x_train.iloc[valid_idx],
                    data.y_train.iloc[valid_idx],
                    fold=fold,
                    sample_size=importance_sample_size,
                    logger=logger,
                )
            )
        elapsed = monotonic() - cv_start
        logger.info(
            "Fold %d done: ROC AUC=%.6f, trees=%d, %.0f s; ~%.0f s for remaining folds",
            fold, fold_auc, model.best_iteration_, monotonic() - fold_start,
            elapsed / fold * (N_SPLITS - fold),
        )

    if np.isnan(oof).any():
        raise RuntimeError("Some training rows have no OOF prediction")
    scores = np.array([row["roc_auc"] for row in fold_rows])
    result = {
        "oof_roc_auc": float(roc_auc_score(data.y_train, oof)),
        "fold_mean_roc_auc": float(scores.mean()),
        "fold_std_roc_auc": float(scores.std(ddof=1)),
        "folds": fold_rows,
        "params": params,
        "features": list(data.x_train.columns),
        "train_rows": len(data.x_train),
        "test_rows": len(data.x_test),
        "positive_rate": float(data.y_train.mean()),
        "cv_seed": SEED,
    }
    if artifact_dir is not None:
        pd.DataFrame({ID: data.train_ids, TARGET: data.y_train, "oof_probability": oof}).to_csv(
            artifact_dir / "oof.csv", index=False
        )
        submission = data.sample_submission.copy()
        submission[TARGET] = test_preds
        submission.to_csv(artifact_dir / "submission.csv", index=False)
    if save_importance:
        result["importance"] = pd.DataFrame(importance_rows)
        result["permutation_importance"] = pd.DataFrame(permutation_rows)
        result["tree_nodes"] = pd.DataFrame(node_rows)
    return result


def save_report(result: dict, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    importance = result.pop("importance", None)
    permutation = result.pop("permutation_importance", None)
    nodes = result.pop("tree_nodes", None)
    (report_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if importance is not None:
        importance.to_csv(report_dir / "feature_importance_by_fold.csv", index=False)
        grouped = importance.groupby(["type", "feature"], as_index=False)["importance"].mean()
        grouped.sort_values(["type", "importance"], ascending=[True, False]).to_csv(
            report_dir / "feature_importance_mean.csv", index=False
        )
    if permutation is not None:
        permutation.to_csv(report_dir / "permutation_by_fold.csv", index=False)
        grouped = permutation.groupby("feature", as_index=False)["auc_drop"].agg(["mean", "std"])
        grouped.sort_values("mean", ascending=False).to_csv(report_dir / "permutation_mean.csv", index=False)
    if nodes is not None:
        nodes.to_csv(report_dir / "tree_nodes.csv", index=False)


def run_baseline(
    data_dir: Path,
    run_id: str,
    n_jobs: int,
    dropped_features: list[str],
    importance_sample_size: int,
    params_file: Path | None = None,
    feature_set: str = "all",
    compare_to: Path | None = None,
    categorical_encoding: str = "native",
) -> dict:
    logger = setup_logger()
    logger.info("Run %s: loading data from %s", run_id, data_dir)
    data = load_data(data_dir)
    unknown = set(dropped_features) - set(data.x_train.columns)
    if unknown:
        raise ValueError(f"Unknown features to drop: {sorted(unknown)}")
    if len(set(dropped_features)) == len(data.x_train.columns):
        raise ValueError("At least one feature must remain after ablation")
    if dropped_features:
        data.x_train = data.x_train.drop(columns=dropped_features)
        data.x_test = data.x_test.drop(columns=dropped_features)
    apply_categorical_encoding(data, categorical_encoding)
    logger.info(
        "Loaded %d train, %d test, %d features; fraud rate %.3f%%",
        len(data.x_train), len(data.x_test), len(data.x_train.columns), 100 * data.y_train.mean(),
    )
    params = {**BASE_PARAMS, "n_jobs": n_jobs}
    if params_file is not None:
        saved = json.loads(params_file.read_text(encoding="utf-8"))
        params.update(saved["params"])
        params["n_jobs"] = n_jobs
        logger.info("Using parameters from %s", params_file)
    reference = None
    if compare_to is not None:
        reference = json.loads(compare_to.read_text(encoding="utf-8"))
        if (
            reference["cv_seed"] != SEED
            or reference["train_rows"] != len(data.x_train)
            or reference["test_rows"] != len(data.x_test)
            or reference["positive_rate"] != float(data.y_train.mean())
            or reference["params"] != params
        ):
            raise ValueError("Reference run must use the same data, folds, and model parameters")
        logger.info("Comparing OOF ROC AUC with %s", compare_to)
    result = evaluate(
        data, params, artifact_dir=Path("artifacts") / run_id,
        importance_sample_size=importance_sample_size, logger=logger,
    )
    result["run_id"] = run_id
    result["kind"] = "baseline"
    result["dropped_features"] = dropped_features
    result["feature_set"] = feature_set
    result["categorical_encoding"] = categorical_encoding
    result["params_file"] = str(params_file) if params_file is not None else None
    if reference is not None:
        result["comparison"] = {
            "reference": str(compare_to),
            "reference_oof_roc_auc": reference["oof_roc_auc"],
            "delta_oof_roc_auc": result["oof_roc_auc"] - reference["oof_roc_auc"],
        }
    save_report(result, Path("experiments") / run_id)
    logger.info(
        "Finished: OOF ROC AUC=%.6f; fold mean=%.6f +/- %.6f",
        result["oof_roc_auc"], result["fold_mean_roc_auc"], result["fold_std_roc_auc"],
    )
    logger.info("Reports: experiments/%s; submission: artifacts/%s/submission.csv", run_id, run_id)
    if compare_to is not None:
        logger.info(
            "Comparison with %s: reference OOF ROC AUC=%.6f, change=%+.6f",
            compare_to, result["comparison"]["reference_oof_roc_auc"],
            result["comparison"]["delta_oof_roc_auc"],
        )
    permutation = pd.read_csv(Path("experiments") / run_id / "permutation_mean.csv")
    logger.info("Top features by held-out ROC AUC drop:\n%s", permutation.head(10).to_string(index=False))
    logger.info("Non-positive mean ROC AUC drop: %d features", int((permutation["mean"] <= 0).sum()))
    return result
