"""Load competition CSV files and keep categorical codes consistent."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TARGET = "is_fraud"
ID = "id"


@dataclass
class CompetitionData:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    train_ids: pd.Series
    test_ids: pd.Series
    sample_submission: pd.DataFrame


def load_data(data_dir: Path) -> CompetitionData:
    paths = {name: data_dir / f"{name}.csv" for name in ("train", "test", "sample_submission")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing competition files: {', '.join(missing)}")

    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])
    sample = pd.read_csv(paths["sample_submission"])
    if TARGET not in train or ID not in train or ID not in test:
        raise ValueError("Expected id in train/test and is_fraud in train")
    if TARGET in test:
        raise ValueError("Test data unexpectedly contains the target")
    if list(sample.columns) != [ID, TARGET]:
        raise ValueError("Sample submission must have columns id,is_fraud")
    if train[ID].isna().any() or test[ID].isna().any():
        raise ValueError("IDs must not be missing")
    if train[ID].duplicated().any() or test[ID].duplicated().any():
        raise ValueError("IDs must be unique within train and test")
    if not sample[ID].equals(test[ID]):
        raise ValueError("Sample submission IDs must match test IDs in the same order")

    x_train = train.drop(columns=[ID, TARGET])
    x_test = test.drop(columns=[ID])
    if set(x_train.columns) != set(x_test.columns):
        raise ValueError("Train and test feature columns differ")
    x_test = x_test[x_train.columns].copy()
    x_train = x_train.copy()
    if x_train.empty or x_test.empty:
        raise ValueError("Train and test must contain rows and features")

    y = pd.to_numeric(train[TARGET], errors="raise")
    if y.isna().any() or not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError("Target must contain both binary classes without missing values")
    y = y.astype("int8")

    for column in x_train.columns:
        train_is_text = pd.api.types.is_object_dtype(x_train[column]) or pd.api.types.is_string_dtype(x_train[column])
        test_is_text = pd.api.types.is_object_dtype(x_test[column]) or pd.api.types.is_string_dtype(x_test[column])
        if train_is_text or test_is_text:
            train_values = x_train[column].astype("string")
            test_values = x_test[column].astype("string")
            categories = pd.Index(train_values.dropna().unique()).union(test_values.dropna().unique())
            dtype = pd.CategoricalDtype(categories=categories)
            x_train[column] = train_values.astype(dtype)
            x_test[column] = test_values.astype(dtype)
        elif not pd.api.types.is_numeric_dtype(x_train[column]):
            raise TypeError(f"Unsupported feature dtype: {column} ({x_train[column].dtype})")

    return CompetitionData(
        x_train=x_train,
        y_train=y,
        x_test=x_test,
        train_ids=train[ID],
        test_ids=test[ID],
        sample_submission=sample,
    )
