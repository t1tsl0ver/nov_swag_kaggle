"""Optional feature transformations, applied identically to train and test."""

import re

import numpy as np
import pandas as pd

from fraud.data import CompetitionData


def apply_categorical_encoding(data: CompetitionData, encoding: str) -> None:
    """Replace the original text categories with one-hot columns when requested."""
    if encoding == "native":
        return
    if encoding != "onehot":
        raise ValueError(f"Unknown categorical encoding: {encoding}")

    columns = list(data.x_train.select_dtypes(include="category").columns)
    if not columns:
        return
    train_dummies = pd.get_dummies(data.x_train[columns], dtype=np.uint8)
    test_dummies = pd.get_dummies(data.x_test[columns], dtype=np.uint8)
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)
    names = [
        f"oh_{index:03d}__{re.sub(r'[^A-Za-z0-9_]+', '_', column).strip('_')}"
        for index, column in enumerate(train_dummies.columns)
    ]
    train_dummies.columns = names
    test_dummies.columns = names
    data.x_train = pd.concat([data.x_train.drop(columns=columns), train_dummies], axis=1)
    data.x_test = pd.concat([data.x_test.drop(columns=columns), test_dummies], axis=1)
