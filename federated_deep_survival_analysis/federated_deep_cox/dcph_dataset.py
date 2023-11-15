from collections import namedtuple
from auton_survival import datasets, preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


SurvivalDataset = namedtuple("SurvivalDataset", ["features", "outcomes"])


def get_survival_dataset(df: pd.DataFrame):
    outcome_cols = ["event", "time"]

    outcomes = df[outcome_cols].copy()
    features = df.drop(outcome_cols, axis=1)

    return SurvivalDataset(features, outcomes)


def get_support(test_size: float = 0.1, random_state: int = 42):
    """Load and pre-process SUPPORT dataset."""

    outcomes, features, feature_dict = datasets.load_dataset(
        "SUPPORT", return_features=True
    )

    # Preprocess (Impute and Scale) the features
    features = preprocessing.Preprocessor().fit_transform(
        features, feature_dict["cat_feats"], feature_dict["num_feats"]
    )

    (
        features_train,
        features_test,
        outcomes_train,
        outcomes_test,
    ) = train_test_split(
        features, outcomes, test_size=test_size, random_state=random_state
    )

    return (
        SurvivalDataset(features_train, outcomes_train),
        SurvivalDataset(features_test, outcomes_test),
    )


def prepare_support_dataset(
    num_partitions: int,
    batch_size: int = 1,
    val_ratio: float = 0.1,
    random_state=0,
):
    """Generate random partitions."""
    #TODO refactor to generate IID partitions
    train_set, test_set = get_support()

    train_df = pd.concat([train_set.features, train_set.outcomes], axis=1)
    partition_splits = np.array_split(train_df, num_partitions)

    train_sets = []
    val_sets = []

    for partition in partition_splits:
        train_split, val_split = train_test_split(
            partition, test_size=val_ratio, random_state=random_state
        )

        train_sets.append(get_survival_dataset(train_split))

        val_sets.append(get_survival_dataset(val_split))

    return train_sets, val_sets, test_set
