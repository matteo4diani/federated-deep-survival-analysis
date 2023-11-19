from collections import namedtuple
from auton_survival import datasets, preprocessing
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
import pandas as pd
from verstack.stratified_continuous_split import scsplit

SurvivalDataset = namedtuple(
    "SurvivalDataset", ["features", "outcomes"]
)


def _features_outcomes_from_df(df: pd.DataFrame):
    outcome_cols = ["event", "time"]

    outcomes = df[outcome_cols].copy()
    features = df.drop(outcome_cols, axis=1)

    return features, outcomes


def _get_survival_dataset(df: pd.DataFrame):
    features, outcomes = _features_outcomes_from_df(df)

    return SurvivalDataset(features, outcomes)


def _stratified_split(
    features,
    outcomes,
    test_size,
    random_state,
    type="simple",
):
    args = (features, outcomes, test_size, random_state)
    match type:
        case "simple":
            return _simple_stratified_split(*args)
        case "shuffle":
            return _stratified_shuffle_split(*args)
        case "continuous":
            return _stratified_continuous_split(*args)
        case _:
            raise Exception(f"No split for type {type}")


def _simple_stratified_split(
    features, outcomes, test_size, random_state
):
    df = pd.concat([features, outcomes], axis=1)

    train_set, val_set = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=outcomes.event.values,
    )

    return _get_survival_dataset(
        train_set
    ), _get_survival_dataset(val_set)


def _stratified_shuffle_split(
    features, outcomes, test_size, random_state
):
    stratified_shuffle_split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_index, test_index = next(
        stratified_shuffle_split.split(
            features, outcomes.event.values
        )
    )

    features_train, features_test = (
        features.iloc[train_index],
        features.iloc[test_index],
    )
    outcomes_train, outcomes_test = (
        outcomes.iloc[train_index],
        outcomes.iloc[test_index],
    )

    return (
        SurvivalDataset(features_train, outcomes_train),
        SurvivalDataset(features_test, outcomes_test),
    )


def _stratified_continuous_split(
    features, outcomes, test_size, random_state
):
    data = pd.concat([features, outcomes], axis=1)

    train_df, test_df = scsplit(
        data,
        stratify=data.time,
        test_size=test_size,
        random_state=random_state,
    )

    return (
        _get_survival_dataset(train_df),
        _get_survival_dataset(test_df),
    )


def _stratified_partitioning(
    num_partitions,
    clients_set,
    test_size,
    random_state,
):
    stratified_k_split = StratifiedKFold(
        n_splits=num_partitions,
        random_state=random_state,
        shuffle=True,
    )

    partition_splits = stratified_k_split.split(
        clients_set.features, clients_set.outcomes.event
    )

    train_sets = []
    val_sets = []

    clients_df = pd.concat(
        [clients_set.features, clients_set.outcomes], axis=1
    )

    for _, partition_index in partition_splits:
        (
            client_features,
            client_outcomes,
        ) = _features_outcomes_from_df(
            clients_df.iloc[partition_index]
        )

        train_set, val_set = _stratified_split(
            client_features,
            client_outcomes,
            test_size=test_size,
            random_state=random_state,
            type="shuffle",
        )

        train_sets.append(train_set)

        val_sets.append(val_set)

    return train_sets, val_sets


def _get_support(test_size, random_state):
    """Load and pre-process SUPPORT dataset."""

    (
        outcomes,
        features,
        feature_dict,
    ) = datasets.load_dataset(
        "SUPPORT", return_features=True
    )

    # Preprocess (Impute and Scale) the features
    features = preprocessing.Preprocessor().fit_transform(
        features,
        feature_dict["cat"],
        feature_dict["num"],
    )

    return _stratified_split(
        features,
        outcomes,
        test_size,
        random_state,
        type="shuffle",
    )


def prepare_support_dataset(
    num_partitions: int = 2,
    server_test_size: float = 0.1,
    test_size: float = 0.2,
    random_state=0,
):
    """Generate random partitions."""
    clients_set, test_set = _get_support(
        server_test_size, random_state
    )

    train_sets, val_sets = _stratified_partitioning(
        num_partitions,
        clients_set,
        test_size,
        random_state,
    )

    input_dim = test_set.features.shape[-1]

    return train_sets, val_sets, test_set, input_dim
