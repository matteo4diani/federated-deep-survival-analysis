from auton_survival import datasets, preprocessing, metrics
from auton_survival import enable_auton_logger
from auton_survival.models.cph import DeepCoxPH
from loguru import logger
from federated_deep_survival_analysis.log_config import (
    configure_loguru_logging,
)
import numpy as np
import pandas as pd

configure_loguru_logging(level="INFO")
# Load the SUPPORT Dataset
outcomes, features, feature_dict = datasets.load_dataset(
    "SUPPORT", return_features=True
)

# Preprocess (Impute and Scale) the features
features = preprocessing.Preprocessor().fit_transform(
    features,
    feature_dict["cat"],
    feature_dict["num"],
)

print(features.head())
from sklearn.model_selection import train_test_split

(
    features_train,
    features_test,
    outcomes_train,
    outcomes_test,
) = train_test_split(
    features,
    outcomes,
    test_size=0.25,
    random_state=42,
    stratify=outcomes.event.values,
)

# Train a Deep Cox Proportional Hazards (DCPH) model
model: DeepCoxPH = DeepCoxPH(layers=[128, 64, 32])

outcomes_train_times = outcomes_train.time.values

model.fit(
    features_train,
    outcomes_train_times,
    outcomes_train.event.values,
    iters=1000,
    patience=1000,
    vsize=0.2,
    breslow=False,
    weight_decay=0.001,
)

times = [365, 365 * 2, 365 * 4]

cic = metrics.concordance_index_censored(
    outcomes_test.event.values.astype(bool),
    outcomes_test.time.values,
    model.predict_time_independent_risk(
        features_test
    ).squeeze(),
)

cic_train = metrics.concordance_index_censored(
    outcomes_train.event.values.astype(bool),
    outcomes_train.time.values,
    model.predict_time_independent_risk(
        features_train
    ).squeeze(),
)

logger.info(f"C-Index Censored:\n{cic}")
logger.info(f"CIC training: {cic_train}")
