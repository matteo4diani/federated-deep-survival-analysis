from auton_survival import datasets, preprocessing, metrics
from auton_survival import enable_auton_logger
from auton_survival.models.cph import DeepCoxPH
from loguru import logger
from federated_deep_survival_analysis.log_config import configure_loguru_logging
import numpy as np
import pandas as pd

configure_loguru_logging(level="INFO")
# Load the SUPPORT Dataset
outcomes, features, feature_dict = datasets.load_dataset(
    "SUPPORT", return_features=True
)


# Preprocess (Impute and Scale) the features
features = preprocessing.Preprocessor().fit_transform(
    features, feature_dict["cat_feats"], feature_dict["num_feats"]
)

from sklearn.model_selection import train_test_split

(
    features_train,
    features_test,
    outcomes_train,
    outcomes_test,
) = train_test_split(features, outcomes, test_size=0.25, random_state=42)

# Train a Deep Cox Proportional Hazards (DCPH) model
model: DeepCoxPH = DeepCoxPH(layers=[128, 64, 32])

outcomes_train_times = outcomes_train.time.values

model.fit(
    features_train,
    outcomes_train_times,
    outcomes_train.event.values,
    iters=100,
    patience=5,
    vsize=0.1,
)

# Predict risk at specific time horizons.
admissible_times = list(
    range(
        min(outcomes_train_times) + 1,
        max(outcomes_train_times),
        30,
    )
)
times = [365, 365 * 2, 365 * 4]

predictions = model.predict_survival(features_test, t=times)

ctd = metrics.survival_regression_metric(
    "ctd",
    outcomes_test,
    predictions,
    times,
    outcomes_train=outcomes_train,
)
boolean_outcomes = list(
    map(lambda i: True if i == 1 else False, outcomes_test.event.values)
)
cic = metrics.concordance_index_censored(
    boolean_outcomes,
    outcomes_test.time.values,
    model.predict_time_independent_risk(features_test).squeeze(),
)

logger.info(f"C-Index Censored:\n{cic}")
logger.info(f"IPCW (time dependent):\n{dict(zip(times, ctd))}")
