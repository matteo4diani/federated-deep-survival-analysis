from auton_survival import datasets, preprocessing, metrics
from auton_survival.models.cph import DeepCoxPH
import numpy as np
import pandas as pd

# Load the SUPPORT Dataset
outcomes, features = datasets.load_dataset("SUPPORT")

cat_feats = ["sex", "dzgroup", "dzclass", "income", "race", "ca"]
num_feats = [
    "age",
    "num.co",
    "meanbp",
    "wblc",
    "hrt",
    "resp",
    "temp",
    "pafi",
    "alb",
    "bili",
    "crea",
    "sod",
    "ph",
    "glucose",
    "bun",
    "urine",
    "adlp",
    "adls",
]

# Preprocess (Impute and Scale) the features
features = preprocessing.Preprocessor().fit_transform(features, cat_feats, num_feats)

from sklearn.model_selection import train_test_split

features_train, features_test, outcomes_train, outcomes_test = train_test_split(
    features, outcomes, test_size=0.25, random_state=42
)

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
times = [365, 365 * 2, 365 * 3, 365 * 4]

predictions = model.predict_survival(features_test, t=times)

ibs = metrics.survival_regression_metric(
    "ctd", outcomes_test, predictions, times, outcomes_train=outcomes_train
)

print(f"Concordance Indexes for horizons {times}: {ibs}")
