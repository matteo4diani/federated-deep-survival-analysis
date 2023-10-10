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

# Train a Deep Cox Proportional Hazards (DCPH) model
model: DeepCoxPH = DeepCoxPH(layers=[128, 64, 32])

outcome_times = outcomes.time.values

model.fit(features, outcome_times, outcomes.event.values, iters=100, patience=5, vsize=0.1)

# Predict risk at specific time horizons.
times = list(range(min(outcome_times), max(outcome_times), 100))

predictions = model.predict_survival(features, t=times)

ibs = metrics.survival_regression_metric("ibs", outcomes, predictions, times)

print(f"IBS: {ibs}")
