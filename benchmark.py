import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

CLOUD = "aws"
INSTANCE_TYPE = "t3.micro"
DATA_PATH = "creditcard.csv"
TARGET_COL = "Class"
RANDOM_STATE = 42
N_WARMUP = 20
N_LATENCY_RUNS = 200
THROUGHPUT_BATCH_SIZE = 1000

load_start = time.perf_counter()
df = pd.read_csv(DATA_PATH)
load_time_seconds = time.perf_counter() - load_start

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model_params = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "random_state": RANDOM_STATE,
    "scale_pos_weight": scale_pos_weight,
}
clf = lgb.LGBMClassifier(**model_params)

train_start = time.perf_counter()
clf.fit(X_train, y_train)
training_time_seconds = time.perf_counter() - train_start

best_iteration = None
best_iteration_note = (
    "No early stopping used (fixed n_estimators="
    f"{model_params['n_estimators']}); with only ~{int((y_train == 1).sum() * 0.1)} "
    "fraud rows available for a held-out validation split, early-stopping AUC was too "
    "noisy and stopped training after 2 trees, producing a badly underfit model. "
    "Training on the fixed estimator count instead is more stable for this dataset size."
)

y_proba = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)

auc_roc = roc_auc_score(y_test, y_proba)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

warmup_rows = X_test.iloc[:N_WARMUP]
for i in range(N_WARMUP):
    clf.predict(warmup_rows.iloc[[i]])

latency_rows = X_test.iloc[:N_LATENCY_RUNS] if len(X_test) >= N_LATENCY_RUNS else X_test
single_row_latencies_s = []
for i in range(len(latency_rows)):
    row = latency_rows.iloc[[i]]
    t0 = time.perf_counter()
    clf.predict(row)
    single_row_latencies_s.append(time.perf_counter() - t0)

inference_latency_ms_one_row = float(np.mean(single_row_latencies_s) * 1000)

batch_size = min(THROUGHPUT_BATCH_SIZE, len(X_test))
throughput_batch = X_test.iloc[:batch_size]
t0 = time.perf_counter()
clf.predict(throughput_batch)
throughput_elapsed_s = time.perf_counter() - t0
inference_throughput_rows_per_second = float(batch_size / throughput_elapsed_s)

result = {
    "cloud": CLOUD,
    "instance_type": INSTANCE_TYPE,
    "dataset_rows": int(len(df)),
    "fraud_rows": int((y == 1).sum()),
    "fraud_ratio": float((y == 1).mean()),
    "load_time_seconds": round(load_time_seconds, 4),
    "training_time_seconds": round(training_time_seconds, 4),
    "training_params": model_params,
    "best_iteration": best_iteration,
    "best_iteration_note": best_iteration_note,
    "auc_roc": round(float(auc_roc), 6),
    "accuracy": round(float(accuracy), 6),
    "precision": round(float(precision), 6),
    "recall": round(float(recall), 6),
    "f1_score": round(float(f1), 6),
    "inference_latency_ms_one_row": round(inference_latency_ms_one_row, 4),
    "inference_latency_runs": len(single_row_latencies_s),
    "inference_throughput_rows_per_second": round(inference_throughput_rows_per_second, 2),
    "inference_throughput_batch_size": batch_size,
}

with open("benchmark_result.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
