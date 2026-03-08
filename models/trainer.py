"""
models/trainer.py
Trains XGBoost, Random Forest, and LSTM models.
Each model is wrapped in try/except — if one fails, the others still run.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore")

from features.engineer import FEATURE_COLUMNS


# ── Train / Test Split ────────────────────────────────────────────────────────

def time_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """Chronological train/test split — no shuffling for time series."""
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def prepare_xy(df: pd.DataFrame):
    """Extract available feature columns and target."""
    features = [col for col in FEATURE_COLUMNS if col in df.columns]
    if not features:
        raise ValueError("No valid feature columns found in DataFrame")
    X = df[features].values.astype(np.float32)
    y = df["Target_Direction"].values.astype(np.int32)
    # Replace any remaining NaN/Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, features


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute classification metrics safely."""
    try:
        return {
            "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "F1 Score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "ROC-AUC": round(float(roc_auc_score(y_true, y_prob)), 4),
            "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
    except Exception as e:
        return {
            "Accuracy": 0.0, "F1 Score": 0.0, "Precision": 0.0,
            "Recall": 0.0, "ROC-AUC": 0.5,
            "Confusion Matrix": [[0, 0], [0, 0]],
            "Error": str(e)
        }


# ── XGBoost ───────────────────────────────────────────────────────────────────

def train_xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Train XGBoost classifier with error handling."""
    try:
        X_train, y_train, features = prepare_xy(train_df)
        X_test, y_test, _ = prepare_xy(test_df)

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        return {
            "model": model, "name": "XGBoost", "success": True,
            "X_train": X_train, "X_test": X_test,
            "y_test": y_test, "y_pred": y_pred, "y_prob": y_prob,
            "features": features, "metrics": compute_metrics(y_test, y_pred, y_prob),
        }
    except Exception as e:
        print(f"  ❌ XGBoost failed: {e}")
        return {"name": "XGBoost", "success": False, "error": str(e),
                "metrics": {"Accuracy": 0, "F1 Score": 0, "Precision": 0, "Recall": 0, "ROC-AUC": 0.5}}


# ── Random Forest ─────────────────────────────────────────────────────────────

def train_random_forest(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Train Random Forest classifier with error handling."""
    try:
        X_train, y_train, features = prepare_xy(train_df)
        X_test, y_test, _ = prepare_xy(test_df)

        model = RandomForestClassifier(
            n_estimators=300, max_depth=8,
            min_samples_split=10, random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        return {
            "model": model, "name": "Random Forest", "success": True,
            "X_train": X_train, "X_test": X_test,
            "y_test": y_test, "y_pred": y_pred, "y_prob": y_prob,
            "features": features, "metrics": compute_metrics(y_test, y_pred, y_prob),
        }
    except Exception as e:
        print(f"  ❌ Random Forest failed: {e}")
        return {"name": "Random Forest", "success": False, "error": str(e),
                "metrics": {"Accuracy": 0, "F1 Score": 0, "Precision": 0, "Recall": 0, "ROC-AUC": 0.5}}


# ── LSTM ──────────────────────────────────────────────────────────────────────

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 20):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_lstm(train_df: pd.DataFrame, test_df: pd.DataFrame,
               seq_len: int = 20, epochs: int = 30) -> dict:
    """Train LSTM model with error handling."""
    try:
        X_train_raw, y_train_raw, features = prepare_xy(train_df)
        X_test_raw, y_test_raw, _ = prepare_xy(test_df)

        if len(X_train_raw) < seq_len * 2:
            raise ValueError(f"Not enough training data for LSTM (need {seq_len * 2}, got {len(X_train_raw)})")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw, seq_len)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_raw, seq_len)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_tr = torch.FloatTensor(X_train_seq).to(device)
        y_tr = torch.FloatTensor(y_train_seq).to(device)
        X_te = torch.FloatTensor(X_test_seq).to(device)

        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=False)

        model = LSTMModel(input_size=X_train_seq.shape[2]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        model.train()
        for _ in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                out = model(batch_X)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            y_prob = model(X_te).cpu().numpy()

        y_pred = (y_prob > 0.5).astype(int)
        y_test = y_test_seq.astype(int)

        return {
            "model": model, "name": "LSTM", "success": True,
            "scaler": scaler,
            "X_test": X_test_raw[:len(y_test)],
            "y_test": y_test, "y_pred": y_pred, "y_prob": y_prob,
            "features": features, "metrics": compute_metrics(y_test, y_pred, y_prob),
        }
    except Exception as e:
        print(f"  ❌ LSTM failed: {e}")
        return {"name": "LSTM", "success": False, "error": str(e),
                "metrics": {"Accuracy": 0, "F1 Score": 0, "Precision": 0, "Recall": 0, "ROC-AUC": 0.5}}


# ── Compare & Train All ───────────────────────────────────────────────────────

def compare_models(results: list) -> pd.DataFrame:
    """Build comparison DataFrame. Only includes successful models."""
    rows = []
    for r in results:
        if not r.get("success", False):
            continue
        row = {"Model": r["name"]}
        row.update({k: v for k, v in r["metrics"].items()
                    if k not in ("Confusion Matrix", "Error")})
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Model").sort_values("ROC-AUC", ascending=False)


def train_all_models(df: pd.DataFrame) -> dict:
    """
    Train all 3 models. If any model fails it is skipped gracefully.
    Returns results dict with whatever succeeded.
    """
    if len(df) < 150:
        raise ValueError(f"Not enough data to train models: {len(df)} rows (need at least 150)")

    train_df, test_df = time_split(df)
    print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    print("  🌲 Training Random Forest...")
    rf_result = train_random_forest(train_df, test_df)

    print("  ⚡ Training XGBoost...")
    xgb_result = train_xgboost(train_df, test_df)

    print("  🧠 Training LSTM...")
    lstm_result = train_lstm(train_df, test_df)

    all_results = [rf_result, xgb_result, lstm_result]
    successful = [r for r in all_results if r.get("success", False)]

    if not successful:
        raise RuntimeError("All 3 models failed to train. Check your data.")

    comparison = compare_models(all_results)

    return {
        "random_forest": rf_result,
        "xgboost": xgb_result,
        "lstm": lstm_result,
        "comparison": comparison,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "successful_models": [r["name"] for r in successful],
    }
