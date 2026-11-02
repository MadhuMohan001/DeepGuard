"""Prediction logging with Pandas and threshold analysis with NumPy."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LOG_PATH = Path("predictions_log.csv")
COLUMNS = ["timestamp", "filename", "label", "confidence", "p_fake"]


def log_prediction(filename: str, result: dict) -> None:
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "filename": filename,
        "label": result["label"],
        "confidence": result["confidence"],
        "p_fake": result["p_fake"],
    }
    df = pd.DataFrame([row], columns=COLUMNS)
    df.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)


def summarize(low: float = 0.6) -> dict:
    """Overall stats plus how many predictions are low-confidence (worth reviewing)."""
    if not LOG_PATH.exists():
        return {"total": 0}
    df = pd.read_csv(LOG_PATH)
    conf = df["confidence"].to_numpy()
    return {
        "total": int(len(df)),
        "fake_count": int((df["label"] == "fake").sum()),
        "real_count": int((df["label"] == "real").sum()),
        "mean_confidence": round(float(np.mean(conf)), 4),
        "median_confidence": round(float(np.median(conf)), 4),
        "low_confidence_threshold": low,
        "low_confidence_count": int((conf < low).sum()),
    }
