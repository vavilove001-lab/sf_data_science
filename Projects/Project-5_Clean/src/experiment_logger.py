import json
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import sklearn


def log_experiment(
    experiment_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    feature_list: List[str],
    random_state: int,
    baseline_rmsle: float | None = None,
    output_dir: str = "experiments"
) -> Dict[str, Any]:
    """
    Логирует один эксперимент в JSON-файл.
    Если передан baseline_rmsle, добавляет флаг warning_needed, если модель хуже baseline.
    """
    os.makedirs(output_dir, exist_ok=True)

    log_data = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "random_state": random_state,
        "params": params,
        "metrics": metrics,
        "features": feature_list,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "baseline_rmsle": baseline_rmsle,
    }

    # Автоматически помечаем, если модель хуже наивного прогноза
    if baseline_rmsle is not None and "rmsle_valid" in metrics:
        log_data["warning_needed"] = metrics["rmsle_valid"] > baseline_rmsle
        if log_data["warning_needed"]:
            print(f"⚠️ ВНИМАНИЕ: модель хуже baseline (valid={metrics['rmsle_valid']:.4f} > baseline={baseline_rmsle:.4f})")
        else:
            print("✅ Модель лучше baseline")

    timestamp_clean = (
        log_data["timestamp"]
        .replace(":", "_")
        .replace(".", "_")
        .replace("-", "_")
    )
    filename = f"{timestamp_clean}_{experiment_name}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, default=str)

    print(f"✅ Эксперимент залогирован: {filepath}")
    return log_data


def update_summary_table(
    log_data: Dict[str, Any], summary_path: str = "experiments/summary.csv"
) -> None:
    row = {
        "experiment_name": log_data["experiment_name"],
        "timestamp": log_data["timestamp"],
        "random_state": log_data["random_state"],
        "model_type": log_data["params"].get("model_type", "unknown"),
        "max_depth": log_data["params"].get("max_depth"),
        "n_estimators": log_data["params"].get("n_estimators"),
        "rmsle_train": log_data["metrics"].get("rmsle_train"),
        "rmsle_valid": log_data["metrics"].get("rmsle_valid"),
        "num_features": len(log_data["features"]),
        "notes": log_data.get("notes", ""),
        "baseline_rmsle": log_data.get("baseline_rmsle"),
        "warning_needed": log_data.get("warning_needed", False),
    }

    df_row = pd.DataFrame([row])

    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        df_summary = pd.concat([df_summary, df_row], ignore_index=True)
    else:
        df_summary = df_row

    df_summary.to_csv(summary_path, index=False)
    print("✅ Сводная таблица обновлена:", summary_path)