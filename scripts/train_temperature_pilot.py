#!/usr/bin/env python3
"""Exploratory grouped CV for the sparse elevated-temperature pilot endpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    matthews_corrcoef,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from dataset.build_windows import build_windows, fit_training_standardizer
from dataset.event_definition import EventDefinition, detect_temperature_episodes
from dataset.labels import make_interval_labels
from model.losses import discrete_hazard_nll
from model.raise_fpga import RaiseFPGA


SEED = 20260915
EVENT_TYPE = "elevated_pl_temperature_pilot"


def load_development(root: Path, split_path: Path, features: list[str]):
    splits = pd.read_csv(split_path)
    development = set(
        splits.loc[splits["partition"].eq("development"), "workload_group"]
    )
    frames = []
    for workload in sorted(development):
        for path in sorted((root / workload).glob("rep*_hw_samples.parquet")):
            repetition = int(path.name.split("_", 1)[0].removeprefix("rep"))
            frame = pd.read_parquet(path, columns=["timestamp_s", *features])
            frame.insert(0, "rep_id", repetition)
            frame.insert(0, "workload_id", workload)
            frame.insert(0, "run_id", f"{workload}/rep{repetition}")
            frames.append(frame)
    return pd.concat(frames, ignore_index=True), sorted(development)


def metric_rows(
    fold: int,
    risks: np.ndarray,
    event_index: np.ndarray,
    observed: np.ndarray,
    validation_groups: list[str],
    selected_thresholds: list[float],
) -> list[dict[str, object]]:
    rows = []
    for horizon in range(3):
        positive = (event_index >= 0) & (event_index <= horizon)
        known = observed[:, horizon] | positive
        truth = positive[known].astype(int)
        score = risks[known, horizon]
        both = len(np.unique(truth)) == 2
        prediction = score >= 0.5
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, prediction, average="binary", zero_division=0
        )
        true_positive = int(np.sum(prediction & (truth == 1)))
        false_positive = int(np.sum(prediction & (truth == 0)))
        false_negative = int(np.sum((~prediction) & (truth == 1)))
        true_negative = int(np.sum((~prediction) & (truth == 0)))
        selected_cutoff = selected_thresholds[horizon]
        selected_prediction = score >= selected_cutoff
        class_precision, class_recall, class_f1, class_support = (
            precision_recall_fscore_support(
                truth,
                selected_prediction,
                labels=[0, 1],
                average=None,
                zero_division=0,
            )
        )
        selected_tp = int(np.sum(selected_prediction & (truth == 1)))
        selected_fp = int(np.sum(selected_prediction & (truth == 0)))
        selected_fn = int(np.sum((~selected_prediction) & (truth == 1)))
        selected_tn = int(np.sum((~selected_prediction) & (truth == 0)))
        ece = 0.0
        for lower in np.linspace(0.0, 0.9, 10):
            in_bin = (score >= lower) & (score < lower + 0.1)
            if in_bin.any():
                ece += float(in_bin.mean()) * abs(
                    float(score[in_bin].mean()) - float(truth[in_bin].mean())
                )
        best_threshold = np.nan
        best_f1 = np.nan
        if both:
            curve_precision, curve_recall, curve_thresholds = precision_recall_curve(
                truth, score
            )
            curve_f1 = (
                2
                * curve_precision[:-1]
                * curve_recall[:-1]
                / np.maximum(curve_precision[:-1] + curve_recall[:-1], 1e-15)
            )
            best = int(np.nanargmax(curve_f1))
            best_threshold = float(curve_thresholds[best])
            best_f1 = float(curve_f1[best])
        rows.append(
            {
                "fold": fold,
                "horizon_s": (0.5, 2.0, 5.0)[horizon],
                "validation_groups": ";".join(validation_groups),
                "known_windows": int(known.sum()),
                "positive_windows": int(truth.sum()),
                "pr_auc": float(average_precision_score(truth, score)) if both else np.nan,
                "auroc": float(roc_auc_score(truth, score)) if both else np.nan,
                "brier": float(brier_score_loss(truth, score)),
                "cutoff": 0.5,
                "accuracy_at_0_5": float(accuracy_score(truth, prediction)),
                "balanced_accuracy_at_0_5": float(
                    balanced_accuracy_score(truth, prediction)
                ),
                "precision_at_0_5": float(precision),
                "recall_at_0_5": float(recall),
                "f1_at_0_5": float(f1),
                "tp_at_0_5": true_positive,
                "fp_at_0_5": false_positive,
                "tn_at_0_5": true_negative,
                "fn_at_0_5": false_negative,
                "posthoc_best_f1": best_f1,
                "posthoc_best_f1_cutoff": best_threshold,
                "prevalence": float(truth.mean()),
                "ece_10_bin": ece,
                "train_selected_cutoff": selected_cutoff,
                "selected_accuracy": float(
                    accuracy_score(truth, selected_prediction)
                ),
                "selected_balanced_accuracy": float(
                    balanced_accuracy_score(truth, selected_prediction)
                ),
                "selected_mcc": float(matthews_corrcoef(truth, selected_prediction)),
                "negative_precision_npv": float(class_precision[0]),
                "negative_recall_specificity": float(class_recall[0]),
                "negative_f1": float(class_f1[0]),
                "negative_support": int(class_support[0]),
                "positive_precision": float(class_precision[1]),
                "positive_recall_sensitivity": float(class_recall[1]),
                "positive_f1": float(class_f1[1]),
                "positive_support": int(class_support[1]),
                "selected_tp": selected_tp,
                "selected_fp": selected_fp,
                "selected_tn": selected_tn,
                "selected_fn": selected_fn,
                "false_positive_rate": selected_fp / max(selected_fp + selected_tn, 1),
                "false_negative_rate": selected_fn / max(selected_fn + selected_tp, 1),
            }
        )
    return rows


def select_training_f1_thresholds(
    risks: np.ndarray,
    event_index: np.ndarray,
    observed: np.ndarray,
) -> list[float]:
    """Select each cutoff on training windows only, then apply it to validation."""

    thresholds = []
    for horizon in range(3):
        positive = (event_index >= 0) & (event_index <= horizon)
        known = observed[:, horizon] | positive
        truth = positive[known].astype(int)
        score = risks[known, horizon]
        precision, recall, candidates = precision_recall_curve(truth, score)
        f1 = (
            2
            * precision[:-1]
            * recall[:-1]
            / np.maximum(precision[:-1] + recall[:-1], 1e-15)
        )
        thresholds.append(float(candidates[int(np.nanargmax(f1))]))
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/extracted/results"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    feature_config = yaml.safe_load(Path("configs/features.yaml").read_text())
    features = [entry["column"] for entry in feature_config["features"]]
    raw, development_groups = load_development(
        args.root, Path("results/split_manifest.csv"), features
    )
    windows = build_windows(raw, feature_columns=features)
    definition = EventDefinition(52.0, 0.5, 51.5, 1.0, 5.0, 0.5)
    episodes = detect_temperature_episodes(
        raw,
        run_col="run_id",
        timestamp_col="timestamp_s",
        temperature_col="temp_pl_temp_C",
        definition=definition,
    )
    episodes = episodes.loc[~episodes["left_censored"].astype(bool)].copy()
    labels = make_interval_labels(windows.metadata, {EVENT_TYPE: episodes})
    labels = labels.sort_values("anchor_index")
    if not np.array_equal(labels["anchor_index"].to_numpy(), np.arange(len(labels))):
        raise RuntimeError("label/window alignment failed")

    observed = labels[["observed_b1", "observed_b2", "observed_b3"]].to_numpy(bool)
    observed_count = observed.sum(axis=1).astype(np.int64)
    event_index = labels["event_interval_index"].to_numpy(np.int64)
    eligible = labels["label_eligible"].to_numpy(bool) & (observed_count > 0)
    values = windows.values[eligible]
    meta = windows.metadata.loc[eligible].reset_index(drop=True)
    observed = observed[eligible]
    observed_count = observed_count[eligible]
    event_index = event_index[eligible]

    event_groups = sorted(
        meta.loc[event_index >= 0, "workload_id"].drop_duplicates().tolist()
    )
    if len(event_groups) != 3:
        raise RuntimeError(f"pilot CV expects exactly three development event groups, got {event_groups}")
    negative_groups = [g for g in development_groups if g not in event_groups]
    folds = [
        [event_groups[i], *negative_groups[i::3]]
        for i in range(3)
    ]

    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, object]] = []
    training_log = []
    for fold, validation_groups in enumerate(folds, start=1):
        validation_mask = meta["workload_id"].isin(validation_groups).to_numpy()
        training_mask = ~validation_mask
        scaler = fit_training_standardizer(
            values[training_mask],
            feature_columns=features,
            partitions=["training"] * int(training_mask.sum()),
        )
        train_x = torch.from_numpy(scaler.transform(values[training_mask]))
        val_x = torch.from_numpy(scaler.transform(values[validation_mask]))
        train_event = torch.from_numpy(event_index[training_mask])
        val_event = torch.from_numpy(event_index[validation_mask])
        train_observed = torch.from_numpy(observed_count[training_mask])
        val_observed = torch.from_numpy(observed_count[validation_mask])
        generator = torch.Generator().manual_seed(SEED + fold)
        loader = DataLoader(
            TensorDataset(train_x, train_event, train_observed),
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
        )

        torch.manual_seed(SEED + fold)
        model = RaiseFPGA(channels=len(features))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_loss = float("inf")
        best_state = None
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch_x, batch_event, batch_observed in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)["hazard_logits"]
                loss = discrete_hazard_nll(logits, batch_event, batch_observed)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                val_output = model(val_x)
                val_loss = float(
                    discrete_hazard_nll(
                        val_output["hazard_logits"], val_event, val_observed
                    )
                )
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            risks = model(val_x)["cumulative_risk"].numpy()
            training_risks = model(train_x)["cumulative_risk"].numpy()
        selected_thresholds = select_training_f1_thresholds(
            training_risks,
            event_index[training_mask],
            observed[training_mask],
        )
        result_rows.extend(
            metric_rows(
                fold,
                risks,
                event_index[validation_mask],
                observed[validation_mask],
                validation_groups,
                selected_thresholds,
            )
        )
        checkpoint = {
            "model_state_dict": best_state,
            "features": features,
            "scaler_mean": scaler.mean,
            "scaler_scale": scaler.scale,
            "event_definition_version": "elevated-pl-temp-pilot-v2",
            "validation_groups": validation_groups,
            "seed": SEED + fold,
            "status": "EXPLORATORY_PILOT_NOT_CALIBRATED",
        }
        torch.save(checkpoint, output_dir / f"temperature_pilot_fold{fold}.pt")
        training_log.append(
            {
                "fold": fold,
                "training_windows": int(training_mask.sum()),
                "validation_windows": int(validation_mask.sum()),
                "training_event_episodes": int(
                    episodes[~episodes["run_id"].str.split("/").str[0].isin(validation_groups)].shape[0]
                ),
                "validation_event_episodes": int(
                    episodes[episodes["run_id"].str.split("/").str[0].isin(validation_groups)].shape[0]
                ),
                "best_validation_nll": best_loss,
            }
        )

    result_path = Path("results/temperature_pilot_cv.csv")
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)
    Path("results/temperature_pilot_training.json").write_text(
        json.dumps(
            {
                "status": "EXPLORATORY_PILOT_NOT_CALIBRATED",
                "seed": SEED,
                "features": features,
                "parameters": sum(p.numel() for p in RaiseFPGA(len(features)).parameters()),
                "development_event_episodes": len(episodes),
                "folds": training_log,
            },
            indent=2,
        )
        + "\n"
    )
    print(pd.DataFrame(result_rows).to_string(index=False))
    print(json.dumps(training_log, indent=2))


if __name__ == "__main__":
    main()
