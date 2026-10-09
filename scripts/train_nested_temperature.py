#!/usr/bin/env python3
"""Grouped-CV experiment for nested temperature-zone onset forecasting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from dataset.build_windows import build_windows, fit_training_standardizer
from dataset.event_definition import EventDefinition, detect_temperature_episodes
from dataset.labels import make_interval_labels
from dataset.synthetic_events import generate_synthetic_events
from model.losses import discrete_hazard_nll
from model.raise_fpga import NestedSeverityRaiseFPGA, PhysicsGuidedNestedRaiseFPGA
from scripts.train_temperature_pilot import load_development, metric_rows


SEED = 20260915
SEVERITIES = ("watch", "elevated", "high")
THRESHOLDS_C = (51.0, 52.0, 53.0)
HORIZONS_S = (0.5, 2.0, 5.0)


def episode_tables(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}
    for name, threshold in zip(SEVERITIES, THRESHOLDS_C, strict=True):
        definition = EventDefinition(
            threshold_c=threshold,
            persistence_s=0.5,
            recovery_threshold_c=threshold - 0.5,
            recovery_s=1.0,
            refractory_s=5.0,
            max_sample_gap_s=0.5,
        )
        table = detect_temperature_episodes(
            raw,
            run_col="run_id",
            timestamp_col="timestamp_s",
            temperature_col="temp_pl_temp_C",
            definition=definition,
        )
        tables[name] = table.loc[~table["left_censored"].astype(bool)].copy()
    return tables


def label_arrays(metadata: pd.DataFrame, episodes: dict[str, pd.DataFrame]):
    event = np.full((len(metadata), len(SEVERITIES)), -1, dtype=np.int64)
    observed = np.zeros((len(metadata), len(SEVERITIES)), dtype=np.int64)
    eligible = np.zeros((len(metadata), len(SEVERITIES)), dtype=bool)
    for severity_index, name in enumerate(SEVERITIES):
        if episodes[name].empty:
            eligible[:, severity_index] = True
            followup = (
                metadata["followup_end_timestamp"].to_numpy(float)
                - metadata["anchor_timestamp"].to_numpy(float)
            )
            observed[:, severity_index] = (
                followup[:, None] + 1e-12 >= np.asarray(HORIZONS_S)[None, :]
            ).sum(axis=1)
            continue
        part = make_interval_labels(metadata, {name: episodes[name]}).sort_values(
            "anchor_index"
        )
        if not np.array_equal(part["anchor_index"].to_numpy(), np.arange(len(metadata))):
            raise RuntimeError(f"label/window alignment failed for {name}")
        event[:, severity_index] = part["event_interval_index"].to_numpy(np.int64)
        observed[:, severity_index] = part[
            ["observed_b1", "observed_b2", "observed_b3"]
        ].to_numpy(bool).sum(axis=1)
        eligible[:, severity_index] = part["label_eligible"].to_numpy(bool)
    return event, observed, eligible


def future_temperature_deltas(
    raw: pd.DataFrame, metadata: pd.DataFrame, current_temperature: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros((len(metadata), 3), dtype=np.float32)
    mask = np.zeros((len(metadata), 3), dtype=bool)
    for run_id, meta_group in metadata.groupby("run_id", sort=False):
        rows = raw.loc[raw["run_id"].eq(run_id), ["timestamp_s", "temp_pl_temp_C"]]
        rows = rows.drop_duplicates("timestamp_s", keep="last").sort_values("timestamp_s")
        source_t = rows["timestamp_s"].to_numpy(float)
        source_y = rows["temp_pl_temp_C"].to_numpy(float)
        indices = meta_group.index.to_numpy()
        anchor = meta_group["anchor_timestamp"].to_numpy(float)
        followup_end = meta_group["followup_end_timestamp"].to_numpy(float)
        for horizon_index, horizon in enumerate(HORIZONS_S):
            requested = anchor + horizon
            source_index = np.searchsorted(source_t, requested, side="right") - 1
            safe = np.maximum(source_index, 0)
            valid = (
                (source_index >= 0)
                & (requested - source_t[safe] <= 0.2 + 1e-12)
                & (requested <= followup_end + 1e-12)
            )
            target[indices, horizon_index] = (
                source_y[safe] - current_temperature[indices]
            ).astype(np.float32)
            mask[indices, horizon_index] = valid
    return target, mask


def select_thresholds(risks, event_index, observed_count):
    cutoffs = []
    for horizon in range(3):
        positive = (event_index >= 0) & (event_index <= horizon)
        known = (observed_count > horizon) | positive
        truth = positive[known].astype(int)
        score = risks[known, horizon]
        if len(np.unique(truth)) < 2:
            cutoffs.append(0.5)
            continue
        precision, recall, candidates = precision_recall_curve(truth, score)
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
            precision[:-1] + recall[:-1], 1e-15
        )
        cutoffs.append(float(candidates[int(np.nanargmax(f1))]))
    return cutoffs


def select_recall_constrained_thresholds(
    risks, event_index, observed_count, target_recall=0.90
):
    """Choose training-only cutoffs meeting recall while maximizing precision."""
    cutoffs = []
    for horizon in range(3):
        positive = (event_index >= 0) & (event_index <= horizon)
        known = (observed_count > horizon) | positive
        truth = positive[known].astype(int)
        score = risks[known, horizon]
        if len(np.unique(truth)) < 2:
            cutoffs.append(0.5)
            continue
        precision, recall, candidates = precision_recall_curve(truth, score)
        eligible = np.flatnonzero(recall[:-1] >= target_recall)
        if len(eligible) == 0:
            cutoffs.append(float(np.min(score)))
            continue
        best_precision = precision[eligible].max()
        best = eligible[precision[eligible] == best_precision]
        cutoffs.append(float(candidates[best[-1]]))
    return cutoffs


def training_loss(output, event, observed, current_temp, aux, aux_mask, improved):
    total = output["hazard_logits"].new_tensor(0.0)
    for severity, threshold in enumerate(THRESHOLDS_C):
        valid = observed[:, severity] > 0
        if not torch.any(valid):
            continue
        losses = discrete_hazard_nll(
            output["hazard_logits"][valid, severity],
            event[valid, severity],
            observed[valid, severity],
            reduction="none",
        )
        if improved:
            is_event = event[valid, severity] >= 0
            positives = int(is_event.sum())
            negatives = len(is_event) - positives
            positive_weight = min(20.0, np.sqrt(negatives / max(positives, 1)))
            hard_negative = (~is_event) & (current_temp[valid] >= threshold - 1.5)
            weights = torch.ones_like(losses)
            weights[is_event] = positive_weight
            weights[hard_negative] = 2.0
            losses = losses * weights / weights.mean()
        total = total + losses.mean()
    total = total / len(SEVERITIES)
    if improved and torch.any(aux_mask):
        total = total + 0.2 * F.smooth_l1_loss(
            output["future_temperature_delta"][aux_mask], aux[aux_mask]
        )
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/extracted/results"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--transition-experiment",
        action="store_true",
        help="Use post-hoc exploratory 49/50/51 C zones and balanced augmentation",
    )
    parser.add_argument(
        "--lower-transition-experiment",
        action="store_true",
        help="Use forecastable post-hoc 48.5/48.75/49 C transition zones",
    )
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument(
        "--merge-synthetic",
        action="store_true",
        help="Merge generated 49/50/51 C histories into each real training fold",
    )
    parser.add_argument("--synthetic-samples-per-fold", type=int, default=30000)
    parser.add_argument("--physics-guided", action="store_true")
    parser.add_argument(
        "--cutoff-objective", choices=("recall", "f1"), default="recall"
    )
    args = parser.parse_args()

    global SEVERITIES, THRESHOLDS_C
    output_prefix = "nested_temperature"
    experimental = args.transition_experiment or args.lower_transition_experiment
    if args.lower_transition_experiment:
        SEVERITIES = ("transition_48p5c", "transition_48p75c", "transition_49c")
        THRESHOLDS_C = (48.5, 48.75, 49.0)
        output_prefix = "lower_transition_temperature"
    elif args.transition_experiment:
        SEVERITIES = ("transition_49c", "transition_50c", "transition_51c")
        THRESHOLDS_C = (49.0, 50.0, 51.0)
        output_prefix = "transition_temperature"
    if args.pretrained_checkpoint is not None:
        output_prefix = f"synthetic_pretrained_{output_prefix}"
    if args.merge_synthetic:
        if not args.transition_experiment or args.lower_transition_experiment:
            raise ValueError("--merge-synthetic requires --transition-experiment")
        output_prefix = "merged_synthetic_real_transition_temperature"
    if args.physics_guided:
        output_prefix = f"physics_guided_{output_prefix}"

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    features = [
        row["column"]
        for row in yaml.safe_load(Path("configs/features.yaml").read_text())["features"]
    ]
    raw, development_groups = load_development(
        args.root, Path("results/split_manifest.csv"), features
    )
    windows = build_windows(raw, feature_columns=features)
    episodes = episode_tables(raw)
    event, observed, eligible = label_arrays(windows.metadata, episodes)
    current_temp = windows.values[:, -1, features.index("temp_pl_temp_C")]
    aux, aux_mask = future_temperature_deltas(raw, windows.metadata, current_temp)

    if experimental:
        episode_group_counts = (
            episodes[SEVERITIES[-1]].assign(
                workload_group=lambda frame: frame["run_id"].str.split("/").str[0]
            )["workload_group"].value_counts()
        )
        folds = [[], [], []]
        fold_counts = [0, 0, 0]
        for group, count in episode_group_counts.items():
            target_fold = int(np.argmin(fold_counts))
            folds[target_fold].append(group)
            fold_counts[target_fold] += int(count)
        negative_groups = [g for g in development_groups if g not in episode_group_counts]
        for index, group in enumerate(negative_groups):
            folds[index % 3].append(group)
    else:
        elevated_groups = sorted(
            windows.metadata.loc[event[:, 1] >= 0, "workload_id"].drop_duplicates()
        )
        if len(elevated_groups) != 3:
            raise RuntimeError(
                f"expected three elevated-event development groups: {elevated_groups}"
            )
        negative_groups = [group for group in development_groups if group not in elevated_groups]
        folds = [[elevated_groups[i], *negative_groups[i::3]] for i in range(3)]

    results = []
    logs = []
    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = (
        ("balanced_augmented",)
        if experimental
        else ("ordinal_only", "ordinal_aux_weighted")
    )
    for variant in variants:
        improved = variant in ("ordinal_aux_weighted", "balanced_augmented")
        for fold, validation_groups in enumerate(folds, start=1):
            validation = windows.metadata["workload_id"].isin(validation_groups).to_numpy()
            training = ~validation
            scaler = fit_training_standardizer(
                windows.values[training],
                feature_columns=features,
                partitions=["training"] * int(training.sum()),
            )
            tensors = (
                torch.from_numpy(scaler.transform(windows.values)),
                torch.from_numpy(event),
                torch.from_numpy(observed),
                torch.from_numpy(current_temp.astype(np.float32)),
                torch.from_numpy(aux),
                torch.from_numpy(aux_mask),
            )
            train_tensors = tuple(value[training] for value in tensors)
            real_training_samples = len(train_tensors[0])
            if args.merge_synthetic:
                synthetic = generate_synthetic_events(
                    args.synthetic_samples_per_fold, seed=SEED + 100 + fold
                )
                synthetic_x = torch.from_numpy(
                    scaler.transform(synthetic["history"])
                )
                synthetic_event = torch.from_numpy(synthetic["event_interval"])
                synthetic_observed = torch.from_numpy(synthetic["observed_intervals"])
                synthetic_current = torch.from_numpy(
                    synthetic["current_temperature_c"]
                )
                synthetic_aux = torch.from_numpy(
                    synthetic["heating_rate_c_per_s"][:, None]
                    * np.asarray(HORIZONS_S, dtype=np.float32)[None, :]
                )
                synthetic_aux_mask = torch.ones_like(synthetic_aux, dtype=torch.bool)
                synthetic_tensors = (
                    synthetic_x,
                    synthetic_event,
                    synthetic_observed,
                    synthetic_current,
                    synthetic_aux,
                    synthetic_aux_mask,
                )
                train_tensors = tuple(
                    torch.cat((real_value, synthetic_value), dim=0)
                    for real_value, synthetic_value in zip(
                        train_tensors, synthetic_tensors, strict=True
                    )
                )
            generator = torch.Generator().manual_seed(SEED + fold)
            sampler = None
            if experimental:
                real_positive = (event[training] >= 0).any(axis=1)
                if args.merge_synthetic:
                    sample_weights = np.concatenate(
                        (
                            np.where(real_positive, 20.0, 1.0),
                            np.full(args.synthetic_samples_per_fold, 0.5),
                        )
                    )
                else:
                    sample_weights = np.where(real_positive, 30.0, 1.0)
                sampler = WeightedRandomSampler(
                    torch.from_numpy(sample_weights),
                    num_samples=len(sample_weights),
                    replacement=True,
                    generator=generator,
                )
            loader = DataLoader(
                TensorDataset(*train_tensors),
                batch_size=args.batch_size,
                shuffle=sampler is None,
                sampler=sampler,
                generator=generator if sampler is None else None,
            )
            torch.manual_seed(SEED + fold)
            model_class = (
                PhysicsGuidedNestedRaiseFPGA
                if args.physics_guided
                else NestedSeverityRaiseFPGA
            )
            model = model_class(channels=len(features))
            if args.pretrained_checkpoint is not None:
                pretrained = torch.load(
                    args.pretrained_checkpoint, map_location="cpu", weights_only=False
                )
                model.load_state_dict(pretrained["model_state_dict"])
            optimizer = torch.optim.Adam(
                model.parameters(), lr=5e-4 if args.pretrained_checkpoint else 1e-3
            )
            best_loss = float("inf")
            best_state = None
            for _epoch in range(args.epochs):
                model.train()
                for batch in loader:
                    optimizer.zero_grad(set_to_none=True)
                    batch_x = batch[0]
                    if experimental:
                        batch_x = batch_x + 0.03 * torch.randn_like(batch_x)
                    loss = training_loss(model(batch_x), *batch[1:], improved)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.inference_mode():
                    validation_tensors = tuple(value[validation] for value in tensors)
                    validation_loss = float(
                        training_loss(
                            model(validation_tensors[0]),
                            *validation_tensors[1:],
                            False,
                        )
                    )
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_state = {
                        key: value.detach().clone()
                        for key, value in model.state_dict().items()
                    }
            assert best_state is not None
            model.load_state_dict(best_state)
            model.eval()
            with torch.inference_mode():
                all_risks = model(tensors[0])["cumulative_risk"].numpy()
            for severity_index, severity in enumerate(SEVERITIES):
                selector = (
                    select_recall_constrained_thresholds
                    if experimental and args.cutoff_objective == "recall"
                    else select_thresholds
                )
                train_cutoffs = selector(
                    all_risks[training, severity_index],
                    event[training, severity_index],
                    observed[training, severity_index],
                )
                rows = metric_rows(
                    fold,
                    all_risks[validation, severity_index],
                    event[validation, severity_index],
                    observed[validation, severity_index, None] > np.arange(3),
                    validation_groups,
                    train_cutoffs,
                )
                for row in rows:
                    row["variant"] = variant
                    row["severity"] = severity
                    row["severity_threshold_c"] = THRESHOLDS_C[severity_index]
                results.extend(rows)
            logs.append(
                {
                    "variant": variant,
                    "fold": fold,
                    "validation_groups": validation_groups,
                    "best_validation_unweighted_loss": best_loss,
                    "real_training_samples": real_training_samples,
                    "synthetic_training_samples": (
                        args.synthetic_samples_per_fold if args.merge_synthetic else 0
                    ),
                }
            )
            if improved:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "features": features,
                        "scaler_mean": scaler.mean,
                        "scaler_scale": scaler.scale,
                        "severity_names": SEVERITIES,
                        "thresholds_c": THRESHOLDS_C,
                        "event_definition_version": (
                            "posthoc-temperature-transitions-v1"
                            if experimental
                            else "nested-pl-temperature-zones-v1"
                        ),
                        "validation_groups": validation_groups,
                        "seed": SEED + fold,
                        "status": "EXPLORATORY_PILOT_NOT_CALIBRATED",
                        "model_class": model_class.__name__,
                    },
                    output_dir / f"{output_prefix}_fold{fold}.pt",
                )

    frame = pd.DataFrame(results)
    frame.to_csv(f"results/{output_prefix}_cv.csv", index=False)
    summary = (
        frame.groupby(["variant", "severity", "horizon_s"], sort=False)[
            ["pr_auc", "auroc", "brier", "positive_precision", "positive_recall_sensitivity", "positive_f1", "selected_balanced_accuracy", "selected_mcc"]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(f"results/{output_prefix}_summary.csv", index=False)
    metadata = {
        "status": "EXPLORATORY_PILOT_NOT_CALIBRATED",
        "seed": SEED,
        "features": features,
        "parameters": sum(
            p.numel()
            for p in (
                PhysicsGuidedNestedRaiseFPGA(len(features))
                if args.physics_guided
                else NestedSeverityRaiseFPGA(len(features))
            ).parameters()
        ),
        "development_episode_counts": {name: len(table) for name, table in episodes.items()},
        "variants": {
            "ordinal_only": "nested ordinal hazards; unweighted likelihood",
            "ordinal_aux_weighted": "nested ordinal hazards; event/hard-negative weighting; future-temperature auxiliary head",
            "balanced_augmented": "post-hoc lower transition zones; positive-window sampling; Gaussian telemetry augmentation; recall-constrained training-only cutoffs",
        },
        "recall_cutoff_target": 0.90 if experimental else None,
        "pretrained_checkpoint": (
            str(args.pretrained_checkpoint) if args.pretrained_checkpoint else None
        ),
        "merged_synthetic": args.merge_synthetic,
        "synthetic_samples_per_fold": (
            args.synthetic_samples_per_fold if args.merge_synthetic else 0
        ),
        "physics_guided": args.physics_guided,
        "cutoff_objective": args.cutoff_objective,
        "folds": logs,
    }
    Path(f"results/{output_prefix}_training.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
