#!/usr/bin/env python3
"""Grouped development ablation for the trajectory-conditioned risk model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from dataset.build_windows import build_windows, fit_training_standardizer
from dataset.causal_features import make_future_temperature_targets
from model.losses import discrete_hazard_nll
from model.raise_fpga import PhysicsGuidedNestedRaiseFPGA, TrajectoryConditionedRaiseFPGA
from scripts.train_nested_temperature import (
    HORIZONS_S,
    label_arrays,
    select_thresholds,
)
from scripts.train_temperature_pilot import load_development, metric_rows


SEED = 20260915
SEVERITIES = ("transition_49c", "transition_50c", "transition_51c")
THRESHOLDS_C = (49.0, 50.0, 51.0)


def episode_tables(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Import after setting the exploratory transition constants because the old
    # experiment module keeps them as module globals.
    import scripts.train_nested_temperature as nested

    nested.SEVERITIES = SEVERITIES
    nested.THRESHOLDS_C = THRESHOLDS_C
    return nested.episode_tables(raw)


def make_dense_deltas(raw, metadata, current_temp):
    horizons = np.arange(0.1, 5.0 + 1e-9, 0.1)
    frame = make_future_temperature_targets(
        raw,
        metadata,
        horizons_s=horizons,
        max_staleness_s=0.2,
        max_gap_s=0.5,
    )
    target = np.zeros((len(frame), len(horizons)), dtype=np.float32)
    mask = np.zeros_like(target, dtype=bool)
    for index, horizon in enumerate(horizons):
        tag = f"{horizon:g}s".replace(".", "_")
        absolute = frame[f"target_temp_pl_C__{tag}"].to_numpy(float)
        valid = frame[f"target_observed__{tag}"].to_numpy(bool)
        target[:, index] = np.where(valid, absolute - current_temp, 0.0)
        mask[:, index] = valid
    return target, mask


def make_folds(groups, episodes):
    counts = (
        episodes[SEVERITIES[-1]]
        .assign(workload_group=lambda x: x.run_id.str.split("/").str[0])
        .workload_group.value_counts()
    )
    folds = [[], [], []]
    totals = [0, 0, 0]
    for group, count in counts.items():
        target = int(np.argmin(totals))
        folds[target].append(group)
        totals[target] += int(count)
    for index, group in enumerate(group for group in groups if group not in counts):
        folds[index % 3].append(group)
    return folds


def joint_loss(
    output,
    event,
    observed,
    dense_target,
    dense_mask,
    inverse_sampling_weight,
    current_temp,
    *,
    dense_weight,
    consistency_weight,
    focal_gamma=0.0,
):
    per_example = output["hazard_logits"].new_zeros(len(event))
    terms = output["hazard_logits"].new_zeros(len(event))
    for severity in range(3):
        valid = observed[:, severity] > 0
        if torch.any(valid):
            severity_loss = discrete_hazard_nll(
                output["hazard_logits"][valid, severity],
                event[valid, severity],
                observed[valid, severity],
                reduction="none",
            )
            if focal_gamma:
                severity_loss = (
                    1.0 - torch.exp(-severity_loss).clamp(0.0, 1.0)
                ).pow(focal_gamma) * severity_loss
            per_example[valid] += severity_loss
            terms[valid] += 1
    valid_any = terms > 0
    hazard = (
        (per_example[valid_any] / terms[valid_any])
        * inverse_sampling_weight[valid_any]
    ).sum() / inverse_sampling_weight[valid_any].sum().clamp_min(1e-8)
    total = hazard

    if dense_weight and "temperature_trajectory_delta" in output:
        if torch.any(dense_mask):
            dense = F.smooth_l1_loss(
                output["temperature_trajectory_delta"][dense_mask],
                dense_target[dense_mask],
            )
            total = total + dense_weight * dense
        if consistency_weight:
            # A differentiable future-maximum proxy makes the risk head agree
            # weakly with its own temperature trajectory, without replacing the
            # censoring-aware onset likelihood.
            trajectory_temp = current_temp[:, None] + output[
                "temperature_trajectory_delta"
            ]
            horizon_ends = (5, 20, 50)
            prior_rows = []
            for threshold in THRESHOLDS_C:
                values = []
                for horizon, end in enumerate(horizon_ends):
                    smooth_max = torch.logsumexp(
                        trajectory_temp[:, :end] * 4.0, dim=1
                    ) / 4.0
                    scale = output["temperature_trajectory_scale"][:, horizon]
                    values.append(torch.sigmoid((smooth_max - threshold) / scale))
                prior_rows.append(torch.stack(values, dim=1))
            prior = torch.stack(prior_rows, dim=1).detach()
            consistency = F.binary_cross_entropy(
                output["cumulative_risk"].clamp(1e-5, 1 - 1e-5),
                prior.clamp(1e-5, 1 - 1e-5),
            )
            total = total + consistency_weight * consistency
    elif dense_weight:
        endpoint = dense_target[:, (4, 19, 49)]
        endpoint_mask = dense_mask[:, (4, 19, 49)]
        if torch.any(endpoint_mask):
            total = total + dense_weight * F.smooth_l1_loss(
                output["future_temperature_delta"][endpoint_mask],
                endpoint[endpoint_mask],
            )
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/extracted/results"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--variants", nargs="*", default=["physics", "tcn", "tcn_dense"])
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional suffix for checkpoint and metric filenames; preserves earlier runs.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    features = [
        row["column"]
        for row in yaml.safe_load(Path("configs/features.yaml").read_text())["features"]
    ]
    raw, groups = load_development(
        args.root, Path("results/split_manifest.csv"), features
    )
    windows = build_windows(raw, feature_columns=features)
    episodes = episode_tables(raw)
    event, observed, _eligible = label_arrays(windows.metadata, episodes)
    current_temp = windows.values[:, -1, features.index("temp_pl_temp_C")]
    dense_target, dense_mask = make_dense_deltas(raw, windows.metadata, current_temp)
    folds = make_folds(groups, episodes)

    variant_spec = {
        "physics": (PhysicsGuidedNestedRaiseFPGA, 0.2, 0.0, 0.0),
        "tcn": (TrajectoryConditionedRaiseFPGA, 0.0, 0.0, 0.0),
        "tcn_dense": (TrajectoryConditionedRaiseFPGA, 0.15, 0.03, 0.0),
        "tcn_dense_focal": (TrajectoryConditionedRaiseFPGA, 0.15, 0.03, 2.0),
    }
    unknown = set(args.variants).difference(variant_spec)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")

    rows = []
    run_log = []
    run_name = (
        f"trajectory_conditioned_{args.run_tag}"
        if args.run_tag
        else "trajectory_conditioned"
    )
    checkpoint_dir = Path("outputs/models")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        model_class, dense_weight, consistency_weight, focal_gamma = variant_spec[variant]
        for fold, validation_groups in enumerate(folds, 1):
            validation = windows.metadata.workload_id.isin(validation_groups).to_numpy()
            training = ~validation
            scaler = fit_training_standardizer(
                windows.values[training],
                feature_columns=features,
                partitions=["training"] * int(training.sum()),
            )
            x = torch.from_numpy(scaler.transform(windows.values))
            tensors = (
                x,
                torch.from_numpy(event),
                torch.from_numpy(observed),
                torch.from_numpy(dense_target),
                torch.from_numpy(dense_mask),
                torch.from_numpy(current_temp.astype(np.float32)),
            )

            event_window = (event[training] >= 0).any(axis=1)
            hard_negative = (~event_window) & (current_temp[training] >= 47.5)
            sampling_weight = np.where(event_window, 20.0, np.where(hard_negative, 4.0, 1.0))
            inverse_weight = (1.0 / sampling_weight).astype(np.float32)
            train_data = tuple(value[training] for value in tensors) + (
                torch.from_numpy(inverse_weight),
            )
            sampler = WeightedRandomSampler(
                torch.from_numpy(sampling_weight),
                num_samples=len(sampling_weight),
                replacement=True,
                generator=torch.Generator().manual_seed(SEED + fold),
            )
            loader = DataLoader(
                TensorDataset(*train_data),
                batch_size=args.batch_size,
                sampler=sampler,
            )

            torch.manual_seed(SEED + 100 * list(variant_spec).index(variant) + fold)
            model = model_class(channels=len(features))
            optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
            best_loss = float("inf")
            best_state = None
            patience = 0
            for epoch in range(args.epochs):
                model.train()
                for batch in loader:
                    optimizer.zero_grad(set_to_none=True)
                    output = model(batch[0] + 0.01 * torch.randn_like(batch[0]))
                    loss = joint_loss(
                        output,
                        batch[1],
                        batch[2],
                        batch[3],
                        batch[4],
                        batch[6],
                        batch[5],
                        dense_weight=dense_weight,
                        consistency_weight=consistency_weight,
                        focal_gamma=focal_gamma,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                model.eval()
                with torch.inference_mode():
                    val = tuple(value[validation] for value in tensors)
                    ones = torch.ones(len(val[0]))
                    val_loss = float(
                        joint_loss(
                            model(val[0]),
                            val[1], val[2], val[3], val[4], ones, val[5],
                            dense_weight=dense_weight,
                            consistency_weight=consistency_weight,
                            focal_gamma=focal_gamma,
                        )
                    )
                if val_loss < best_loss - 1e-5:
                    best_loss = val_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= 5 and not args.smoke:
                        break
            assert best_state is not None
            model.load_state_dict(best_state)
            model.eval()
            with torch.inference_mode():
                risk = model(x)["cumulative_risk"].numpy()
            for severity_index, severity in enumerate(SEVERITIES):
                cutoffs = select_thresholds(
                    risk[training, severity_index],
                    event[training, severity_index],
                    observed[training, severity_index],
                )
                metrics = metric_rows(
                    fold,
                    risk[validation, severity_index],
                    event[validation, severity_index],
                    observed[validation, severity_index, None] > np.arange(3),
                    validation_groups,
                    cutoffs,
                )
                for row in metrics:
                    row.update(
                        variant=variant,
                        severity=severity,
                        severity_threshold_c=THRESHOLDS_C[severity_index],
                    )
                rows.extend(metrics)
            checkpoint = checkpoint_dir / f"{run_name}_{variant}_fold{fold}.pt"
            torch.save(
                {
                    "model_state_dict": best_state,
                    "model_class": model_class.__name__,
                    "features": features,
                    "scaler_mean": scaler.mean,
                    "scaler_scale": scaler.scale,
                    "validation_groups": validation_groups,
                    "status": "DEVELOPMENT_CV_EXPLORATORY_NOT_FINAL_TEST",
                },
                checkpoint,
            )
            run_log.append(
                {
                    "variant": variant,
                    "fold": fold,
                    "epochs_completed": epoch + 1,
                    "best_validation_loss": best_loss,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "checkpoint": str(checkpoint),
                }
            )

    output = pd.DataFrame(rows)
    output.to_csv(f"results/{run_name}_cv.csv", index=False)
    summary = (
        output.groupby(["variant", "severity", "horizon_s"], sort=False)[
            [
                "pr_auc",
                "auroc",
                "brier",
                "positive_precision",
                "positive_recall_sensitivity",
                "positive_f1",
                "selected_balanced_accuracy",
                "selected_mcc",
            ]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(f"results/{run_name}_summary.csv", index=False)
    Path(f"results/{run_name}_training.json").write_text(
        json.dumps(
            {
                "status": "DEVELOPMENT_CV_EXPLORATORY_NOT_FINAL_TEST",
                "seed": SEED,
                "variants": args.variants,
                "dense_target_valid_fraction": dense_mask.mean(axis=0).tolist(),
                "runs": run_log,
            },
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
