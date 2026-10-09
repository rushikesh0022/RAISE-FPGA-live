"""Fine-tune the existing physics CNN+GRU on old data plus new board sessions.

The long new recording and original untouched-test workloads are never used
for gradient updates, epoch selection, scaling or probability-cutoff selection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from dataset.build_windows import build_windows
from model.raise_fpga import PhysicsGuidedNestedRaiseFPGA
from scripts.train_nested_temperature import label_arrays
from scripts.train_trajectory_conditioned import (
    HORIZONS_S, SEVERITIES, THRESHOLDS_C, episode_tables,
    joint_loss, make_dense_deltas,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261008


def infer(model, x):
    risks, delta = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), 2048):
            out = model(x[start:start + 2048])
            risks.append(out['cumulative_risk'].cpu().numpy())
            delta.append(out['future_temperature_delta'].cpu().numpy())
    return np.concatenate(risks), np.concatenate(delta)


def metrics(risk, delta, event, observed, target, mask, cutoffs, split, balanced=False):
    rows = []
    rng = np.random.default_rng(SEED)
    for severity, threshold in enumerate(THRESHOLDS_C):
        for h, horizon in enumerate(HORIZONS_S):
            y = (event[:, severity] >= 0) & (event[:, severity] <= h)
            valid = (observed[:, severity] > h) | y
            idx = np.flatnonzero(valid)
            if balanced:
                pos, neg = idx[y[idx]], idx[~y[idx]]
                n = min(len(pos), len(neg))
                if n == 0:
                    continue
                idx = np.r_[rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)]
            truth = y[idx]
            prediction = risk[idx, severity, h] >= cutoffs[severity, h]
            tp = int((prediction & truth).sum())
            fp = int((prediction & ~truth).sum())
            tn = int((~prediction & ~truth).sum())
            fn = int((~prediction & truth).sum())
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            specificity = tn / (tn + fp) if tn + fp else None
            f1 = 2 * tp / (2 * tp + fp + fn) if tp + fn else None
            future_valid = mask[:, (4, 19, 49)[h]]
            error = delta[future_valid, h] - target[future_valid, (4, 19, 49)[h]]
            persistence_error = target[future_valid, (4, 19, 49)[h]]
            rows.append({
                'split': split, 'balanced_case_control': balanced,
                'threshold_c': threshold, 'horizon_s': horizon,
                'cutoff': float(cutoffs[severity, h]),
                'samples': len(idx), 'positive_samples': int(truth.sum()),
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'accuracy': (tp + tn) / len(idx) if len(idx) else None,
                'precision': precision, 'recall': recall, 'f1': f1,
                'specificity': specificity,
                'balanced_accuracy': (recall + specificity) / 2 if recall is not None and specificity is not None else None,
                'temperature_mae_c': float(np.abs(error).mean()) if len(error) else None,
                'temperature_rmse_c': float(np.sqrt(np.mean(error ** 2))) if len(error) else None,
                'persistence_temperature_mae_c': float(np.abs(persistence_error).mean()) if len(error) else None,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--samples-per-epoch', type=int, default=24000)
    parser.add_argument('--check-data', action='store_true', help='Verify published data, splits and parent checkpoint without training')
    parser.add_argument('--smoke-test', action='store_true', help='Short synthetic-size subset check; writes only to smoke output paths, not a real training result')
    args = parser.parse_args()
    if args.epochs < 0 or args.samples_per_epoch < 1:
        parser.error('Epochs must be nonnegative and samples per epoch positive')
    result_dir = ROOT / ('results_training/smoke' if args.smoke_test else 'results_training/full')
    model_dir = ROOT / ('outputs/retrained/smoke' if args.smoke_test else 'outputs/retrained/full')
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    manifest = json.loads((ROOT / 'data/training/manifest.json').read_text())
    for name in ('dataset', 'parent_checkpoint'):
        spec = manifest[name]
        digest = hashlib.sha256()
        with (ROOT / spec['path']).open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != spec['sha256']:
            raise ValueError('Published input checksum differs: ' + spec['path'])
    features = [item['column'] for item in yaml.safe_load((ROOT / 'configs/features.yaml').read_text())['features']]
    source_path = ROOT / 'outputs/models/physics_guided_merged_synthetic_real_transition_temperature_fold1.pt'
    checkpoint = torch.load(source_path, map_location='cpu', weights_only=True)
    checkpoint['scaler_mean'] = np.asarray(checkpoint['scaler_mean'])
    checkpoint['scaler_scale'] = np.asarray(checkpoint['scaler_scale'])
    if checkpoint['features'] != features:
        raise ValueError('Source checkpoint feature order differs')
    data_path = ROOT / 'data/training/combined_telemetry.parquet'
    raw = pd.read_parquet(data_path)
    expected_parts = {'old_training', 'new_board_training', 'old_development_validation', 'old_untouched_test', 'new_board_test'}
    if set(raw.partition.unique()) != expected_parts:
        raise ValueError('Published dataset partition labels differ from the recorded experiment')
    if raw.groupby('run_id').partition.nunique().max() != 1:
        raise ValueError('A run crosses partitions')
    if not np.isfinite(raw[['timestamp_s', *features]].to_numpy(float)).all():
        raise ValueError('Required telemetry contains missing or nonfinite readings')
    if args.check_data:
        check_model = PhysicsGuidedNestedRaiseFPGA(channels=len(features))
        check_model.load_state_dict(checkpoint['model_state_dict'])
        print(raw.groupby('partition').agg(rows=('run_id', 'size'), runs=('run_id', 'nunique')).to_string())
        print('Data schema, finite values, session splits and parent model load verified. No training performed.')
        return
    if args.smoke_test:
        # Keep two new train sessions and one run from every other partition;
        # only the first 12s per selected run. Never report smoke metrics as test results.
        selected = set(raw.loc[raw.partition.eq('new_board_training'), 'run_id'].unique())
        for partition in expected_parts - {'new_board_training'}:
            selected.add(raw.loc[raw.partition.eq(partition), 'run_id'].iloc[0])
        raw = raw.loc[raw.run_id.isin(selected)].copy()
        starts = raw.groupby('run_id').timestamp_s.transform('min')
        raw = raw.loc[raw.timestamp_s <= starts + 12].copy()
        args.epochs, args.samples_per_epoch = min(args.epochs, 1), min(args.samples_per_epoch, 64)
    split_by_run = raw.groupby('run_id').partition.first()
    split_by_run.rename('partition').reset_index().to_csv(result_dir / 'split_manifest.csv', index=False)
    windows = build_windows(raw, feature_columns=features)
    episodes = episode_tables(raw)
    event, observed, eligible = label_arrays(windows.metadata, episodes)
    observed[~eligible] = 0
    event[~eligible] = -1
    current = windows.values[:, -1, features.index('temp_pl_temp_C')]
    dense_target, dense_mask = make_dense_deltas(raw, windows.metadata, current)
    parts = windows.metadata.run_id.map(split_by_run).to_numpy()
    training = np.isin(parts, ['old_training', 'new_board_training'])
    validation = parts == 'old_development_validation'
    x = torch.from_numpy(((windows.values - checkpoint['scaler_mean']) / checkpoint['scaler_scale']).astype(np.float32))
    tensors = (x, torch.from_numpy(event), torch.from_numpy(observed),
               torch.from_numpy(dense_target), torch.from_numpy(dense_mask), torch.from_numpy(current))
    del windows.values
    positive = (event[training] >= 0).any(1)
    boost = np.where(positive, 20., np.where(current[training] >= 47.5, 4., 1.))
    new_train = parts[training] == 'new_board_training'
    if not new_train.any() or (not positive.any() and not args.smoke_test):
        raise ValueError('Combined training requires new windows and original transition positives')
    domain_weight = np.where(new_train, .25 / new_train.sum(), .75 / (~new_train).sum())
    sampler = WeightedRandomSampler(torch.from_numpy(domain_weight * boost), args.samples_per_epoch,
                                   replacement=True, generator=torch.Generator().manual_seed(SEED))
    loader = DataLoader(TensorDataset(*(t[training] for t in tensors), torch.from_numpy((1 / boost).astype(np.float32))),
                        batch_size=1024, sampler=sampler)
    model = PhysicsGuidedNestedRaiseFPGA(channels=len(features))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    val_idx = np.flatnonzero(validation)
    if not len(val_idx) or not training.any():
        raise ValueError('Selected input has no eligible training or validation windows')
    if len(val_idx) > 15000:
        val_idx = rng.choice(val_idx, 15000, replace=False)
    best_loss, best_state, best_epoch = float('inf'), None, 0
    history = []
    # Include epoch zero so adaptation is retained only if validation loss improves.
    for epoch in range(args.epochs + 1):
        losses = []
        if epoch:
            model.train()
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = joint_loss(model(batch[0]), batch[1], batch[2], batch[3], batch[4], batch[6], batch[5],
                                  dense_weight=.2, consistency_weight=0.)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            val = tuple(t[val_idx] for t in tensors)
            vloss = float(joint_loss(model(val[0]), val[1], val[2], val[3], val[4], torch.ones(len(val_idx)), val[5],
                                     dense_weight=.2, consistency_weight=0.))
        if vloss < best_loss:
            best_loss, best_epoch = vloss, epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append({'epoch': epoch, 'training_loss': float(np.mean(losses)) if losses else None, 'validation_loss': vloss})
        print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(best_state)
    risk, delta = infer(model, x[validation])
    cutoffs = np.full((3, 3), .5)
    for s in range(3):
        for h in range(3):
            y = (event[validation, s] >= 0) & (event[validation, s] <= h)
            known = (observed[validation, s] > h) | y
            if len(np.unique(y[known])) == 2:
                p, r, thresholds = precision_recall_curve(y[known], risk[known, s, h])
                f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
                cutoffs[s, h] = thresholds[int(np.argmax(f1))]
    output_path = model_dir / 'combined_board_physics_cnn_gru.pt'
    saved_payload = {**checkpoint, 'model_state_dict': best_state, 'probability_cutoffs': cutoffs.tolist(),
                'scaler_mean': checkpoint['scaler_mean'].tolist(), 'scaler_scale': checkpoint['scaler_scale'].tolist(),
                'model_class': 'PhysicsGuidedNestedRaiseFPGA', 'parent_checkpoint': source_path.name,
                'training_groups': sorted(raw.loc[raw.partition.isin(['old_training', 'new_board_training']), 'run_id'].unique()),
                'test_groups': sorted(raw.loc[raw.partition.isin(['old_untouched_test', 'new_board_test']), 'run_id'].unique()),
                'best_epoch': best_epoch, 'status': 'SMOKE_TEST_NOT_A_BENCHMARK' if args.smoke_test else 'COMBINED_BOARD_PILOT_WITH_SESSION_HELD_OUT_TEST',
                'cutoff_selection': 'old development validation only; fixed before test evaluation',
                'seed': SEED}
    torch.save(saved_payload, output_path)
    all_rows = []
    for split in ['old_development_validation', 'old_untouched_test', 'new_board_test']:
        pick = parts == split
        if not pick.any():
            if not args.smoke_test:
                raise ValueError('Full dataset has no eligible windows for ' + split)
            print('Smoke subset has no eligible windows for ' + split + '; evaluation skipped.', flush=True)
            continue
        risks, deltas = infer(model, x[pick])
        for balanced in ([False, True] if split != 'new_board_test' else [False]):
            all_rows.extend(metrics(risks, deltas, event[pick], observed[pick], dense_target[pick], dense_mask[pick], cutoffs, split, balanced))
    pd.DataFrame(all_rows).to_csv(result_dir / 'metrics.csv', index=False)
    summary = []
    for split in ['old_untouched_test', 'new_board_test']:
        for balanced in ([False, True] if split != 'new_board_test' else [False]):
            rows = [r for r in all_rows if r['split'] == split and r['balanced_case_control'] == balanced]
            if not rows:
                summary.append({'split': split, 'balanced_case_control': balanced,
                                'status': 'no_eligible_positive_negative_pairs'})
                continue
            summary.append({'split': split, 'balanced_case_control': balanced,
                            'mean_accuracy': float(np.mean([r['accuracy'] for r in rows])),
                            'minimum_accuracy': min(r['accuracy'] for r in rows),
                            'mean_f1': float(np.mean([r['f1'] for r in rows if r['f1'] is not None])) if any(r['f1'] is not None for r in rows) else None,
                            'mean_temperature_mae_c': float(np.mean([r['temperature_mae_c'] for r in rows])),
                            'mean_persistence_temperature_mae_c': float(np.mean([r['persistence_temperature_mae_c'] for r in rows])),
                            'known_positive_windows_per_output': [r['positive_samples'] for r in rows]})
    result = {'checkpoint': str(output_path.relative_to(ROOT)), 'smoke_test': args.smoke_test,
              'best_epoch': best_epoch, 'history': history,
              'window_counts': {str(p): int((parts == p).sum()) for p in np.unique(parts)},
              'summary': summary, 'features': features,
              'limitations': ['New board test contains zero 49/50/51C transition positives; its accuracy measures false alarms only.',
                              'Old untouched-test workloads were held out from this retraining; they have been reported in earlier project evaluations.',
                              'Earlier 88.47% number used post-hoc development cutoffs; it is not directly comparable to this held-out test.',
                              'Sensor acquisition is sequential; INA226 power-channel equivalence to old source is provisional.']}
    (result_dir / 'training.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2), flush=True)
    # Reload the saved checkpoint and confirm inference and horizon/severity ordering.
    saved = torch.load(output_path, map_location='cpu', weights_only=True)
    check = PhysicsGuidedNestedRaiseFPGA(channels=7)
    check.load_state_dict(saved['model_state_dict'])
    check.eval()
    with torch.inference_mode():
        scores = check(x[-16:])['cumulative_risk']
    assert torch.isfinite(scores).all()
    assert (scores[:, :, 1:] >= scores[:, :, :-1] - 1e-6).all()
    assert (scores[:, 1:] <= scores[:, :-1] + 1e-6).all()
    print('Saved checkpoint reload and risk ordering verified.', flush=True)


if __name__ == '__main__':
    main()
