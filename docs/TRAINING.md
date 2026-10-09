# Train the deployed telemetry model

This reproduces the **October combined-data fine-tuning procedure**, starting
from the included earlier physics CNN+GRU checkpoint. It is not a complete
reproduction of all previous synthetic pretraining/architecture experiments.
Training runs on a PC, not on the FPGA board. The tiny CPU CNN workload is a
different, untrained network and is not trained by this script.

## 1. Get the files

In Windows PowerShell, inside your cloned RAISE-FPGA-live folder:

```powershell
git pull --ff-only
```

The clone includes the prepared 10 MB Parquet dataset. No separate download of
the original 345 MB archive and no source Excel workbooks are needed.

## 2. Install training dependencies

Use the existing Python 3.11 .venv from deployment setup. If missing, create it
first with py -3.11 -m venv .venv. Then:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
```

## 3. Check data and checkpoint without training

```powershell
.\.venv\Scripts\python.exe -m scripts.train_combined_board_20261006 --check-data
```

Expected: the partition row/session counts followed by confirmation that schema,
finite values, session splits and parent model load were verified.

## 4. Run a short installation smoke test

```powershell
.\.venv\Scripts\python.exe -m scripts.train_combined_board_20261006 --smoke-test --epochs 1
```

This uses only a short subset and at most 64 sampled windows per epoch. Its
checkpoint/metrics are **not valid benchmark results**. Smoke outputs are saved
only under outputs/retrained/smoke and results_training/smoke.

## 5. Run full fine-tuning

```powershell
.\.venv\Scripts\python.exe -m scripts.train_combined_board_20261006 --epochs 12 --samples-per-epoch 24000
```

This builds all eligible causal windows, performs fine-tuning, selects an epoch
and cutoffs on validation data, evaluates test data and reloads the saved model.
Allow substantial time and several GB of free RAM; speed depends on the PC.
The seed is 20261008. Floating-point differences across library/platform versions
can change scores, so byte-identical results are not guaranteed.

Outputs:

- outputs/retrained/full/combined_board_physics_cnn_gru.pt
- results_training/full/training.json
- results_training/full/metrics.csv
- results_training/full/split_manifest.csv

The deployed checkpoint under outputs/models is **not overwritten**. Re-running
the training command overwrites that command's generated training outputs;
archive them before another experiment if needed.

## Methods and boundaries

The fine-tuning architecture, causal windows, event labels, losses, optimizer,
domain weights, positive oversampling and validation cutoff selection match the
original script. The publication adaptation loads the already assembled input
table instead of requiring the old archive folder and Excel-conversion stage.
It also uses separate output paths and safely serializable checkpoint metadata.

The included parent checkpoint has identical weights and scaler values to the
local source. It retains the parent model's training-only scaler rather than
refitting scaling on combined validation/test data. The parent contains earlier
training history in its learned weights; this is fine-tuning, not random initialization.

Helper modules train_nested_temperature, train_temperature_pilot and
train_trajectory_conditioned are included because the final fine-tuning script
imports their labeling/loss functions. Their historical standalone experiment
commands may need additional original artifacts not included in this package.
Use the combined training entry point above.

Reference metrics and split assignments from the original run are preserved in
docs/training_reference. They are historical results, not results from a new run.
Both test sets have been examined previously, so repeated model development
against those reported results would compromise claims of an untouched test.
The new held-out session has no positive thermal transition events. Compare
class-sensitive metrics and persistence/always-negative baselines, not raw
accuracy alone. Source attribution and dataset scope are in data/training/README.md.
