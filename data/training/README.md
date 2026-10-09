# Combined telemetry used for October fine-tuning

combined_telemetry.parquet contains 862,551 rows and 13 columns: seven telemetry
inputs plus run_id, workload_id, rep_id, timestamp_s and partition. It is the
existing combined input table, not a newly generated or synthetic dataset.

| Partition | Rows | Sessions |
|---|---:|---:|
| old_training | 223,133 | 21 |
| new_board_training | 679 | 2 |
| old_development_validation | 150,018 | 15 |
| old_untouched_test | 478,630 | 15 |
| new_board_test | 10,091 | 1 |

Gradient updates use only old_training and new_board_training. Epoch selection
and decision cutoff selection use old_development_validation. The two test
partitions are evaluated afterward. Do not randomly shuffle individual rows
between partitions. Preserve session identity and chronological order.

## Attribution and publication scope

The older recordings are derived from Marcus Fredriksson, **ZCU104 DPU Inference
Benchmark Dataset**, version 1.0.0, 2026, DOI 10.5281/zenodo.19763074.
[Original record](https://zenodo.org/records/19763074).
[Publisher license metadata](https://zenodo.org/api/records/19763074).
The original data is licensed [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

Changes to that source: selected the seven model input channels and relevant
development/test sessions; added run, repetition and partition fields; combined
them with the team's three October board recordings. No source application
classification labels, full sensor inventory, or original 345 MB archive is included.
This extract does not represent all 76 source sensors or all source partitions.

The new team recordings were extracted into the same seven-channel schema and
published at the user's explicit request. The source-data CC-BY license applies
to the older source-derived portion; no blanket license for the team recordings,
code or checkpoints is inferred here. No credentials or raw Excel files are included.

The new held-out recording has no positive 49/50/51C transition examples. Its
negative-only accuracy must not be represented as validated hazard detection.
