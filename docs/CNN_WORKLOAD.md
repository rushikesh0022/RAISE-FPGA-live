# Run the CNN workload and check our temperature predictions

There are two separate neural networks:

| Network | Purpose | Execution |
|---|---|---|
| Tiny CNN workload | Performs repeated convolution computations to create activity | ZCU104 ARM CPU |
| Trained CNN+GRU .pt model | Predicts future PL temperature and transition risk from telemetry | Windows PC |

The workload is **not FPGA-fabric/DPU inference**. It has fixed, untrained weights
and synthetic images. Its three numerical outputs have no class meaning.
Your pasted original script used SHA-256 hashing on CPU workers, not a CNN.

## 1. Connect the board

Follow the README's cable/SW6/PuTTY steps. Keep the existing working Linux SD
image and normal cooling. Connect board Ethernet and PC to the same router/network.
PuTTY is for the Linux console; SSH over Ethernet is required for this walkthrough.
Do not run the previous CPU-load script at the same time.

## 2. Find the board IP in PuTTY

At the board Linux prompt:

```bash
whoami
python3 --version
ip -br address
```

Your pasted prompt used root. Use whichever account whoami reports. Note the
Ethernet IPv4 address without its /24 suffix, not 127.0.0.1.
No additional Python packages are required on the board.

## 3. Clone or update on the Windows PC

Open Windows PowerShell. For a first clone:

```powershell
cd $env:USERPROFILE
git clone https://github.com/rushikesh0022/RAISE-FPGA-live.git
cd RAISE-FPGA-live
```

If already cloned into that folder, instead:

```powershell
cd "$env:USERPROFILE\RAISE-FPGA-live"
git pull --ff-only
```

## 4. Set up the PC runtime once

Skip this if your .venv already works. Run each line separately:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install numpy==1.26.4
```

Keep the .pt file on Windows; the board workload is a different network.

## 5. Check SSH

In PowerShell, substitute your actual board IP:

```powershell
Test-NetConnection 192.168.1.25 -Port 22
ssh root@192.168.1.25
```

Use your actual username instead of root when necessary. Check new host
fingerprints with the lab. Expected: TcpTestSucceeded True and a board Linux prompt.
Type exit to return to PowerShell. If SSH fails, fix networking/service access
before proceeding; do not disable security protections.

## 6. Run one command in PowerShell

Inside RAISE-FPGA-live, replacing the example account/IP:

```powershell
.\.venv\Scripts\python.exe -m scripts.start_cnn_demo --board root@192.168.1.25 --duration-s 60
```

This automatically:

1. Creates raise_fpga_live in the board account's home folder.
2. Copies the collector and CNN workload scripts there.
3. Checks the seven sensor readings. Any failed check stops setup.
4. Loads the trained predictor on the PC.
5. Starts the board's CNN/telemetry session.
6. Streams telemetry and prints PC predictions.
7. Saves recordings and workload evidence.

You may be asked for the board password several times. Do not manually launch
another collector or workload at the same time.

Default 60-second plan:

| Time from telemetry start | Stage |
|---|---|
| 0–15 seconds | Idle baseline |
| 15–45 seconds | One CPU worker computing CNN forwards, approximately 25% duty |
| 45–60 seconds | Worker stopped; cool-down observation |

Temperature may not reach 49C or rise substantially. That is a valid observation,
not a reason to disable cooling or force higher temperature.

## 7. Read the results

The console should show workload stages and predictions after a valid 3.2-second
history. Forecasts are for 0.5, 2 and 5 seconds. Risk rows correspond to 49, 50
and 51C events. They are uncalibrated risk scores, not verified safety probabilities.

On Windows, open the results/live folder in the cloned repository. It contains:

- telemetry CSV, including idle / cnn_arm_cpu / cool_down stage labels.
- predictions JSONL, including later temperature_followup records and errors.

The board account's recordings folder contains:

- The same telemetry CSV.
- Sensor metadata JSON.
- A .workload.jsonl file with backend, CNN architecture, stage times, forward
  counts, numerical outputs, latency and any abort reason.

The CNN workload architecture is:
16x16x1 input → Conv2D(4, 3x3)+ReLU → 2x2 max pool → Conv2D(8, 3x3)+ReLU →
global average pooling → dense(3). It has two convolution layers.

## Stop conditions and limitations

The workload stops permanently for that session if any of the three measured
temperatures reaches 50C, telemetry becomes invalid, or the CNN worker exits.
Collection continues without load so the stop can be inspected; the session
returns a failure code when aborted. 50C is a conservative demo operational
cutoff, not a manufacturer safety limit or a certified protection mechanism.
Sensor checks have sampling/processing delay; leave hardware protection enabled.

Ctrl+C stops the PC runner. The board workload has a bounded lifetime, but after
an interrupted SSH connection check for a remaining session before restarting.
Do not terminate unrelated board workloads. Do not leave this demo unattended.

Local tests exercised real CNN forward calculations and the complete session
with simulated sensor files. No physical ZCU104 or Windows run is claimed.
CPU-load results are not equivalent to FPGA-load results and must be reported
separately. Classification accuracy needs held-out, event-labeled evaluation.

To repeat, rerun the same one-command launcher. New files receive timestamped
names. For a longer session, replace 60 with 240 (four minutes); the phases
remain 25% idle, 50% computation, 25% cool-down.
