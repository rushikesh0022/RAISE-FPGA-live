# RAISE-FPGA: Windows PC + ZCU104 deployment

Follow the steps in order. Do not proceed past a failed check.

**What this deployment does:** the ZCU104 sends seven sensor readings to your Windows PC. The PC loads the included trained .pt file, makes predictions, and saves telemetry and prediction logs. The model runs on the PC CPU, not in FPGA fabric.

You will use two windows:
- **Windows PowerShell:** Git, Python setup, copying files and running predictions.
- **PuTTY:** commands on the ZCU104's Linux system.

## Step 1 — Open Windows PowerShell and check prerequisites

On the Windows PC, open Start, search PowerShell and open it. Run each line separately:

```powershell
git --version
py -3.11 --version
ssh -V
```

Expected: Git version, Python 3.11 version and OpenSSH version.
If any command is missing, install that prerequisite from its official source before continuing. You also need PuTTY for the board console.

## Step 2 — Clone the project onto Windows

In the same PowerShell window:

```powershell
cd $env:USERPROFILE
git clone https://github.com/rushikesh0022/RAISE-FPGA-live.git
cd RAISE-FPGA-live
```

If you already cloned it, do not clone again. Instead:

```powershell
cd "$env:USERPROFILE\RAISE-FPGA-live"
git pull --ff-only
```

Check the checkpoint is present:

```powershell
Test-Path .\outputs\models\combined_board_20261006_physics_cnn_gru.pt
```

Expected: True. You do not need to train again.

## Step 3 — Install the PC runtime

Still in PowerShell, inside RAISE-FPGA-live:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install numpy==1.26.4
.\.venv\Scripts\python.exe -c "import torch,numpy; print(torch.__version__,numpy.__version__)"
```

Wait for each command to finish before running the next. Expected: the last command prints both versions without an error. No environment activation or PowerShell execution-policy change is needed. PyTorch is not required on the board.

## Step 4 — Check the model locally before connecting

In PowerShell:

```powershell
.\.venv\Scripts\python.exe -m scripts.make_demo_telemetry
.\.venv\Scripts\python.exe -m scripts.run_live_zcu104_prediction --input examples\replay_telemetry.csv --output-dir results\replay
```

Expected: "Model loaded", then current temperature and forecasts, then saved file paths.
This uses synthetic readings: it checks installation, not model accuracy.

On subsequent tests, skip make_demo_telemetry if examples\replay_telemetry.csv already exists. The generator deliberately refuses to overwrite it.

## Step 5 — Connect the board with power OFF

Keep your existing working ZCU104 Linux microSD image. Do not reflash it.

Locate SW6 using the [official setup picture](https://pynq.readthedocs.io/en/latest/_images/zcu104_setup.png). Set switch 1 ON and switches 2, 3 and 4 OFF. ON means toward the printed ON mark. Use the switch numbers, not left/right orientation.

Connect:
1. The existing boot microSD in the J100 slot.
2. The supplied board power adapter to J52 and mains.
3. A data-capable micro-USB cable from J164 USB-UART to a USB port on the PC.
4. An Ethernet cable from P12 to a router/switch. The PC must use the same network, through Ethernet or Wi-Fi.

Keep normal fan/cooling connected. USB alone does not power this board. This guide uses a router/network with DHCP; a direct board-to-PC Ethernet cable needs different IP configuration.

Turn the board power switch ON. Do not move SW6 while powered on.
For these steps see [PYNQ setup](https://pynq.readthedocs.io/en/latest/getting_started/zcu104_setup.html) and [AMD UG1267](https://docs.amd.com/v/u/en-US/ug1267-zcu104-eval-bd).
No HDMI, new Vivado project, JTAG programming or new bitstream is required for PC inference.

## Step 6 — Open the board's Linux console in PuTTY

1. On Windows, open Device Manager.
2. Expand Ports (COM & LPT).
3. Identify the ZCU104 USB serial COM port for FTDI UART channel B. COM numbers differ between PCs.
4. Open PuTTY. Select Serial.
5. Enter your COM port and speed 115200.
6. Under Connection > Serial: 8 data bits, 1 stop bit, no parity, no flow control.
7. Click Open and press Enter.
8. Log in using the account for your installed board image.

An unchanged PYNQ image commonly uses xilinx / xilinx. Other images may not; use your existing credentials.

Expected: a Linux shell prompt. If there is no console, check the data cable, UART COM port, SD image and boot setting. Do not proceed until Linux is running.

## Step 7 — Find the board's IP address

Type these commands in **PuTTY**, not PowerShell:

```bash
whoami
python3 --version
ip -br address
```

Write down:
- The username printed by whoami.
- The Ethernet IPv4 address, for example 192.168.1.25. Ignore the /24 suffix and do not use 127.0.0.1.

The example IP is not your actual address. If Ethernet has no IPv4 address, fix the router connection/DHCP first. Keep PuTTY open.

## Step 8 — Test the network connection from the PC

Return to **Windows PowerShell** in the project folder.
Set these variables using your actual username and IP:

```powershell
$boardUser = "xilinx"
$boardIp = "192.168.1.25"
$board = "$boardUser@$boardIp"
Test-NetConnection $boardIp -Port 22
```

Expected: TcpTestSucceeded : True.
If False, stop and check Ethernet, network access and the board's SSH service; do not disable firewall/security protections.

Test login:

```powershell
ssh $board
```

Verify any first-connection host fingerprint with your lab before accepting it. Enter the board account password if requested; password typing may not display characters.

Expected: a board Linux prompt. Type:

```bash
exit
```

You must now be back at the Windows PowerShell prompt.

## Step 9 — Copy only the collector to the board

In **PowerShell**:

```powershell
ssh $board "mkdir -p raise_fpga_live"
scp .\scripts\collect_zcu104_training_telemetry.py "${board}:raise_fpga_live/collect_zcu104_training_telemetry.py"
```

The collector goes into a dedicated folder in the board account's home directory. The .pt file stays on Windows; do not copy it for this PC-inference workflow.

## Step 10 — Check all seven sensor readings

In **PowerShell**:

```powershell
ssh $board "python3 raise_fpga_live/collect_zcu104_training_telemetry.py --check-only"
```

Expected: mappings/readings for all seven inputs:
- PL temperature, PS temperature, remote temperature — degrees C.
- VCCINT, VCCAUX, VCCBRAM — volts.
- Input power — watts.

Confirm plausible units and matching sensor identities against your previous readings.
The collector expects the AMS channel layout from the supplied logs. INA226 power equivalence still needs board verification.
If a sensor is missing, access is denied, or a value is implausible, stop and keep the error output. Do not fill missing channels with zero or replace them with unrelated sensors.

## Step 11 — Run a short live test

Leave normal cooling/protections enabled. If needed, run your existing approved FPGA workload separately using its established procedure. This repository does not start or program that workload.

In **PowerShell**:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_live_zcu104_prediction --ssh $board --remote-script raise_fpga_live/collect_zcu104_training_telemetry.py --duration-s 60
```

Enter the board password if requested. The runner automatically starts the collector over SSH; do not start a second collector manually.

Expected:
1. "Model loaded. Waiting for a valid 3.2-second history..."
2. Once sufficient valid readings arrive, prediction lines roughly every second.
3. After 60 seconds, file paths for saved telemetry and predictions.

If input stops or has invalid/gapped readings, prediction pauses and requires a new valid history. Do not treat paused/stale readings as a valid prediction.

## Step 12 — Understand the printed output

Each line shows:
- PL: the temperature measured now.
- forecast 0.5/2/5s: predicted future PL temperatures.
- risk rows 49/50/51C: three rows, one for each event threshold.
- Within each risk row, columns correspond to 0.5, 2 and 5 seconds.
- inference: time taken for the PC model calculation, not total network latency.

49/50/51C are trained event thresholds, not manufacturer danger limits. The risk scores are uncalibrated, so do not present them as verified safety probabilities. This checkpoint does not predict 10-second horizons or newly defined 46/47/48C events.

## Step 13 — Inspect the evidence

After the run, in **PowerShell**:

```powershell
Get-ChildItem .\results\live
$log = Get-ChildItem .\results\live\predictions_*.jsonl | Sort-Object LastWriteTime | Select-Object -Last 1
Get-Content $log.FullName | Select-Object -First 3
Get-Content $log.FullName | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.kind -eq 'temperature_followup' } | Select-Object -First 5 | Format-List
```

The PC saves raw telemetry CSV and predictions JSONL in results\live.
The board also saves CSV and sensor metadata under recordings in the board account's home directory.

temperature_followup records compare a previous forecast with the later measured temperature and include the absolute error. Predictions near the recording end may lack future follow-up.

This demonstrates forecast-versus-measurement behavior, not classification accuracy. Accuracy, precision, recall and F1 require complete event labeling and a separate held-out evaluation.

## Step 14 — Collect a longer session

After the 60-second test succeeds, run in **PowerShell**:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_live_zcu104_prediction --ssh $board --remote-script raise_fpga_live/collect_zcu104_training_telemetry.py --duration-s 1800
```

1800 seconds is 30 minutes. Keep warm-up, steady-state and cool-down as separate documented workload phases. Save complete sessions, record workload timing, and keep whole sessions separate between training and held-out testing.

Ctrl+C stops the PC runner, not a separately launched FPGA workload. After an interrupted SSH session, check the board for any remaining collector before restarting. Shut down Linux normally before turning board power off.

## What is and is not finished

The package passed local replay and 8 unit tests. Live streaming on your physical board still requires verification.

The earlier held-out macro accuracy was 89.53%, versus a 96.82% always-negative baseline. Raw accuracy alone does not prove a useful hazard detector. The new gradual recording had no positive 49/50/51C events; its negative-only 100% result does not validate transition detection.

Automatic workload switching to a second FPGA, safe state transfer, and FPGA-fabric execution are not included. A .pt file is a trained checkpoint, not a bitstream. Do not intentionally overheat the board or bypass protection to create training events.
