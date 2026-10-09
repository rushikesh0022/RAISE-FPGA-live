# Live ZCU104 predictions on Windows

This package runs inference on the Windows PC. The ZCU104 supplies seven live
sensor measurements using read-only Linux sysfs access over SSH. The model is
loaded once. Predicted outputs are 49/50/51C transition risk within 0.5/2/5s,
and future PL temperature. The current checkpoint does not implement new
46/47/48C transition thresholds. Risk scores are exploratory and uncalibrated.

## 1. Connect the ZCU104 (power OFF first)

Use the printed connector and switch labels, not left/right orientation.
With board power OFF, set SW6 switch **1 ON, 2 OFF, 3 OFF, 4 OFF** for SD
boot. ON means toward the ON marking.
[Official setup picture](https://pynq.readthedocs.io/en/latest/_images/zcu104_setup.png)
and [PYNQ setup instructions](https://pynq.readthedocs.io/en/latest/getting_started/zcu104_setup.html).

1. Keep your existing working ZCU104 Linux microSD in J100. Do not reflash it.
2. Connect the board's supplied power adapter to J52 and mains. USB alone
   does not power this board. Keep the normal fan/cooling connected.
3. Connect a **data-capable micro-USB cable** from J164 USB-UART to a USB
   port on the Windows computer. This provides the serial console, not telemetry
   Ethernet. Do not use a charge-only cable or substitute another USB connector.
4. Connect an Ethernet cable from P12 to your router/switch. Connect the Windows
   PC to the same network, by Ethernet or Wi-Fi. This guide assumes DHCP;
   a direct PC-to-board cable needs separate IP configuration.
5. Turn on board power. Never change boot switches while powered on.

Connector identities are documented in the
[AMD ZCU104 user guide](https://docs.amd.com/v/u/en-US/ug1267-zcu104-eval-bd).
No HDMI, JTAG programming cable, Vivado programming or new bitstream is required
for this PC-inference workflow. Run the existing approved FPGA workload separately.

## 2. Open the board console in PuTTY

Open Windows Device Manager, expand Ports (COM & LPT), and identify the
ZCU104 USB serial port (FTDI UART channel B). In PuTTY select Serial, that COM
number, speed **115200**, 8 data bits, 1 stop bit, no parity, no flow control.
Click Open and press Enter. If you see no Linux console, verify the cable,
selected UART port, SD image and boot setting before continuing.

Log in using the credentials for your installed image. The unchanged PYNQ image
commonly uses xilinx/xilinx; do not assume this for other or modified images.
At the board's Linux prompt run:

```bash
uname -a
cat /etc/os-release
python3 --version
ip -br address
```

Note the Ethernet IPv4 address, not 127.0.0.1. Keep PuTTY open for diagnosis.
If Linux or Python 3 is missing, stop: the following collector cannot run yet.

## 3. Clone and prepare Windows

Open **PowerShell on Windows**, not the PuTTY board shell. Check your existing
Git, Python 3.11 and OpenSSH Client installations:

```powershell
git --version
py -3.11 --version
ssh -V
git clone https://github.com/rushikesh0022/RAISE-FPGA-live.git
cd RAISE-FPGA-live
```

If a prerequisite command is unavailable, install it from its official source
before continuing. Do not install PyTorch on the board for this workflow.

If using the ZIP instead, open PowerShell in its extracted RAISE_FPGA_live folder.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install numpy==1.26.4
.\.venv\Scripts\python.exe -c "import torch,numpy; print(torch.__version__,numpy.__version__)"
```

Only the board collector requires Python on the ZCU104; it uses the standard
library. PyTorch and NumPy are installed on Windows, not required on the board.

## 4. Verify SSH and all seven sensors

Use PuTTY's existing Linux console to obtain the Ethernet IP:

```bash
ip -br address
python3 --version
```

In Windows PowerShell, replace BOARD_USER and BOARD_IP below. Verify the SSH
host fingerprint with the lab before accepting a new host key. Login credentials
depend on the board image. OpenSSH Client supplies ssh and scp on Windows.

```powershell
ssh BOARD_USER@BOARD_IP
```

If the command opens the board shell, run `exit` to return to PowerShell.
If SSH is unavailable/refused, the board image's SSH service and network must
be configured before live streaming; the serial console alone is insufficient.

Copy the updated collector to the board:

```powershell
scp .\scripts\collect_zcu104_training_telemetry.py BOARD_USER@BOARD_IP:
ssh BOARD_USER@BOARD_IP "python3 collect_zcu104_training_telemetry.py --check-only"
```

This expects the legacy AMS PL-channel layout from the supplied board logs.
Missing sensors fail explicitly; do not substitute other channels. The board
must permit read access to those sysfs files. INA226 board-input power
equivalence to the original training measurement remains provisional.

## 5. Run live predictions

From the same extracted folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_live_zcu104_prediction --ssh BOARD_USER@BOARD_IP --duration-s 1800
```

The command launches the board collector over SSH. The board saves telemetry
under recordings/. The PC saves its own CSV and JSONL under results/live/.
Allow at least 3.2 seconds for an input history; predictions print approximately
once per board second. Values are aligned causally to 10Hz using actual
per-sensor acquisition timestamps, with 0.2s staleness and 0.5s gap limits.
Invalid rows and source gaps clear the history. Transport silence pauses
predictions; a fresh history is required after a transport interruption.

Console output contains current PL temperature, forecasts for 0.5/2/5s,
three risk rows (49/50/51C) with three horizons each, and CPU inference time.
JSONL records include all nine decisions using the checkpoint's fixed
cutoffs. Later temperature_followup records include actual observations and
absolute forecast errors. This does not calculate online transition accuracy:
complete follow-up and the event-definition labeling procedure are needed.

Run the existing FPGA workload separately, leaving normal cooling and board
protections enabled. This application does not program the FPGA or change
workloads, voltages, fan controls or clocks. Ctrl+C stops the PC runner; it does
not stop a separately launched workload. The board stream may exit when SSH
disconnects; check its process if reconnecting.

## 6. Replay before connecting (optional)

The public repository includes a synthetic recording for a plumbing check,
not a measured test dataset. Generate it first:

```powershell
.\.venv\Scripts\python.exe -m scripts.make_demo_telemetry
```

```powershell
.\.venv\Scripts\python.exe -m scripts.run_live_zcu104_prediction --input examples\replay_telemetry.csv --output-dir results\replay
```

Replay advances through the file rapidly; it is not a live-board test or an
accuracy benchmark. The separately shared original ZIP may contain a real replay.

## 7. Inspect and save evidence

In PowerShell after the run:

```powershell
Get-ChildItem .\results\live
$log = Get-ChildItem .\results\live\predictions_*.jsonl | Sort-Object LastWriteTime | Select-Object -Last 1
Get-Content $log.FullName | Select-Object -First 3
Get-Content $log.FullName | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.kind -eq 'temperature_followup' } | Select-Object -First 5 | Format-List
```

Keep the CSV, JSONL and the board's metadata JSON together with the workload
name and timing. Follow-up observations demonstrate forecast versus actual
temperature; they do not establish transition classification accuracy by themselves.
For retraining, preserve complete separate runs, including warm-up, steady state
and cool-down. Keep entire sessions separated between training and held-out test.
Do not intentionally overheat the board or defeat hardware protections.

## What remains before claiming completion

The code has passed local replay and unit tests; a live SSH run on your physical
board is still required. Confirm each channel matches the training measurement
(especially INA226 power) and measure live prediction errors and latency.
Then collect enough safe positive and negative transition examples and evaluate
with the original event definitions and fixed decision cutoffs on held-out runs.
The recorded older test macro accuracy is 89.53%, but its always-negative
baseline is 96.82%; raw accuracy therefore does not establish a strong detector.
The new gradual recording had no 49/50/51C positive events. Do not report its
100% negative-only result as verified hazard prediction.

49/50/51C are model event thresholds, not manufacturer danger limits. Any demo
alarm or workload handoff is advisory until validated. Automatic switching to
another FPGA, workload state transfer and a verified fail-safe controller are
not implemented by this package. Running .pt on Windows is not FPGA-fabric deployment.

## Board execution alternatives

A .pt file contains model weights and metadata. It can be stored on the board's
SD card and loaded into PS DDR by a compatible ARM runtime. ARM CPU inference
requires the model code and matching runtime dependencies and still needs
board testing. FPGA fabric execution requires an implementation/accelerator,
operator support checks (including the GRU), quantization where applicable,
compilation and a matching bitstream/runtime. Copying .pt into FPGA memory
does not create executable FPGA logic. None of those FPGA conversion stages
are included in this PC inference package.
