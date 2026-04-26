# SensorLogger 📡

SensorLogger turns your laptop into a **multi-channel Data Acquisition (DAQ) system**. It simultaneously samples microphone RMS amplitude, camera optical-flow motion magnitude, keystroke cadence, CPU utilisation, and battery level (or CPU temperature on desktops) at 10 Hz — 100ms intervals — and displays every channel as a live sparkline dashboard in your terminal. Any channel that crosses its threshold triggers a red anomaly alert. It's built for the **BUILDCORED ORCAS — Day 20** challenge.

## How it works

- **DAQ sampler** fires every 100 ms, reads all sensor values with a shared timestamp, and pushes them into per-channel **ring buffers** (300 samples = 30 s of history). This mirrors a hardware NI-DAQ board: multiple sensors, one sample clock, one buffer per channel.
- **Mic RMS** — a dedicated audio thread keeps pyaudio streaming at 44100 Hz. Each 1024-sample chunk is converted to float32 and its root-mean-square amplitude is stored; the DAQ loop reads the latest value.
- **Motion magnitude** — an OpenCV thread captures frames and runs `calcOpticalFlowFarneback` between consecutive grayscale frames. The mean flow vector magnitude (pixels/frame) is the motion metric.
- **Keystroke cadence** — a pynput listener timestamps every keypress into a rolling deque. The DAQ loop counts events within the last 1 second → keys/second.
- **CPU %** — `psutil.cpu_percent()` with no blocking interval (non-blocking, uses cached kernel counter delta).
- **Battery %** — `psutil.sensors_battery()`. On desktops where this returns `None`, the logger automatically switches to CPU temperature via `psutil.sensors_temperatures()` and relabels the channel.
- **Anomaly detection** — each channel has a configurable threshold in the `THRESH` dict at the top of the script. Any channel whose current value exceeds its threshold lights up red and appears in the anomaly summary bar.
- **Rich live display** — a `rich.live.Live` context manager redraws the dashboard at 4 Hz without flicker. Each channel row shows: label, live value + unit, threshold, OK/SPIKE badge, and a 60-character sparkline of the ring buffer history.

## Requirements

- Python 3.10.x
- tkinter not required (rich renders in terminal)
- A webcam (channel degrades gracefully to 0 if absent)
- A microphone (channel degrades gracefully to 0 if absent)

## Python packages

```bash
pip install numpy psutil pyaudio opencv-python pynput rich
```

Or:

```bash
pip install -r requirements.txt
```

## Setup

1. Install packages (see above).
2. On Windows, PyAudio may need a pre-built wheel:
   ```
   pip install pipwin && pipwin install pyaudio
   ```
3. Run the script — no arguments needed.

## Usage

```bash
python sensorlogger.py
```

The live dashboard shows five rows:

| Channel | What is measured | Unit |
|---|---|---|
| 🎙 Mic RMS | Microphone amplitude (0–1 normalised) | amp |
| 👁 Motion | Mean optical flow magnitude between frames | px/f |
| ⌨ Keys/s | Keypresses in the last 1 second | k/s |
| 🖥 CPU % | Total CPU utilisation | % |
| 🔋 Batt % | Battery charge (or CPU temperature on desktops) | % / °C |

Press **Ctrl+C** to stop cleanly.

### Tuning thresholds

Edit the `THRESH` dict near the top of the script:

```python
THRESH = {
    "mic_rms":         0.04,   # raise if ambient noise keeps triggering
    "motion_mag":      8.0,    # raise in busy environments
    "key_cadence":     5.0,    # 5 keys/s ≈ fast typing
    "cpu_pct":         85.0,
    "battery_pct":     15.0,   # or °C threshold if using CPU temp
}
```

### Changing sample rate or buffer size

```python
SAMPLE_RATE_HZ   = 10    # samples per second (default 10 = 100ms interval)
RING_BUFFER_SIZE = 300   # total samples stored (300 / 10 Hz = 30 seconds)
```

## Common fixes

**Rich live display flickers** — this uses `Live` with `refresh_per_second=4` which is the correct fix. If it still flickers, try running in Windows Terminal instead of the legacy cmd.exe.

**Optical flow crashes on startup** — camera index 0 not found. Try plugging in your webcam, or change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`. The motion channel safely returns 0 if the camera never opens.

**Battery returns None on desktops** — handled automatically. The logger detects this, switches to CPU temperature, and relabels the channel `🌡 CPU°C` with an 80°C anomaly threshold.

**PyAudio won't open** — your microphone is in use by another app (Teams, Discord, browser). Close the other app. On Windows you can also check Settings → Privacy → Microphone.

**pynput keyboard listener requires permissions on macOS** — go to System Preferences → Security & Privacy → Privacy → Accessibility and add Terminal.

**CPU % always reads 0.0 on first sample** — `psutil.cpu_percent()` without a blocking interval returns 0 on its first call (it needs two measurements to compute a delta). The script pre-warms it before starting the dashboard.

**Sparklines all look flat** — the sparkline normalises to the min/max of the current buffer window. If a channel barely changes, the sparkline will still show relative variation within that range — this is expected DAQ behaviour.

## Hardware concept

This project mirrors a **National Instruments NI-DAQ** or similar multi-channel data acquisition board:

- **Channels** — each sensor is a channel with its own sample buffer
- **Sample rate** — all channels share one 10 Hz sample clock (configurable)
- **Ring buffer** — fixed-size circular buffer, oldest sample discarded when full — identical to the `collections.deque(maxlen=N)` used here
- **Threshold triggering** — hardware DAQ boards have comparator circuits per channel; here it's a Python `>` comparison, same logic

In v2.0 you'll attach real sensors (accelerometer, thermistor, light sensor) to a Raspberry Pi Pico with an ADC, log to SD card, and replay the CSV — the same ring buffer pattern applies.

## Credits

- System metrics: [psutil](https://psutil.readthedocs.io/)
- Audio capture: [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)
- Optical flow: [OpenCV](https://opencv.org/)
- Keyboard events: [pynput](https://pynput.readthedocs.io/)
- Terminal UI: [Rich](https://rich.readthedocs.io/)

Built as part of the **BUILDCORED ORCAS — Day 20: SensorLogger** challenge.
