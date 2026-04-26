"""
SensorLogger 📡 — BUILDCORED ORCAS Day 20
Multi-channel DAQ logger: mic RMS, camera motion, keystroke cadence, CPU/battery.
100ms sample rate, ring buffer, anomaly detection, rich live dashboard.
"""

import time
import threading
import collections
import sys
import os

import numpy as np
import psutil
import pyaudio
import cv2
from pynput import keyboard
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.progress import BarColumn, Progress

# ─── CFG ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE_HZ   = 10          # how many DAQ samples per second (every 100ms)
RING_BUFFER_SIZE = 300         # 30 seconds at 10 Hz
DISPLAY_WIDTH    = 60          # sparkline width in chars

# Anomaly thresholds — tune to your environment
THRESH = {
    "mic_rms":         0.04,   # 0.0–1.0 normalised amplitude
    "motion_mag":      8.0,    # optical flow magnitude (pixels/frame)
    "key_cadence":     7.0,    # keypresses per second
    "cpu_pct":         85.0,   # CPU %
    "battery_pct":     95.0,   # low battery % (or CPU temp °C if no battery)
}

# Audio config
AUDIO_CHUNK    = 1024
AUDIO_RATE     = 44100
AUDIO_CHANNELS = 1
AUDIO_FORMAT   = pyaudio.paInt16

# ─── SHARED STATE ─────────────────────────────────────────────────────────────
st = {
    "running": True,
    "lock":    threading.Lock(),
    "buffers": {
        "mic_rms":      collections.deque(maxlen=RING_BUFFER_SIZE),
        "motion_mag":   collections.deque(maxlen=RING_BUFFER_SIZE),
        "key_cadence":  collections.deque(maxlen=RING_BUFFER_SIZE),
        "cpu_pct":      collections.deque(maxlen=RING_BUFFER_SIZE),
        "battery_pct":  collections.deque(maxlen=RING_BUFFER_SIZE),
    },
    "latest": {
        "mic_rms":      0.0,
        "motion_mag":   0.0,
        "key_cadence":  0.0,
        "cpu_pct":      0.0,
        "battery_pct":  0.0,
    },
    "anomalies": {k: False for k in THRESH},
    "key_events": collections.deque(maxlen=200),   # timestamps of recent keypresses
    "audio_rms":  0.0,                              # written by audio thread
    "motion_mag": 0.0,                              # written by vision thread
    "sample_count": 0,
    "start_time": time.time(),
    "labels": {
        "mic_rms":     "🎙  Mic RMS",
        "motion_mag":  "👁  Motion",
        "key_cadence": "⌨  Keys/s",
        "cpu_pct":     "🖥  CPU %",
        "battery_pct": "🔋 Batt %",
    },
    "units": {
        "mic_rms":     "amp",
        "motion_mag":  "px/f",
        "key_cadence": "k/s",
        "cpu_pct":     "%",
        "battery_pct": "%",
    },
    "use_cpu_temp": False,   # set True if no battery found
}

# ─── SPARKLINE ────────────────────────────────────────────────────────────────
SPARK_CHARS = "▁▂▃▄▅▆▇█"

def sparkline(buf, width=DISPLAY_WIDTH, threshold=None):
    if len(buf) < 2:
        return " " * width
    data = list(buf)[-width:]
    lo, hi = min(data), max(data)
    span = hi - lo or 1e-9
    chars = []
    for v in data:
        idx = int((v - lo) / span * (len(SPARK_CHARS) - 1))
        chars.append(SPARK_CHARS[idx])
    line = "".join(chars).rjust(width)
    return line

# ─── AUDIO THREAD ─────────────────────────────────────────────────────────────
def audio_thread():
    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK,
        )
    except Exception as e:
        print(f"[audio] Could not open mic: {e}")
        pa.terminate()
        return

    while st["running"]:
        try:
            raw = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples ** 2)))
            st["audio_rms"] = rms
        except Exception:
            pass

    stream.stop_stream()
    stream.close()
    pa.terminate()

# ─── VISION THREAD ────────────────────────────────────────────────────────────
def vision_thread():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[vision] Camera not opened — motion channel will be 0.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while st["running"]:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag = float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))
        st["motion_mag"] = mag
        prev_gray = gray
        time.sleep(0.05)   # ~20 fps is plenty for motion magnitude

    cap.release()

# ─── KEYBOARD LISTENER ────────────────────────────────────────────────────────
def start_keyboard_listener():
    def on_press(key):
        st["key_events"].append(time.time())

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

# ─── DAQ SAMPLER (main logging loop) ──────────────────────────────────────────
def daq_loop():
    """Fires every 100ms, reads all sensor values, stamps and pushes to ring buffers."""
    interval = 1.0 / SAMPLE_RATE_HZ

    while st["running"]:
        t0 = time.time()

        # — Mic RMS
        mic = st["audio_rms"]

        # — Motion magnitude
        mot = st["motion_mag"]

        # — Keystroke cadence: events in last 1 second
        now = time.time()
        recent_keys = [e for e in st["key_events"] if now - e <= 1.0]
        cadence = float(len(recent_keys))

        # — CPU %
        cpu = psutil.cpu_percent(interval=None)

        # — Battery or CPU temp
        batt_val = 0.0
        if not st["use_cpu_temp"]:
            bat = psutil.sensors_battery()
            if bat is None:
                st["use_cpu_temp"] = True
                st["labels"]["battery_pct"] = "🌡  CPU°C"
                st["units"]["battery_pct"]  = "°C"
                THRESH["battery_pct"] = 80.0
            else:
                batt_val = bat.percent
        if st["use_cpu_temp"]:
            try:
                temps = psutil.sensors_temperatures()
                # try common keys
                for key in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
                    if key in temps and temps[key]:
                        batt_val = temps[key][0].current
                        break
            except Exception:
                batt_val = 0.0

        vals = {
            "mic_rms":     mic,
            "motion_mag":  mot,
            "key_cadence": cadence,
            "cpu_pct":     cpu,
            "battery_pct": batt_val,
        }

        with st["lock"]:
            for ch, v in vals.items():
                st["latest"][ch] = v
                st["buffers"][ch].append(v)
                st["anomalies"][ch] = v > THRESH[ch]
            st["sample_count"] += 1

        elapsed = time.time() - t0
        sleep_for = max(0.0, interval - elapsed)
        time.sleep(sleep_for)

# ─── DASHBOARD RENDERER ───────────────────────────────────────────────────────
CHANNEL_COLORS = {
    "mic_rms":     "cyan",
    "motion_mag":  "magenta",
    "key_cadence": "yellow",
    "cpu_pct":     "green",
    "battery_pct": "blue",
}

def make_dashboard():
    with st["lock"]:
        latest    = dict(st["latest"])
        buffers   = {k: list(v) for k, v in st["buffers"].items()}
        anomalies = dict(st["anomalies"])
        count     = st["sample_count"]
        labels    = st["labels"]
        units     = st["units"]

    elapsed = time.time() - st["start_time"]
    h, m, s = int(elapsed//3600), int((elapsed%3600)//60), int(elapsed%60)

    # Header
    header = Text(
        f"  📡  SensorLogger — BUILDCORED ORCAS Day 20  │  "
        f"Samples: {count}  │  "
        f"Uptime: {h:02d}:{m:02d}:{s:02d}  │  "
        f"Rate: {SAMPLE_RATE_HZ} Hz  │  "
        f"Buffer: {RING_BUFFER_SIZE} pts",
        style="bold white on #1a1a2e",
    )

    # Channel rows
    table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold white",
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Channel",   style="bold", width=16)
    table.add_column("Value",     width=10, justify="right")
    table.add_column("Threshold", width=10, justify="right")
    table.add_column("Status",    width=10, justify="center")
    table.add_column(f"Last {RING_BUFFER_SIZE} samples", ratio=1)

    for ch in ["mic_rms", "motion_mag", "key_cadence", "cpu_pct", "battery_pct"]:
        val   = latest[ch]
        thr   = THRESH[ch]
        anom  = anomalies[ch]
        color = CHANNEL_COLORS[ch]
        buf   = buffers[ch]
        unit  = units[ch]
        label = labels[ch]

        status_text = Text("⚠ SPIKE", style="bold red") if anom else Text("● OK", style="bold green")
        val_text    = Text(f"{val:7.3f} {unit}", style=f"bold {color}" if anom else color)
        spark       = Text(sparkline(buf), style=color if not anom else "red")

        table.add_row(label, val_text, f"{thr:.1f} {unit}", status_text, spark)

    # Anomaly summary bar
    active = [st["labels"][k] for k, v in anomalies.items() if v]
    if active:
        alert_panel = Panel(
            Text("  🚨  ANOMALIES: " + "  |  ".join(active), style="bold red"),
            style="red",
            height=3,
        )
    else:
        alert_panel = Panel(
            Text("  ✅  All channels nominal", style="bold green"),
            style="green",
            height=3,
        )

    # Footer hint
    footer = Text(
        "  Press Ctrl+C to stop  │  "
        f"Ring buffer: {min(len(buffers['cpu_pct']), RING_BUFFER_SIZE)}/{RING_BUFFER_SIZE} filled",
        style="dim",
    )

    layout = Layout()
    layout.split_column(
        Layout(Panel(header, style="on #1a1a2e", height=3), name="header", size=3),
        Layout(Panel(table, title="[bold white]DAQ Channels[/]", border_style="bright_blue"), name="main"),
        Layout(alert_panel, name="alerts", size=3),
        Layout(footer, name="footer", size=1),
    )
    return layout

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    console = Console()
    console.print("[bold cyan]📡 SensorLogger starting...[/]")

    # Pre-warm CPU % (first call always returns 0.0)
    psutil.cpu_percent(interval=None)

    # Start sensor threads
    threading.Thread(target=audio_thread, daemon=True).start()
    threading.Thread(target=vision_thread, daemon=True).start()
    start_keyboard_listener()

    # DAQ loop in its own thread
    threading.Thread(target=daq_loop, daemon=True).start()

    # Give sensors a moment to warm up
    time.sleep(0.5)
    console.print("[bold green]All channels active. Starting dashboard...[/]")
    time.sleep(0.3)

    try:
        with Live(make_dashboard(), refresh_per_second=4, screen=True) as live:
            while True:
                time.sleep(0.25)
                live.update(make_dashboard())
    except KeyboardInterrupt:
        st["running"] = False
        console.print("\n[bold yellow]SensorLogger stopped. Goodbye. 📡[/]")


if __name__ == "__main__":
    main()