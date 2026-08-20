"""Background poller: launch the detector when the AirPods connect.

macOS gives shell scripts no "on Bluetooth connect" hook, so this polls.
Primary source is IOBluetooth (an in-process framework call, effectively
free); if that is unavailable it falls back to parsing `system_profiler
SPBluetoothDataType -json`, which is correct but takes a second or two per
call.

Only the disconnected -> connected *transition* launches the detector, so
holding a connection does not spawn repeatedly. The child pid is tracked, and
on disconnect it is terminated if `stop_detector_on_disconnect` is set.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_config, log  # noqa: E402

PYTHON = ROOT / ".venv" / "bin" / "python"
DETECTOR = ROOT / "src" / "detector.py"


def normalize(s: str) -> str:
    """Fold curly apostrophes and case so 'AirPods Pro' matches real names."""
    s = unicodedata.normalize("NFKD", s)
    return s.replace("’", "'").replace("‘", "'").lower()


def _connected_via_iobluetooth(needle: str) -> bool | None:
    try:
        import IOBluetooth
    except Exception:
        return None
    try:
        devices = IOBluetooth.IOBluetoothDevice.pairedDevices()
    except Exception:
        return None
    if devices is None:
        return None
    for dev in devices:
        try:
            name = dev.nameOrAddress()
        except Exception:
            continue
        if name and needle in normalize(str(name)) and bool(dev.isConnected()):
            return True
    return False


def _connected_via_profiler(needle: str) -> bool:
    try:
        out = subprocess.run(
            ["system_profiler", "SPBluetoothDataType", "-json"],
            capture_output=True, text=True, timeout=30,
        )
        data = json.loads(out.stdout)
    except Exception as exc:
        log("bt", f"system_profiler failed: {exc}")
        return False

    for controller in data.get("SPBluetoothDataType", []):
        for entry in controller.get("device_connected", []) or []:
            for name in entry:
                if needle in normalize(name):
                    return True
    return False


def is_connected(needle: str) -> bool:
    via_framework = _connected_via_iobluetooth(needle)
    if via_framework is not None:
        return via_framework
    return _connected_via_profiler(needle)


def spawn_detector(extra: list[str]) -> subprocess.Popen | None:
    cmd = [str(PYTHON), str(DETECTOR), *extra]
    log("bt", f"launching detector: {' '.join(cmd)}")
    try:
        # New process group so the whole detector tree can be signalled at once.
        return subprocess.Popen(cmd, cwd=str(ROOT), start_new_session=True)
    except OSError as exc:
        # A crash here would otherwise take the whole watcher down with it --
        # the one process meant to keep running for the rest of the day.
        log("bt", f"failed to launch detector: {exc}")
        return None


def stop_detector(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    log("bt", "stopping detector")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        for _ in range(20):
            if proc.poll() is not None:
                return
            time.sleep(0.1)
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> int:
    cfg = load_config()
    bt = cfg["bluetooth"]
    ap = argparse.ArgumentParser(description="AirPods connection watcher.")
    ap.add_argument("--device", default=bt["device_name_contains"])
    ap.add_argument("--interval", type=float, default=bt["poll_seconds"])
    ap.add_argument("--once", action="store_true",
                    help="print current connection state and exit")
    ap.add_argument("detector_args", nargs="*",
                    help="extra args forwarded to detector.py")
    args = ap.parse_args()

    needle = normalize(args.device)

    if args.once:
        print("connected" if is_connected(needle) else "not connected")
        return 0

    log("bt", f"watching for a device matching '{args.device}' "
              f"every {args.interval}s")

    proc: subprocess.Popen | None = None
    was_connected = is_connected(needle)
    if was_connected:
        log("bt", "device already connected at startup; waiting for a "
                  "fresh connect before launching.")

    while True:
        time.sleep(args.interval)
        now_connected = is_connected(needle)

        if now_connected and not was_connected:
            log("bt", "connected")
            if proc is None or proc.poll() is not None:
                proc = spawn_detector(args.detector_args)
        elif was_connected and not now_connected:
            log("bt", "disconnected")
            if proc is not None and bt.get("stop_detector_on_disconnect", True):
                stop_detector(proc)
                proc = None

        was_connected = now_connected


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
