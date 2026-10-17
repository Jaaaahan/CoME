"""Terminate this batch job if a serving process or its input thread dies."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

FATAL_MARKERS = (
    "Exception in thread",
    "EngineDeadError",
    "Engine core initialization failed",
    "CUDA out of memory",
    "RuntimeError: can't start new thread",
)


def process_alive(pid):
    try:
        os.kill(pid, 0)
        # A zombie still responds to kill(pid, 0), but cannot serve requests.
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except (ProcessLookupError, FileNotFoundError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--proxy-pid", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    offset = 0
    while process_alive(args.parent_pid):
        reason = None
        for pid in (args.server_pid, args.proxy_pid):
            if not process_alive(pid):
                reason = f"Serving process {pid} exited"
        if args.log.exists():
            with args.log.open(errors="replace") as handle:
                handle.seek(offset)
                fresh = handle.read()
                offset = handle.tell()
            for marker in FATAL_MARKERS:
                if marker in fresh:
                    reason = f"Fatal model service log: {marker}"
        if reason:
            print(json.dumps({"event": "service_failed", "reason": reason}), flush=True)
            # Close serving sockets so a foreground benchmark wakes immediately
            # rather than waiting through its 1,200-second HTTP timeout.
            for pid in (args.server_pid, args.proxy_pid):
                if process_alive(pid):
                    os.kill(pid, signal.SIGTERM)
            time.sleep(5)
            for pid in (args.server_pid, args.proxy_pid):
                if process_alive(pid):
                    os.kill(pid, signal.SIGKILL)
            # This is the batch shell that launched us, never another user's job.
            os.kill(args.parent_pid, signal.SIGTERM)
            return 1
        time.sleep(15)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
