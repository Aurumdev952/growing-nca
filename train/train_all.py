"""Train many (emoji, experiment) models concurrently; one process each shares the GPU."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from nca import EMOJIS, EXPERIMENTS

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emojis", nargs="+", default=list(EMOJIS))
    p.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--out", type=Path, default=ROOT / "models")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="passed through to train.py after --")
    a = p.parse_args()
    extra = a.extra[1:] if a.extra[:1] == ["--"] else a.extra

    logs = a.out / "runs"
    logs.mkdir(parents=True, exist_ok=True)
    queue = [(e, x) for e in a.emojis for x in a.experiments]
    running: list[tuple[str, subprocess.Popen]] = []
    failed = []
    t0 = time.time()
    while queue or running:
        while queue and len(running) < a.jobs:
            e, x = queue.pop(0)
            log = open(logs / f"{e}_{x}.log", "w")
            cmd = [sys.executable, str(Path(__file__).with_name("train.py")),
                   "--emoji", e, "--experiment", x, "--out", str(a.out), *extra]
            running.append((f"{e}_{x}", subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
        time.sleep(2)
        for name, proc in list(running):
            if proc.poll() is not None:
                running.remove((name, proc))
                status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                if proc.returncode:
                    failed.append(name)
                print(f"{time.time() - t0:6.0f}s {name}: {status}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
