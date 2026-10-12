#!/usr/bin/env python3
"""Run a Python entry point with every endpoint_b inference request throttled by one cross-process token bucket.

The endpoint_b project serves about 8 requests per second in total and cascades into 429s above 7.5; 6 per second
holds. Every v6 call process (R6 first and second pass on endpoint_b, U9 samples) runs through this wrapper, so the
bucket bounds their sum. The bucket is a file holding the earliest time the next request may start; a request takes
the lock, reserves its slot, releases the lock and sleeps until the slot.

The throttle sits on endpoint_b_sdk's BaseModelInference._post, which every text generation request passes through
(the SDK fans a list of prompts out over threads, each one a _post), so a burst of 20 prompts is spread out too.

Usage: endpoint_b_throttle.py <module> [args ...]      e.g. endpoint_b_throttle.py map_reflect.map_reflect_runner --datasets ...
       endpoint_b_throttle.py <script.py> [args ...]   e.g. endpoint_b_throttle.py scripts/naacl/reflect_v4.py --base ...
env:   WX_RATE (requests per second for all wrapped processes together, default 5.0),
       WX_BUCKET (bucket file, default results/naacl/v6/wx_bucket)
"""
import fcntl
import os
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RATE = float(os.environ.get("WX_RATE", "5.0"))
BUCKET = Path(os.environ.get("WX_BUCKET", str(ROOT / "results/naacl/v6/wx_bucket")))


def acquire():
    BUCKET.parent.mkdir(parents=True, exist_ok=True)
    with open(BUCKET, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        try:
            nxt = float(fh.read().strip() or 0)
        except ValueError:
            nxt = 0.0
        now = time.time()
        slot = max(now, nxt)
        fh.seek(0)
        fh.truncate()
        fh.write(f"{slot + 1.0 / RATE:.6f}")
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)
    if slot > now:
        time.sleep(slot - now)
        if os.environ.get("MAPREFLECT_LATENCY_LOG"):  # the latency log reports the wait (map_reflect/latency_log.py)
            from map_reflect import latency_log
            latency_log.note_throttle_wait(slot - now)


def patch():
    from endpoint_b_sdk.foundation_models.inference import base_model_inference as B
    orig = B.BaseModelInference._post

    def _post(self, http_client, *a, **k):
        acquire()
        return orig(self, http_client, *a, **k)

    B.BaseModelInference._post = _post


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    target, sys.argv = sys.argv[1], sys.argv[1:]
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
    patch()
    if target.endswith(".py"):
        sys.path.insert(0, str(Path(target).resolve().parent))
        runpy.run_path(target, run_name="__main__")
    else:
        runpy.run_module(target, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
