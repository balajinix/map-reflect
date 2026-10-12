"""Resume guard for the v5 collectors (scripts/naacl/v5_run.py, scripts/naacl/v6_d3_run.py through run_v5.sh).

A collector resumes by skipping the keys its output already holds. Without a check, a resume under different flags,
lookups or code would append rows of another configuration to the same file. The guard writes an immutable
configuration fingerprint beside the output (<out>.fingerprint.json) the first time rows are written, and refuses to
resume when the stored fingerprint differs from the current one, or when an output with rows has no fingerprint at all
(it predates the guard, so its configuration cannot be checked).

The fingerprint is: every MAPREFLECT_* variable, the md5 of each code file the prompts and parsing depend on, the
md5 of each data input (lookup, pools, first-pass slot file, frozen thresholds, keys file), and the run's own settings
(model, split, actions, arms, verifier). TOKEN_LOG, worker counts and timestamps are not part of it.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ["scripts/naacl/v5_run.py", "scripts/naacl/v6_d3_run.py", "scripts/naacl/reflect_v2.py",
        "scripts/naacl/reflect_v3.py", "scripts/naacl/reflect_v4.py", "scripts/naacl/router.py",
        "scripts/naacl/field_first.py", "scripts/naacl/ff1_log.py", "map_reflect/map_reflect_runner.py",
        "eval/calc_accuracy.py"]


def md5(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if p.is_dir():  # a runner slot: its final output file
        fs = sorted(f for f in p.glob("*.json") if not f.name.startswith("checkpoint_"))
        p = fs[0] if fs else p
    return hashlib.md5(p.read_bytes()).hexdigest() if p.is_file() else None


def fingerprint(inputs, settings):
    """inputs: {name: path}; settings: {name: value}."""
    return dict(flags={k: v for k, v in sorted(os.environ.items()) if k.startswith("MAPREFLECT_")},
                code={f: md5(f) for f in CODE}, inputs={k: md5(v) for k, v in sorted(inputs.items()) if v},
                settings={k: settings[k] for k in sorted(settings)})


def check(out, fp):
    """Refuse (exit) when <out> has rows under another configuration; otherwise record fp beside it."""
    out = Path(out)
    fpf = Path(str(out) + ".fingerprint.json")
    has_rows = out.exists() and out.stat().st_size > 0
    if fpf.exists():
        old = json.loads(fpf.read_text())
        if old != fp:
            diff = sorted({f"{k}.{kk}" for k in fp for kk in set(fp[k]) | set(old.get(k, {}))
                           if fp[k].get(kk) != old.get(k, {}).get(kk)})
            sys.exit(f"resume guard: {out} was written under another configuration; differs in {diff[:12]}. "
                     "Write to a fresh output path, or set the old file aside.")
    elif has_rows:
        sys.exit(f"resume guard: {out} has rows but no {fpf.name}, so its configuration cannot be checked. "
                 "Write to a fresh output path, or set the old file aside.")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        fpf.write_text(json.dumps(fp, indent=1) + "\n")
    return fpf
