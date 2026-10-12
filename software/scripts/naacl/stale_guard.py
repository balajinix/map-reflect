"""Refuse reflection-based analysis on data with stale reflection rows (orchestrator rule, 2026-10-08).

Runs scripts/naacl/v2_check_repairs.py on a results root and sums every row whose first-pass answer is stale,
including rows still pending on a redo list. Any stale row means the reflection outcomes do not match the
re-scored first passes, so selective, harm-containment and v3 analyses must not run on that copy.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def stale_count(root):
    """(stale rows, checker output) for the data under root."""
    p = subprocess.run([sys.executable, str(REPO / "scripts/naacl/v2_check_repairs.py"), str(root)],
                       capture_output=True, text=True, cwd=REPO)
    out = p.stdout + p.stderr
    n = 0
    for line in out.splitlines():
        m = re.search(r"stale base (\d+) \(\+(\d+) pending", line)
        if m:
            n += int(m.group(1)) + int(m.group(2))
            continue
        m = re.match(r"PENDING .*?: (\d+) rows with a stale base", line)
        if m:
            n += int(m.group(1))
            continue
        m = re.match(r"STALE BASE .*?: (\d+) of", line)
        if m:
            n += int(m.group(1))
    return n, out


def require_no_stale(root, what):
    n, out = stale_count(root)
    if n:
        lines = [l for l in out.splitlines() if "stale" in l.lower() or l.startswith("PENDING")]
        sys.exit(f"refusing {what}: v2_check_repairs.py finds {n} stale reflection rows under {root}\n" +
                 "\n".join(lines[:20]))
    print(f"stale-row check on {root}: 0 stale rows")
