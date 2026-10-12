"""Group-level statistics for the routed second pass (docs/naacl/REDO_PLAN_V4.md, item F).

Resampling unit. A split group is the schema unit of the protocol split (data/splits/<dataset>.json, computed by
scripts/naacl/build_splits_v2.group_of: source product, tool, OMOP table, Valentine scenario and shape, GraphQL
schema). For ITOpsGraph, groups that share a v2.1 duplicate group (same target type and same mapping) are joined into
one connected component, so near-duplicate samples never fall on both sides of a resample. The bootstrap resamples
these units with replacement within each dataset (stratified), B times, with one draw shared by every model, policy and
contrast, so all contrasts are paired. Fields of the four models on a resampled unit enter together.

Estimands.
  pooled   100 * sum(change) / fields over the chosen datasets (field-weighted, the primary estimand)
  macro    the unweighted mean over datasets of each dataset's pooled change (secondary)

Tests. Two-sided bootstrap p-values, p = min(1, 2 * min(P*(d <= 0), P*(d >= 0))), floored at 1/B, and Holm's step-down
correction over a pre-declared family of contrasts.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def split_groups(datasets, itops_dup=None):
    """(dataset, sample id) -> resampling unit for every sample of the datasets, from the split builder's own grouping.
    itops_dup: sample id -> ITOpsGraph duplicate group; groups sharing one are joined (union-find)."""
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "naacl")]
    import build_splits_v2 as BS  # noqa: E402  the grouping the split was built with
    from map_reflect.map_reflect_runner import load_dataset  # noqa: E402
    tools = BS.toolbench_tools()
    out = {}
    for name in datasets:
        ds, ev, _ = BS.DATASETS[name]
        man = json.loads((ROOT / f"data/splits/{name}.json").read_text())
        g_of = {}
        for s in load_dataset(ROOT / "data", ds, None, ROOT / ev if ev else None):
            g_of[str(s.get("sample_id"))] = str(BS.group_of(name, s, tools))
        sides = man.get("groups") or {}
        for sid, g in g_of.items():  # the manifest's side for every group must agree with the samples' split
            if g in sides:
                want = "dev" if sid in set(man["dev"]) else "test"
                assert sides[g] == want or sid not in set(man["dev"]) | set(man["test_grouped"]), (name, sid, g)
        if name == "itopsgraph" and itops_dup:
            parent = {}

            def find(x):
                while parent.setdefault(x, x) != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for sid, g in g_of.items():
                d = itops_dup.get(sid)
                if d is not None:
                    a, b = find("g:" + g), find("d:" + str(d))
                    if a != b:
                        parent[a] = b
            g_of = {sid: find("g:" + g) for sid, g in g_of.items()}
        for sid, g in g_of.items():
            out[(name, sid)] = g
    return out


class GroupBootstrap:
    """Stratified bootstrap over split groups. Runs carry .ds, .n and .sid (each field's sample id); vec_fn(run)
    returns a per-field vector."""

    def __init__(self, runs, groups, B=2000, seed=20261009):
        self.groups, self.B = groups, B
        self.index = defaultdict(dict)
        for r in runs:
            for x in r.sid:
                g = self.unit(r.ds, x)
                self.index[r.ds].setdefault(g, len(self.index[r.ds]))
        rng = np.random.default_rng(seed)
        self.counts = {ds: rng.multinomial(len(ix), np.full(len(ix), 1.0 / len(ix)), size=B).astype(float)
                       for ds, ix in sorted(self.index.items())}

    def unit(self, ds, sid):
        g = self.groups.get((ds, str(sid)))
        if g is None:
            raise KeyError(f"{ds}: sample {sid} has no split group")
        return g

    def n_groups(self, ds):
        return len(self.index.get(ds, {}))

    def sums(self, runs, vec_fn):
        out = {}
        for r in runs:
            ix = self.index[r.ds]
            ci = np.array([ix[self.unit(r.ds, x)] for x in r.sid])
            S, N = out.setdefault(r.ds, (np.zeros(len(ix)), np.zeros(len(ix))))
            np.add.at(S, ci, vec_fn(r))
            np.add.at(N, ci, 1.0)
        return out

    def pooled(self, runs, vec_fn, datasets=None):
        """(estimate, draws) of the field-weighted change over the datasets."""
        sums = self.sums(runs, vec_fn)
        dss = [d for d in sums if datasets is None or d in datasets]
        num = sum(sums[d][0].sum() for d in dss)
        den = sum(sums[d][1].sum() for d in dss)
        bn = sum(self.counts[d] @ sums[d][0] for d in dss)
        bd = sum(self.counts[d] @ sums[d][1] for d in dss)
        return 100 * num / den, 100 * bn / np.where(bd > 0, bd, np.nan)

    def macro(self, runs, vec_fn, datasets=None):
        """(estimate, draws) of the unweighted mean over datasets of each dataset's pooled change."""
        sums = self.sums(runs, vec_fn)
        dss = [d for d in sums if datasets is None or d in datasets]
        pts = [100 * sums[d][0].sum() / sums[d][1].sum() for d in dss]
        draws = np.mean([100 * (self.counts[d] @ sums[d][0]) / (self.counts[d] @ sums[d][1]) for d in dss], axis=0)
        return float(np.mean(pts)), draws


def interval(draws, lo=2.5, hi=97.5):
    d = draws[~np.isnan(draws)]
    return float(np.percentile(d, lo)), float(np.percentile(d, hi))


def boot_p(draws):
    """Two-sided bootstrap p-value of a contrast against zero."""
    d = draws[~np.isnan(draws)]
    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
    return float(min(1.0, max(p, 1.0 / len(d))))


def holm(pvals):
    """Holm step-down adjusted p-values, in the input order."""
    order = np.argsort(pvals)
    m, adj, run = len(pvals), np.zeros(len(pvals)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * pvals[i]))
        adj[i] = run
    return adj.tolist()


def digest(groups):
    """A short fingerprint of a group assignment, for the report."""
    h = hashlib.sha1(json.dumps(sorted(groups.items())).encode()).hexdigest()
    return h[:12]
