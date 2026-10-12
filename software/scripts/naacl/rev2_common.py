"""Shared helpers for the offline analyses answering the second simulated ARR review, Part I
(docs/reviews/naacl_gpt_review2_full.txt; response in docs/naacl/v6/reviews/gpt_review2_technical_response.md).

Every rev2_*.py script reads stored outputs only: the method v5 rows (results/naacl/v6/v5/{dev,test}), the frozen
benefit model and threshold (docs/naacl/results/v5_frozen.json) and the frozen development cost estimates
(docs/naacl/results/v5_matched_cost_frozen.json). No model call, nothing refit on test, no frozen file written.
Each script writes docs/naacl/results/rev2_<name>_facts.json with status "exploratory" (specified after the test
results were seen) and asserts that it reproduces the paper's frozen headline numbers before it reports anything.
"""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
os.environ.setdefault("ITOPS_V21", str(ROOT.parent / "schema-mapping-datasets/data/itopsgraph_v2_1/itopsgraph_v2.1_cross_product.jsonl"))
sys.argv = ["v5_matched.py", "--stage", "score", "--B", "2000"]
import v5_matched as VM  # noqa: E402  (v5 loader, frozen-model scoring, cost model and sequential fill)

V5, GS, MODELS = VM.V5, VM.GS, VM.MODELS
FZ = VM.FZ
BM = FZ["c_benefit"]
MU, SD = np.array(BM["standardise"]["mu"]), np.array(BM["standardise"]["sd"])
S = FZ["intervention"]["outcome_field"]
EST = json.loads(VM.COST_F.read_text())["est"]
B = 2000
FAMILY = {"itopsgraph": "ITOpsGraph", "nl2gql": "NL2GQL", "toolbench": "ToolBench", "omop_cms": "OMOP",
          "omop_mimic": "OMOP", "omop_synthea": "OMOP", "valentine_chembl": "Valentine", "valentine_opendata": "Valentine",
          "valentine_tpcdi": "Valentine", "valentine_wikidata": "Valentine"}


def git_head():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


class Facts:
    def __init__(self, name, definition):
        self.name, self.definition, self.rows = name, definition, []
        self.path = ROOT / f"docs/naacl/results/rev2_{name}_facts.json"

    def add(self, fid, claim, value, lo=None, hi=None):
        assert fid.isalpha(), fid
        self.rows.append(dict(id="F-RevTwo" + fid, claim=claim, value=str(value),
                              lo=None if lo is None else str(lo), hi=None if hi is None else str(hi),
                              src=str(self.path.relative_to(ROOT)), fn=f"rev2_{self.name}", status="exploratory"))
        print(f"F-RevTwo{fid}: {value}" + (f" [{lo}, {hi}]" if lo is not None else ""), "|", claim[:110], flush=True)

    def write(self, extra=None):
        prov = dict(generated_by=f"scripts/naacl/rev2_{self.name}.py", date=str(datetime.date.today()),
                    code_commit=git_head(), status="exploratory",
                    note="Specified after the test results were seen, in answer to the second simulated ARR review "
                         "(Part I). Offline on stored outputs; no model call; nothing refit on test; unadjusted 95% "
                         "split-group bootstrap intervals (B = 2000, seed 20261009) where given.",
                    inputs=["results/naacl/v6/v5/{dev,test}", "docs/naacl/results/v5_frozen.json",
                            "docs/naacl/results/v5_matched_cost_frozen.json"],
                    definition=self.definition)
        if extra:
            prov.update(extra)
        self.path.write_text(json.dumps(dict(provenance=prov, facts=self.rows), indent=1) + "\n")
        print("wrote", self.path, len(self.rows), "facts")


def sg(x, nd=2):
    return f"{x:+.{nd}f}"


def design(c, cols=None, drop_model=False):
    M_, D_ = V5.feats(c)
    X = np.hstack([(M_ - MU) / SD, D_])
    if drop_model:
        X = X[:, :X.shape[1] - len(MODELS)]
    return X if cols is None else X[:, cols]


def load(split, heldout=False):
    cells = V5.load(split, datasets=V5.HELDOUT if heldout else None)
    VM.prepare(cells)
    for c in cells:
        c.lab = V5.label(c, S)
    return cells


def check_headline(test):
    """The paper's frozen core-test numbers (F-VFiveKeepAcc, F-VFiveMainAcc, F-VFiveEveryAcc, F-VFiveStructAcc,
    F-VFiveMainCalls) must reproduce before any new number is reported."""
    N = sum(c.n for c in test)
    keep = 100 * sum(c.keep.sum() for c in test) / N
    main = 100 * sum(VM.outcome(c, c.r5).sum() for c in test) / N
    every = 100 * sum(c.act.sum() for c in test) / N
    struct = 100 * sum(VM.outcome(c, c.struct, "wid").sum() for c in test) / N
    calls = sum(np.sum(c.r5 * np.nan_to_num(c.act_calls)) for c in test) / N
    assert N == 13384, N
    assert f"{keep:.2f}" == "85.49", keep
    assert f"{main:.2f}" == "93.05", main
    assert f"{every:.2f}" == "93.10", every
    assert f"{struct:.2f}" == "92.57", struct
    assert f"{calls:.2f}" == "0.44", calls
    return dict(N=N, keep=keep, main=main, every=every, struct=struct, calls=calls)


def groups():
    return GS.split_groups(V5.DATASETS, V5.ITOPS_DUP)


def boot(cells, heldout=False):
    g = GS.split_groups(V5.HELDOUT, None) if heldout else groups()
    return GS.GroupBootstrap(cells, g, B=B, seed=V5.SEED)
