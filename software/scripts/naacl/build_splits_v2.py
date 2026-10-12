#!/usr/bin/env python3
"""Protocol v2 split manifests: grouped development and test splits for every dataset.

Samples that share a schema unit go to the same side, so the test split never contains a source schema the
method was designed on. Grouping unit per dataset:

  ITOpsGraph v2, ITOpsGraph-MB v3  source product (source_schema)
  Spider                           database (metadata.database)
  NL2GQL                           GraphQL schema (expected_source)
  ToolBench                        source tool (longest tool name prefixing the source API id)
  OMOP-CMS, -MIMIC, -Synthea       source OMOP table (expected_source.id)
  Valentine suites                 scenario and split shape (e.g. joinable/vertical); each suite is one table
                                   pair, so this is the finest unit that still keeps variants of one
                                   perturbation family together
  Valentine Magellan               table pair (each of the seven pairs is its own unit)
  MIMIC-OMOP (ReMatch release)     MIMIC source table
  SemTab 2024 Metadata2KG round 1  web table
  OAEI Conference                  source ontology
  SchemaNet CPRD Aurum and GOLD    CPRD source table (per CPRD source)
  Bio-ML OMIM-ORDO (500 sampled)   source class (each record)
  v5 held-out sets                 SemTab round 2 table, Bio-ML NCIT-DOID source class, OAEI Anatomy mouse parent
                                   class, T2D web table, MIMIC-IV source table

Groups are ordered by md5(seed + group) and assigned to development while the development share stays at or
below one third of the samples (the first group always goes there); the rest is test. Sample ids are the ids
the runner assigns (load_dataset), so manifests apply directly with --split-manifest.

The reflection prompts were designed on the earlier hash split (md5(source|target|sample_id) % 3 == 0, the
development fields of results/naacl/reflect_pilot). No sample of that split may be tested on, so for the eleven
datasets of the reflection study the test split is the grouped test minus every such sample
("old_dev_excluded" in the manifest; "test_grouped" keeps the grouped test before the exclusion). ITOpsGraph-MB
was never used in that design, so the rule does not apply to it. The exclusion is checked against the keys
actually present in the old development files.

Writes data/splits/<dataset>.json ({"dataset", "unit", "seed", "dev", "test", "groups"}) and
docs/naacl/results/v2_split_audit.md (sizes and what crosses the split).

Usage: .venv/bin/python scripts/naacl/build_splits_v2.py
"""
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
logging.disable(logging.INFO)
from map_reflect.map_reflect_runner import load_dataset  # noqa: E402

SEED = "protocol-v2-2026-10-07"
MIN_TEST_FIELDS = 100  # held-out additions only (docs/naacl/plan/newdata.md)
OUT = ROOT / "data" / "splits"

# name -> (runner dataset name, eval file or None, grouping unit)
DATASETS = {
    "itopsgraph": ("ITOpsGraph", "data/v2/itopsgraph_v2_cross_product.jsonl", "source product"),
    "itgraph_mb": ("ITGraph_MB", "data/v2/itgraph_mb_v3_cross_product_positives.jsonl", "source product"),
    "nl2gql": ("NL2GQL", None, "GraphQL schema"),
    "toolbench": ("ToolBench", None, "source tool"),
    "omop_cms": ("OMOP_CMS", None, "source OMOP table"),
    "omop_mimic": ("OMOP_MIMIC", None, "source OMOP table"),
    "omop_synthea": ("OMOP_Synthea", None, "source OMOP table"),
    "spider": ("Spider", None, "database"),
    "valentine_chembl": ("valentine_chembl", None, "scenario and split shape"),
    "valentine_opendata": ("valentine_opendata", None, "scenario and split shape"),
    "valentine_tpcdi": ("valentine_tpcdi", None, "scenario and split shape"),
    "valentine_wikidata": ("valentine_wikidata", None, "scenario and split shape"),
    # Held-out evaluation datasets added after the method was frozen (docs/naacl/plan/newdata.md): never part of
    # the reflection design, so no old-development exclusion applies.
    "valentine_magellan": ("valentine_magellan", None, "table pair"),
    "mimic_omop_rematch": ("mimic_omop_rematch", None, "MIMIC source table"),
    "semtab_m2kg_r1": ("semtab_m2kg_r1", None, "web table"),
    "oaei_conference": ("oaei_conference", None, "source ontology"),
    "schemanet_cprd": ("schemanet_cprd", None, "CPRD source table"),
    "bioml_omim_ordo": ("bioml_omim_ordo", None, "source class"),
    # New held-out sets for the method v5 confirmatory test (docs/naacl/PLAN_V5_UQ.md), untouched by v3 and v4.
    "semtab_m2kg_r2": ("semtab_m2kg_r2", None, "table"),
    "bioml_ncit_doid": ("bioml_ncit_doid", None, "source class"),
    "oaei_anatomy": ("oaei_anatomy", None, "mouse parent class"),
    "t2d_v2": ("t2d_v2", None, "web table"),
    "mimic4_omop_parciak": ("mimic4_omop_parciak", None, "MIMIC-IV source table"),
}


def _id(x):
    return x.get("id", x.get("name", "")) if isinstance(x, dict) else str(x or "")


def toolbench_tools():
    tools = set()
    for line in open(ROOT / "data" / "toolbench_dataset.jsonl"):
        r = json.loads(line)
        for side in ("source_api", "target_api"):
            n = (r.get(side) or {}).get("tool_name")
            if n:
                tools.add(n.replace(" ", "_").lower())
    return tools


def valentine_group(source_table):
    m = re.match(r"(semantically_joinable|view_unionable|joinable|unionable)_(.*)", source_table or "")
    if not m:
        return source_table or "?"
    scen, rest = m.groups()
    shape = next((s for s in ("both", "vertical", "horizontal") if f"_{s}_" in f"_{rest}_"), "other")
    return f"{scen}/{shape}"


def group_of(name, s, tools):
    md = s.get("metadata") or {}
    if name in ("itopsgraph", "itgraph_mb"):
        return str(s.get("source_schema") or md.get("source_schema") or _id(s.get("expected_source"))).lower()
    if name == "spider":
        return md.get("database", "?")
    if name == "nl2gql":
        return _id(s.get("expected_source")).lower()
    if name == "toolbench":
        api = _id(s.get("expected_source")).lower()
        hits = [t for t in tools if api.startswith(t + "_")]
        return max(hits, key=len) if hits else api
    if name.startswith("omop"):
        return _id(s.get("expected_source"))
    if name == "schemanet_cprd":
        return _id(s.get("expected_source"))
    if name in ("mimic_omop_rematch", "semtab_m2kg_r1", "oaei_conference", "bioml_omim_ordo", "semtab_m2kg_r2",
                "bioml_ncit_doid", "oaei_anatomy", "t2d_v2", "mimic4_omop_parciak"):
        return md["source_table"]
    if name.startswith("valentine"):
        return valentine_group(md.get("source_table", ""))
    raise ValueError(name)


def target_of(name, s):
    md = s.get("metadata") or {}
    if name in ("itopsgraph", "itgraph_mb"):
        return str(s.get("expected_target_type") or md.get("target_type") or _id(s.get("expected_target")))
    return _id(s.get("expected_target"))


DESIGNED_ON = {"itopsgraph", "nl2gql", "toolbench", "omop_cms", "omop_mimic", "omop_synthea", "spider",
               "valentine_chembl", "valentine_opendata", "valentine_tpcdi", "valentine_wikidata"}


def _pid(x):
    """reflect_pilot._id: a dict's 'id' (empty if it has none), else the string."""
    return x.get("id", "") if isinstance(x, dict) else str(x or "")


def old_dev(s):
    skey = f"{_pid(s.get('expected_source'))}|{_pid(s.get('expected_target'))}|{s.get('sample_id', '')}"
    return int(hashlib.md5(skey.encode()).hexdigest(), 16) % 3 == 0


def old_dev_keys_on_disk(name):
    """Sample ids in the old development files (results/naacl/reflect_pilot/<name>__*), as a check."""
    ids = set()
    for f in (ROOT / "results/naacl/reflect_pilot").glob(f"{name}__*/fields*.jsonl"):
        for line in open(f):
            k = json.loads(line)["key"]
            ids.add(k.split("||")[0].split("|")[-1])
    return ids


def split(groups):
    n = sum(len(v) for v in groups.values())
    order = sorted(groups, key=lambda g: hashlib.md5((SEED + g).encode()).hexdigest())
    dev, cnt = [], 0
    for g in order:
        if not dev or cnt + len(groups[g]) <= n / 3:
            dev.append(g)
            cnt += len(groups[g])
    return set(dev)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tools = toolbench_tools()
    lines = ["# Protocol v2 split audit\n",
             f"Built by `scripts/naacl/build_splits_v2.py` (seed `{SEED}`). Groups never cross the split. "
             "Target overlap counts the target schemas or types seen on both sides; it is expected in a "
             "catalogue-mapping task, where every source maps into the same target catalogue.\n",
             "The test split also excludes every sample of the earlier hash-based development split, on which the "
             "reflection prompts were designed (not applicable to ITOpsGraph-MB, which was not used in that design).\n",
             "| Dataset | Unit | Groups (dev / test) | Dev samples | Grouped test | Old-dev excluded | Test | Source groups crossing | Targets on both sides |",
             "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for name, (ds, ev, unit) in DATASETS.items():
        samples = load_dataset(ROOT / "data", ds, None, ROOT / ev if ev else None)
        groups = defaultdict(list)
        for s in samples:
            groups[group_of(name, s, tools)].append(str(s.get("sample_id")))
        devg = split(groups)
        nf = {str(s.get("sample_id")): len(s.get("ground_truth_mapping") or {}) for s in samples}
        all_test = False
        if name not in DESIGNED_ON and name != "itgraph_mb":
            # Held-out additions: nothing is tuned on them, so a grouped test below MIN_TEST_FIELDS fields becomes
            # the whole dataset (development empty) rather than a test too small to report.
            if sum(nf[i] for g in groups if g not in devg for i in groups[g]) < MIN_TEST_FIELDS:
                devg, all_test = set(), True
        dev = sorted(i for g in devg for i in groups[g])
        test_grouped = sorted(i for g in groups if g not in devg for i in groups[g])
        assert len(set(dev) | set(test_grouped)) == len(samples) and not set(dev) & set(test_grouped)
        sid = {str(s.get("sample_id")): s for s in samples}
        if name in DESIGNED_ON:
            olddev = {i for i in sid if old_dev(sid[i])}
            on_disk = old_dev_keys_on_disk(name)
            missed = (on_disk & set(sid)) - olddev
            assert not missed, f"{name}: {len(missed)} old development samples not caught by the rule"
        else:
            olddev = set()
        excluded = sorted(set(test_grouped) & olddev)
        test = sorted(set(test_grouped) - olddev)
        tdev = {target_of(name, sid[i]) for i in dev}  # noqa: E501
        ttest = {target_of(name, sid[i]) for i in test}
        man = dict(dataset=ds, eval_file=ev, unit=unit, seed=SEED, dev=dev, test=test,
                   test_grouped=test_grouped, old_dev_excluded=excluded,
                   old_dev_rule=("md5(source|target|sample_id) % 3 == 0" if name in DESIGNED_ON else "not applicable"),
                   groups={g: ("dev" if g in devg else "test") for g in sorted(groups)},
                   **({"all_test": f"grouped test below {MIN_TEST_FIELDS} fields; whole dataset is test"}
                      if all_test else {}))
        json.dump(man, open(OUT / f"{name}.json", "w"), indent=1)
        lines.append(f"| {name} | {unit} | {len(devg)} / {len(groups) - len(devg)} | {len(dev)} | {len(test_grouped)} | "
                     f"{len(excluded) if name in DESIGNED_ON else 'n/a'} | {len(test)} | 0 | {len(tdev & ttest)} of {len(tdev | ttest)} |")
        print(name, "dev", len(dev), "test_grouped", len(test_grouped), "excluded", len(excluded), "test", len(test))
    (ROOT / "docs/naacl/results").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs/naacl/results/v2_split_audit.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
