#!/usr/bin/env python3
"""Field-first reflection lookups (treatment ff1, scripts/naacl/field_first.py): the stored reflection lookup of each
affected dataset, every entry unchanged, plus its description split into explicit parts.

Each entry gains one key, "components":
  column_description  the column's (field's) own definition
  etl_conventions     the ETL notes of the column, when the schema has them (OMOP CDM in the ReMatch release)
  table_description   the description of the table (type) the column belongs to, shared by its columns
  provenance          where the split comes from (below)
Nothing else changes, so a run with an ff1 lookup and MAPREFLECT_FIELD_FIRST unset renders the default prompts. The
stored lookups are read, never written; each output is data/<ds>_reflect_lookup_ff1.json.

Every other dataset with a split manifest and a reflection lookup on which the treatment changes a prompt (a
description over 300 characters, or an explicit type description) gets a generated ff1 lookup (generated(): whole
description as the field text, split false, never cut; see field_first.py).

The affected datasets are those whose descriptions put a shared table text before the column text (so a 300-character
cut hides the column), or cut the column text itself:
  omop_cms, omop_mimic, omop_synthea   the stored entries carry table_description, and description is that text,
                                       ";", the column text (provenance "stored table_description")
  mimic_omop_rematch                   the ReMatch release (TableDesc, ColumnDesc, ETL Conventions), read by
                                       convert_mimic_omop_rematch.components; two stored table-only entries
                                       (empty column name) keep the whole description as table text
  schemanet_cprd                       the LLMatch release's COMMENT ON TABLE / COLUMN, read by
                                       convert_schemanet_cprd.components
  mimic4_omop_parciak                  the converter's own layout "<column doc> (table <T>: <table doc>)", cut at 600
                                       characters; the table text is the longest one stored for that table
For the two release-backed sets every split is checked to recompose the stored description exactly (with the
converter's own formula). The split never reads gold, splits or results.

Usage (from the repo root):
  .venv/bin/python scripts/naacl/ff1_lookups.py [--rematch-raw DIR] [--llmatch DIR] [--check]
--check rebuilds in memory and compares with the files on disk (no write).
"""
import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts" / "naacl")]

AFFECTED = ["omop_cms", "omop_mimic", "omop_synthea", "mimic_omop_rematch", "schemanet_cprd", "mimic4_omop_parciak"]


def needs_ff1(ds):
    """The treatment would change a prompt of this dataset: some description is longer than the 300 characters the
    default shows, or an entry has an explicit type or table description."""
    lk = json.load(open(stored_path(ds)))
    return any(isinstance(v, dict) and (len(str(v.get("description", ""))) > 300 or v.get("type_description")
                                        or v.get("table_description")) for v in lk.values())


def generated():
    """Every other dataset with a split manifest and a reflection lookup on which the treatment changes a prompt: its
    ff1 lookup is generated, each entry's components marking the whole description as the field text (split: false,
    never cut). A dataset with no description over 300 characters and no type description renders the same prompts
    with the treatment on or off, so it gets no ff1 lookup and its runs keep the default lookup."""
    return sorted(p.stem for p in (ROOT / "data/splits").glob("*.json")
                  if p.stem not in AFFECTED and stored_path(p.stem).exists() and needs_ff1(p.stem))
REMATCH_RAW = Path("<REMATCH_DATA_DIR>")
LLMATCH = ROOT / "llmatch/LLMatch/dataset"


def stored_path(ds):
    p = ROOT / f"data/{ds}_reflect_lookup_v41.json"
    return p if p.exists() else ROOT / f"data/{ds}_api_lookup.json"


def out_path(ds):
    return ROOT / f"data/{ds}_reflect_lookup_ff1.json"


def comp(column, table, etl="", provenance=""):
    return dict(column_description=column or "", etl_conventions=etl or "", table_description=table or "",
                provenance=provenance)


def attach(lookup, parts):
    """The stored lookup, entries unchanged and in the same order, each with its "components"."""
    missing = [k for k, v in lookup.items() if isinstance(v, dict) and k not in parts]
    assert not missing, f"no components for {missing[:5]}"
    return {k: (dict(v, components=parts[k]) if isinstance(v, dict) else v) for k, v in lookup.items()}


def omop_family(lookup):
    parts = {}
    for k, v in lookup.items():
        td, d = v["table_description"], v["description"]
        assert d.startswith(td + ";"), k
        parts[k] = comp(d[len(td) + 1:].strip(), td, provenance="stored table_description")
    return parts


def rematch(lookup, raw):
    import convert_mimic_omop_rematch as C
    rel = C.components(raw)
    parts = {}
    for k, v in lookup.items():
        if k in rel:
            c = rel[k]
            assert C.compose(c) == v["description"], k
            parts[k] = comp(c["column_description"], c["table_description"], c["etl_conventions"],
                            "ReMatch release schema CSV")
        else:
            assert not v.get("field_name"), k  # only the two table-only entries are absent from this release copy
            parts[k] = comp("", v["description"], provenance="table-only stored entry")
    return parts


def cprd(lookup, raw):
    import convert_schemanet_cprd as C
    rel = {}
    for src in ("cprd_aurum", "cprd_gold"):
        rel.update(C.components(raw, src))
    parts = {}
    for k, v in lookup.items():
        c = rel[k]
        assert C.compose(c) == v["description"], k
        parts[k] = comp(c["column_description"], c["table_description"], "", "LLMatch release schema SQL comments")
    return parts


PARCIAK_TABLE = re.compile(r" \(table (\w+): ")


def parciak(lookup):
    split, tables = {}, defaultdict(str)
    for k, v in lookup.items():
        d = v["description"]
        m = PARCIAK_TABLE.search(d)
        if m and m.group(1).upper() == v["type_name"].upper():
            col, tab = d[:m.start()], d[m.end():]
            tab = tab[:-1] if tab.endswith(")") else tab
            tables[v["type_name"]] = max(tables[v["type_name"]], tab, key=len)
        else:
            col = d  # the 600-character cut fell inside the column text: no table text stored
        split[k] = col
    parts = {}
    for k, v in lookup.items():
        col = split[k]
        # the converter falls back to the attribute name when the release has no documentation for it
        parts[k] = comp("" if col == v["field_name"] else col, tables[v["type_name"]],
                        provenance="converter layout '<column doc> (table <T>: <table doc>)'")
    return parts


def whole(lookup):
    """No separable parts: the description is the field text, shown whole. An explicit type description (ITOpsGraph's
    type_description, a key of its own) is the table text."""
    parts = {}
    for k, v in lookup.items():
        if isinstance(v, dict):
            t = str(v.get("type_description") or v.get("table_description") or "")
            parts[k] = dict(comp(str(v.get("description", "")), t,
                                 provenance="whole description (no separable parts)"
                                 + ("; explicit type_description" if t else "")), split=False)
    return parts


def build(ds, rematch_raw=REMATCH_RAW, llmatch=LLMATCH):
    lookup = json.load(open(stored_path(ds)))
    if ds.startswith("omop_"):
        parts = omop_family(lookup)
    elif ds == "mimic_omop_rematch":
        parts = rematch(lookup, rematch_raw)
    elif ds == "schemanet_cprd":
        parts = cprd(lookup, llmatch)
    elif ds == "mimic4_omop_parciak":
        parts = parciak(lookup)
    elif ds in generated():
        parts = whole(lookup)
    else:
        raise KeyError(ds)
    return attach(lookup, parts)


def dump(obj):
    return json.dumps(obj, indent=1) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rematch-raw", type=Path, default=REMATCH_RAW)
    ap.add_argument("--llmatch", type=Path, default=LLMATCH)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    bad = 0
    for ds in AFFECTED + generated():
        before = hashlib.sha256(stored_path(ds).read_bytes()).hexdigest()
        text = dump(build(ds, a.rematch_raw, a.llmatch))
        assert hashlib.sha256(stored_path(ds).read_bytes()).hexdigest() == before
        if a.check:
            same = out_path(ds).exists() and out_path(ds).read_text() == text
            bad += not same
            print(f"{ds}: {'identical' if same else 'DIFFERS'}")
        else:
            out_path(ds).write_text(text)
            print(f"{ds}: wrote {out_path(ds).relative_to(ROOT)} (from {stored_path(ds).name}, sha256 {before[:12]})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
