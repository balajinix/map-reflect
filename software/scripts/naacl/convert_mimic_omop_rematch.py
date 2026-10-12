#!/usr/bin/env python3
"""MIMIC-III to OMOP as released with ReMatch (github.com/meniData1/MIMIC_2_OMOP, MIT licence) -> map_reflect format.

Inputs (the release's data/ directory): MIMIC_III_Schema.csv, OMOP_Schema.csv, MIMIC_to_OMOP_Mapping.csv
(SRC_ENT, SRC_ATT, TGT_ENT, TGT_ATT; a null mapping has TGT_ENT = TGT_ATT = NA). Schema documentation only, no
patient data, so no PhysioNet access is involved.

Task: MIMIC column (source) to OMOP column (target), one gold target per source column, given the gold OMOP table.
  - one record per (MIMIC table, OMOP table) with at least one gold pair; expected_target_type = that OMOP table;
    target_schema.properties = every column of the OMOP table in the release's schema order
  - field ids mimic_<TABLE>__<COLUMN> and omop_<TABLE>__<column>; the lookup holds every column of both schemas
    (298 + 427), description = table description + "; " + column description (OMOP: + "; ETL: " conventions),
    type = ColumnType. No instance values exist in the release.
  - null gold (NA) rows are not scored (the task has no "no match" answer); they are kept per MIMIC table in
    metadata.null_sources of each record of that table
  - gold rows whose target is not a column of the released OMOP schema are dropped and listed in metadata.dropped
    (two MEASUREMENT.value_as_string rows; one row whose target is "0"."0")

Field-first treatment (ff1, scripts/naacl/field_first.py): components() keeps the table description, the column
description and the ETL conventions apart, and the run also writes data/mimic_omop_rematch_reflect_lookup_ff1.json,
the same lookup with each entry's parts under "components". The api lookup and the dataset are unchanged.

Usage: convert_mimic_omop_rematch.py <release data dir>   (writes data/mimic_omop_rematch_{dataset.jsonl,api_lookup.json,
       reflect_lookup_ff1.json})
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "mimic_omop_rematch"


def read(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def clean(s):
    return " ".join(str(s or "").split())


def compose(c):
    """The lookup description from its parts: table; column (OMOP: ; ETL: conventions)."""
    desc = f"{c['table_description']}; {c['column_description']}".strip("; ")
    return desc + (f"; ETL: {c['etl_conventions']}" if c["etl_conventions"] else "")


def components(raw):
    """field id -> {table_description, column_description, etl_conventions} for both schemas of the release."""
    out = {}
    for prefix, f in (("mimic", "MIMIC_III_Schema.csv"), ("omop", "OMOP_Schema.csv")):
        for r in read(Path(raw) / f):
            t, c = r["TableName"].strip(), r["ColumnName"].strip()
            fid = f"mimic_{t.upper()}__{c.upper()}" if prefix == "mimic" else f"omop_{t.upper()}__{c.lower()}"
            out[fid] = dict(table_description=clean(r["TableDesc"]), column_description=clean(r["ColumnDesc"]),
                            etl_conventions=clean(r.get("ETL Conventions")) if prefix == "omop" else "")
    return out


def main():
    raw = Path(sys.argv[1])
    mimic, omop, gold = (read(raw / "MIMIC_III_Schema.csv"), read(raw / "OMOP_Schema.csv"),
                         read(raw / "MIMIC_to_OMOP_Mapping.csv"))
    parts = components(raw)
    lookup, mk, ok, omop_cols = {}, {}, {}, defaultdict(list)
    for r in mimic:
        t, c = r["TableName"].strip(), r["ColumnName"].strip()
        fid = f"mimic_{t.upper()}__{c.upper()}"
        mk[(t.upper(), c.upper())] = fid
        lookup[fid] = dict(product_name="mimic", type_name=t.upper(), field_name=c.upper(), name=c.upper(),
                           type=clean(r["ColumnType"]) or "unknown", database="mimic_iii",
                           description=compose(parts[fid]))
    for r in omop:
        t, c = r["TableName"].strip(), r["ColumnName"].strip()
        fid = f"omop_{t.upper()}__{c.lower()}"
        ok[(t.upper(), c.upper())] = fid
        omop_cols[t.upper()].append(fid)
        lookup[fid] = dict(product_name="omop", type_name=t.upper(), field_name=c.lower(), name=c.lower(),
                           type=clean(r["ColumnType"]) or "unknown", database="omop_cdm",
                           description=compose(parts[fid]))

    pairs, nulls, dropped = defaultdict(dict), defaultdict(list), []
    for g in gold:
        s = (g["SRC_ENT"].strip().upper(), g["SRC_ATT"].strip().upper())
        t = (g["TGT_ENT"].strip().upper(), g["TGT_ATT"].strip().upper())
        if s not in mk:
            dropped.append(dict(g, reason="source column not in the MIMIC schema"))
            continue
        if t[0] == "NA" and t[1] == "NA":
            nulls[s[0]].append(mk[s])
            continue
        if t not in ok:
            dropped.append(dict(g, reason="target not a column of the released OMOP schema"))
            continue
        assert mk[s] not in pairs[(s[0], t[0])]
        pairs[(s[0], t[0])][mk[s]] = ok[t]

    out = []
    for (st, tt), mapping in sorted(pairs.items()):
        src_id, tgt_id = f"mimic_{st}", f"omop_{tt}"
        out.append({
            "id": f"{NAME}_{st}_{tt}".lower(),
            "utterance": f"Map columns from source table {src_id} to the target table {tgt_id}",
            "source_schema": src_id, "source_type": src_id, "expected_target_type": tgt_id,
            "target_schema": {"name": tgt_id, "properties": [{"name": f} for f in omop_cols[tt]]},
            "mapping": mapping,
            "metadata": {"source_table": st, "target_tables": [tgt_id], "num_fields": len(mapping),
                         "benchmark": NAME, "null_sources": sorted(nulls.get(st, [])),
                         "dropped": [d for d in dropped if d["SRC_ENT"].strip().upper() == st]},
        })
    json.dump(lookup, open(ROOT / f"data/{NAME}_api_lookup.json", "w"), indent=1)
    from ff1_lookups import attach, comp
    ff1 = {k: comp(p["column_description"], p["table_description"], p["etl_conventions"], "ReMatch release schema CSV")
           for k, p in parts.items()}
    (ROOT / f"data/{NAME}_reflect_lookup_ff1.json").write_text(json.dumps(attach(lookup, ff1), indent=1) + "\n")
    with open(ROOT / f"data/{NAME}_dataset.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    nf = sum(len(r["mapping"]) for r in out)
    beyond = sum(1 for r in out for t in r["mapping"].values()
                 if [p["name"] for p in r["target_schema"]["properties"]].index(t) >= 20)
    print(f"{NAME}: {len(out)} records, {nf} gold fields over {len({r['metadata']['source_table'] for r in out})} "
          f"MIMIC tables; null sources {sum(len(v) for v in nulls.values())}; dropped {len(dropped)}; "
          f"gold beyond the first 20 shown candidates {beyond}; lookup {len(lookup)}")


if __name__ == "__main__":
    main()
