#!/usr/bin/env python3
"""SchemaNet CPRD Aurum and CPRD GOLD to OMOP (LLMatch release, github.com/knowledge-fusion/LLMatch) -> map_reflect.

Inputs (the release's dataset/ directory): schema_export/<cprd_aurum|cprd_gold|omop>-original.sql (CREATE TABLE
statements with COMMENT ON TABLE / COLUMN descriptions from the CPRD data specifications and the OMOP CDM) and
ground_truth_files/<CPRD_AURUM|CPRD_GOLD>-OMOP-ground_truth.csv. A gold row is "src table, src column, OMOP table,
OMOP column", or, for fields routed through the OMOP ETL stem table, "src table, src column, stem, stem column,
OMOP table, OMOP column"; the last two items are the target. Rows the release comments out with "#" are skipped. Schema text
only.

Task: CPRD column (source) to OMOP column (target), given the gold OMOP table, the same set-up as the MIMIC-OMOP
(ReMatch) conversion (scripts/naacl/convert_mimic_omop_rematch.py):
  - one record per (CPRD table, OMOP table) with gold; target_schema.properties = the OMOP table's columns in the
    release's schema order; a source column with several gold columns in that table keeps them all as acceptable
    targets
  - field ids <cprd_aurum|cprd_gold>_<TABLE>__<column>, omop_<TABLE>__<column>; description = table comment + "; "
    + column comment; type from the CREATE TABLE statement; the lookup holds every column of both schemas
  - gold rows naming a table or column the schema does not declare are dropped and counted
  - grouped split unit: the CPRD source table

This replaces the earlier conversion in data/schemanet/, which kept one target per source column and gave every
column the generic description "Column <c> in table <t>".

Field-first treatment (ff1, scripts/naacl/field_first.py): components() keeps the table comment and the column comment
apart, and the run also writes data/schemanet_cprd_reflect_lookup_ff1.json, the same lookup with each entry's parts
under "components". The api lookup and the dataset are unchanged.

Usage: convert_schemanet_cprd.py <LLMatch dataset dir>   (writes data/schemanet_cprd_{dataset.jsonl,api_lookup.json,
       reflect_lookup_ff1.json}, both sources in one set; record ids and source field ids carry the source name)
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "schemanet_cprd"  # both CPRD sources in one evaluation set


def parse_sql(path):
    text = Path(path).read_text(encoding="utf-8")
    cols, tdesc, cdesc = {}, {}, {}
    for m in re.finditer(r"CREATE TABLE\s+(\w+)\s*\((.*?)\);", text, re.S):
        t = m.group(1).lower()
        cols[t] = []
        for line in m.group(2).split(",\n"):
            parts = line.strip().rstrip(",").split(None, 1)
            if parts and re.match(r"^\w+$", parts[0]) and parts[0].upper() not in ("PRIMARY", "FOREIGN", "CONSTRAINT"):
                cols[t].append((parts[0], parts[1] if len(parts) > 1 else "unknown"))
    for m in re.finditer(r"COMMENT ON TABLE\s+(\w+)\s+IS\s+'((?:[^']|'')*)'", text):
        tdesc[m.group(1).lower()] = " ".join(m.group(2).replace("''", "'").replace("[SYSTEM] ", "").split())
    for m in re.finditer(r"COMMENT ON COLUMN\s+(\w+)\.(\w+)\s+IS\s+'((?:[^']|'')*)'", text):
        cdesc[(m.group(1).lower(), m.group(2).lower())] = " ".join(m.group(3).replace("''", "'").split())
    return cols, tdesc, cdesc


def compose(c):
    """The lookup description from its parts: table comment; column comment (either may be absent)."""
    return "; ".join(x for x in (c["table_description"], c["column_description"]) if x)


def components(ds_dir, src):
    """field id -> {table_description, column_description} for the source schema and OMOP."""
    out = {}
    for prefix, sql in ((src, f"{src}-original.sql"), ("omop", "omop-original.sql")):
        cols, tdesc, cdesc = parse_sql(Path(ds_dir) / "schema_export" / sql)
        for t, cl in cols.items():
            for c, _ in cl:
                out[f"{prefix}_{t.upper()}__{c.lower()}"] = dict(table_description=tdesc.get(t, ""),
                                                                 column_description=cdesc.get((t, c.lower()), ""))
    return out


def convert(ds_dir, src, gt_file):
    name = f"schemanet_{src}"
    lookup, ids = {}, {}
    parts = components(ds_dir, src)
    for prefix, sql in ((src, f"{src}-original.sql"), ("omop", "omop-original.sql")):
        cols, tdesc, cdesc = parse_sql(ds_dir / "schema_export" / sql)
        for t, cl in cols.items():
            for c, typ in cl:
                fid = f"{prefix}_{t.upper()}__{c.lower()}"
                ids[(prefix, t, c.lower())] = fid
                lookup[fid] = dict(product_name=prefix, type_name=t.upper(), field_name=c.lower(), name=c.lower(),
                                   type=typ.strip().rstrip(","), database=prefix, description=compose(parts[fid]))
    omop_cols = defaultdict(list)
    for (p, t, c), fid in ids.items():
        if p == "omop":
            omop_cols[t].append(fid)
    gold, dropped, rows = defaultdict(lambda: defaultdict(list)), [], 0
    for line in open(ds_dir / "ground_truth_files" / gt_file, encoding="utf-8-sig"):
        if line.lstrip().startswith("#"):  # rows the release comments out
            continue
        parts = [x.strip() for x in line.strip().split(",")]
        if len(parts) not in (4, 6) or not all(parts):
            continue
        rows += 1
        st, sc, tt, tc = parts[0].lower(), parts[1].lower(), parts[-2].lower(), parts[-1].lower()
        s, t = ids.get((src, st, sc)), ids.get(("omop", tt, tc))
        if not s or not t:
            dropped.append(dict(row=line.strip(), reason="source not declared" if not s else "target not declared"))
            continue
        if t not in gold[(st, tt)][s]:
            gold[(st, tt)][s].append(t)
    out = []
    for (st, tt), mp in sorted(gold.items()):
        out.append({
            "id": f"{name}_{st}_{tt}",
            "utterance": f"Map columns from source table {src}_{st.upper()} to the target table omop_{tt.upper()}",
            "source_schema": f"{src}_{st.upper()}", "source_type": f"{src}_{st.upper()}",
            "expected_target_type": f"omop_{tt.upper()}",
            "target_schema": {"name": f"omop_{tt.upper()}", "properties": [{"name": f} for f in omop_cols[tt]]},
            "mapping": {k: (v if len(v) > 1 else v[0]) for k, v in mp.items()},
            "metadata": {"source_table": st, "target_tables": [f"omop_{tt.upper()}"], "num_fields": len(mp),
                         "benchmark": name},
        })
    nf = sum(len(r["mapping"]) for r in out)
    multi = sum(1 for r in out for v in r["mapping"].values() if isinstance(v, list))
    beyond = sum(1 for r in out for v in r["mapping"].values()
                 if min([p["name"] for p in r["target_schema"]["properties"]].index(t)
                        for t in (v if isinstance(v, list) else [v])) >= 20)
    print(f"{name}: {rows} gold rows, {len(dropped)} dropped; {len(out)} records, {nf} scored fields "
          f"({multi} with several acceptable targets, {beyond} whose gold is past the first 20 shown); "
          f"lookup {len(lookup)}")
    for d in dropped:
        print("   dropped:", d)
    return lookup, out


def main():
    ds = Path(sys.argv[1])
    lookup, out = {}, []
    for src, gt in (("cprd_aurum", "CPRD_AURUM-OMOP-ground_truth.csv"), ("cprd_gold", "CPRD_GOLD-OMOP-ground_truth.csv")):
        lk, recs = convert(ds, src, gt)
        for k, v in lk.items():
            assert lookup.get(k, v) == v, k  # the two share the OMOP schema file
            lookup[k] = v
        out += recs
    json.dump(lookup, open(ROOT / f"data/{NAME}_api_lookup.json", "w"), indent=1)
    from ff1_lookups import attach, comp
    parts = {}
    for src in ("cprd_aurum", "cprd_gold"):
        parts.update(components(ds, src))
    ff1 = {k: comp(p["column_description"], p["table_description"], "", "LLMatch release schema SQL comments")
           for k, p in parts.items()}
    (ROOT / f"data/{NAME}_reflect_lookup_ff1.json").write_text(json.dumps(attach(lookup, ff1), indent=1) + "\n")
    with open(ROOT / f"data/{NAME}_dataset.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{NAME}: {len(out)} records, {sum(len(r['mapping']) for r in out)} scored fields, lookup {len(lookup)}")


if __name__ == "__main__":
    main()
