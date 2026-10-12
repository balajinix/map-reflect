#!/usr/bin/env python3
"""Convert Valentine schema-matching pairs into the map_reflect input format.

Valentine layout (either file-naming convention, both handled):
  <pair>/source.json                {col: {"type": ...}}
  <pair>/target.json                {col: {"type": ...}}
  <pair>/mapping.json               {"matches": [{source_column, target_column, ...}]}
  or  <pair>/<name>_source.json etc.

Emits (SchemaNet-compatible), for a whole Valentine SUBSET aggregated into one task:
  data/<out>_api_lookup.json   {tgt_table_id: {name, description, fields:{col:{name,type,description}}}}
  data/<out>_dataset.jsonl     one line/pair: {id, utterance, source_schema, target_schema,
                               source_type, expected_target_type, mapping, metadata}

The target pool = every target table in the subset, so the mapper must find the right
target table among many (a real matching task, not a trivial 1:1).

With --v41 it also writes the method v4.1 reflection lookup (the two files above are unchanged, byte for byte):
  data/<out>_reflect_lookup_v41.json   the target entries above plus one entry per source column of every pair,
                               read from <pair>/source.json (all source columns, mapped or not, with their
                               Valentine datatype and the source table as parent). The gold mapping is never
                               read for these entries. reflect_v4.py --lookup takes this file, so reflection and
                               the pairwise check see the source field's datatype, table and sibling columns;
                               the frozen --pools still fix the candidates, and the first pass keeps the
                               target-only lookup.

Usage:
  build_valentine_dataset.py <subset_root> <out_name> [--cap=N] [--v41]
  e.g. build_valentine_dataset.py llmatch/LLMatch/dataset/test_data/valentine/Wikidata valentine_wikidata
"""
import sys, os, json, glob, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def sanitize(s):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(s)).strip("_").lower()


def find_pairs(root):
    """Yield (pair_id, source.json, target.json, mapping.json) triples."""
    root = Path(root)
    for mp in sorted(root.rglob("*mapping.json")):
        d = mp.parent
        base = mp.name[:-len("mapping.json")]  # '' or '<name>_'
        src = d / f"{base}source.json"
        tgt = d / f"{base}target.json"
        if src.exists() and tgt.exists():
            rel = d.relative_to(root)
            pid = sanitize(f"{rel}_{base}".strip("_/")) or sanitize(d.name)
            yield pid, src, tgt, mp


def load_cols(path):
    """Return {col: type_str} from a Valentine schema json."""
    d = json.load(open(path))
    out = {}
    if isinstance(d, dict):
        for col, meta in d.items():
            t = (meta.get("type") if isinstance(meta, dict) else str(meta)) or "unknown"
            out[str(col)] = str(t)
    return out


def load_matches(path):
    """Return list of (source_column, target_column)."""
    d = json.load(open(path))
    ms = d.get("matches", d) if isinstance(d, dict) else d
    pairs = []
    for m in ms:
        if isinstance(m, dict):
            sc = m.get("source_column") or m.get("source") or m.get("col1")
            tc = m.get("target_column") or m.get("target") or m.get("col2")
            if sc and tc:
                pairs.append((str(sc), str(tc)))
        elif isinstance(m, (list, tuple)) and len(m) >= 2:
            pairs.append((str(m[0]), str(m[1])))
    return pairs


def source_entries(out, pid, src_id, src_cols):
    """Lookup entries for every column of a pair's source table, in the format of the target entries.

    src_cols comes from the pair's source.json (load_cols), so the datatype is Valentine's own and every source
    column is kept, including the ones the gold mapping does not use. The key is the field id the runner gives a
    source field (f"{src_id}__{col}"), and the parent (type_name) is the source table, so reflect_v2.Prompts finds
    the field's datatype, its table and its sibling columns.
    """
    return {f"{src_id}__{c}": {
        "product_name": out, "type_name": src_id, "field_name": c, "name": c, "type": t, "database": out,
        "description": f"Column {c} ({t}) in source table {pid}", "side": "source"}
        for c, t in src_cols.items()}


def reflect_lookup_v41(api_lookup, src_lookup):
    """The v4.1 reflection lookup: the target entries unchanged, then the source entries. A source key that
    collides with a target key would change a candidate's metadata, so it is an error."""
    clash = set(api_lookup) & set(src_lookup)
    if clash:
        raise ValueError(f"source keys collide with target keys: {sorted(clash)[:5]}")
    return {**api_lookup, **src_lookup}


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    root, out = sys.argv[1], sys.argv[2]
    cap = None
    v41 = "--v41" in sys.argv[3:]
    for a in sys.argv[3:]:
        if a.startswith("--cap="):
            cap = int(a.split("=", 1)[1])
    api_lookup, samples, skipped = {}, [], 0
    src_lookup = {}  # --v41: source-column entries, from source.json only

    pairs = list(find_pairs(root))
    if cap and len(pairs) > cap:
        # deterministic evenly-spaced sample to spread across scenarios
        step = len(pairs) / cap
        pairs = [pairs[int(i * step)] for i in range(cap)]

    # Faithful Valentine: each pair is an independent column-matching problem.
    # We scope each sample's candidate pool to ITS OWN target columns (via a
    # target_schema dict with 'properties'), and keep a FLAT field-level
    # api_lookup so the runner resolves each candidate field id. Field ids are
    # globally unique (prefixed by the pair's target id) to avoid collisions.
    for pid, srcp, tgtp, mapp in pairs:
        src_cols = load_cols(srcp)
        tgt_cols = load_cols(tgtp)
        matches = load_matches(mapp)
        if not src_cols or not tgt_cols or not matches:
            skipped += 1; continue
        src_id = f"src_{pid}"
        tgt_id = f"tgt_{pid}"
        # flat api_lookup entries for every target column of this pair
        tgt_field_ids = []
        for c, t in tgt_cols.items():
            fid = f"{tgt_id}__{c}"
            tgt_field_ids.append(fid)
            api_lookup[fid] = {
                "product_name": out, "type_name": tgt_id, "field_name": c,
                "name": c, "type": t, "database": out,
                "description": f"Column {c} ({t}) in table {pid}",
            }
        # ground-truth mapping over globally-unique ids
        mapping = {}
        for sc, tc in matches:
            if tc in tgt_cols and sc in src_cols:
                mapping[f"{src_id}__{sc}"] = f"{tgt_id}__{tc}"
        if not mapping:
            skipped += 1; continue
        src_lookup.update(source_entries(out, pid, src_id, src_cols))
        samples.append({
            "id": f"{out}_{pid}",
            "utterance": f"Map columns from source table {pid} to the target table",
            "source_schema": src_id,
            "source_type": src_id,
            "expected_target_type": tgt_id,
            # dict target_schema -> runner extracts per-sample target_fields AND
            # preserves ground_truth_mapping (the 'mapping' key below)
            "target_schema": {"name": tgt_id,
                              "properties": [{"name": fid} for fid in tgt_field_ids]},
            "mapping": mapping,
            "metadata": {"source_table": pid, "target_tables": [tgt_id],
                         "num_fields": len(mapping), "benchmark": "valentine"},
        })

    if not samples:
        print(f"No usable pairs found under {root}"); sys.exit(1)

    lookup_path = DATA / f"{out}_api_lookup.json"
    ds_path = DATA / f"{out}_dataset.jsonl"
    json.dump(api_lookup, open(lookup_path, "w"), indent=1)
    with open(ds_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    if v41:
        rl_path = DATA / f"{out}_reflect_lookup_v41.json"
        json.dump(reflect_lookup_v41(api_lookup, src_lookup), open(rl_path, "w"), indent=1)
        print(f"  -> {rl_path.relative_to(ROOT)} ({len(src_lookup)} source columns)")
    n_fields = sum(len(s["mapping"]) for s in samples)
    print(f"{out}: {len(samples)} pairs, {n_fields} GT field-matches, "
          f"{len(api_lookup)} target fields (flat lookup), skipped {skipped}")
    print(f"  -> {lookup_path.relative_to(ROOT)}")
    print(f"  -> {ds_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
