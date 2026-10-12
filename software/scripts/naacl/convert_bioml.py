#!/usr/bin/env python3
"""OAEI Bio-ML 2024 equivalence pairs (Zenodo record 13119437, CC BY 4.0) -> map_reflect format.

Pairs: OMIM-ORDO (omim-ordo.zip; data/bioml_omim_ordo_*) and NCIT-DOID (ncit-doid.zip; data/bioml_ncit_doid_*).
Inputs: <src>.owl, <tgt>.owl and refs_equiv/test.cands.tsv (SrcEntity, TgtEntity, TgtCandidates: 2,605 test rows,
each with 101 candidates, the gold and 100 negatives, in no fixed order). This is Bio-ML's local ranking setting
(Hits@1), which is our field accuracy with one source field per record.

  - a fixed sample of N rows (random.Random(SEED).sample over the file's rows); the rest is not used
  - class id omim__<IRI tail> and ordo__<IRI tail>; description: rdfs:label, then exact synonyms
    (oboInOwl:hasExactSynonym), then the definition (obo:IAO_0000115), then the labels of named superclasses,
    cut to 400 characters. Cross-references (skos:exactMatch, oboInOwl:hasDbXref, skos:notation) are never read,
    because they can name the gold
  - one record per sampled source; target_schema lists its 101 candidates in the order of the runner's hybrid
    retriever for the runner's own query "<id>: <description>" (scripts/naacl/newdata_order.py, gold-blind); the
    first pass shows the first 20, the full-list control all 101, and v3 widening draws from the rest of the 101
  - grouped split unit: the source class (every record is its own unit)
  - the retriever can break exact score ties differently between runs, so a rerun may order a few candidate lists
    differently (12 of the 500 OMIM-ORDO records on 2026-10-10); the committed files are the ones the runs used

Usage: convert_bioml.py <pair dir> [omim-ordo|ncit-doid]   (default omim-ordo; writes data/bioml_<src>_<tgt>_*)
"""
import ast
import csv
import json
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from newdata_order import ROOT, order_pools  # noqa: E402

N = 500
PAIRS = {"omim-ordo": ("omim", "ordo", "bioml-omim-ordo-2026-10-09"),
         "ncit-doid": ("ncit", "doid", "bioml-ncit-doid-2026-10-10")}
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
OWL = "{http://www.w3.org/2002/07/owl#}"
SYN = "{http://www.geneontology.org/formats/oboInOwl#}hasExactSynonym"
DEF = "{http://purl.obolibrary.org/obo/}IAO_0000115"


def classes(path):
    out = {}
    for _, el in ET.iterparse(path):
        if el.tag == OWL + "Class" and el.get(RDF + "about"):
            c = dict(labels=[], syns=[], defs=[], parents=[])
            for ch in el:
                txt = " ".join((ch.text or "").split())
                if ch.tag == RDFS + "label" and txt:
                    c["labels"].append(txt)
                elif ch.tag == SYN and txt:
                    c["syns"].append(txt)
                elif ch.tag == DEF and txt:
                    c["defs"].append(txt)
                elif ch.tag == RDFS + "subClassOf" and ch.get(RDF + "resource"):
                    c["parents"].append(ch.get(RDF + "resource"))
            out[el.get(RDF + "about")] = c
            el.clear()
    return out


def tail(iri):
    return iri.rstrip("/").rsplit("/", 1)[-1].replace("#", "_")


def main():
    raw = Path(sys.argv[1])
    S, T, SEED = PAIRS[sys.argv[2] if len(sys.argv) > 2 else "omim-ordo"]
    NAME = f"bioml_{S}_{T}"
    rows = list(csv.DictReader(open(raw / "refs_equiv/test.cands.tsv"), delimiter="\t"))
    sample = random.Random(SEED).sample(rows, N)
    onto = {S: classes(raw / f"{S}.owl"), T: classes(raw / f"{T}.owl")}

    def desc(side, iri):
        c = onto[side].get(iri, {})
        parts = []
        if c.get("labels"):
            parts.append(c["labels"][0])
        if c.get("syns"):
            parts.append("synonyms: " + "; ".join(dict.fromkeys(c["syns"])))
        if c.get("defs"):
            parts.append("definition: " + c["defs"][0])
        pl = [onto[side].get(p, {}).get("labels", [None])[0] for p in c.get("parents", [])]
        if any(pl):
            parts.append("subclass of " + "; ".join(p for p in pl if p))
        return (". ".join(parts) or tail(iri))[:400]

    lookup, queries, pools, recs = {}, {}, {}, []
    for i, r in enumerate(sample):
        src = f"{S}__{tail(r['SrcEntity'])}"
        cands = list(ast.literal_eval(r["TgtCandidates"]))
        assert r["TgtEntity"] in cands and len(cands) == 101
        lookup[src] = dict(product_name=S, type_name=S, field_name=tail(r["SrcEntity"]),
                           name=(onto[S].get(r["SrcEntity"], {}).get("labels") or [tail(r["SrcEntity"])])[0],
                           type="class", database=S, description=desc(S, r["SrcEntity"]),
                           iri=r["SrcEntity"])
        for c in cands:
            cid = f"{T}__{tail(c)}"
            lookup.setdefault(cid, dict(product_name=T, type_name=T, field_name=tail(c),
                                        name=(onto[T].get(c, {}).get("labels") or [tail(c)])[0], type="class",
                                        database=T, description=desc(T, c), iri=c))
        key = f"{i}||{src}"
        queries[key], pools[key] = f"{src}: {lookup[src]['description']}", key
        recs.append((key, src, [f"{T}__{tail(c)}" for c in cands], f"{T}__{tail(r['TgtEntity'])}"))
    # one pool per record: its own 101 candidates
    plookup = {}
    for key, _, cands, _ in recs:
        for c in cands:
            plookup[f"{key}##{c}"] = dict(description=lookup[c]["description"], _pool=key)
    order = order_pools(queries, pools, plookup)
    out, in20 = [], 0
    for key, src, cands, gold in recs:
        props = [p.split("##", 1)[1] for p in order[key]]
        assert sorted(props) == sorted(cands)
        in20 += gold in props[:20]
        out.append({
            "id": f"{NAME}_{src.split('__', 1)[1]}".lower(),
            "utterance": f"Map the {S.upper()} disease class to its equivalent {T.upper()} class",
            "source_schema": S, "source_type": S, "expected_target_type": T,
            "target_schema": {"name": T, "properties": [{"name": p} for p in props]},
            "mapping": {src: gold},
            "metadata": {"source_table": src, "target_tables": [T], "num_fields": 1, "benchmark": NAME},
        })
    assert len({r["id"] for r in out}) == len(out)
    json.dump(lookup, open(ROOT / f"data/{NAME}_api_lookup.json", "w"))
    with open(ROOT / f"data/{NAME}_dataset.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{NAME}: {len(out)} sampled of {len(rows)} test rows (seed {SEED}); lookup {len(lookup)}; "
          f"gold in the first 20 shown {in20}")


if __name__ == "__main__":
    main()
