#!/usr/bin/env python3
"""OAEI Conference track, reference alignment ra1 (oaei.ontologymatching.org/2024/conference) -> map_reflect format.

Inputs: the track's conference.zip (16 OWL ontologies, RDF/XML) and reference-alignment.zip (21 Alignment API RDF
files <onto1>-<onto2>.rdf over cmt, conference, confOf, edas, ekaw, iasted, sigkdd). The ontologies are schema only.

Task: entity of onto1 (class, object property or datatype property) to entity of onto2; every ra1 cell is "=".
  - entity id <onto>__<local name>; description: the kind, the name split on case and underscores, rdfs:label and
    rdfs:comment where present, named superclasses for a class, domain and range for a property
  - one record per (source entity, ontology pair) with a gold cell; a source with several cells in one pair keeps
    them all as acceptable targets
  - the target pool is every declared entity of onto2; each record lists it in the order of the runner's hybrid
    retriever for the runner's own query "<id>: <description>" (scripts/naacl/newdata_order.py, gold-blind), so the
    first pass shows the first 20 and the full-list control the whole ontology
  - a cell naming an entity that the OWL file does not declare is dropped and counted
  - grouped split unit: the source ontology

Usage: convert_oaei_conference.py <dir with the OWL files> <dir with the reference RDF files>
       (writes data/oaei_conference_{dataset.jsonl,api_lookup.json})
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from newdata_order import ROOT, order_pools  # noqa: E402

NAME = "oaei_conference"
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
OWL = "{http://www.w3.org/2002/07/owl#}"
XML = "{http://www.w3.org/XML/1998/namespace}"
KINDS = {OWL + "Class": "class", OWL + "ObjectProperty": "object property", OWL + "DatatypeProperty": "datatype property",
         # characteristics that also declare a property; the kind comes from another declaration, an rdf:type, or
         # the range (an XSD datatype makes it a datatype property)
         OWL + "FunctionalProperty": "property", OWL + "InverseFunctionalProperty": "property",
         OWL + "TransitiveProperty": "property", OWL + "SymmetricProperty": "property"}
ONTOS = ["cmt", "conference", "confOf", "edas", "ekaw", "iasted", "sigkdd"]


def words(local):
    return " ".join(re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", local.replace("_", " ")).split()).lower()


def local_of(ref, base):
    if not ref:
        return None
    if "#" in ref:
        pre, loc = ref.rsplit("#", 1)
        return loc if pre in ("", base) or pre.lower().rstrip("/") == base.lower().rstrip("/") else None
    return None


def parse_owl(path):
    root = ET.parse(path).getroot()
    base = root.get(XML + "base") or ""
    ents = defaultdict(lambda: dict(kind=None, labels=[], comments=[], parents=[], domain=[], range=[]))

    def ref_of(el):
        if el.get(RDF + "ID"):
            return el.get(RDF + "ID")
        return local_of(el.get(RDF + "about") or el.get(RDF + "resource"), base)

    for el in root.iter():
        kind = KINDS.get(el.tag)
        if not kind:
            continue
        loc = ref_of(el)
        if not loc:
            continue
        e = ents[loc]
        if kind != "property" or not e["kind"]:
            e["kind"] = kind if kind != "property" or not e["kind"] else e["kind"]
        for ch in el:
            if ch.tag == RDF + "type":
                t = KINDS.get(OWL + (ch.get(RDF + "resource") or "").rsplit("#", 1)[-1])
                if t and t != "property":
                    e["kind"] = t
            txt = " ".join((ch.text or "").split())
            if ch.tag == RDFS + "label" and txt:
                e["labels"].append(txt)
            elif ch.tag == RDFS + "comment" and txt:
                e["comments"].append(txt)
            elif ch.tag in (RDFS + "subClassOf", RDFS + "domain", RDFS + "range"):
                key = {"subClassOf": "parents", "domain": "domain", "range": "range"}[ch.tag.split("}")[1]]
                r = local_of(ch.get(RDF + "resource"), base)
                if r:
                    e[key].append(r)
                for sub in ch:
                    if sub.tag == OWL + "Class" and ref_of(sub):
                        e[key].append(ref_of(sub))
                    elif ch.tag != RDFS + "subClassOf":
                        # unionOf domains or ranges, and XSD datatypes
                        for c in sub.iter(OWL + "Class"):
                            if ref_of(c):
                                e[key].append(ref_of(c))
                if key == "range" and not r and ch.get(RDF + "resource", "").startswith("http://www.w3.org/2001/XMLSchema#"):
                    e[key].append(ch.get(RDF + "resource").split("#")[1])
    for e in ents.values():
        if e["kind"] == "property":
            xsd = {"string", "int", "integer", "dateTime", "date", "boolean", "float", "double", "anyURI",
                   "nonNegativeInteger", "positiveInteger", "decimal", "time", "gYear", "long"}
            e["kind"] = "datatype property" if any(r in xsd for r in e["range"]) else "object property"
    return base, {k: v for k, v in ents.items() if v["kind"]}


def describe(loc, e):
    parts = [f"{e['kind']} '{words(loc)}'"]
    if e["labels"]:
        parts.append("label: " + "; ".join(dict.fromkeys(e["labels"])))
    if e["comments"]:
        parts.append("comment: " + " ".join(dict.fromkeys(e["comments"]))[:400])
    if e["kind"] == "class" and e["parents"]:
        parts.append("subclass of " + ", ".join(dict.fromkeys(words(p) for p in e["parents"])))
    if e["kind"] != "class":
        if e["domain"]:
            parts.append("domain " + ", ".join(dict.fromkeys(words(p) for p in e["domain"])))
        if e["range"]:
            parts.append("range " + ", ".join(dict.fromkeys(words(p) for p in e["range"])))
    return "; ".join(parts)


def main():
    owl_dir, ref_dir = Path(sys.argv[1]), Path(sys.argv[2])
    files = {p.stem.lower(): p for p in owl_dir.glob("*.owl")}
    onto = {}
    lookup = {}
    for o in ONTOS:
        base, ents = parse_owl(files[o.lower()])
        onto[o] = ents
        for loc, e in ents.items():
            lookup[f"{o}__{loc}"] = dict(product_name=f"oaei_{o}", type_name=o, field_name=loc, name=words(loc),
                                         type=e["kind"], database="oaei_conference", description=describe(loc, e),
                                         _pool=o)
    gold, dropped = defaultdict(list), []
    for rf in sorted(ref_dir.glob("*.rdf")):
        o1, o2 = rf.stem.split("-")
        o1, o2 = next(o for o in ONTOS if o.lower() == o1.lower()), next(o for o in ONTOS if o.lower() == o2.lower())
        for cell in ET.parse(rf).getroot().iter("{http://knowledgeweb.semanticweb.org/heterogeneity/alignment}Cell"):
            e1 = cell.find("{http://knowledgeweb.semanticweb.org/heterogeneity/alignment}entity1").get(RDF + "resource")
            e2 = cell.find("{http://knowledgeweb.semanticweb.org/heterogeneity/alignment}entity2").get(RDF + "resource")
            rel = cell.findtext("{http://knowledgeweb.semanticweb.org/heterogeneity/alignment}relation")
            l1, l2 = e1.rsplit("#", 1)[1], e2.rsplit("#", 1)[1]
            s, t = f"{o1}__{l1}", f"{o2}__{l2}"
            if rel != "=" or s not in lookup or t not in lookup:
                dropped.append(dict(pair=f"{o1}-{o2}", entity1=e1, entity2=e2, relation=rel,
                                    reason="not declared in the OWL file" if rel == "=" else "relation is not ="))
                continue
            gold[(o1, o2, s)].append(t)
    queries = {s: f"{s}: {lookup[s]['description']}" for (_, _, s) in gold}
    keyed = {f"{o2}||{s}": o2 for (o1, o2, s) in gold}
    qk = {f"{o2}||{s}": queries[s] for (o1, o2, s) in gold}
    order = order_pools(qk, keyed, lookup)
    for v in lookup.values():
        v.pop("_pool", None)
    out, in20 = [], 0
    for (o1, o2, s), ts in sorted(gold.items()):
        props = order[f"{o2}||{s}"]
        in20 += any(t in props[:20] for t in ts)
        out.append({
            "id": f"{NAME}_{o1}_{o2}_{s.split('__', 1)[1]}".lower(),
            "utterance": f"Map entities from ontology {o1} to ontology {o2}",
            "source_schema": f"oaei_{o1}", "source_type": f"oaei_{o1}", "expected_target_type": o2,
            "target_schema": {"name": o2, "properties": [{"name": p} for p in props]},
            "mapping": {s: ts if len(ts) > 1 else ts[0]},
            "metadata": {"source_table": o1, "pair": f"{o1}-{o2}", "target_tables": [o2], "num_fields": 1,
                         "benchmark": NAME},
        })
    json.dump(lookup, open(ROOT / f"data/{NAME}_api_lookup.json", "w"), indent=1)
    with open(ROOT / f"data/{NAME}_dataset.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    json.dump(dropped, open(ROOT / f"data/{NAME}_dropped.json", "w"), indent=1)
    print(f"{NAME}: {len(out)} records ({sum(len(v) for v in gold.values())} gold cells kept, {len(dropped)} dropped); "
          f"entities {len(lookup)}; gold in the first 20 shown {in20}; "
          f"several targets {sum(1 for v in gold.values() if len(v) > 1)}")


if __name__ == "__main__":
    main()
