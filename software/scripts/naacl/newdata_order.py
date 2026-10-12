"""Gold-blind candidate order for the held-out datasets whose target pool is too large for one ranking prompt.

The given-target first pass shows the first 20 of a record's target_schema.properties (runner shown_ids). When a
pool is larger than that (an ontology, a property glossary, Bio-ML's 101 candidates), each source field gets its own
record and its pool is stored in the order of the runner's own hybrid retriever (MiniLM dense + BM25, reciprocal
rank fusion; map_reflect_runner.retrieve_candidates) for the query the runner itself would build,
"<source field id>: <description>". The gold never enters the order. Pool entries the retriever does not return
follow in their original order.
"""
import logging
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)


def order_pools(queries, pools, lookup, keep=None):
    """queries {src: text}; pools {src: pool name}; lookup {id: entry}; returns {src: [ids in retriever order]}.

    Each pool name maps to the ids whose lookup entry has that (product_name, type_name) "pool" key, given here
    as lookup[id]["_pool"]; keep caps the stored list (None keeps all)."""
    from map_reflect.map_reflect_runner import index_chroma, retrieve_candidates
    by_pool = {}
    for k, v in lookup.items():
        if v.get("_pool"):
            by_pool.setdefault(v["_pool"], []).append(k)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, (name, ids) in enumerate(sorted(by_pool.items())):
            coll, idset = f"nd_pool_{i}", set(ids)
            index_chroma(coll, {k: {"description": lookup[k].get("description", "")} for k in ids}, tmp)
            for src, pname in pools.items():
                if pname != name:
                    continue
                hits = [str(h["id"]) for h in retrieve_candidates(queries[src], coll, tmp, len(ids))]
                seen, order = set(), []
                for h in hits + ids:
                    if h in seen or h not in idset:
                        continue
                    seen.add(h)
                    order.append(h)
                out[src] = order[:keep] if keep else order
    return out
