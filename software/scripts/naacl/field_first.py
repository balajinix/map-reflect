"""Field-first descriptions for reflection: treatment ff1, opt-in with MAPREFLECT_FIELD_FIRST=1.

The defect (docs/naacl/v6/reviews/round2_run_audit/REPORT.md, section 3). reflect_v2.Prompts.desc shows the first
300 characters of a lookup description. The clinical converters write the shared table description first, then the
column's own definition and its ETL notes, so on MIMIC-OMOP, OMOP-CMS/MIMIC/Synthea and CPRD-OMOP the cut often leaves
only the table text: sibling columns of one table read the same, although the first pass saw their full text.

The treatment, with an ff1 lookup (data/<ds>_reflect_lookup_ff1.json, built by scripts/naacl/ff1_lookups.py, whose
entries carry "components": column_description, etl_conventions, table_description):
  * every property's text is its own definition, then "ETL: " and its ETL notes, never cut (FIELD_BUDGET is None;
    the first ff1 version cut it at 300 characters); a property with neither says so ("no column-specific
    description");
  * the shared table description is given once per table in a "Table context" block, whole (TABLE_BUDGET is None;
    the first ff1 version cut it at 300 characters per table), listing every table of the properties the prompt shows, in order of first appearance;
  * the same rendering is used for the source property, the candidates (the incumbent, the alternatives and the
    candidates a widen action adds, all rendered by one method), the evidence, the pairwise check and the D3 cards
    (scripts/naacl/v6_d3_run.py: symmetric evidence and the blind verifier).
Every other dataset on which the treatment changes a prompt has a generated ff1 lookup (ff1_lookups.generated()). Its descriptions have
no parts that can be separated without guessing, so each entry's components say split: false, and the rule is: show
the whole description, never cut (ITOpsGraph, Bio-ML, OAEI, SemTab and T2D have descriptions over 300 characters);
an explicit type description (ITOpsGraph's type_description) is the table text, also uncut. A property whose lookup
entry has no "components" at all keeps the default text, and the table block lists no table for it.

Off (the default, and whenever the variable is not exactly "1"), every prompt is byte-identical to the code before
the treatment (tests/test_ff1_field_first.py against tests/fixtures/ff1_default_prompt_md5.json). The ff1 lookup
changes no top-level key of any entry, so it too renders the default prompts when the treatment is off. Candidate
ids, pools, splits, gold and first passes are not touched. MAPREFLECT_V41=1 does not switch it on.

Runs with the treatment on record it: reflect_v4 and v6_d3_run refuse to start without an ff1 lookup and TOKEN_LOG,
write <out>.ff1_manifest.json (treatment, lookup path and sha256, flags), tag every usage record in TOKEN_LOG with the
field key, the action and the prompt md5, and extend the raw log (<out>.raw.jsonl or MAPREFLECT_RAW_LOG) with the
fields of scripts/naacl/ff1_log.py: per call the lookup hash, parsed ranking, selected answer and margin, failure
category and actual tokens; per action the router decision, proposal, verifier decision and final answer.
"""
import hashlib
import json
import os
from pathlib import Path

TREATMENT = "ff1"
FLAG = "MAPREFLECT_FIELD_FIRST"
FIELD_BUDGET = None  # per property: never cut (the default desc shows 300 characters)
TABLE_BUDGET = None  # per table, given once: never cut
NO_FIELD_TEXT = "no column-specific description"


def on():
    """Read at call time, like the method v4.1 flags. Independent of MAPREFLECT_V41."""
    return os.environ.get(FLAG) == "1"


def parts(meta):
    c = meta.get("components") if isinstance(meta, dict) else None
    return c if isinstance(c, dict) else None


def field_text(meta, key=""):
    """The property's own definition, then its ETL notes, whole (FIELD_BUDGET None). An entry whose description
    has no separable parts (components.split is false) shows its whole description, uncut."""
    c = parts(meta)
    if c.get("split") is False:
        return str(meta.get("description", key))
    col, etl = str(c.get("column_description") or "").strip(), str(c.get("etl_conventions") or "").strip()
    text = "; ".join(x for x in (col, f"ETL: {etl}" if etl else "") if x) or NO_FIELD_TEXT
    return text if FIELD_BUDGET is None else text[:FIELD_BUDGET]


def table_key(meta):
    return (meta.get("product_name"), meta.get("type_name"))


def table_block(metas):
    """The shared table descriptions of the given lookup entries, once per table, in order of first appearance."""
    seen, lines = set(), []
    for m in metas:
        c = parts(m)
        if not c or not str(c.get("table_description") or "").strip():
            continue
        tk = table_key(m)
        if tk in seen:
            continue
        seen.add(tk)
        owner = f" ({tk[0]})" if tk[0] else ""
        tab = str(c["table_description"]).strip()
        lines.append(f"- {tk[1]}{owner}: {tab if c.get('split') is False or TABLE_BUDGET is None else tab[:TABLE_BUDGET]}")
    if not lines:
        return ""
    return "\n\nTable context (shared by every property of a table, given once per table):\n" + "\n".join(lines)


def has_components(lookup):
    return any(parts(v) for v in lookup.values())


def manifest(out, lookup_path, extra=None):
    """Write <out>.ff1_manifest.json: the treatment, the lookup and its sha256, the method flags."""
    lp = Path(lookup_path)
    m = dict(treatment=TREATMENT, flag=f"{FLAG}=1", field_budget=FIELD_BUDGET, table_budget=TABLE_BUDGET,
             lookup=str(lp), lookup_sha256=hashlib.sha256(lp.read_bytes()).hexdigest(),
             flags={k: v for k, v in sorted(os.environ.items()) if k.startswith("MAPREFLECT_")},
             token_log=os.environ.get("TOKEN_LOG"), **(extra or {}))
    p = Path(str(out) + ".ff1_manifest.json")
    p.write_text(json.dumps(m, indent=1) + "\n")
    return p


def require(lookup, lookup_path):
    """Exit unless an ff1 run can record what it did: an ff1 lookup and TOKEN_LOG for the actual prompt tokens."""
    import sys
    if not has_components(lookup):
        sys.exit(f"{FLAG}=1 needs an ff1 lookup (data/<ds>_reflect_lookup_ff1.json); {lookup_path} has no components")
    if not os.environ.get("TOKEN_LOG"):
        sys.exit(f"{FLAG}=1 runs log the actual prompt tokens of every call: set TOKEN_LOG")
