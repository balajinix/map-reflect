#!/usr/bin/env python3
"""Write paper/appendix_prompts.tex: every prompt of the paper's methods, rendered verbatim from the code that sends it.

The prompts are built by the same functions the experiments call, on one worked example, so the appendix cannot
drift from the experiments:
  first pass        the runner's ranking prompt
  method v5         the frozen configuration (docs/naacl/results/v5_frozen.json): the improved reflection
                    (v6_d3_run.D3Prompts.symmetric(scores=False): symmetric evidence, no old scores, the dataset's
                    correspondence rule instead of the uniqueness hint) and the either-order verifier
                    (D3Prompts.blind in both option orders), under MAPREFLECT_V41=1, MAPREFLECT_CANONICAL_MARGIN=1;
                    plus one rendering under the description repair (MAPREFLECT_FIELD_FIRST=1, nothing cut) on the
                    MIMIC-OMOP case CHARTEVENTS.CHARTTIME, from the stored Granite first pass and the ff1 lookup
  reference         method v4.1 (reflect_v2: neutral reflection with evidence, then the pairwise check)
  interventions     the other protocol v2 arms of the interventions table (doubt, Self-Refine, neutral, check)
Chat templates are not shown: the system and user messages are printed as they are passed to the model's template.

Usage: .venv/bin/python scripts/naacl/make_appendix_prompts.py
"""
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "naacl")]
import os  # noqa: E402

os.environ.update(MAPREFLECT_V41="1", MAPREFLECT_CANONICAL_MARGIN="1")  # the frozen flags of v4.1 and v5
os.environ.pop("MAPREFLECT_FIELD_FIRST", None)  # switched on only for the ff1 example
import reflect_v2 as R  # noqa: E402
from map_reflect.map_reflect_runner import MapReflectConfig, MapReflectRunner  # noqa: E402
from v6_d3_run import D3Prompts  # noqa: E402

WIDTH = 43  # characters per line at \small (inconsolata) in one ACL column

LOOKUP = {
    "monitor_Alert.severity": dict(description="How serious the alert is.", field_type="String",
                                   product_name="monitor", type_name="Alert", field_name="severity"),
    "monitor_Alert.message": dict(description="Text shown to the operator.", field_type="String",
                                  product_name="monitor", type_name="Alert", field_name="message"),
    "monitor_Alert.raisedAt": dict(description="When the alert was raised.", field_type="DateTime",
                                   product_name="monitor", type_name="Alert", field_name="raisedAt"),
    "itsm_Incident.criticalityLevel": dict(description="Business criticality of the affected service.",
                                           field_type="Int", product_name="itsm", type_name="Incident",
                                           field_name="criticalityLevel"),
    "itsm_Incident.priority": dict(description="Priority assigned to the incident, from P1 to P4.",
                                   field_type="String", product_name="itsm", type_name="Incident",
                                   field_name="priority"),
    "itsm_Incident.status": dict(description="Lifecycle state of the incident.", field_type="String",
                                 product_name="itsm", type_name="Incident", field_name="status"),
    "itsm_Incident.openedAt": dict(description="When the incident was opened.", field_type="DateTime",
                                   product_name="itsm", type_name="Incident", field_name="openedAt"),
}
RANKING = [
    {"id": "itsm_Incident.criticalityLevel", "score": 0.62, "reason": "Both describe how serious the issue is."},
    {"id": "itsm_Incident.priority", "score": 0.58, "reason": "Priority is set from the severity of the alert."},
    {"id": "itsm_Incident.status", "score": 0.20, "reason": "Status is the lifecycle state, not seriousness."},
]
REC = dict(key="monitor|itsm|example||monitor_alert.severity", field="monitor_Alert.severity", ranking=RANKING,
           base="itsm_Incident.criticalityLevel", margin=0.04, valid=True,
           siblings={"monitor_Alert.raisedAt": "itsm_Incident.openedAt"}, src="monitor", tgt="itsm")


def render(system, user):
    return f"[system]\n{system}\n\n[user]\n{user}"


class P(R.Prompts):
    def fmt(self, system, user):  # print the messages instead of applying a chat template
        return render(system, user)


class P5(D3Prompts):
    def fmt(self, system, user):
        return render(system, user)


def ff1_example():
    """Method v5's either-order verifier under the description repair, on MIMIC-OMOP CHARTEVENTS.CHARTTIME: the
    stored Granite first answer measurement_time against measurement_datetime, both orders' first (order 1)."""
    import json
    import ff1_render as FR
    os.environ["MAPREFLECT_FIELD_FIRST"] = "1"
    try:
        lk = json.load(open(ROOT / "data/mimic_omop_rematch_reflect_lookup_ff1.json"))
        rec = next(r for r in FR.sample("mimic_omop_rematch", 10 ** 6)[0]
                   if r["field"].upper() == "MIMIC_CHARTEVENTS__CHARTTIME")
        p = P5("example", lk, "mimic_omop_rematch")
        return rec["base"], p.blind(dict(rec, cand_list=None), rec["base"], "omop_MEASUREMENT__measurement_datetime")
    finally:
        os.environ.pop("MAPREFLECT_FIELD_FIRST", None)


def ranking_prompt():
    shim = type("Shim", (), {"is_omop": False, "config": MapReflectConfig(), "doc_retriever": None,
                             "_format_prompt": lambda _s, m: render(m[0]["content"], m[1]["content"]),
                             "_shown_limit": MapReflectRunner._shown_limit})()
    cands = [{"id": k, "doc": f"{k} -- {v['description']}", "distance": 0.5}
             for k, v in LOOKUP.items() if k.startswith("itsm_")]
    return MapReflectRunner._build_ranking_prompt(shim, REC["field"], LOOKUP[REC["field"]]["description"],
                                                  cands, "MONITOR", "ITSM")


def wrap(text):
    out = []
    for line in text.split("\n"):
        if len(line) <= WIDTH:
            out.append(line)
            continue
        indent = len(line) - len(line.lstrip())
        parts = textwrap.wrap(line, WIDTH, subsequent_indent=" " * (indent + 2), break_long_words=True,
                              break_on_hyphens=False, drop_whitespace=True)
        out.extend(parts)
    return "\n".join(out)


def box(title, label, text):
    return (f"\\subsection*{{{title}}}\\label{{{label}}}\n"
            "{\\small\n\\begin{verbatim}\n" + wrap(text) + "\n\\end{verbatim}\n}\n")


def main():
    p = P("example", LOOKUP)
    naive, _ = p.naive(REC)
    neutral, _ = p.neutral(REC)
    evidence, _ = p.neutral(REC, with_evidence=True)
    fb = p.sr_feedback(REC)
    refine, _ = p.sr_refine(REC, "<the model's feedback from the previous call>")
    ver, _ = p.verify(REC, "itsm_Incident.criticalityLevel", "itsm_Incident.priority")
    ver_e, _ = p.verify(REC, "itsm_Incident.criticalityLevel", "itsm_Incident.priority", with_evidence=True)
    p5 = P5("example", LOOKUP, "itopsgraph")
    r5 = dict(REC, cand_list=None)
    v5_reflect, _ = p5.symmetric(r5, scores=False)
    old, new = REC["base"], "itsm_Incident.priority"
    v5_c1, v5_c2 = p5.blind(r5, old, new), p5.blind(r5, new, old)
    ff1_base, ff1 = ff1_example()
    # the verifier runs in both option orders; the second call swaps options 1 and 2 and is otherwise the same text
    import re
    norm = lambda t: sorted(re.sub(r"Option [12]", "Option N", t).split("\n"))  # noqa: E731
    assert norm(v5_c1) == norm(v5_c2) and v5_c1 != v5_c2, "the two verifier orders differ beyond the option order"
    parts = [
        "% Generated by scripts/naacl/make_appendix_prompts.py from the code that sends each prompt. Do not edit.",
        "\\section{Prompts}\\label{app:prompts}",
        "All but one of the prompts below are shown on one worked example: the source field "
        "\\texttt{monitor\\_Alert.severity} and the four fields of an incident type, the same case as "
        "Figure~\\ref{fig:pipeline}. The first-pass prompt lists the four fields; the reflection prompts use the top "
        "three of its ranking, and the verifier and check prompts its leading two, the first answer and the proposal. "
        "The exception is one Map-Reflect verifier on a MIMIC-OMOP field, shown to illustrate "
        "long field descriptions. The Self-Refine refinement prompt shows a placeholder where the model's feedback goes. "
        "The system and user messages are passed to each model's chat template. Map-Reflect uses the reflection with "
        "symmetric evidence and the either-order verifier below; the Margin-Reflect configuration "
        "(Appendix~\\ref{app:ablmr}) uses the neutral reflection with evidence and the pairwise check. The remaining "
        "prompts are the other second-pass reflection strategies we compare.",
        box("First pass: ranking", "app:prompt-map", ranking_prompt()),
        box("Map-Reflect: reflection with symmetric evidence", "app:prompt-v5-reflect", v5_reflect),
        box("Map-Reflect: either-order verifier, run with both option orders", "app:prompt-v5-verify-1", v5_c1)
        .replace("\\label{app:prompt-v5-verify-1}", "\\label{app:prompt-v5-verify-1}\\label{app:prompt-v5-verify-2}"),
        "The verifier is called twice, once as shown, with the first answer as option 1, and once with options 1 and 2 "
        "swapped, so that the proposal is option 1; the two calls are otherwise identical.",
        "Map-Reflect's reflection and verifier prompts show each field's own definition and ETL notes first, then "
        "each table's shared description once, in full. In the worked example each description is one short sentence "
        "with no table text. On MIMIC-OMOP, where descriptions are long, the verifier for "
        "\\texttt{CHARTEVENTS.CHARTTIME}, with the first answer \\texttt{" + ff1_base.replace("_", "\\_")
        + "} as option 1, reads:",
        box("Map-Reflect with full field descriptions: MIMIC-OMOP verifier", "app:prompt-v5-ff1", ff1),
        box("Margin-Reflect configuration: reflection, neutral + evidence", "app:prompt-b", evidence),
        box("Margin-Reflect configuration: pairwise check", "app:prompt-verify-b", ver_e),
        box("Strategy: doubt, a deliberately change-seeking control", "app:prompt-naive", naive),
        box("Strategy: Self-Refine (feedback)", "app:prompt-sr-feedback", fb),
        box("Strategy: Self-Refine (refine)", "app:prompt-sr-refine", refine),
        box("Strategy: neutral", "app:prompt-a", neutral),
        box("Strategy: check after neutral", "app:prompt-verify", ver),
    ]
    text = "\n\n".join(parts) + "\n"
    assert "\u2014" not in text, "an em-dash reached the appendix"
    out = ROOT / "paper" / "appendix_prompts.tex"
    out.write_text(text)
    print(out)


if __name__ == "__main__":
    main()
