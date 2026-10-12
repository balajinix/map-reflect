"""Raw-log fields of treatment ff1 runs (MAPREFLECT_FIELD_FIRST=1): the method v4.1 raw log (MAPREFLECT_RAW_LOG,
<out>.raw.jsonl) extended so that one field's whole path can be rebuilt from it. Off, nothing here runs and the raw
log is as before.

Schema RAW_SCHEMA. Two kinds of line, both keyed by the stable field key.

Call lines (stage reflect:<X> or verify:<X>), the v4.1 fields
    ts, model, run, key, action (reflect_v4) or arm (v6_d3_run), stage, prompt_md5, candidates, response, parse_status
  plus
    raw_schema       RAW_SCHEMA
    treatment        "ff1"
    lookup_sha256    sha256 of the lookup file the prompts were rendered from
    failure          ok | transport (the client returned "Error: ...") | empty (blank response) | parse (reflect:
                     unparsed or invalid; verify: unparsed or no_choice)
    parsed_ranking   reflect: the revised ranking resolved against the shown candidates, [{id, score}] in the order
                     given (unresolvable ids dropped); verify: null
    selected         reflect: the selected answer (reflect_v2.top1: first listed shown candidate; "" for abstain or
                     invalid; null when unparsed); verify: the option the verifier chose, or null
    selected_margin  reflect: the revised margin of the selected answer (reflect_v3.revised); verify: null
    choice           verify: 1, 2 or null; reflect: null
    calls            provider calls behind this line (usage records the client wrote; 0 when none reached it)
    prompt_tokens, completion_tokens   the provider's actual counts for those calls (null when the client reported
                     none)

Decision lines (stage decision:<X>), one per action or arm run on a field
    raw_schema, treatment, lookup_sha256, ts, model, run, key, action or arm, stage
    base, base_margin, valid  the first answer and its protocol v2 margin
    band, router_action       the router's decision for the field and the action it picks (null when the run has no
                              frozen thresholds)
    proposed                  the reflection's answer after the pool rule (null when no call was made)
    selected_margin           its revised margin
    reflect_failure, verify_failure   the failure category of each call (null when not called)
    verifier_decision         accept (the check confirmed the change) | reject | null (no check ran)
    final                     the answer the action leaves
    calls, prompt_tokens, completion_tokens   totals over the action's calls
"""
RAW_SCHEMA = "ff1-raw-v1"


def failure(resp, status):
    if isinstance(resp, str) and resp.startswith("Error:"):
        return "transport"
    if not (resp or "").strip():
        return "empty"
    if status in ("unparsed", "invalid", "no_choice"):
        return "parse"
    return "ok"


def usage_totals(sink):
    calls = len(sink)
    if not calls:
        return dict(calls=0, prompt_tokens=None, completion_tokens=None)
    return dict(calls=calls, prompt_tokens=sum(u.get("prompt_tokens", 0) for u in sink),
                completion_tokens=sum(u.get("completion_tokens", 0) for u in sink))


def parsed_ranking(resp, cands):
    import reflect_v2 as R2
    from map_reflect.map_reflect_runner import resolve_ranked_id, safe_parse_json_array, safe_parse_json_object
    obj = safe_parse_json_object(resp or "", want="revised_ranking")
    arr = obj.get("revised_ranking") if isinstance(obj, dict) else None
    if not arr:
        arr = safe_parse_json_array(resp or "") or []
    out = []
    for it in arr if isinstance(arr, list) else []:
        if isinstance(it, dict) and "id" in it:
            s = str(it["id"])
            rid = (list(cands)[int(s) - 1] if s.isdigit() and 1 <= int(s) <= len(cands)
                   else resolve_ranked_id(s, list(cands)))
            if rid:
                out.append(dict(id=rid, score=R2._num(it.get("score")) if R2._num(it.get("score")) != float("-inf")
                                else None))
    return out


def call_fields(resp, kind, status, cands, sink, lookup_sha):
    """The ff1 fields of one call line."""
    import reflect_v2 as R2
    from reflect_v3 import revised
    f = dict(raw_schema=RAW_SCHEMA, treatment="ff1", lookup_sha256=lookup_sha, failure=failure(resp, status),
             parsed_ranking=None, selected=None, selected_margin=None, choice=None, **usage_totals(sink))
    if kind == "reflect":
        ans, rm, st = revised(resp, list(cands or []))
        f.update(parsed_ranking=parsed_ranking(resp, cands or []), selected=ans, selected_margin=rm)
    else:
        c = R2.choice(resp)
        f.update(choice=c, selected={1: cands[0], 2: cands[1]}.get(c) if cands else None)
    return f


def decision_fields(rec, info, lookup_sha, proposed, rmargin, check, final):
    """The ff1 decision line of one action or arm on one field (info: what the action's calls recorded)."""
    route = info.get("route") or (None, None)
    return dict(raw_schema=RAW_SCHEMA, treatment="ff1", lookup_sha256=lookup_sha, base=rec["base"] or "",
                base_margin=rec["margin"], valid=rec["valid"], band=route[0], router_action=route[1],
                proposed=proposed, selected_margin=rmargin, reflect_failure=info.get("reflect_failure"),
                verify_failure=info.get("verify_failure"),
                verifier_decision=None if check is None else ("accept" if check else "reject"), final=final,
                **usage_totals(info.get("usage", [])))
