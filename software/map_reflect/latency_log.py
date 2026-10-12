"""Per-call latency log for the LLM clients in map_reflect_runner (EndpointBBatchClient, EndpointABatchClient).

Off unless MAPREFLECT_LATENCY_LOG names a jsonl file. When it does, every LLM call appends one record:

  t_start, t_end, seconds   wall clock (epoch seconds) from the first attempt to the returned response, retries and
                            any throttle wait included; throttle_wait_s is the part spent waiting on the shared
                            endpoint_b token bucket (scripts/naacl/v6_calls/endpoint_b_throttle.py), so the net is
                            seconds - throttle_wait_s
  endpoint, model           the base URL the request went to and the model id
  prompt_tokens, completion_tokens   as the endpoint reports them
  call_type                 first_pass, reflection or verifier, from the usage stage (map_reflect_runner
                            set_usage_stage / TOKEN_STAGE): reflect* is reflection, verify* is verifier, anything
                            else is first_pass
  stage, key, action        the usage stage and the caller's usage context (v5_run.py sets key and action); the
                            first pass has no per-call context, so the runner registers each ranking prompt's field
                            key (register_prompt) and the record carries it
  basis                     "call" when the record times exactly one request; "batch" when a endpoint_b batch of n
                            prompts went out together (then t_start/t_end are the batch's and n_batch says how many)
  attempts, error           retries taken (ENDPOINT_A) and whether the call ended in an error

Nothing here changes a request or a response. With the variable unset, enabled() is False and the clients behave as
before, byte for byte.
"""
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Optional

ENV = "MAPREFLECT_LATENCY_LOG"
_lock = threading.Lock()
_wait = 0.0
_prompt_keys: dict = {}


def enabled() -> bool:
    return bool(os.environ.get(ENV))


def call_type(stage: Optional[str]) -> str:
    s = str(stage or "")
    if s.startswith("reflect"):
        return "reflection"
    if s.startswith("verify"):
        return "verifier"
    return "first_pass"


def register_prompt(prompt: str, key: str) -> None:
    """Remember which field a batched prompt belongs to (the runner's first pass sends many prompts in one batch)."""
    _prompt_keys[hashlib.md5(prompt.encode()).hexdigest()] = key


def note_throttle_wait(seconds: float) -> None:
    """Called by the endpoint_b throttle after it slept, so the current call's record can report the wait. The SDK sends
    requests from its own worker threads, so the wait is kept per process: exact when one call is in flight per
    process (the latency measurement's setting), an upper bound shared across calls otherwise."""
    global _wait
    with _lock:
        _wait += float(seconds)


def take_throttle_wait() -> float:
    global _wait
    with _lock:
        w, _wait = _wait, 0.0
    return w


def log_call(*, t_start: float, t_end: float, endpoint: str, model: str, prompt: str, prompt_tokens=None,
             completion_tokens=None, stage: Optional[str] = None, context: Optional[dict] = None,
             basis: str = "call", n_batch: int = 1, attempts: int = 1, error: bool = False,
             throttle_wait_s: float = 0.0) -> None:
    path = os.environ.get(ENV)
    if not path:
        return
    md5 = hashlib.md5(prompt.encode()).hexdigest()
    ctx = {k: v for k, v in (context or {}).items() if not k.startswith("_")}
    rec = dict(t_start=round(t_start, 4), t_end=round(t_end, 4), seconds=round(t_end - t_start, 4),
               throttle_wait_s=round(throttle_wait_s, 4), endpoint=endpoint, model=model,
               prompt_tokens=None if prompt_tokens is None else int(prompt_tokens),
               completion_tokens=None if completion_tokens is None else int(completion_tokens),
               call_type=call_type(stage), stage=stage or "", key=ctx.get("key") or _prompt_keys.get(md5),
               action=ctx.get("action"), prompt_md5=md5, basis=basis, n_batch=n_batch, attempts=attempts,
               error=bool(error), pid=os.getpid(), run=os.environ.get("TOKEN_RUN", ""))
    with _lock:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
