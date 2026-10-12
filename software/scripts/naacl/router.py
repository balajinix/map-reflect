"""The one routing rule of Map-Reflect (method v3 and v4). Fitting, running and every analysis import it.

A field's first pass gives three free signals: the margin m = s1 - s2 of its score-sorted ranking (never negative; 0
for a declined or invalid ranking), the top score s1, and whether the ranking is valid (names a shown candidate).
With a model's thresholds t_m and t_s the field falls in exactly one band, decided in this order:

  1. keep       valid and m > t_m          keep the first answer, no call
  2. near_tie   valid and s1 >= t_s        the near-tie action over the shown candidates
  3. widen      everything else            the widen action over a wider candidate pool
                (m <= t_m with s1 < t_s, or an invalid first pass at any m)

The order matters: a field with m > t_m is kept whatever its s1, and an invalid first pass always widens. Validity
is part of rule 1 so that the rule holds for any t_m; with m = 0 for invalid rankings and t_m >= 0 it changes nothing.

action(...) names what a field would run if it were routed: the near-tie action when it is valid with s1 >= t_s,
the widen action otherwise. For method v3 these are BC and W<K>; method v4 passes its own names.

Canonical margins (method v4.1). The margin is a difference of decimal scores computed in binary floating point,
so a mathematically equal margin can land a hair above the threshold (0.85 - 0.70 = 0.15000000000000002 > 0.15) and
be kept when the rule says route. With MAPREFLECT_CANONICAL_MARGIN=1 in the environment, margins are rounded to 12
decimals and a margin or top score within 1e-12 of its threshold counts as equal to it. The default stays off so
that method v4 and every v4 analysis reproduce exactly; method v4.1 turns it on (docs/naacl/v6/reviews/
opendata_uq_audit/REPORT.md).
"""
import math
import os

import numpy as np

CANONICAL = os.environ.get("MAPREFLECT_CANONICAL_MARGIN") == "1"
TOL = 1e-12


def canonical_margin(m):
    """The margin rounded to 12 decimals (scalar or array); the identity when canonical mode is off."""
    if not CANONICAL:
        return m
    return np.round(np.asarray(m, float), 12) if isinstance(m, (list, tuple, np.ndarray)) else round(float(m), 12)

KEEP, NEAR_TIE, WIDEN = "keep", "near_tie", "widen"
BANDS = (KEEP, NEAR_TIE, WIDEN)
LEGACY = {"confident": KEEP}  # the name stored v3 rows use for KEEP


def canon(name):
    """The router's name for a stored band label."""
    return LEGACY.get(name, name)


def band(margin, valid, s1, t_m, t_s):
    """The routing band of one field (see the module docstring for the order)."""
    if CANONICAL:
        margin = canonical_margin(margin)
        if valid and margin > t_m and not math.isclose(margin, t_m, rel_tol=0.0, abs_tol=TOL):
            return KEEP
        if valid and (s1 >= t_s or math.isclose(s1, t_s, rel_tol=0.0, abs_tol=TOL)):
            return NEAR_TIE
        return WIDEN
    if valid and margin > t_m:
        return KEEP
    if valid and s1 >= t_s:
        return NEAR_TIE
    return WIDEN


def action(valid, s1, t_s, near_action, widen_action):
    """The action the field would run if routed, whatever its margin."""
    near = s1 >= t_s or (CANONICAL and math.isclose(s1, t_s, rel_tol=0.0, abs_tol=TOL))
    return near_action if (valid and near) else widen_action


def v3_actions(t):
    """Method v3's action names for a frozen threshold record {t_m, t_s, K, ...}."""
    return "BC", f"W{t['K']}"


def route(margin, valid, s1, t, actions=None):
    """(band, action) of one field under a frozen threshold record t = {t_m, t_s, K, ...}. actions is the
    (near-tie, widen) pair of names, v3's BC and W<K> by default."""
    near_a, widen_a = actions or v3_actions(t)
    return band(margin, valid, s1, t["t_m"], t["t_s"]), action(valid, s1, t["t_s"], near_a, widen_a)


def bands(margin, valid, s1, t_m, t_s):
    """Vectorised band over numpy arrays: an array of KEEP, NEAR_TIE and WIDEN, identical to band() element-wise."""
    margin, s1 = np.asarray(margin, float), np.asarray(s1, float)
    valid = np.asarray(valid, bool)
    if CANONICAL:
        margin = canonical_margin(margin)
        keep = valid & (margin > t_m) & ~np.isclose(margin, t_m, rtol=0.0, atol=TOL)
        near = ~keep & valid & ((s1 >= t_s) | np.isclose(s1, t_s, rtol=0.0, atol=TOL))
        return np.where(keep, KEEP, np.where(near, NEAR_TIE, WIDEN))
    keep = valid & (margin > t_m)
    near = ~keep & valid & (s1 >= t_s)
    return np.where(keep, KEEP, np.where(near, NEAR_TIE, WIDEN))


def routed(margin, valid, s1, t_m, t_s):
    """Vectorised: True where the field is routed (near_tie or widen)."""
    return bands(margin, valid, s1, t_m, t_s) != KEEP
