"""The split behind every branch or school ASSIGNED from outside the source (spec §2.6a, §2.6b, §7e).

islam_assign.py and buddhism_assign.py hold the tables (which countries, which branch, how much
is left unassigned, which units are left whole) and their reasons; this file holds the one
operation they share, so the two cannot drift apart in how a row is split or tagged.

`split()` takes a country's counts() frame and divides every row on `parent` (bare `islam`,
bare `buddhism`) into `node` and a remainder that stays on `parent`:

* the remainder is `share` of the row, or the unit's own share from `unit_share`;
* a unit in `whole` is left entirely on `parent`;
* a `measured` row's assigned half becomes `assigned` with `roll=parent`, so `inferred dots:
  not shown` puts the source's own count back (§7e); the remainder stays `measured`;
* a `modelled` or `derived` row keeps its tier on both halves;
* the assigned half never rings (`may_ring` False), as a derived count never does (§3.10).

Every unit's people on `parent` plus `node` are conserved, and `split` stops if they are not.
"""

import numpy as np
import pandas as pd


def split(cc, df, parent, node, share, unit_share=None, whole=()):
    unit_share = unit_share or {}
    whole = set(whole)
    if df.empty:
        return df
    df = df.copy()
    if "tier" not in df.columns:
        df["tier"] = "measured"
    df["tier"] = df["tier"].fillna("measured")
    is_int = pd.api.types.is_integer_dtype(df["count"])

    units = df["unit"].astype(str)
    take = (df["node"] == parent) & ~units.isin(whole)
    if not take.any():
        return df
    before = float(df.loc[df["node"].isin([parent, node]), "count"].sum())
    rem = units[take].map(lambda u: unit_share.get(u, share)).astype(float)
    base = df.loc[take, "count"].astype(float)
    assigned = base * (1.0 - rem)
    if is_int:
        assigned = np.rint(assigned).astype("int64")

    new = df.loc[take].copy()
    new["node"] = node
    new["count"] = assigned.values
    if "congregations" in new.columns:
        new["congregations"] = 0
    measured = new["tier"] == "measured"
    new.loc[measured, "tier"] = "assigned"
    if measured.any() or "roll" in df.columns:
        if "roll" not in df.columns:
            df["roll"] = None
            new["roll"] = None
        new.loc[measured, "roll"] = parent
    if "may_ring" in df.columns or measured.any():
        if "may_ring" not in df.columns:
            df["may_ring"] = True        # scatter.py's default for an adapter that says nothing
        new["may_ring"] = False

    if is_int:
        df.loc[take, "count"] = df.loc[take, "count"].values - assigned.values
    else:
        df.loc[take, "count"] = (base - assigned).values
    out = pd.concat([df, new], ignore_index=True)
    out = out[out["count"] > 0].reset_index(drop=True)
    after = float(out.loc[out["node"].isin([parent, node]), "count"].sum())
    if abs(after - before) > 1e-6 * max(before, 1) + 1:
        raise SystemExit(f"{cc}: assigning {parent} -> {node} did not conserve people "
                         f"({after:,.0f} after against {before:,.0f} before)")
    return out
