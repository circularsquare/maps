"""Split Latin America's unspecified Protestant / Evangelical bucket with Pew 2014's national shares.

Called at the end of each country's counts() as `return latam_split.split("<cc>", df)`. The shares
come from data/normalized/pew_latam.csv, which sources/pew_latam.py builds from Pew Research
Center's *Religion in Latin America* (2014) microdata. The research behind it is
unspecified_splits/latam.md.

ANITA'S RULING, 2026-10-16:
  * Pew 2014 national shares, applied INSIDE the counted `christianity.protestant` /
    `christianity.evangelical` rows, in every country of the proposal except Brazil. Brazil's 2010
    census already counts as many Pentecostals as Pew implies (28.7M against 28.3M), so its 9.2M
    `Evangélica não determinada` are people who named no church, and they stay where they are.
  * Pentecostal everywhere; Baptist only in El Salvador and Panama, where Pew finds it large
    (17.2% and 12.1% of Protestants). Nothing else, and no Trinitarian/Oneness split.
  * What Pew could not place ("another church", "just a Protestant", don't know) stays on the
    bucket node. It does not go to Non-denominational.
  * Argentina is drawn on Pew although the Global Flourishing Study disagrees (sources/ar.md).

THE RULE. U is the country's Protestant universe as drawn: every `christianity.*` row except
Catholic, Orthodox, Oriental, Jehovah's Witnesses, Latter-day Saints and `christianity.other`
(Pew counts Witnesses and Mormons outside "Protestant" too). For each family,

    add_f = max(0, Pew share_f x U - people the source already names in family f)

so a source that already counts Pentecostals (Mexico, Paraguay) gives up only what Pew finds on
top of them. The adds become ONE NATIONAL FRACTION of the bucket, applied to every bucket row
alike: the bucket's own counted geography carries the pattern, and nothing claims to know where
inside it the Pentecostals are, because Pew's public file has no geography. If the adds exceed
the bucket they are scaled to it.

THE TIER (spec §2.7a, §7). A split row takes its bucket row's tier, except that `measured`
becomes `derived`: a survey share inside a counted column is derived, not modelled, so
`inferred dots: not shown` must put those people back rather than delete them. The split row's
`roll` is the bucket row's own `roll` where it had one (Mexico's allocated rows) and the bucket
node otherwise. In a country that is modelled throughout (the LAPOP ones) the split rows stay
`modelled` and leave with everything else. A split row never rings (§3.10).

THE CHECKS, every run, each a SystemExit: per unit, the bucket plus its split rows equals the
bucket before the split; no count is negative; the country total does not move; the adds never
exceed the bucket. And where a source itself names a family Pew also measures, both figures are
printed (Mexico Presbyterian, Peru Adventist): printed rather than asserted, because Pew has 116
to 638 Protestants a country and a gap there is information, not a failure.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TABLE = HERE / "data" / "normalized" / "pew_latam.csv"

BUCKET = ("christianity.protestant", "christianity.evangelical")
OUTSIDE_U = ("christianity.catholic", "christianity.orthodox", "christianity.oriental",
             "christianity.witnesses", "christianity.latterday", "christianity.other")
FAMILY_NODE = {"pentecostal": "christianity.pentecostal", "baptist": "christianity.baptist",
               "adventist": "christianity.adventist",
               "presbyterian": "christianity.reformed.presbyterian",
               "methodist": "christianity.methodist"}

# Anita, 2026-10-16. Brazil is deliberately absent.
FAMILIES = {cc: ("pentecostal",) for cc in
            ("ar", "bo", "cl", "co", "cr", "do", "ec", "gt", "hn", "mx", "ni", "pe", "pr", "py",
             "uy", "ve")}
FAMILIES["sv"] = ("pentecostal", "baptist")
FAMILIES["pa"] = ("pentecostal", "baptist")

# Families printed against the source's own named row, where both exist.
CROSSCHECK = {"mx": ("presbyterian", "adventist", "baptist"), "pe": ("adventist",),
              "do": ("adventist",)}

TOL = 1e-6


def table():
    return pd.read_csv(TABLE).set_index("cc")


def _in_u(node):
    return node.startswith("christianity.") and not node.startswith(OUTSIDE_U)


def _named(df, fam):
    node = FAMILY_NODE[fam]
    m = (df["node"] == node) | df["node"].str.startswith(node + ".")
    return float(df.loc[m & ~df["node"].isin(BUCKET), "count"].sum())


def universe(df):
    return float(df.loc[df["node"].map(_in_u), "count"].sum())


def crosscheck(cc, df, t=None):
    """Print Pew's implied figure beside the source's own, for the families both measure."""
    t = table() if t is None else t
    U = universe(df)
    for fam in CROSSCHECK.get(cc, ()):
        s = t.loc[cc, fam]
        if pd.isna(s):
            continue
        print(f"  pew_latam {cc}: {fam:<12} Pew {s * U:>12,.0f}   source names "
              f"{_named(df, fam):>12,.0f}   (Pew {100 * s:.1f}% of {U:,.0f})")


def split(cc, df):
    """`df` is the country's counts() frame; returns it with the bucket rows split."""
    t = table()
    if cc not in FAMILIES:
        raise SystemExit(f"pew_latam: no ruling for {cc}")
    df = df.copy().reset_index(drop=True)
    if "tier" not in df.columns:
        df["tier"] = "measured"
    df["tier"] = df["tier"].fillna("measured")
    if "roll" not in df.columns:
        df["roll"] = None
    if "may_ring" not in df.columns:
        df["may_ring"] = df["tier"] == "measured"

    U = universe(df)
    is_b = df["node"].isin(BUCKET)
    B = float(df.loc[is_b, "count"].sum())
    if B <= 0:
        raise SystemExit(f"pew_latam {cc}: no bucket rows")

    add = {}
    for fam in FAMILIES[cc]:
        s = t.loc[cc, fam]
        if pd.isna(s):
            raise SystemExit(f"pew_latam {cc}: {fam} is not on Pew's card for this country")
        add[fam] = max(0.0, float(s) * U - _named(df, fam))
    tot = sum(add.values())
    scale = min(1.0, B / tot) if tot > 0 else 1.0
    frac = {fam: a * scale / B for fam, a in add.items()}
    keep = 1.0 - sum(frac.values())
    if keep < -TOL:
        raise SystemExit(f"pew_latam {cc}: the splits exceed the bucket ({keep:.6f})")

    before = df.loc[is_b].groupby("unit")["count"].sum()
    bucket = df.loc[is_b].copy()
    parts = [bucket.assign(count=bucket["count"] * keep)]
    for fam, f in frac.items():
        r = bucket.copy()
        r["roll"] = [rl if isinstance(rl, str) and rl else nd
                     for rl, nd in zip(r["roll"], r["node"])]
        r["node"] = FAMILY_NODE[fam]
        r["count"] = r["count"] * f
        r["tier"] = r["tier"].replace({"measured": "derived"})
        r["may_ring"] = False
        parts.append(r)
    from_b = pd.concat(parts, ignore_index=True)
    after = from_b.groupby("unit")["count"].sum().reindex(before.index, fill_value=0)
    if (after - before).abs().max() > TOL * max(1.0, float(before.max())):
        raise SystemExit(f"pew_latam {cc}: a unit's bucket moved")
    res = pd.concat([df.loc[~is_b], from_b], ignore_index=True)
    if (res["count"] < -TOL).any():
        raise SystemExit(f"pew_latam {cc}: negative counts")
    if abs(res["count"].sum() - df["count"].sum()) > TOL * df["count"].sum():
        raise SystemExit(f"pew_latam {cc}: the country total moved")

    desc = ", ".join(f"{fam} {frac[fam] * B:,.0f} ({100 * frac[fam]:.1f}%)" for fam in frac)
    print(f"  pew_latam {cc}: bucket {B:,.0f} of Protestant universe {U:,.0f}; {desc}; "
          f"{keep * B:,.0f} ({100 * keep:.1f}%) stays on the bucket"
          + (f"; scaled x{scale:.3f}" if scale < 1 else ""))
    crosscheck(cc, df, t)
    return res[res["count"] > 0].reset_index(drop=True)
