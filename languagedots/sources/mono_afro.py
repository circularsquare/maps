"""Afrobarometer home-language check for the near-monolingual countries (rw, bi, ls, sz, km).

    python sources/mono_afro.py Burundi Lesotho Eswatini ...

Not a build input. These countries are drawn on their national language off population alone
(sources/<cc>.md); this prints the weighted home-language shares per round, pooled, and by
REGION, from religiondots' six merged rounds (read-only), as a check on that national share and
a search for any minority big enough to draw. Same rounds and columns as sources/ng_afro.py.
Country names are matched case-insensitively; "Eswatini" also matches "Swaziland".
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import pyreadstat

HERE = Path(__file__).resolve().parent.parent
AB_DIR = HERE.parent / "religiondots" / "data" / "raw" / "afrobarometer"
ROUNDS = [
    (4, "merged_r4_data.sav", "Q3", "Withinwt"),
    (5, "merged-round-5-data-34-countries-2011-2013-last-update-july-2015_0.sav", "Q2",
     "withinwt"),
    (6, "merged_r6_data_2016_36countries2.sav", "Q2", "withinwt"),
    (7, "r7_merged_data_34ctry.release.sav", "Q2B", "withinwt"),
    (8, "afrobarometer_release-dataset_merge-34ctry_r8_en_2023-03-01.sav", "Q2", "withinwt_hh"),
    (9, "R9.Merge_39ctry.20Nov23.final_.release_Updated.4Jun25-3.sav", "Q2", "withinwt_hh"),
]
ALIAS = {"eswatini": {"eswatini", "swaziland"}}


def _read(path, cols=None, meta_only=False):
    for enc in (None, "LATIN1"):
        try:
            kw = {"encoding": enc} if enc else {}
            if meta_only:
                return pyreadstat.read_sav(str(path), metadataonly=True, **kw)
            return pyreadstat.read_sav(str(path), usecols=cols, **kw)
        except Exception:  # noqa: BLE001
            if enc:
                raise


def extract(aliases, r7q="Q2B"):
    """[round, region, lang, w] for one country, every round it is in. `aliases`: casefolded
    country labels (a set). `r7q`: R7's language column; "Q2A" takes R7's mother-tongue
    question (ask 018's ruling) instead of its "language spoken in home"."""
    rows = []
    for rnd, fn, q, wt in ROUNDS:
        if rnd == 7:
            q = r7q
        p = AB_DIR / fn
        _, meta = _read(p, meta_only=True)
        up = {c.upper(): c for c in meta.column_names}
        cols = [up["COUNTRY"], up["REGION"], up[q.upper()], up[wt.upper()]]
        df, meta = _read(p, cols)
        clab = meta.variable_value_labels.get(up["COUNTRY"], {})
        cn = df[up["COUNTRY"]].map(clab).astype(str).str.strip().str.casefold()
        sub = df[cn.isin(aliases)]
        if not len(sub):
            continue
        w = pd.to_numeric(sub[up[wt.upper()]], errors="coerce").fillna(0)
        assert 0.98 <= w.sum() / len(sub) <= 1.02, (rnd, w.sum() / len(sub))
        lang = sub[up[q.upper()]].map(meta.variable_value_labels.get(up[q.upper()], {}))
        assert lang.notna().all(), f"R{rnd}: an answer code has no label"
        rows.append(pd.DataFrame({
            "round": rnd,
            "region": sub[up["REGION"]].map(meta.variable_value_labels.get(up["REGION"], {})),
            "lang": lang.astype(str).str.strip(), "w": w}))
    return pd.concat(rows, ignore_index=True)


def gkey(s):
    """A REGION label as bare lower-case letters (Mohale's Hoek / Mohale’s Hoek agree)."""
    return "".join(ch for ch in str(s).casefold() if ch.isalpha())


def unit_counts(a, norm, labels, pop, source_id, year, geo_level, lf=None, lf_rounds=(7,)):
    """Pooled weighted shares per unit x the unit's population -> normalized rows.

    a: extract() output; norm: gkey(REGION) -> unit id; labels: survey answer -> source_category
    (None drops the answer, e.g. a refusal); pop: unit -> (name, population).
    lf: lingua-franca categories (ask 018's ruling) whose share in every unit is their NATIONAL
    weighted share in lf_rounds (R7's mother tongue, with extract(r7q="Q2A")); every other
    category keeps its pooled per-unit share among the non-lingua-franca answers, scaled to what
    the lingua francas leave."""
    a = a.copy()
    a["unit"] = a["region"].map(gkey).map(norm)
    bad = sorted(a.loc[a["unit"].isna(), "region"].unique())
    assert not bad, f"REGION labels with no unit: {bad}"
    miss = sorted(set(a["lang"]) - set(labels))
    assert not miss, f"answers with no label mapping: {miss}"
    a["cat"] = a["lang"].map(labels)
    a = a[a["cat"].notna()]
    assert set(a["unit"]) == set(pop), (set(pop) ^ set(a["unit"]))
    if lf:
        r = a[a["round"].isin(lf_rounds)]
        assert len(r), f"no respondents in rounds {lf_rounds}"
        nat = {c: r.loc[r["cat"] == c, "w"].sum() / r["w"].sum() for c in lf}
        rest = a[~a["cat"].isin(lf)]
        sh = rest.groupby(["unit", "cat"])["w"].sum()
        sh = sh / sh.groupby(level=0).transform("sum") * (1 - sum(nat.values()))
        add = pd.Series({(u, c): v for u in pop for c, v in nat.items() if v > 0}, dtype=float)
        sh = pd.concat([sh, add]).sort_index()
        sh.index = pd.MultiIndex.from_tuples(sh.index)
        print("  lingua francas at their R" + "/R".join(map(str, lf_rounds)) + " national share: "
              + ", ".join(f"{c} {v:.3%}" for c, v in nat.items()))
    else:
        sh = a.groupby(["unit", "cat"])["w"].sum()
        sh = sh / sh.groupby(level=0).transform("sum")
    out = []
    for (u, cat), s in sh.items():
        name, n = pop[u]
        out.append(dict(geo_id=u, geo_level=geo_level, geo_name=name, source_category=cat,
                        count=s * n, tier="modelled", source_id=source_id, year=year,
                        note=f"{(a['unit'] == u).sum()} respondents"))
    df = pd.DataFrame(out)
    # whole people, largest remainder within each unit, so every unit sums to its population
    df["share"] = df["count"] / df.groupby("geo_id")["count"].transform("sum")
    fl = df["count"].astype(int)
    rem = df["count"] - fl
    df["count"] = fl
    for u, (_, n) in pop.items():
        m = df["geo_id"] == u
        short = int(n) - int(df.loc[m, "count"].sum())
        idx = rem[m].sort_values(ascending=False).index[:short]
        df.loc[idx, "count"] += 1
        assert int(df.loc[m, "count"].sum()) == int(n), u
    df["note"] = "share " + df["share"].round(4).astype(str) + "; " + df["note"]
    return df.drop(columns="share")


def main(names):
    want = {n.casefold(): ALIAS.get(n.casefold(), {n.casefold()}) for n in names}
    rows = []
    for key, al in want.items():
        try:
            rows.append(extract(al).assign(country=key))
        except ValueError:
            print(f"{key}: in no round")
    a = pd.concat(rows, ignore_index=True)
    for key in want:
        s = a[a["country"] == key]
        print(f"\n===== {key}: {len(s):,} respondents, rounds {sorted(s['round'].unique())}")
        if not len(s):
            continue
        per = s.groupby(["round", "lang"])["w"].sum().unstack(0).fillna(0)
        per = per / per.sum() * 100
        per["pooled"] = s.groupby("lang")["w"].sum() / s["w"].sum() * 100
        n = s.groupby("round").size()
        print("respondents per round:", dict(n))
        print(per.sort_values("pooled", ascending=False).round(2).to_string())
        top = per["pooled"].idxmax()
        reg = s.assign(other=s["lang"] != top).groupby("region").apply(
            lambda g: pd.Series({"n": len(g), "pct_not_" + str(top)[:12]:
                                 100 * g.loc[g["other"], "w"].sum() / g["w"].sum()}))
        print(reg.round(2).to_string())
        oth = s[s["lang"] != top].groupby(["region", "lang"]).size()
        print("non-majority answers by region (unweighted n):")
        print(oth[oth > 0].to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
