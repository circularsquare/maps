"""United Kingdom: three censuses' main-language tables -> data/normalized/uk.csv (the tables as
published) and data/normalized/uk_units.csv (one row per drawn unit and category, the input to
countries/uk.py).

    python sources/uk_fetch.py      first; downloads everything into data/raw/uk/
    python sources/uk_census.py

All three ask the same question, "What is your main language?", of everyone aged 3 and over;
under-3s are coded "Does not apply" / "No code required" and are not drawn (`gap`).

ENGLAND AND WALES (ONS, 21 March 2021). The trade sources/uk.md describes: fine categories or
fine geography, never both (religiondots spec §3.9).
  * main_language_detailed_26a at all 188,880 Output Areas: English, French, Portuguese,
    Spanish, Polish, Russian, Turkish, Arabic, Panjabi, Urdu, Bengali, Gujarati, Tamil, BSL and
    two other sign categories by name, and ten residual groups ("Other European language (EU):
    Any other European languages", "African languages", ...).
  * TS024, 95 categories, at 331 local authorities (Dec 2021 boundaries).
  Each OA's residual group is shared among TS024's members of that group in the proportions
  of the OA's local authority (religiondots spec §3.10, a within-group proportional
  allocation), tier `derived`; the named 26a categories stay `measured`. A group with OA people
  but none in its district's TS024 (cell-key perturbation) takes England and Wales's mix.
  WALES: the form offered one tick box, "English or Welsh", so the census cannot tell English
  from Welsh as a main language there; ONS prints it as "English (English or Welsh in Wales)".
  Anita's ruling on ask 004 (2026-10-04): split it. split_wales() does, in each Welsh OA, by
  the same census's "can you speak Welsh?" at that OA, tier `derived` (docstring there).

SCOTLAND (NRS, 20 March 2022). UV212 at 46,363 Output Areas: English, Scots, Gaelic, Sign
Language, Other language. NRS codes 22 categories but publishes no finer table at any
geography (sources/uk.md). "Other language" (272,820) is split into languages by country of
birth, fitted to a national estimate (sources/uk_scot_other.py, Anita 2026-10-06), tier
`derived`; every OA keeps its total.

NORTHERN IRELAND (NISRA, 21 March 2021). MAIN_LANGUAGE_1000 at 3,780 Data Zones from the
Flexible Table Builder: English, Irish and 17 other languages by name, and "Other languages".
MS-B13 lists 81 more languages for Northern Ireland as a whole; "Other languages" in each Data
Zone is shared among them in Northern Ireland's proportions, tier `derived`.

Category keys are prefixed with the census ("ew:", "sc:", "ni:"), because the three agencies
print the same word for different things (Scotland's "Gaelic" is Scottish Gaelic; NISRA's
"Gaelic (Not otherwise specified)" is not).
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "uk"
NORM = HERE / "data" / "normalized"
OUT = NORM / "uk.csv"
UNITS = NORM / "uk_units.csv"
P = "Main language (detailed): "

# 26a category -> the TS024 leaves it holds. A one-member group is a named category, measured
# at OA. Checked against TS024 at every district below.
EU_NAMED = {"French", "Portuguese", "Spanish"}
SA_NAMED = {"Urdu", "Panjabi", "Bengali (with Sylheti and Chatgaya)", "Gujarati", "Tamil"}


def ts024_group(leaf):
    """Which 26a category a TS024 leaf belongs to."""
    head, _, tail = leaf.partition(": ")
    if leaf == "English (English or Welsh in Wales)":
        return "English (English or Welsh if in Wales)"
    if leaf == "Welsh or Cymraeg (in England only)" or head == "Other UK language":
        return "Any other UK languages"
    if leaf in EU_NAMED or leaf in ("Russian", "Turkish", "Arabic"):
        return leaf
    if head == "Other European language (EU)":
        return ("Other European language (EU): Polish" if tail == "Polish"
                else "Other European language (EU): Any other European languages")
    if head in ("Other European language (non EU)", "Other European language (EU and non-EU)",
                "Other European language (non-national)"):
        return "European languages (non-EU)"
    if head == "West or Central Asian language":
        return "West or Central Asian languages"
    if head == "South Asian language":
        return (f"South Asian language: {tail}" if tail in SA_NAMED
                else "South Asian language: Any other South Asian languages")
    if head == "East Asian language":
        return ("East Asian language: Mandarin, Cantonese and other Chinese languages"
                if tail in ("Mandarin Chinese", "Cantonese Chinese", "All other Chinese")
                else "East Asian language: Any other East Asian languages")
    if head == "African language":
        return "African languages"
    if head == "Sign language":
        return {"British Sign Language": "Sign language: British Sign Language",
                "Any other sign language": "Sign language: Any other sign language",
                "Any sign communication system": "Sign language: Any sign communication system"}[tail]
    if leaf in ("Oceanic or Australian language", "North or South American language",
                "Other language") or head == "Caribbean Creole":
        return "Any other languages"
    raise SystemExit(f"TS024 leaf with no 26a group: {leaf}")


def read_ts024(level):
    with zipfile.ZipFile(RAW / "census2021-ts024.zip") as z:
        d = pd.read_csv(io.BytesIO(z.read(f"census2021-ts024-{level}.csv")))
    cats = [c for c in d.columns if c.startswith(P)]
    names = [c[len(P):].strip() for c in cats]
    # TS024 is hierarchical: a header is a leaf unless another header extends it
    leaves = [n for n in names if not any(m.startswith(n + ": ") for m in names)
              and not n.startswith("Total")]
    d = d.rename(columns=dict(zip(cats, names)))
    long = d.melt(id_vars=["geography code"], value_vars=leaves, var_name="category",
                  value_name="count").rename(columns={"geography code": "geo_id"})
    return long, d


def england_wales():
    lu = pd.read_csv(RAW / "oa21_lsoa_msoa_lad21_ew_lu.csv", usecols=["OA21CD", "LAD22CD"])
    oa_lad = dict(zip(lu["OA21CD"], lu["LAD22CD"]))
    oa = pd.read_csv(RAW / "ew_oa_main_language_26a.csv")
    lt = pd.read_csv(RAW / "ew_ltla_main_language_26a.csv")
    ts, ts_wide = read_ts024("ltla")
    ts_ctry, _ = read_ts024("ctry")

    # ---- checks on the download ----
    n_oa = oa["oa"].nunique()
    assert n_oa == 188_880 == len(oa_lad), (n_oa, len(oa_lad))
    assert set(oa["oa"]) == set(oa_lad), "OA download and lookup disagree"
    assert len(oa) == n_oa * 26 and not oa.duplicated(["oa", "cat_id"]).any()
    oa["ltla"] = oa["oa"].map(oa_lad)
    assert set(oa["ltla"]) == set(ts["geo_id"]) == set(lt["ltla"]), "district sets differ"
    print(f"E&W: {n_oa:,} OAs in {oa['ltla'].nunique()} districts; "
          f"{oa['count'].sum():,} people, {oa.loc[oa['category'] != 'Does not apply', 'count'].sum():,} aged 3+")

    # OA sums against the same 26a at district (both perturbed independently)
    a = oa.groupby(["ltla", "category"])["count"].sum()
    b = lt.set_index(["ltla", "category"])["count"]
    diff = (a - b).abs()
    print(f"  OA sums vs 26a at district: total abs diff {diff.sum():,} over {b.sum():,} "
          f"({diff.sum() / b.sum():.4%}); worst cell {diff.max():,} ({diff.idxmax()})")
    assert diff.sum() / b.sum() < 0.002

    # TS024 leaves summed into 26a groups against 26a at district: this is what proves the map
    ts["group"] = ts["category"].map(ts024_group)
    g = ts.groupby(["geo_id", "group"])["count"].sum()
    g.index.names = ["ltla", "category"]
    b2 = b.drop("Does not apply", level="category")
    d2 = (g.reindex(b2.index).fillna(0) - b2).abs()
    print(f"  TS024 grouped vs 26a at district: abs diff {d2.sum():,.0f} of {b2.sum():,} "
          f"({d2.sum() / b2.sum():.4%}); worst {d2.max():,.0f} at {d2.idxmax()}")
    assert d2.sum() / b2.sum() < 0.001
    tot = ts_wide.set_index("geography code")["Total: All usual residents aged 3 years and over"]
    s = ts.groupby("geo_id")["count"].sum()
    assert (s - tot).abs().max() == 0, "TS024 leaves do not sum to the district total"

    # ---- allocation ----
    ts_ew = ts_ctry.groupby("category")["count"].sum()       # England + Wales rows
    groups = {}
    for leaf in ts["category"].unique():
        groups.setdefault(ts024_group(leaf), []).append(leaf)
    share = ts.pivot_table(index="geo_id", columns="category", values="count", aggfunc="sum")
    oa = oa[oa["category"] != "Does not apply"]
    oa = oa[oa["count"] > 0]
    rows, fallback = [], {}
    for grp, members in groups.items():
        part = oa[oa["category"] == grp]
        if len(members) == 1:
            rows.append(pd.DataFrame({"unit": part["oa"], "category": "ew:" + members[0],
                                      "count": part["count"].astype(float), "tier": "measured"}))
            continue
        sh = share[members]
        tot_g = sh.sum(axis=1)
        nat = ts_ew[members] / ts_ew[members].sum()
        w = sh.div(tot_g.replace(0, float("nan")), axis=0)
        empty = tot_g[tot_g == 0].index
        for e in empty:
            w.loc[e] = nat.values
        fallback[grp] = part[part["ltla"].isin(empty)]["count"].sum()
        wm = w.reindex(part["ltla"]).to_numpy()
        for j, m in enumerate(members):
            c = part["count"].to_numpy() * wm[:, j]
            keep = c > 0
            rows.append(pd.DataFrame({"unit": part["oa"].to_numpy()[keep], "category": "ew:" + m,
                                      "count": c[keep], "tier": "derived"}))
    units = pd.concat(rows, ignore_index=True)
    for grp, n in fallback.items():
        if n:
            print(f"  {n:,} people in '{grp}' sit in districts whose TS024 has none; E&W mix used")
    want = oa["count"].sum()
    got = units["count"].sum()
    assert abs(got - want) < 1, (got, want)
    print(f"  allocated: {got:,.0f} people, {units.loc[units['tier'] == 'derived', 'count'].sum():,.0f} "
          f"derived ({units.loc[units['tier'] == 'derived', 'count'].sum() / got:.2%})")

    units, wales_table = split_wales(units)

    table = pd.concat([
        wales_table,
        oa.assign(geo_level="output_area", source_id="ew_2021_main_language_26a")
          .rename(columns={"oa": "geo_id"})[["geo_id", "geo_level", "category", "count", "source_id"]],
        ts.assign(geo_level="ltla", source_id="ew_2021_ts024")[
            ["geo_id", "geo_level", "category", "count", "source_id"]],
    ])
    return units, table


BOX = "ew:English (English or Welsh in Wales)"
BOX_WELSH = "ew:English or Welsh (Wales): split as Welsh"
BOX_ENGLISH = "ew:English or Welsh (Wales): split as English"


def split_wales(units):
    """Wales's one "English or Welsh" box -> Welsh and English, by Welsh speaking ability.

    Anita's ruling on ask 004 (2026-10-04): split the box somehow; a proxy is allowed. The
    proxy is the same census's welsh_skills_speak ("can you speak Welsh?", TS033's question,
    everyone aged 3 and over, the main-language universe) at the same Output Area. Welsh in an
    OA = min(can speak Welsh, box); English = box - Welsh. That counts every Welsh speaker as
    inside the box, which the cross-tab of the two questions supports where ONS publishes it
    (15 of 22 districts): 99.6% of Welsh speakers there gave "English or Welsh" as main
    language. It is an upper bound on Welsh as a main language: everyone whose main language
    is Welsh can speak it, but not everyone who can speak it uses it most (sources/uk.md §1).
    """
    sp = pd.read_csv(RAW / "ew_oa_welsh_speak.csv")
    assert sp["oa"].nunique() == 10_275 and sp["oa"].str.startswith("W").all()
    sp = sp.pivot_table(index="oa", columns="category", values="count", aggfunc="sum")
    can, cannot = sp["Can speak Welsh"], sp["Cannot speak Welsh"]

    wales = units["unit"].str.startswith("W")
    is_box = wales & (units["category"] == BOX)
    assert (units.loc[is_box, "tier"] == "measured").all()
    box = units[is_box].set_index("unit")["count"]
    assert box.index.is_unique and set(box.index) <= set(sp.index), "box OAs not in the speak table"

    # the speak table's 3+ total against the 26a table's 3+ total in each Welsh OA (both perturbed)
    tot26 = units[wales].groupby("unit")["count"].sum()
    d = ((can + cannot).reindex(tot26.index) - tot26).abs()
    print(f"  Wales: {len(tot26):,} OAs; speak table 3+ vs main language 3+: abs diff "
          f"{d.sum():,.0f} of {tot26.sum():,.0f} ({d.sum() / tot26.sum():.3%}), worst {d.max():.0f}")
    assert d.sum() / tot26.sum() < 0.005

    welsh = pd.concat([can.reindex(box.index).fillna(0), box], axis=1).min(axis=1)
    english = box - welsh
    capped = (can.reindex(box.index).fillna(0) > box).sum()
    # the split sums back to the box in every OA, and neither part is negative
    assert ((welsh + english - box).abs() < 1e-9).all() and (welsh >= 0).all() and (english >= 0).all()
    print(f"  'English or Welsh' {box.sum():,.0f} in {len(box):,} OAs -> Welsh {welsh.sum():,.0f} "
          f"({welsh.sum() / box.sum():.1%}), English {english.sum():,.0f}; {capped} OAs had more "
          f"Welsh speakers than box answers and were capped at the box")

    # the check: the cross-tab of the two questions, where ONS publishes it (15 districts)
    lu = pd.read_csv(RAW / "oa21_lsoa_msoa_lad21_ew_lu.csv", usecols=["OA21CD", "LAD22CD"])
    oa_lad = dict(zip(lu["OA21CD"], lu["LAD22CD"]))
    x = pd.read_csv(RAW / "ew_ltla_main_language_x_welsh_speak.csv")
    x = x[x["ltla"].str.startswith("W")]
    xb = x[(x["language"] == "English or Welsh") & (x["speak"] == "Can speak Welsh")]
    xb = xb.set_index("ltla")["count"]
    xall = x[x["speak"] == "Can speak Welsh"].groupby("ltla")["count"].sum()
    est = welsh.groupby(welsh.index.map(oa_lad)).sum().reindex(xb.index)
    share = (box * can / (can + cannot)).reindex(box.index)
    est_share = share.groupby(share.index.map(oa_lad)).sum().reindex(xb.index)
    print(f"  cross-tab check, {len(xb)} districts: Welsh speakers in the box {xb.sum():,} of "
          f"{xall.sum():,} Welsh speakers ({xb.sum() / xall.sum():.2%}); this split "
          f"{est.sum():,.0f} (abs diff by district {(est - xb).abs().sum():,.0f}, worst "
          f"{(est - xb).abs().max():,.0f}); a proportional share would give {est_share.sum():,.0f} "
          f"(abs diff {(est_share - xb).abs().sum():,.0f})")
    assert len(xb) >= 10 and (est - xb).abs().sum() / xb.sum() < 0.02

    six = pd.read_csv(RAW / "ew_ltla_welsh_skills_6a.csv").groupby("category")["count"].sum()
    print(f"  for comparison, Wales: can speak, read and write Welsh "
          f"{six['Can speak, read and write Welsh']:,} (the stricter measure, not used)")

    split = pd.concat([
        pd.DataFrame({"unit": welsh.index, "category": BOX_WELSH, "count": welsh.values}),
        pd.DataFrame({"unit": english.index, "category": BOX_ENGLISH, "count": english.values}),
    ])
    split = split[split["count"] > 0].assign(tier="derived")
    before = units.groupby("unit")["count"].sum()
    units = pd.concat([units[~is_box], split], ignore_index=True)
    after = units.groupby("unit")["count"].sum().reindex(before.index)
    assert ((after - before).abs() < 1e-6).all(), "the Wales split changed an OA's total"
    assert not ((units["category"] == BOX) & units["unit"].str.startswith("W")).any()

    table = sp.reset_index().melt(id_vars="oa", var_name="category", value_name="count") \
        .rename(columns={"oa": "geo_id"}).assign(geo_level="output_area",
                                                 source_id="ew_2021_welsh_skills_speak")
    return units, table[["geo_id", "geo_level", "category", "count", "source_id"]]


def scotland():
    with zipfile.ZipFile(RAW / "sc_Census-2022-Output-Area-v1.zip") as z:
        lines = z.read("UV212 - Main language.csv").decode("utf-8-sig").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(',"All people aged 3 and over"'))
    r = csv.reader(io.StringIO("\n".join(lines[start:])))
    head = next(r)
    rows = []
    for rec in r:
        if not rec or not rec[0].startswith("S00"):
            continue
        for cat, v in zip(head[1:], rec[1:]):
            rows.append((rec[0], cat, 0 if v.strip() in ("-", "") else int(v)))
    d = pd.DataFrame(rows, columns=["geo_id", "category", "count"])
    assert d["geo_id"].nunique() == 46_363
    tot = d[d["category"] == "All people aged 3 and over"].set_index("geo_id")["count"]
    parts = d[d["category"] != "All people aged 3 and over"]
    s = parts.groupby("geo_id")["count"].sum()
    gap = (s - tot).abs()
    print(f"Scotland: {len(tot):,} OAs; all 3+ {tot.sum():,}, categories sum {s.sum():,} "
          f"(NRS perturbs each cell; abs gap {gap.sum():,}, {gap.sum() / tot.sum():.3%})")
    print("  " + ", ".join(f"{c} {n:,}" for c, n in parts.groupby("category")["count"].sum()
                          .sort_values(ascending=False).items()))
    assert gap.sum() / tot.sum() < 0.01
    parts = parts[parts["count"] > 0]
    units = pd.DataFrame({"unit": parts["geo_id"], "category": "sc:" + parts["category"],
                          "count": parts["count"].astype(float), "tier": "measured"})
    table = d.assign(geo_level="output_area", source_id="sc_2022_uv212")
    return units, table[["geo_id", "geo_level", "category", "count", "source_id"]]


def split_scotland_other(units):
    """Scotland's "Other language" -> languages, by country of birth fitted to a national
    estimate (sources/uk_scot_other.py; Anita, 2026-10-06). Every OA keeps its total; the split
    rows are tier `derived`, keyed "sc:Other language: <2011 census label>"."""
    import uk_scot_other
    is_oth = units["category"] == "sc:Other language"
    oth = units[is_oth].set_index("unit")["count"]
    res = uk_scot_other.split(oth)
    split = pd.DataFrame({"unit": res["unit"], "category": "sc:Other language: " + res["label"],
                          "count": res["count"], "tier": "derived"})
    out = pd.concat([units[~is_oth], split], ignore_index=True)
    a = units.groupby("unit")["count"].sum()
    b = out.groupby("unit")["count"].sum().reindex(a.index)
    assert ((a - b).abs() < 1e-6).all(), "the Other language split changed an OA's total"
    return out


def northern_ireland():
    dz = pd.read_csv(RAW / "ni_main_language_DZ21.csv")
    dz.columns = ["geo_id", "name", "code", "category", "count"]
    lgd = pd.read_csv(RAW / "ni_main_language_LGD14.csv")
    lgd.columns = ["geo_id", "name", "code", "category", "count"]
    assert dz["geo_id"].nunique() == 3_780
    print(f"Northern Ireland: {dz['geo_id'].nunique():,} Data Zones, {dz['count'].sum():,} people "
          f"(NISRA 1,903,175), under-3s {dz.loc[dz['category'] == 'No code required', 'count'].sum():,}")
    a = dz.groupby("category")["count"].sum()
    b = lgd.groupby("category")["count"].sum()
    print(f"  Data Zones vs districts: abs diff {(a - b).abs().sum():,} "
          "(each level is perturbed on its own, so small)")
    assert (a - b).abs().sum() < 1000

    b13 = pd.read_excel(RAW / "ni_census-2021-ms-b13.xlsx", sheet_name="MS-B13", header=None)
    start = b13.index[b13[0] == "All usual residents aged 3 and over"][0]
    b13 = b13.iloc[start + 1:].dropna()
    b13.columns = ["category", "count"]
    b13["category"] = b13["category"].str.replace(r"\s*\[note \d+\]", "", regex=True)
    b13["count"] = b13["count"].astype(int)
    named = set(dz["category"]) - {"Other languages", "No code required"}
    rest = b13[~b13["category"].isin(named)]
    missing = named - set(b13["category"])
    assert not missing, f"Data Zone labels not in MS-B13: {missing}"
    print(f"  'Other languages' {a['Other languages']:,} in Data Zones; MS-B13's other "
          f"{len(rest)} rows sum to {rest['count'].sum():,}")
    assert abs(rest["count"].sum() - a["Other languages"]) < 50

    keep = dz[(dz["category"] != "No code required") & (dz["count"] > 0)]
    meas = keep[keep["category"] != "Other languages"]
    oth = keep[keep["category"] == "Other languages"]
    w = (rest.set_index("category")["count"] / rest["count"].sum())
    der = pd.DataFrame([(u, "ni:" + c, n * s) for u, n in zip(oth["geo_id"], oth["count"])
                        for c, s in w.items()], columns=["unit", "category", "count"])
    der["tier"] = "derived"
    units = pd.concat([
        pd.DataFrame({"unit": meas["geo_id"], "category": "ni:" + meas["category"],
                      "count": meas["count"].astype(float), "tier": "measured"}), der],
        ignore_index=True)
    table = pd.concat([dz.assign(geo_level="data_zone", source_id="ni_2021_main_language_1000"),
                       b13.assign(geo_id="N92000002", geo_level="country", source_id="ni_2021_ms_b13")])
    return units, table[["geo_id", "geo_level", "category", "count", "source_id"]]


def main():
    ew_u, ew_t = england_wales()
    sc_u, sc_t = scotland()
    sc_u = split_scotland_other(sc_u)
    ni_u, ni_t = northern_ireland()
    units = pd.concat([ew_u, sc_u, ni_u], ignore_index=True)
    # Welsh, Gaelic, Scots and Irish in NI at a home or daily-use measure (Anita, 2026-10-06)
    import uk_home_use
    units = uk_home_use.apply(units)
    # three code namespaces (E00/W00, S00, N20) share one unit column; check they are disjoint
    pre = units["unit"].str[:3].unique()
    assert set(pre) == {"E00", "W00", "S00", "N20"}, pre
    table = pd.concat([ew_t, sc_t, ni_t], ignore_index=True)
    NORM.mkdir(parents=True, exist_ok=True)
    table.rename(columns={"category": "source_category"}).to_csv(OUT, index=False)
    units["count"] = units["count"].round(4)
    units.rename(columns={"category": "source_category"}).to_csv(UNITS, index=False)
    print(f"wrote {OUT.name} ({len(table):,} rows) and {UNITS.name} ({len(units):,} rows, "
          f"{units['unit'].nunique():,} units, {units['count'].sum():,.0f} people)")


if __name__ == "__main__":
    sys.exit(main())
