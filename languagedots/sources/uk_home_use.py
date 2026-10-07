"""The UK's indigenous languages at a home or daily-use measure, not main language. Called by
sources/uk_census.py after the Wales and Scotland splits; sources/uk.md §10 has the record.

Anita, 2026-10-06: "home-use rule for uk/ireland would be good". The map's Irish in Ireland is
drawn at the census's daily speakers outside education (countries/ie.py); this brings the UK's
Celtic languages and Scots to the nearest measure each census or survey offers:

  * Scottish Gaelic: 2011 census "Do you use a language other than English at home?", Gaelic
    24,974 (AT_002_2011: "Gaelic (Scottish)" 10,443 + "Gaelic (Not otherwise specified)"
    14,531). Placed on 2022 Gaelic speakers (UV208, the three "speaks" columns) in each OA x the
    2011 share of Gaelic speakers who used it at home in the OA's council group (NRS, Scotland's
    Census 2011 Gaelic Report part 1: Na h-Eileanan Siar 73.7%, Highland 41.5%, Argyll and
    Bute 33.4%, the other 29 councils 23.6%), fitted to 24,974.
  * Scots: 2011 home use, 55,817 (AT_002_2011 "Scots"), placed on 2022 Scots speakers (UV209)
    at one national share; no council breakdown of Scots home use was found.
  * Irish, Northern Ireland: 2021 census "How often do you speak Irish?", daily, 43,541 by Data
    Zone (Flexible Table Builder IRISH_SKILLS_SPEAK_FREQUENCY). The question has no school
    exclusion, so the full-time students or schoolchildren among them (14,886 by DEA) are taken
    out where they live (northern_ireland()), matching Ireland's "outside education".
  * Welsh: the census's Welsh speakers per OA (split_wales) x the share of Welsh speakers who
    speak it daily in the OA's local authority, from the Annual Population Survey (StatsWales,
    "Frequency of speaking Welsh by local authority and year"), counts pooled over the years
    ending 31 March 2020, 2021 and 2022. The Welsh Language Use Survey 2019-20 has daily use
    nationally only (56% of its speakers); the APS gives the same 54% nationally.

Each language keeps its main-language count as a floor in every unit; the people added come
out of the unit's English (Wales: out of Welsh into English). Every changed row is `derived`;
every unit keeps its total.
"""
import io
import zipfile

import pandas as pd

from uk_census import RAW, BOX_WELSH, BOX_ENGLISH

GAELIC_HOME_2011 = 10_443 + 14_531
SCOTS_HOME_2011 = 55_817
GAELIC_SHARE = {"Na h-Eileanan Siar": 0.737, "Highland": 0.415, "Argyll and Bute": 0.334}
GAELIC_SHARE_REST = 0.236
APS_YEARS = ["31st March 2020", "31st March 2021", "31st March 2022"]

SC_GAELIC_EXTRA = "sc:Gaelic: used at home, beyond main language"
SC_SCOTS_EXTRA = "sc:Scots: used at home, beyond main language"
NI_IRISH_EXTRA = "ni:Irish: speaks daily, beyond main language"


def _uv(z, prefix):
    n = next(x for x in z.namelist() if x.startswith(prefix + " "))
    d = pd.read_csv(io.BytesIO(z.read(n)), skiprows=4, na_values=["-"])
    d = d.rename(columns={d.columns[0]: "oa"})
    return d[d["oa"].astype(str).str.startswith("S00")].fillna(0).set_index("oa")


def _check_at002():
    x = pd.read_excel(RAW / "sc2011_AT_002_2011.xls", header=None)
    got = {str(a).strip(): b for a, b in zip(x[0], x[1])}
    assert got["Gaelic (Scottish)"] + got["Gaelic (Not otherwise specified)"] == GAELIC_HOME_2011
    assert got["Scots"] == SCOTS_HOME_2011


def _speakers(d, lang):
    cols = [c for c in d.columns if c.startswith("Speaks")]
    assert len(cols) == 3, cols
    return d[cols].sum(axis=1)


def _fit(est, floor, target):
    """Scale est so that sum(max(k*est, floor)) == target."""
    lo, hi = 0.0, 50.0
    for _ in range(100):
        k = (lo + hi) / 2
        if pd.concat([est * k, floor], axis=1).max(axis=1).sum() < target:
            lo = k
        else:
            hi = k
    return pd.concat([est * k, floor], axis=1).max(axis=1), k


def _move(units, unit_prefix, lang_key, eng_key, want, extra_key, label):
    """Raise lang_key in each unit to `want`, taking the people from eng_key; tier derived."""
    sel = units["unit"].str.startswith(unit_prefix)
    have = units[sel & (units["category"] == lang_key)].groupby("unit")["count"].sum()
    eng = units[sel & (units["category"] == eng_key)].groupby("unit")["count"].sum()
    want = want.reindex(eng.index.union(want.index)).fillna(0)
    need = (want - have.reindex(want.index).fillna(0)).clip(lower=0)
    move = pd.concat([need, eng.reindex(need.index).fillna(0)], axis=1).min(axis=1)
    short = (need - move).sum()
    move = move[move > 0]
    before = units.groupby("unit")["count"].sum()
    is_eng = sel & (units["category"] == eng_key) & units["unit"].isin(move.index)
    units.loc[is_eng, "count"] = units.loc[is_eng, "count"] - units.loc[is_eng, "unit"].map(move)
    units.loc[is_eng, "tier"] = "derived"
    add = pd.DataFrame({"unit": move.index, "category": extra_key, "count": move.values,
                        "tier": "derived"})
    units = pd.concat([units, add], ignore_index=True)
    units = units[units["count"] > 1e-9].reset_index(drop=True)
    after = units.groupby("unit")["count"].sum().reindex(before.index).fillna(0)
    assert ((after - before).abs() < 1e-6).all(), f"{label}: a unit's total changed"
    total = units[units["category"].isin([lang_key, extra_key])]["count"].sum()
    print(f"  {label}: main language {have.sum():,.0f} -> {total:,.0f} "
          f"(+{move.sum():,.0f} from English; {short:,.0f} could not be placed, English ran out)")
    return units


def scotland(units):
    _check_at002()
    with zipfile.ZipFile(RAW / "sc_Census-2022-Output-Area-v1.zip") as z:
        gd, sc = _uv(z, "UV208"), _uv(z, "UV209")
    with zipfile.ZipFile(RAW / "sc_Census_2022_Index.zip") as z:
        oa = pd.read_csv(io.BytesIO(z.read("Census_2022_Index/OA_TO_HIGHER_AREAS.csv")),
                         usecols=["OA2022", "CA2019"])
        ca = pd.read_csv(io.BytesIO(z.read(
            "Census_2022_Index/Higher_Geographies_LookUps/Council Area 2019 Lookup.csv")))
    ca_name = dict(zip(ca["CouncilArea2019Code"], ca["CouncilArea2019Name"]))
    oa_ca = oa.set_index("OA2022")["CA2019"].map(ca_name)
    assert set(GAELIC_SHARE) <= set(oa_ca), "council names changed"
    assert set(gd.index) <= set(oa_ca.index)

    g_speak = _speakers(gd, "Gaelic")
    share = oa_ca.reindex(g_speak.index).map(GAELIC_SHARE).fillna(GAELIC_SHARE_REST)
    g_main = units[units["category"] == "sc:Gaelic"].groupby("unit")["count"].sum()
    est = g_speak * share
    print(f"Scotland home use: Gaelic speakers {g_speak.sum():,.0f}; x 2011 council-group home "
          f"shares = {est.sum():,.0f} (2011 count {GAELIC_HOME_2011:,})")
    g_want, k = _fit(est, g_main.reindex(est.index).fillna(0), GAELIC_HOME_2011)
    print(f"  fitted x{k:.3f}; Western Isles {g_want[oa_ca.reindex(g_want.index) == 'Na h-Eileanan Siar'].sum():,.0f}")
    units = _move(units, "S00", "sc:Gaelic", "sc:English", g_want, SC_GAELIC_EXTRA, "Gaelic")

    s_speak = _speakers(sc, "Scots")
    s_main = units[units["category"] == "sc:Scots"].groupby("unit")["count"].sum()
    s_want, k = _fit(s_speak, s_main.reindex(s_speak.index).fillna(0), SCOTS_HOME_2011)
    print(f"  Scots speakers {s_speak.sum():,.0f}; home share fitted {k:.4f}")
    units = _move(units, "S00", "sc:Scots", "sc:English", s_want, SC_SCOTS_EXTRA, "Scots")
    return units


def _student_daily(g):
    """Daily Irish speakers who are full-time students or schoolchildren, by unit; NaN where
    NISRA blanked the cell."""
    x = pd.read_csv(RAW / f"ni_irish_speak_frequency_x_student_{g}.csv")
    x.columns = ["u", "name", "sc", "student", "fc", "freq", "n"]
    return x[(x["fc"] == 1) & (x["sc"] == 1)].set_index("u")["n"].astype(float)


def _share_down(parent, child_known, child_weight, child_parent):
    """Split each parent's count among its children: known children keep their value (scaled
    down if they exceed the parent), the rest share the remainder by child_weight."""
    known = child_known.dropna()
    ksum = known.groupby(child_parent.reindex(known.index)).sum()
    scale = (parent / ksum).clip(upper=1).reindex(ksum.index).fillna(1)
    out = known * child_parent.reindex(known.index).map(scale)
    rest = (parent - ksum.reindex(parent.index).fillna(0)).clip(lower=0)
    blank = child_weight.index.difference(known.index)
    w = child_weight.reindex(blank)
    wp = child_parent.reindex(blank)
    wsum = w.groupby(wp).sum()
    est = w * wp.map(rest).fillna(0) / wp.map(wsum).replace(0, float("nan"))
    return pd.concat([out, est.fillna(0)])


def northern_ireland(units):
    """Daily Irish speakers less the full-time students and schoolchildren among them, so the
    measure matches Ireland's daily use outside education. NISRA's cross of student status by
    speaking frequency is complete by DEA (80) and partly blanked by SDZ and DZ; each level's
    known cells are kept and the blanks share the parent's remainder by daily speakers."""
    d = pd.read_csv(RAW / "ni_irish_speak_frequency_DZ21.csv")
    d.columns = ["dz", "name", "code", "freq", "n"]
    assert d["dz"].nunique() == 3_780
    daily = d[d["freq"] == "Can speak Irish: Speaks Irish daily"].set_index("dz")["n"].astype(float)
    lk = pd.read_csv(RAW / "ni_dz2021_lookup.csv").set_index("DZ2021_cd")
    assert set(lk.index) == set(daily.index)
    sdz_dea = lk.groupby("SDZ2021_cd")["DEA2014_cd"].agg(lambda s: s.iloc[0])
    assert lk.groupby("SDZ2021_cd")["DEA2014_cd"].nunique().max() == 1, "SDZ splits a DEA"
    dea = _student_daily("DEA14")
    assert len(dea) == 80 and dea.notna().all()
    sdz = _student_daily("SDZ21")
    dz = _student_daily("DZ21")
    assert set(sdz.index) == set(sdz_dea.index) and set(dz.index) == set(daily.index)
    sdz_daily = daily.groupby(lk["SDZ2021_cd"]).sum()
    s_sdz = _share_down(dea, sdz, sdz_daily, sdz_dea)
    s_dz = _share_down(s_sdz, dz, daily, lk["SDZ2021_cd"])
    s_dz = s_dz.reindex(daily.index).fillna(0).clip(upper=daily)
    print(f"Northern Ireland: Irish daily {daily.sum():,.0f} by Data Zone; full-time students "
          f"among them {dea.sum():,.0f} by DEA ({sdz.isna().sum()} of {len(sdz)} SDZ and "
          f"{dz.isna().sum()} of {len(dz)} DZ cells blank), placed {s_dz.sum():,.0f}")
    want = daily - s_dz
    return _move(units, "N20", "ni:Irish", "ni:English", want, NI_IRISH_EXTRA, "Irish (NI)")


def wales(units):
    a = pd.read_csv(RAW / "wales_aps_welsh_frequency_la.csv")
    a = a[(a["Data description"] == "Number") & a["Year"].isin(APS_YEARS)].copy()
    a["Data values"] = pd.to_numeric(a["Data values"].astype(str).str.replace(",", ""))
    assert a["Year"].nunique() == 3 and a["Data values"].notna().all()
    p = a.pivot_table(index="Local Authority", columns="Frequency of speaking Welsh",
                      values="Data values", aggfunc="sum")
    share = p["Daily"] / p.sum(axis=1)
    lu = pd.read_csv(RAW / "oa21_lsoa_msoa_lad21_ew_lu.csv", usecols=["OA21CD", "LAD22NM"])
    oa_la = lu.set_index("OA21CD")["LAD22NM"]
    las = set(oa_la[oa_la.index.str.startswith("W")])
    assert las == set(share.index) - {"Wales"}, las ^ set(share.index)
    print(f"Wales: APS daily share of Welsh speakers, years ending March 2020-22: Wales "
          f"{share['Wales']:.1%}, Gwynedd {share['Gwynedd']:.1%}, Cardiff {share['Cardiff']:.1%}, "
          f"lowest {share.min():.1%} ({share.idxmin()})")
    w = units["category"] == BOX_WELSH
    e = units["category"] == BOX_ENGLISH
    before = units.groupby("unit")["count"].sum()
    old = units.loc[w, "count"].sum()
    keep = units.loc[w, "count"] * units.loc[w, "unit"].map(oa_la).map(share)
    assert keep.notna().all()
    drop = (units.loc[w, "count"] - keep).groupby(units.loc[w, "unit"]).sum()
    units.loc[w, "count"] = keep
    has_e = units.loc[e, "unit"]
    units.loc[e, "count"] = units.loc[e, "count"] + has_e.map(drop).fillna(0)
    missing = drop[~drop.index.isin(has_e)]
    units = pd.concat([units, pd.DataFrame({"unit": missing.index, "category": BOX_ENGLISH,
                                            "count": missing.values, "tier": "derived"})],
                      ignore_index=True)
    units = units[units["count"] > 1e-9].reset_index(drop=True)
    after = units.groupby("unit")["count"].sum().reindex(before.index).fillna(0)
    assert ((after - before).abs() < 1e-6).all(), "Wales: a unit's total changed"
    print(f"  Welsh (split) {old:,.0f} -> {units.loc[units['category'] == BOX_WELSH, 'count'].sum():,.0f}")
    return units


def apply(units):
    units = wales(units)
    units = scotland(units)
    units = northern_ireland(units)
    return units
