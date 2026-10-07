"""Israel: CBS Social Survey 2021 native language, by sub-district and population group, applied
to the 2022 census population of each statistical area -> data/normalized/il.csv.

    python sources/il_social.py --fetch     # the three survey tables -> data/raw/il/
    python sources/il_social.py             # build il.csv from the cached tables

NO CENSUS ASKS LANGUAGE (1995 and 2008 forms have no item; 2022 is register-based). The CBS
Social Survey asked "native language" (the first language) in 2011 and 2021; 2021 is used. The
CBS table generator (boardsgenerator.cbs.gov.il, Social Survey, 2021, persons) is an open
Angular app whose JSON handlers answer plain posts, no login:
  WizardHandler.ashx  mode=VariablesFilterByGroup / SetCategories   variable ids and codes
  GridHandler.ashx    query={Fr, Sr, Fc, BoardChoice, Fyear, SekerType, ...}  the table
Variables (2021): 2344 Native language (9 codes), 359 Sub District, 314 District, 30 Age,
1888 Population group (Jews and others / Arabs). Cells are weighted estimates of people aged 20+
with a standard error and relative SE ("3,263,009##25,590##&&0.0078").

THE MODEL (rows `modelled`):
  Population: religiondots' il.csv (CBS 2022 census, register religion) on statistical areas and
  whole localities, read-only, minus data/geo/il/dropped_units.json: the 267 units beyond the
  Green Line (West Bank settlements and East Jerusalem) are drawn by neither entry here; East
  Jerusalem's Palestinians are on Palestine's entry, as in religiondots.
  Groups: Arabs = Muslims + Druze + Arab Christians; Jews and others = Jews + Others + the rest
  of the Christians. A locality's Arab Christians = its Arabs (bycode2023) - Muslims - Druze,
  clipped to [0, Christians], spread over its areas in proportion to Christians.
  Shares: Jews and others take their sub-district's 2021 shares, shrunk towards their district's
  by est / (est + 100,000); Arabs take the national Arab shares (the sub-district Arab cells are
  a few hundred respondents at most and all ~99% Arabic).
  Children: the survey is 20+. Each area's under-20s (sa2022 age0_19_pcnt) take the national
  20-24 shares of their group, which already lean Hebrew (83% against 68% for all adults).
  Placement inside a sub-district: Jews-and-others counts are raked (IPF) so every area keeps its
  population and every sub-district keeps the survey's language totals, from a seed weighted by
  each area's census origin profile (sa2022): Russian by European origin (x2 where the main
  country of birth is an ex-Soviet one), Amharic by African origin (x4 where it is Ethiopia),
  French by European + African origin, English and Spanish by American origin, Yiddish by the
  area's Haredi share (il.csv's ultra-religious households), Arabic and other by born abroad,
  Hebrew by born in Israel.

CHECKS: sub-district rows sum to the national row per language; district rows likewise; every
drawn unit gets a sub-district; output sums to the drawn population exactly; raked sub-district
language totals match their targets.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD  # noqa: E402

RAW = HERE / "data" / "raw" / "il"
OUT = HERE / "data" / "normalized" / "il.csv"
RD_IL = RD / "data" / "normalized" / "il.csv"
RD_DROPPED = RD / "data" / "geo" / "il" / "dropped_units.json"
RD_BYCODE = RD / "data" / "raw" / "il" / "bycode2023.xlsx"
RD_SA = RD / "data" / "raw" / "il" / "sa2022.csv"

GEN = "https://boardsgenerator.cbs.gov.il"
LANGS = ["Hebrew", "Arabic", "Russian", "English", "French", "Spanish", "Yiddish", "Amharic",
         "AnotherLanguage"]
TABLES = {"subdistrict": 359, "district": 314, "age": 30}
K_SHRINK = 100_000
FSU = {"רוסיה", "אוקראינה", "אוזבקיסטן", 'בריה"מ (לשעבר)', "בלארוס", "מולדובה", "גאורגיה",
       "אזרבייג'ן", "קזחסטן"}
ETHIOPIA = "אתיופיה"
JERUSALEM = 3000
NAFA_2021 ={"25": "23", "52": "51", "53": "51"}


def fetch():
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                      "Referer": f"{GEN}/pages/Seker/wizardpage.aspx?l=1",
                      "Accept": "application/json, text/plain, */*"})
    s.get(f"{GEN}/pages/Seker/wizardpage.aspx?l=1")
    RAW.mkdir(parents=True, exist_ok=True)
    for name, fr in TABLES.items():
        q = dict(Fr=fr, Sr=1888, Tr=None, Fc=2344, Sc=None, Tc=None, SadeForMapot=False,
                 BoardChoice="4", SekerType=1, Fyear=2021, Language="English", Unknow=False,
                 IncludeAllRelevant=True, Filters=[], Grouping=[])
        r = s.post(f"{GEN}/Handlers/Seker/GridHandler.ashx", data={"query": json.dumps(q)})
        r.raise_for_status()
        d = r.json()
        (RAW / f"social2021_native_{name}.json").write_text(json.dumps(d, ensure_ascii=False),
                                                            encoding="utf-8")
        print(f"  {name}: {len(d['data'])} rows")
        time.sleep(1)


def _num(v):
    if not isinstance(v, str) or not v.strip():
        return 0.0
    return float(re.sub(r"##.*", "", v).replace(",", "") or 0)


def read_table(name):
    d = json.loads((RAW / f"social2021_native_{name}.json").read_text(encoding="utf-8"))
    key = [c for c in d["data"][0] if c.endswith("Code") and c != "PopGroupCode"][0]
    rows = []
    for r in d["data"]:
        rows.append(dict(code=r[key], group=r["PopGroup"], total=_num(r["TOTALTOTAL"]),
                         **{l: _num(r.get(l, "")) for l in LANGS}))
    df = pd.DataFrame(rows)
    # check: the parts sum to the total, per group and language (estimates are rounded)
    for g in ("Jews and others", "Arabs"):
        part = df[(df.group == g) & (df.code != "-1")][LANGS + ["total"]].sum()
        whole = df[(df.group == g) & (df.code == "-1")][LANGS + ["total"]].iloc[0]
        bad = (part - whole).abs() > 0.002 * whole["total"] + 5
        if bad.any():
            raise SystemExit(f"{name} {g}: parts do not sum to the total: {list(part[bad].index)}")
        cells = df[(df.group == g) & (df.code != "-1")]
        rowsum = cells[LANGS].sum(axis=1)
        if ((rowsum - cells["total"]).abs() > 5).any():
            raise SystemExit(f"{name} {g}: a row's languages do not sum to its total")
    return df


def shares(row):
    v = row[LANGS].to_numpy(float)
    return v / v.sum()


def build():
    sub = read_table("subdistrict")
    dist = read_table("district")
    age = read_table("age")
    J, A = "Jews and others", "Arabs"
    nat_A = shares(sub[(sub.group == A) & (sub.code == "-1")].iloc[0])
    young = {g: shares(age[(age.group == g) & (age.code == "1")].iloc[0]) for g in (J, A)}
    dshare = {}
    for _, r in dist[(dist.group == J) & (dist.code != "-1")].iterrows():
        dshare[str(r.code)] = shares(r)
    adult = {}
    for _, r in sub[(sub.group == J) & (sub.code != "-1")].iterrows():
        code = str(r.code)
        lam = r.total / (r.total + K_SHRINK)
        adult[code] = lam * shares(r) + (1 - lam) * dshare[code[0]]

    # population, from religiondots' 2022 census rows
    il = pd.read_csv(RD_IL, dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    il = il[il.geo_level.isin(("statarea", "locality"))]
    dropped = set(json.loads(RD_DROPPED.read_text()))
    il = il[~il.geo_id.isin(dropped)].copy()
    il["rel"] = il.source_category.str.replace(r" \[.*\]$", "", regex=True)
    il["haredi"] = np.where(il.source_category == "Jews [Ultra-religious]", il["count"], 0.0)
    u = il.pivot_table(index="geo_id", columns="rel", values="count", aggfunc="sum", fill_value=0)
    u["haredi"] = il.groupby("geo_id")["haredi"].sum()
    names = il.groupby("geo_id")["geo_name"].first()
    levels = il.groupby("geo_id")["geo_level"].first()
    for c in ("Jews", "Others", "Christians", "Muslims", "Druze"):
        if c not in u:
            u[c] = 0.0
    u["loc"] = u.index.str.split("_").str[0].astype(int)

    by = pd.read_excel(RD_BYCODE)
    by = by.rename(columns={"סמל יישוב": "loc", "סמל נפה": "nafa", "ערבים - ארעי": "arabs",
                            "סך הכל ישראלים 2023 - ארעי": "israelis"})
    by = by[["loc", "nafa", "arabs", "israelis"]].fillna({"arabs": 0, "israelis": 0})
    u = u.join(by.set_index("loc"), on="loc")
    if u["nafa"].isna().any():
        miss = sorted(u[u.nafa.isna()].loc.unique())
        raise SystemExit(f"{len(miss)} localities have no sub-district in bycode2023: {miss[:10]}")
    u["nafa"] = u["nafa"].astype(int).astype(str)
    # bycode2023 splits two sub-districts the 2021 survey keeps whole: 25 is the second half of
    # Jezreel (its name is Jezreel too: the Arab villages around Nazareth), 52/53 are Tel Aviv.
    u["nafa"] = u["nafa"].replace(NAFA_2021)
    missing_nafa = set(u.nafa) - set(adult)
    if missing_nafa:
        raise SystemExit(f"sub-districts with no survey row: {missing_nafa}")

    # Arabs per locality from bycode2023's population groups (Arabs / Israelis), placed inside
    # the locality on its Muslims + Christians + Druze and capped at them. The register rows
    # cannot be used as groups directly: in Jewish towns religiondots resolved CBS's "Other
    # religions" lump by sub-district shares, so Bat Yam carries 3,587 "Muslims" against 1,389
    # Arabs, and non-Arab Christians are not told apart from Arab ones.
    u["mcd"] = u.Muslims + u.Christians + u.Druze
    u["tot"] = u.Jews + u.Others + u.mcd
    loc = u.groupby("loc")[["mcd", "tot"]].sum().join(by.set_index("loc")[["arabs", "israelis"]])
    loc["A"] = (loc.arabs / loc.israelis.replace(0, np.nan)).clip(upper=1).fillna(0) * loc.tot
    # Jerusalem's bycode row includes East Jerusalem, which is not drawn here: there the Arabs
    # are the drawn units' Muslims, Christians and Druze, nothing more.
    loc.loc[JERUSALEM, "A"] = loc.at[JERUSALEM, "mcd"]
    # first on the Muslims + Christians + Druze, then (Arab towns where religiondots' lump
    # allocation put some people on Jews or Others: Nazareth) the rest on the remaining people
    first = np.minimum(u.tot, u["loc"].map(loc.A) * u.mcd / u["loc"].map(loc.mcd).replace(0, np.inf))
    first = first.fillna(0)
    resid = u["loc"].map(loc.A - first.groupby(u["loc"]).sum()).clip(lower=0)
    room = u.tot - first
    share = room / u["loc"].map(room.groupby(u["loc"]).sum()).replace(0, np.inf)
    u["A"] = (first + resid * share).clip(upper=u.tot)
    u["J"] = u.tot - u.A
    tot = u.tot.sum()
    print(f"  drawn population {tot:,.0f} on {len(u):,} units; Arabs {u.A.sum():,.0f}, "
          f"Jews and others {u.J.sum():,.0f}; of the register's Muslims, Christians and Druze "
          f"{u.A.sum() / u.mcd.sum():.1%} counted Arab")

    # census profile per area
    sa = pd.read_csv(RD_SA)
    sa = sa.dropna(subset=["LocalityCode"]).copy()
    sa["loc"] = sa.LocalityCode.astype(int)
    sa["geo_id"] = np.where(sa.StatAreaCmb.isna(), sa["loc"].astype(str),
                            sa["loc"].astype(str) + "_" + sa.StatAreaCmb.astype(str))
    sa = sa.drop_duplicates("geo_id").set_index("geo_id")
    prof_cols = ["age0_19_pcnt", "j_isr_pcnt", "j_abr_pcnt", "europe_pcnt", "africa_pcnt",
                 "america_pcnt", "shem_eretz1"]
    p = u[[]].join(sa[prof_cols])
    locp = sa[sa.StatAreaCmb.isna()].set_index("loc")[prof_cols]
    nump = prof_cols[:-1]
    fill = u[["loc"]].join(locp, on="loc")[nump]
    p[nump] = p[nump].fillna(fill)
    nafa_mean = p[nump].join(u.nafa).groupby("nafa")[nump].transform("mean")
    p[nump] = p[nump].fillna(nafa_mean).fillna(p[nump].mean())
    print(f"  census profile: {u.index.isin(sa.index).mean():.1%} of units matched directly")
    c = (p.age0_19_pcnt / 100).clip(0, 0.8)

    # Arabs: national shares, adults and young
    rows = []
    A_counts = np.outer(u.A * (1 - c), nat_A) + np.outer(u.A * c, young[A])
    # Jews and others: seed by origin profile, raked per sub-district
    fl = 0.5
    abr = p.j_abr_pcnt + fl
    w = pd.DataFrame({
        "Hebrew": p.j_isr_pcnt + fl,
        "Arabic": abr,
        "Russian": (p.europe_pcnt + fl) * np.where(p.shem_eretz1.isin(FSU), 2, 1),
        "English": p.america_pcnt + fl,
        "French": p.europe_pcnt + p.africa_pcnt + fl,
        "Spanish": p.america_pcnt + fl,
        "Yiddish": (u.haredi / u.J.replace(0, np.nan)).fillna(0) * 100 + fl,
        "Amharic": (p.africa_pcnt + fl) * np.where(p.shem_eretz1 == ETHIOPIA, 4, 1),
        "AnotherLanguage": abr,
    }, index=u.index)[LANGS]
    Jc = pd.DataFrame(0.0, index=u.index, columns=LANGS)
    for n, idx in u.groupby("nafa").groups.items():
        Ju = u.loc[idx, "J"].to_numpy()
        cu = c.loc[idx].to_numpy()
        target = (Ju * (1 - cu)).sum() * adult[n] + (Ju * cu).sum() * young[J]
        ww = w.loc[idx].to_numpy()
        ww = ww / np.average(ww, axis=0, weights=Ju + 1e-9)
        M = Ju[:, None] * target[None, :] / max(target.sum(), 1e-9) * ww
        for _ in range(200):
            rs = M.sum(1)
            M *= np.where(rs > 0, Ju / np.where(rs > 0, rs, 1), 0)[:, None]
            cs = M.sum(0)
            M *= np.where(cs > 0, target / np.where(cs > 0, cs, 1), 0)[None, :]
        err = np.abs(M.sum(1) - Ju).max()
        if err > 1:
            raise SystemExit(f"sub-district {n}: raking left a unit {err:.1f} off its population")
        Jc.loc[idx] = M
    # final exact row closure (column error stays below a person)
    Jc = Jc.mul(u.J / Jc.sum(1).replace(0, 1), axis=0)

    for i, gid in enumerate(u.index):
        for k, l in enumerate(LANGS):
            if A_counts[i, k] > 0:
                rows.append((gid, levels[gid], names[gid], f"{l} (Arabs)", A_counts[i, k]))
            v = Jc.at[gid, l]
            if v > 0:
                rows.append((gid, levels[gid], names[gid], f"{l} (Jews and others)", v))
    out = pd.DataFrame(rows, columns=["geo_id", "geo_level", "geo_name", "source_category",
                                      "count"])
    out["count"] = out["count"].round(3)
    out["tier"] = "modelled"
    out["source_id"] = "cbs_social_survey_2021"
    out["year"] = 2021
    out["note"] = "native language 20+, sub-district x population group; CBS 2022 population"
    if abs(out["count"].sum() - tot) > 5:
        raise SystemExit(f"il.csv sums to {out['count'].sum():,.0f}, expected {tot:,.0f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    lang = out.assign(l=out.source_category.str.replace(r" \(.*\)$", "", regex=True)) \
              .groupby("l")["count"].sum().sort_values(ascending=False)
    print(f"  wrote {OUT.name}: {len(out):,} rows, {out['count'].sum():,.0f} people")
    for l, v in lang.items():
        print(f"    {l:16s} {v:12,.0f} {v / tot:6.1%}")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    build()
