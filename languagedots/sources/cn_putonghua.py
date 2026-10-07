"""China: the share of people who use Putonghua (standard Mandarin) after work in the non-Mandarin
provinces, by prefecture and migrant status, from CLDS 2016 -> data/normalized/cn_putonghua.csv.
Checked against WVS wave 7 (China 2018) "language at home" by province.

    python sources/cn_putonghua.py [--fetch]     # --fetch: the WVS crosstab only

THE SURVEY. China Labor-force Dynamics Survey 2016 (Center for Social Survey, Sun Yat-sen
University), individual file `CLDS2016individual_STATA_171106.dta`, read-only from religiondots'
raw tree; never copied here. Anita approved tabulating it (2026-10-06): only aggregates leave this
script. Terms: non-commercial, no raw data passed on, cite as below.
  I1_8_4  "请问您下班/放学后，主要使用的语言是？" (main language after work or school):
          1 普通话 Putonghua, 2 本地方言 local dialect, 3 老家方言 home-town dialect, 99 other.
  PROV2016, CITY (prefecture; real codes), I1_3_1_psu (hukou place; the county part scrambled
  within the city by the release, so the city part is usable), birthyear, wpp (person weight).
Migrant status from the hukou place against the interview city: `local` same prefecture (same
province for the four municipalities), `intra` another prefecture of the province, `inter`
another province. No hukou place or no city: left out.

WHERE IT MEANS ANYTHING. In Mandarin areas respondents call their own Mandarin "local dialect"
(Henan 86%), so "Putonghua" is a Mandarin share only where the local speech is not Mandarin. A
CLDS city counts as non-Mandarin when its counties' atlas groups (data/normalized/cn_dialect_dlac.csv,
weighted by cn.csv's people) are under half Mandarin. Only those cities feed the shares.

THE SHARES. Two per place: `settled` (locals and people from elsewhere in the province, pooled,
weighted; the map draws both on the county's own dialect groups) and `inter` (people registered in
another province). Weighted Putonghua share, shrunk towards the level above with K pseudo-cases:
national pool of non-Mandarin cities -> province -> city. A city's share is used for its own
counties; every other county takes its province's, and Hainan (not sampled) the national pool.
Cell n is printed and written. Age is printed for the record, not used (15-64 only sampled).

THE CHECK. WVS wave 7, China 2018 (3,036 adults), Q272 language at home: Putonghua against
"other Chinese dialects", by N_REGION_ISO, through the online tool (the route of sources/ir_wvs.py;
saved to data/raw/cn/wvs/). Printed beside CLDS per province; sources/cn.md §10 compares both with
the map.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
CLDS = (HERE.parent / "religiondots" / "data" / "raw" / "cn" / "clds" / "2016" / "中国劳动力动态调查2016"
        / "CLDS2016全部数据(STATA)171106" / "CLDS2016individual_STATA_171106.dta")
NORM = HERE / "data" / "normalized"
OUT = NORM / "cn_putonghua.csv"
WVS_RAW = HERE / "data" / "raw" / "cn" / "wvs" / "w7_q272_region.html"

# provinces where the map applies the shares (Anita's list, 2026-10-06), and Hainan, unsampled
PROVS = {44: "Guangdong", 45: "Guangxi", 46: "Hainan", 35: "Fujian", 33: "Zhejiang",
         31: "Shanghai", 32: "Jiangsu", 36: "Jiangxi", 43: "Hunan", 34: "Anhui"}
MUNI = {11, 12, 31, 50}
K = 30            # shrinkage pseudo-cases towards the level above
EXPECT_N = 21086  # people in the individual file

WVS_REGION = {   # N_REGION_ISO column -> province code
    "CN-GD Guangdong": 44, "CN-GX Guangxi": 45, "CN-HI Hainan": 46, "CN-FJ Fujian": 35,
    "CN-ZJ Zhejiang": 33, "CN-SH Shanghai": 31, "CN-JS Jiangsu": 32, "CN-JX Jiangxi": 36,
    "CN-HN Hunan": 43, "CN-AH Anhui": 34,
}
WVS_PTH = "Standard Chinese; Mandarin; Putonghua; Guoyu"


def fetch_wvs():
    import requests
    import urllib3
    urllib3.disable_warnings()
    import ir_wvs
    s = requests.Session()
    s.verify = False
    s.headers["User-Agent"] = ir_wvs.UA
    base = ir_wvs.BASE
    s.get(base + "WVSOnline.jsp", timeout=60)
    s.get(base + "AJOnline.jsp?WAVE=&COUNTRY=", timeout=60)
    form = {"ulthost": "WVS", "CMSID": "", "WAVE": "1562", "MAIDX": "", "SAIDS": "3267",
            "AMIDS": "156", "SATITULOS": "China", "COUNTRY": "", "CRUCEX": ""}
    s.post(base + "AJOnlineCountries.jsp", data=form, timeout=60)
    s.post(base + "AJOnlineIndex.jsp", data=form, timeout=60)
    form["MAIDX"] = "C_Q272"
    s.post(base + "AJOnlineQtn.jsp", data=form, timeout=120)
    form2 = {"ulthost": "WVS", "CMSID": "", "WAVE": "1562", "SAIDS": "3267", "SATITULOS": "China",
             "AMIDS": "156", "MAIDX": "C_Q272", "MACRUCE1": "2437884", "MACRUCE2": "",
             "CRUCES_ROTARXY": "", "CRUCE_TYPE": "TAB", "AJArchive": "WVS Data Archive"}
    r = s.post(base + "AJOnlineQtn.jsp", data=form2, timeout=120)
    r.raise_for_status()
    if "JDSTableCellHeader" not in r.text:
        raise SystemExit("WVS: no crosstab in the response")
    WVS_RAW.parent.mkdir(parents=True, exist_ok=True)
    tmp = WVS_RAW.with_suffix(".part")
    tmp.write_text(r.text, encoding="utf-8")
    tmp.replace(WVS_RAW)


def wvs_shares():
    """{province code: (N, Putonghua %)} from the saved crosstab. China's file is weighted, so the
    printed percentages are not whole counts (ir_wvs.counts' rounding check does not apply); the
    column percentages are taken as printed, with "No answer" left in the base (it is under 1%)."""
    import ir_wvs
    out, seen = {}, []
    for filt, cols, data, ns in ir_wvs.tables(WVS_RAW):
        if filt is not None or ns is None:
            continue
        for j, col in enumerate(cols):
            seen.append(col)
            p = next((v for k, v in WVS_REGION.items() if col.startswith(k)), None)
            if p is None:
                continue
            cell = data.get(WVS_PTH, ["-"] * len(cols))[j]
            out[p] = (ns[j], float(cell.rstrip("%")) if cell not in ("-", "") else 0.0)
    missing = set(PROVS) - set(out)
    if missing:
        raise SystemExit(f"WVS: no column for {missing}; columns are {seen}")
    return out


def mandarin_share_by_pref():
    """Atlas Mandarin share of each prefecture's (and municipality's) people."""
    dia = pd.read_csv(NORM / "cn_dialect_dlac.csv", dtype={"unit": str})
    pop = pd.read_csv(NORM / "cn.csv", dtype={"geo_id": str}).groupby("geo_id")["count"].sum()
    dia["pop"] = dia["unit"].map(pop).fillna(0) * dia["share"]
    dia["m"] = dia["pop"] * (dia["sgroup"] == "Mandarin")
    dia["pref"] = [u[:2] + "0000" if int(u[:2]) in MUNI else u[:4] + "00" for u in dia["unit"]]
    g = dia.groupby("pref")[["m", "pop"]].sum()
    return g["m"] / g["pop"]


def load():
    cols = ["PROV2016", "CITY", "I1_3_1_psu", "I1_8_4", "birthyear", "wpp"]
    d = pd.read_stata(CLDS, columns=cols, convert_categoricals=False)
    if len(d) != EXPECT_N:
        raise SystemExit(f"CLDS: {len(d)} people, expected {EXPECT_N}")
    d = d[d["I1_8_4"].isin([1, 2, 3, 99]) & d["CITY"].notna() & d["I1_3_1_psu"].notna()].copy()
    city = d["CITY"].astype(int)
    prov = city // 10000          # the city code decides the province (a few PROV2016 disagree)
    h = d["I1_3_1_psu"].astype(int)
    if (h < 110000).any():
        raise SystemExit("CLDS: hukou place codes that are not six-digit")
    same = np.where(prov.isin(MUNI), h // 10000 == prov, h // 100 == city // 100)
    d["status"] = np.where(same, "local", np.where(h // 10000 == prov, "intra", "inter"))
    d["prov"] = prov
    d["pref"] = np.where(prov.isin(MUNI), prov * 10000, city).astype(int).astype(str)
    d["ans"] = d["I1_8_4"].map({1: "pth", 2: "local", 3: "home", 99: "other"})
    d["age"] = 2016 - d["birthyear"]
    print(f"  CLDS: {len(d):,} people with an answer, a city and a hukou place; "
          f"{(d['PROV2016'] != d['prov']).sum()} whose PROV2016 differs from their city's province")
    return d


def share(g):
    w = g.groupby("ans")["wpp"].sum()
    return pd.Series({"n": len(g), "wsum": float(w.sum()),
                      **{f"p_{a}": float(w.get(a, 0) / w.sum()) for a in ("pth", "local", "home", "other")}})


def main():
    if "--fetch" in sys.argv:
        fetch_wvs()
    d = load()
    mshare = mandarin_share_by_pref()
    d["mandarin_city"] = d["pref"].map(mshare)
    nm = d[d["prov"].isin(PROVS) & (d["mandarin_city"] < 0.5)].copy()
    dropped = d[d["prov"].isin(PROVS) & (d["mandarin_city"] >= 0.5)]
    print(f"  non-Mandarin cities sampled in the ten provinces: {nm['pref'].nunique()} "
          f"({len(nm):,} people); Mandarin cities left out: "
          f"{sorted(dropped['pref'].unique())} ({len(dropped):,} people)")
    nm["grp"] = np.where(nm["status"] == "inter", "inter", "settled")

    rows = []
    nat = {gp: share(g) for gp, g in nm.groupby("grp")}
    for gp, s in nat.items():
        rows.append(dict(level="pool", code="00", group=gp, **s, p=s["p_pth"]))
    prov_p = {}
    for p in PROVS:
        for gp in ("settled", "inter"):
            g = nm[(nm["prov"] == p) & (nm["grp"] == gp)]
            parent = nat[gp]["p_pth"]
            if len(g):
                s = share(g)
                pp = (s["n"] * s["p_pth"] + K * parent) / (s["n"] + K)
            else:
                s = pd.Series({"n": 0, "wsum": 0.0, "p_pth": np.nan, "p_local": np.nan,
                               "p_home": np.nan, "p_other": np.nan})
                pp = parent
            prov_p[(p, gp)] = pp
            rows.append(dict(level="province", code=str(p), group=gp, **s, p=pp))
    for (pref, gp), g in nm.groupby(["pref", "grp"]):
        s = share(g)
        parent = prov_p[(int(pref[:2]), gp)]
        rows.append(dict(level="city", code=pref, group=gp, **s,
                         p=(s["n"] * s["p_pth"] + K * parent) / (s["n"] + K)))
    out = pd.DataFrame(rows)
    out["n"] = out["n"].astype(int)
    for c in ("p_pth", "p_local", "p_home", "p_other", "p"):
        out[c] = out[c].round(4)
    out = out.drop(columns="wsum")
    tmp = OUT.with_suffix(".part")
    out.to_csv(tmp, index=False)
    tmp.replace(OUT)
    print(f"  wrote {OUT.relative_to(HERE)}: {len(out)} rows (aggregates only)")

    # record: status x province, every answer, and the cities
    pd.set_option("display.width", 200)
    t = nm.groupby(["prov", "status"]).apply(share).drop(columns="wsum")
    print("\n  province x status, after-work language, weighted % (non-Mandarin cities only)")
    print((t.assign(**{c: (t[c] * 100).round(1) for c in t if c.startswith("p_")})).to_string())
    t = nm.groupby(["pref", "status"]).apply(share).drop(columns="wsum")
    print("\n  city x status")
    print((t.assign(**{c: (t[c] * 100).round(1) for c in t if c.startswith("p_")})).to_string())
    a = nm[nm["prov"].isin([44, 31, 33, 35])].copy()
    a["band"] = pd.cut(a["age"], [0, 30, 45, 70], labels=["15-30", "31-45", "46-64"])
    t = a.groupby(["prov", "grp", "band"], observed=True).apply(share).drop(columns="wsum")
    print("\n  age, Putonghua %")
    print((t[["n"]].assign(pth=(t["p_pth"] * 100).round(1))).to_string())
    gi = nm[(nm["prov"] == 44) & (nm["status"] == "inter")].copy()
    gi["origin"] = (gi["I1_3_1_psu"] // 10000).astype(int)
    t = gi.groupby("origin").apply(share).drop(columns="wsum")
    print("\n  Guangdong's people from other provinces by origin, Putonghua %")
    print(t[t["n"] >= 15][["n"]].assign(pth=(t["p_pth"] * 100).round(1)).to_string())
    sz = d[d["pref"] == "440300"]
    print(f"\n  Shenzhen: {len(sz)} people; by status:")
    print(sz.groupby("status").apply(share).drop(columns="wsum").round(3).to_string())

    if WVS_RAW.exists():
        w = wvs_shares()
        print("\n  province: CLDS settled / inter (shrunk) | CLDS all, every city | WVS 2018 home")
        for p, name in PROVS.items():
            allc = d[d["prov"] == p]
            ac = share(allc)["p_pth"] * 100 if len(allc) else np.nan
            print(f"  {name:<10} {prov_p[(p, 'settled')] * 100:5.1f} / {prov_p[(p, 'inter')] * 100:5.1f}"
                  f" | {ac:5.1f} (n {len(allc):>5}) | {w[p][1]:5.1f} (n {w[p][0]})")
    else:
        print("  WVS crosstab not saved; run with --fetch")


if __name__ == "__main__":
    main()
