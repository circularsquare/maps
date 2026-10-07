"""Burundi: home language from two pooled Afrobarometer rounds (R5 2012, R6 2014), by province,
on current province populations (2024 census total, spread by COD-PS 2022 communes) ->
data/normalized/bi.csv.

    python sources/bi_afro.py

NO LANGUAGE QUESTION IN THE CENSUS. The 2008 RGPH asks only which language a person can read
and write (literacy); the 2024 RGPHAE has published preliminary totals only. So this is the
survey route of AGENT_BRIEF §2: shares from the Afrobarometer's home-language question ("Which
language is your home language?"), one answer, times a population base, every row `modelled`.
Rounds 7-9 did not survey Burundi; R4 did not either.

2,400 respondents: Kirundi 99.37% pooled, Swahili 0.48% (8 of 176 in Bujumbura Mairie, 2 in
Cibitoke, 1 in Bururi; R5 prints "Kiswahili", R6 "Swahili"), French 0.15% (one each in Kayanza,
Mwaro, Ngozi). Each answer's share is the pooled weighted share in its province.

POPULATION (rescaled 2026-10-05, session edd42a8c-mono2; first built on the 2008 RGPH's
8,053,574). The units stay the 17 provinces of 2008, religiondots' placement layer. Each
province's share comes from the COD-PS 2022 commune projections (HDX cod-ps-bdi,
bdi_admpop_2022_adm2_v3.csv, 119 communes on 18 provinces, 13,462,695 people): communes are
summed onto their 2008 province, Rumonge province's five going back to Bururi (Burambi,
Buyengero, Rumonge) and Bujumbura Rural (Bugarama, Muhuta) as religiondots' bi_communes.csv
files them, and the three Mairie communes onto Bujumbura Mairie. Those shares are scaled to the
2024 RGPHAE preliminary total, 12,332,788 (published 27 March 2025), because the census count is
newer and lower than the projection; the 2024 results by province are on the 2025 five-province
map, which does not nest in the 2008 provinces.

REGION -> province is religiondots' sources/bi.py NORM: "Bujumbura" is Bujumbura Rural, the
Mairie is spelled both "Mairie" and "Marie", Cankuzo and Ruyigi each have a misspelling.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sources"))
from rdlink import RD_GEO  # noqa: E402
from mono_afro import extract, unit_counts  # noqa: E402

OUT = HERE / "data" / "normalized" / "bi.csv"
RGPH2008 = 8_053_574
RGPHAE2024 = 12_332_788           # preliminary total, 27 March 2025
CODPS = HERE / "data" / "raw" / "bi" / "bdi_admpop_2022_adm2_v3.csv"
CODPS_TOTAL = 13_462_695
CODPS_URL = ("https://data.humdata.org/dataset/61e643bc-e36d-4106-8815-4b952fc40e9c/resource/"
             "caf569e5-177b-4a6f-abc9-b01e8bf85c43/download/bdi_admpop_2022_adm2_v3.csv")
NORM = {"bubanza": "BI01", "bujumbura": "BI02", "bururi": "BI03", "cankuzo": "BI04",
        "cankuza": "BI04", "cibitoke": "BI05", "gitega": "BI06", "karusi": "BI07",
        "karuzi": "BI07", "kayanza": "BI08", "kirundo": "BI09", "makamba": "BI10",
        "muramvya": "BI11", "muyinga": "BI12", "mwaro": "BI13", "ngozi": "BI14", "rutana": "BI15",
        "ruyigi": "BI16", "ruyiga": "BI16", "bujumburamairie": "BI17", "bujumburamarie": "BI17"}
LABELS = {"Kirundi": "Kirundi", "Swahili": "Swahili", "Kiswahili": "Swahili",
          "French": "French"}


def gk(s):
    return "".join(ch for ch in str(s).casefold() if ch.isalpha())


def current_pop(lut):
    """unit -> (name, people): COD-PS 2022 communes summed onto the 2008 provinces, scaled to
    the 2024 census total by largest remainder."""
    if not CODPS.exists():
        import urllib.request
        CODPS.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(CODPS_URL, headers={"User-Agent": "Mozilla/5.0"})
        CODPS.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    c = pd.read_csv(CODPS, encoding="utf-8-sig")
    assert len(c) == 119 and int(c["T_TL"].sum()) == CODPS_TOTAL, (len(c), c["T_TL"].sum())
    com = pd.read_csv(RD_GEO / "bi" / "bi_communes.csv", dtype=str)
    # (province key, commune key) -> 2008 unit; Rumonge province's communes by commune alone
    by_pc = {(gk(r.province), gk(r.commune)): r.unit for r in com.itertuples()}
    by_c = {gk(r.commune): r.unit for r in com.itertuples()}
    prov = {gk(n): u for u, n in zip(lut["unit"], lut["name"])}
    spell = {"buhiga": "buhuga", "mpingakayove": "mpinga", "mugougomanga": "mugongomanga",
             "nyabitsunda": "nyabitsinda"}       # COD-PS spelling -> 2008 table spelling

    def unit(r):
        p, k = gk(r.ADM1_NAME), spell.get(gk(r.ADM2_NAME), gk(r.ADM2_NAME))
        if p == "bujumburamairie":
            return "BI17"                        # Muha, Mukaza, Ntahangwa: the 2008 quartiers
        if p == "rumonge":
            return by_c[k]
        u = by_pc.get((p, k))
        assert u == prov[p], (r.ADM1_NAME, r.ADM2_NAME, u)
        return u
    c["unit"] = [unit(r) for r in c.itertuples()]
    assert set(c["unit"]) == set(lut["unit"])
    rc = c[c["ADM1_NAME"].map(gk) == "rumonge"]
    rum = rc.set_index(rc["ADM2_NAME"].map(gk))["unit"]
    assert rum.to_dict() == {"bugarama": "BI02", "burambi": "BI03", "buyengero": "BI03",
                             "muhuta": "BI02", "rumonge": "BI03"}, rum.to_dict()
    s = c.groupby("unit")["T_TL"].sum()
    x = s / s.sum() * RGPHAE2024
    n = x.astype(int)
    n[(x - n).sort_values(ascending=False).index[:RGPHAE2024 - int(n.sum())]] += 1
    assert int(n.sum()) == RGPHAE2024
    names = dict(zip(lut["unit"], lut["name"]))
    return {u: (names[u], int(v)) for u, v in n.items()}


def main():
    a = extract({"burundi"})
    assert len(a) == 2400 and sorted(a["round"].unique()) == [5, 6], (len(a),)
    lut = pd.read_csv(RD_GEO / "bi" / "bi_lookup.csv", dtype={"unit": str})
    assert len(lut) == 17 and int(lut["pop"].sum()) == RGPH2008
    pop = current_pop(lut)
    for u, (nm, v) in sorted(pop.items()):
        old = int(lut.loc[lut["unit"] == u, "pop"].iloc[0])
        print(f"  {u} {nm:18} 2008 {old:>9,}  now {v:>10,}  x{v / old:.2f}")
    df = unit_counts(a, NORM, LABELS, pop, "afrobarometer_r5_r6_burundi", "2012-2014",
                     "province")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"wrote {OUT}: {df['geo_id'].nunique()} provinces, {df['count'].sum():,} people")
    print((nat / nat.sum() * 100).round(2).to_string())
    print(df[df["source_category"] != "Kirundi"][["geo_name", "source_category", "count",
                                                  "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
