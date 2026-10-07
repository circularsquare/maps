"""Morocco RGPH 2024, local languages used, per commune -> data/normalized/ma.csv.

    python sources/ma_rgph.py [--fetch]

SOURCE. Haut-Commissariat au Plan (HCP), *Indicateurs démographiques et socioéconomiques du
Royaume du Maroc selon les résultats du RGPH 2024*, Excel, https://www.hcp.ma/file/242671/
(no login; 7,597,863 bytes, digest pinned). Sheet `Population` (all residents, both sexes): one
row per nation, 12 régions, 75 provinces and préfectures, 213 cercles, 1,503 communes, 41
arrondissements, 8 préfectures d'arrondissement(s) and 164 "dont le centre urbain" rows, in that
nesting order.

QUESTION. "Langues locales utilisées (non exclusives) (%)": five columns, Darija, Tachelhit,
Tamazight, Tarifit, Hassania, each the share of the population using that local language,
several allowed (national: 91.9 + 14.2 + 7.4 + 3.2 + 0.8 = 117.5). The workbook's
`Avis_aux_Utilisateurs` sheet says the homeless are not in this indicator. No other language is
tabulated: someone who uses none of the five is in no column.

THE BUILD (AGENT_BRIEF §2, multi-answer). The units are the leaves: the 41 arrondissements of
the six cities that have them (Casablanca, Fès, Marrakech, Rabat, Salé, Tanger) and the 1,462
communes without arrondissements. Each leaf's people are its population municipale (residents
in dwellings, nomads and the homeless; not the 337,739 "comptée à part" in barracks, boarding
schools, prisons and the like, whom the indicator's base does not visibly include: check 4).
Within a leaf the mentions are scaled so that they add to its people:

    count = municipale * share / max(sum of the five shares, 100)

so each person is shared across the languages they use. Where the five add to less than 100
(293 leaves, 101 of them under 99) the shortfall is people using none of the five; they are not
drawn and their number is printed. Four starred communes in Oued Ed-Dahab-Aousserd (Lagouira,
Aghouinite, Zoug, Mijik: 5,371 people, figures from the local administration for a mobile
population) print "." and are not drawn. Every row is `derived`.

CHECKS (all must pass):
  1. the workbook is the pinned file (digest)
  2. the header carries the five languages at the expected columns
  3. leaves add to every province's and the nation's population légale and municipale exactly;
     each arrondissement nests under one commune and the arrondissements add to it
  4. every province's printed shares against its leaves' shares weighted by population
     municipale: within 0.35 points per language (one decimal rounding of up to ~100 leaves;
     the worst printed), except the two provinces holding the four "." communes (Aousserd
     misses by up to 13 points: its printed shares cannot be rebuilt from its communes). The
     same with population légale as the weight is printed, as a test of which base the
     indicator uses (2026-10-05: mean miss 0.027 on municipale, 0.195 on légale)
  5. the nation's printed shares against all leaves, within 0.1
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "ma"
NORM = HERE / "data" / "normalized" / "ma.csv"
NAME = "hcp_indicateurs_rgph2024.xlsx"
URL = "https://www.hcp.ma/file/242671/"
SHA = "f3d96fcae3c10a2c194b2fe97ee8352a242fb3f36b1b63d4d2a7006176dc7c3e"
LANGS = ("Darija", "Tachelhit", "Tamazight", "Tarifit", "Hassania")
COL0 = 48                                   # first language column in sheet `Population`
NATIONAL_LEGAL, NATIONAL_MUNI = 36_828_330, 36_490_591
NATIONAL_PCT = (91.9, 14.2, 7.4, 3.2, 0.8)


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(URL, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"})
    body = urllib.request.urlopen(req, timeout=120).read()
    (RAW / NAME).write_bytes(body)
    print(f"  fetched {URL} ({len(body):,} bytes)")


def level_of(name):
    for prefix, lvl in (("Ensemble", "national"), ("Région", "region"),
                        ("Préfecture d'arrondissement", "pref_arr"), ("Préfecture", "province"),
                        ("Province", "province"), ("Cercle", "cercle"), ("Commune", "commune"),
                        ("Arrondissement", "arrondissement"), ("dont", "centre")):
        if name.startswith(prefix):
            return lvl
    raise SystemExit(f"row with an unknown level: {name!r}")


def read():
    import openpyxl
    body = (RAW / NAME).read_bytes()
    if hashlib.sha256(body).hexdigest() != SHA:
        raise SystemExit(f"{NAME} is not the pinned file; check it, then update SHA")   # check 1
    wb = openpyxl.load_workbook(RAW / NAME, read_only=True, data_only=True)
    rows = list(wb["Population"].iter_rows(values_only=True))
    if (rows[1][COL0] != "Langues locales utilisées (non exclusives) (%)"
            or tuple(rows[2][COL0:COL0 + 5]) != LANGS or rows[1][2] != "Population légale"
            or rows[1][3] != "Population municipale"):
        raise SystemExit("sheet Population: header moved")                                 # check 2
    out = []
    region = province = commune = None
    for r in rows[3:]:
        if r[1] is None:
            continue
        name = str(r[1]).strip()
        lvl = level_of(name)
        code = None if r[0] is None else int(r[0])
        if lvl == "region":
            region = code
        elif lvl == "province":
            province = code
        elif lvl == "commune":
            commune = code
        pct = r[COL0:COL0 + 5]
        ok = all(isinstance(x, (int, float)) for x in pct)
        if not ok and set(pct) != {"."}:
            raise SystemExit(f"{name}: language cells {pct}")
        out.append(dict(code=code, name=name, level=lvl, region=region, province=province,
                        commune=commune if lvl == "arrondissement" else None,
                        legal=int(r[2]), muni=int(r[3]),
                        **{k: (float(v) if ok else None) for k, v in zip(LANGS, pct)}))
    df = pd.DataFrame(out)
    for c in ("code", "region", "province", "commune"):
        df[c] = df[c].astype("Int64")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch or not (RAW / NAME).exists():
        fetch()
    df = read()
    n = df["level"].value_counts()
    print("  rows:", dict(n))
    if (n.get("region"), n.get("province"), n.get("commune"), n.get("arrondissement")) != (12, 75, 1503, 41):
        raise SystemExit("expected 12 régions, 75 provinces, 1,503 communes, 41 arrondissements")

    # ---- leaves: arrondissements, and communes that have none ----
    arr = df[df["level"] == "arrondissement"]
    parents = set(arr["commune"])
    com = df[df["level"] == "commune"]
    for c in parents:
        p = com[com["code"] == c]
        kids = arr[arr["commune"] == c]
        # commune <province>010, its arrondissements <province>01<nn>
        if len(p) != 1 or not all(str(k)[:-2] == str(c)[:-1] for k in kids["code"]):
            raise SystemExit(f"arrondissements of {c} do not nest by code")
        if (int(kids["legal"].sum()), int(kids["muni"].sum())) != (int(p["legal"].iloc[0]), int(p["muni"].iloc[0])):
            raise SystemExit(f"arrondissements of {p['name'].iloc[0]} do not add to it")   # check 3
    print(f"  {len(parents)} communes split into {len(arr)} arrondissements: "
          + ", ".join(sorted(com[com['code'].isin(parents)]['name'].str.replace('Commune de ', ''))))
    leaves = pd.concat([com[~com["code"].isin(parents)], arr], ignore_index=True)
    if len(leaves) != 1503 - len(parents) + 41:
        raise SystemExit("leaf count")
    if not all(str(c).startswith(str(p)) for c, p in zip(leaves["code"], leaves["province"])):
        raise SystemExit("a leaf's code does not start with its province's")

    prov = df[df["level"] == "province"].set_index("code")
    for col in ("legal", "muni"):
        s = leaves.groupby("province")[col].sum()
        bad = (s - prov[col]).abs()
        if bad.max() > 0:
            raise SystemExit(f"leaves do not add to provinces' {col}: {bad[bad > 0].head()}")
    if (int(leaves["legal"].sum()), int(leaves["muni"].sum())) != (NATIONAL_LEGAL, NATIONAL_MUNI):
        raise SystemExit("leaves do not add to the nation")                               # check 3
    print(f"  {len(leaves):,} leaves add to 75 provinces and the nation: légale "
          f"{NATIONAL_LEGAL:,}, municipale {NATIONAL_MUNI:,}")

    # ---- check 4: provinces' printed shares against their leaves ----
    have = leaves[leaves["Darija"].notna()]
    starred = set(leaves.loc[leaves["Darija"].isna(), "province"])
    for base in ("legal", "muni"):
        miss = []
        for p, g in have.groupby("province"):
            for L in LANGS:
                est = (g[L] * g[base]).sum() / g[base].sum()
                miss.append((abs(est - prov.loc[p, L]), p, f"{prov.loc[p, 'name']} {L}: printed "
                             f"{prov.loc[p, L]}, leaves {est:.2f}"))
        miss.sort(reverse=True)
        clean = [m for m in miss if m[1] not in starred]
        print(f"  check 4, weight {base}: mean miss {sum(m[0] for m in miss) / len(miss):.3f}; "
              f"without the starred communes' provinces mean {sum(m[0] for m in clean) / len(clean):.3f}, "
              f"worst {clean[0][0]:.2f} ({clean[0][2]})")
        if base == "muni":
            for m in miss[:6]:
                print(f"      {m[0]:6.2f} {m[2]}")
            if clean[0][0] > 0.35:
                raise SystemExit("provinces' shares do not follow from their communes")
    for i, L in enumerate(LANGS):                                                          # check 5
        est = (have[L] * have["muni"]).sum() / have["muni"].sum()
        print(f"  check 5 {L:10s} printed {NATIONAL_PCT[i]:5.1f}  leaves {est:6.2f}")
        if abs(est - NATIONAL_PCT[i]) > 0.1:
            raise SystemExit("national shares do not follow from the leaves")

    # ---- counts ----
    nodata = leaves[leaves["Darija"].isna()]
    print(f"  not drawn, no figures ('.'): {', '.join(nodata['name'])} = {int(nodata['muni'].sum()):,}")
    have = have.copy()
    have["sum"] = have[list(LANGS)].sum(axis=1)
    have["div"] = have["sum"].clip(lower=100)
    short = (have["muni"] * (100 - have["sum"]).clip(lower=0) / 100).sum()
    print(f"  shares add to {have['sum'].min():.1f}-{have['sum'].max():.1f}; {int((have['sum'] < 99.95).sum())} "
          f"leaves under 100; people using none of the five, not drawn: {short:,.0f}")
    rows = []
    for _, r in have.iterrows():
        for L in LANGS:
            rows.append(dict(geo_id=str(r["code"]), geo_level=r["level"], geo_name=r["name"],
                             province=str(r["province"]), source_category=L, pct=r[L],
                             pop_municipale=int(r["muni"]),
                             count=r["muni"] * r[L] / r["div"], tier="derived"))
    out = pd.DataFrame(rows)
    drawn = out["count"].sum()
    print(f"  drawn {drawn:,.0f} of municipale {NATIONAL_MUNI:,} "
          f"(not drawn: '.' {int(nodata['muni'].sum()):,}, none of the five {short:,.0f})")
    for L, c in out.groupby("source_category")["count"].sum().sort_values(ascending=False).items():
        print(f"    {L:10s} {c:>12,.0f}  {100 * c / drawn:5.2f}%")
    NORM.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(NORM, index=False, encoding="utf-8")
    print(f"  wrote {NORM} ({len(out):,} rows, {out['geo_id'].nunique():,} units)")


if __name__ == "__main__":
    main()
