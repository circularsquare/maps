"""Gabon: Afrobarometer home language by province (R6-R9 pooled, French from R6 only), on the
RGPL 2026's Gabonese per province; foreign residents by UN DESA 2020 origin through each
origin's home mix.

    python sources/ga_afro.py          -> data/normalized/ga.csv

No census asks language (RGPL 2013 Resultats globaux: nationality, no language; RGPL 2026: only
totals out). Afrobarometer is in Gabon in R6 (2015), R7 (2017), R8 (2020), R9 (2022), ~1,200
adult citizens a round, region coded in all four (religiondots' merged files, read-only, through
sources/mono_afro.py).

LINGUA FRANCA (ask 018; Cameroon's rule, sources/cm.md): R6 asked "home language" and 29% said
French; R7-R9 asked "language spoken in home" and 73-75% did. As Cameroon, Kenya and Tanzania's
first-language reading, French's share per province comes from R6 alone (LF_ROUNDS), and the
other languages' from all four rounds among the non-French answers, scaled to what French
leaves. LF_ROUNDS = [6, 7, 8, 9] draws French as answered. Since Anita's ruling (2026-10-05)
LF_ROUNDS = "R7Q2A": French at R7's separate mother-tongue question, by province.

Population: religiondots' ga_lookup.csv (RGPL 2026 province totals; Gabonese and foreign per
province by its IPF on the 2013 shares; read-only). Foreigners (1,200,256): UN DESA International
Migrant Stock 2020's named origins for Gabon (405,666 of 416,651; `Others` taken to be alike),
each through origin_mix.mix(iso, "ga"), one national mix in every province.
"""
import io
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from rdlink import RD, RD_GEO  # noqa: E402
from mono_afro import extract, gkey  # noqa: E402
from origin_mix import mix  # noqa: E402

LOOKUP = RD_GEO / "ga" / "ga_lookup.csv"
DESA = RD / "data" / "raw" / "mr" / "undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx"
OUT = ROOT / "data" / "normalized" / "ga.csv"
LF = "French"
LF_ROUNDS = "R7Q2A"     # ask 018 ruling, 2026-10-05; [6] was the reading before it
GABONESE_2026, FOREIGN_2026 = 2_318_365, 1_200_256

LABELS = {   # survey answer -> source_category (spelling variants merged); None = not drawn
    "French": "French", "Fang": "Fang", "Punu/Mériè": "Punu", "Nzébi/Métié": "Nzebi",
    "Mbédè": "Mbede", "Kota": "Kota", "Tsogho": "Tsogho", "Myénè": "Myene",
    "Baloumbou": "Baloumbou", "Kélé": "Kele", "Kélè": "Kele", "Masangu": "Masangu",
    "Eshira": "Eshira", "Bateke": "Bateke", "English": "English", "Bavungu": "Bavungu",
    "Vili": "Vili", "Other": "Other",
}
DESA_YEAR = "2020.0"
DESA_ORIGINS = {
    "Burundi": "BI", "Rwanda": "RW", "Angola": "AO", "Cameroon": "CM", "Chad": "TD",
    "Congo": "CG", "Democratic Republic of the Congo": "CD", "Equatorial Guinea": "GQ",
    "Sao Tome and Principe": "ST", "Algeria": "DZ", "Morocco": "MA", "Tunisia": "TN",
    "South Africa": "ZA", "Benin": "BJ", "Burkina Faso": "BF", "Côte d'Ivoire": "CI",
    "Ghana": "GH", "Guinea": "GN", "Mali": "ML", "Nigeria": "NG", "Senegal": "SN", "Togo": "TG",
    "China": "CN", "Dem. People's Republic of Korea": "KP", "Japan": "JP",
    "Republic of Korea": "KR", "Israel": "IL", "Lebanon": "LB", "Syrian Arab Republic": "SY",
    "Belgium": "BE", "France": "FR", "Germany": "DE", "Canada": "CA",
    "United States of America": "US"}
DESA_WORLD_2020 = 416_651


def desa_origins():
    with zipfile.ZipFile(DESA) as z:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
            for n in z.namelist():
                if n != "xl/styles.xml":            # openpyxl is slow on the stylesheet
                    out.writestr(n, z.read(n))
    buf.seek(0)
    df = pd.read_excel(buf, sheet_name="Table 1", header=None, engine="openpyxl")
    hdr = next(i for i in range(2, 20)
               if any("of destination" in str(x) for x in df.iloc[i])
               and any("of origin" in str(x) for x in df.iloc[i]))
    cols = [str(x).strip() for x in df.iloc[hdr]]
    dcol = next(i for i, x in enumerate(cols) if "of destination" in x)
    ocol = next(i for i, x in enumerate(cols) if "of origin" in x)
    ccol = next(i for i, x in enumerate(cols) if x == "Location code of origin")
    ycol = cols.index(DESA_YEAR)
    body = df.iloc[hdr + 1:]
    m = body[body[dcol].astype(str).str.strip().str.rstrip("*").str.strip() == "Gabon"]
    name = m[ocol].astype(str).str.strip().str.rstrip("*").str.strip()
    world = int(pd.to_numeric(m.loc[name == "World", ycol]).iloc[0])
    if world != DESA_WORLD_2020:
        raise SystemExit(f"DESA 2020 world stock for Gabon is {world:,}")
    code = pd.to_numeric(m[ccol], errors="coerce")
    ctry = m[(code < 900) | (name == "Others")]
    stock = dict(zip(ctry[ocol].astype(str).str.strip().str.rstrip("*").str.strip(),
                     pd.to_numeric(ctry[ycol]).astype(int)))
    if set(stock) != set(DESA_ORIGINS) | {"Others"}:
        raise SystemExit(f"DESA's origins for Gabon changed: {sorted(set(stock) ^ set(DESA_ORIGINS))}")
    if sum(stock.values()) != world:
        raise SystemExit("DESA's origins do not sum to the world total")
    named = {DESA_ORIGINS[k]: v for k, v in stock.items() if k in DESA_ORIGINS}
    print(f"UN DESA 2020: {world:,} migrants in Gabon, named {sum(named.values()):,}; top: " +
          ", ".join(f"{k} {v:,}" for k, v in sorted(named.items(), key=lambda kv: -kv[1])[:8]))
    return named


def round_rows(m, totals):
    out = {}
    for g, row in m.iterrows():
        fl = row.apply(int)
        short = int(totals[g]) - int(fl.sum())
        fl[(row - fl).sort_values(ascending=False).index[:short]] += 1
        out[g] = fl
    return pd.DataFrame(out).T


def main():
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str}).set_index("geo_id")
    if len(lut) != 9 or int(lut["pop"].sum()) != GABONESE_2026 \
            or int(lut["foreign"].sum()) != FOREIGN_2026:
        raise SystemExit(f"{LOOKUP} moved: {len(lut)} provinces, {lut['pop'].sum():,} Gabonese")
    norm = {gkey(n): g for g, n in lut["name"].items()}

    a = extract({"gabon"})
    a["unit"] = a["region"].map(gkey).map(norm)
    bad = sorted(a.loc[a["unit"].isna(), "region"].unique())
    if bad:
        raise SystemExit(f"REGION labels with no province: {bad}")
    miss = sorted(set(a["lang"]) - set(LABELS))
    if miss:
        raise SystemExit(f"answers with no label: {miss}")
    a["cat"] = a["lang"].map(LABELS)
    a = a[a["cat"].notna()]
    print(f"Afrobarometer Gabon: {len(a):,} respondents, rounds {sorted(a['round'].unique())}; "
          f"per province: " + ", ".join(f"{lut.loc[u, 'name']} {n}" for u, n in
                                       a["unit"].value_counts().items()))

    if LF_ROUNDS == "R7Q2A":
        # Anita's ruling (ask 018, 2026-10-05): French at R7's mother-tongue question (Q2A),
        # each province's share shrunk to the national one by wafr_afro.K_SHRINK respondents
        from wafr_afro import r7_mother, shrink
        t = r7_mother("Gabon", {LF: ["French"]})
        reg = {r: norm.get(gkey(r)) for r in t.index if r != "_national"}
        if None in reg.values() or len(set(reg.values())) != len(lut):
            raise SystemExit(f"R7 REGION labels not one province each: {reg}")
        lfs = shrink(t, LF).rename(reg).reindex(lut.index)
        print(f"R7 Q2A French nationally {t.loc['_national', LF + '_A'] / t.loc['_national', 'n']:.2%}")
    else:
        lf = a[a["round"].isin(LF_ROUNDS)]
        lfs = (lf[lf["cat"] == LF].groupby("unit")["w"].sum() / lf.groupby("unit")["w"].sum()) \
            .reindex(lut.index).fillna(0)
    rest = a[a["cat"] != LF]
    rs = rest.groupby(["unit", "cat"])["w"].sum().unstack(fill_value=0)
    rs = rs.div(rs.sum(axis=1), axis=0).reindex(lut.index)
    if rs.isna().any().any():
        raise SystemExit("a province has no non-French answers")
    share = rs.mul(1 - lfs, axis=0)
    share[LF] = lfs
    print("French share by province (R%s): " % LF_ROUNDS +
          ", ".join(f"{lut.loc[u, 'name']} {v:.0%}" for u, v in lfs.items()))
    nat = round_rows(share.mul(lut["pop"], axis=0), lut["pop"])

    named = desa_origins()
    tot = float(sum(named.values()))
    comp = {}
    for iso, w in named.items():
        for node, s in mix(iso, "ga").items():
            comp[node] = comp.get(node, 0.0) + (w / tot) * s
    fsh = pd.DataFrame({g: comp for g in lut.index}).T
    ext = round_rows(fsh.mul(lut["foreign"], axis=0), lut["foreign"])

    rows = []
    for df, tier, src, note in ((nat, "modelled", "afrobarometer_r6-r9_x_rgpl2026", "Gabonese"),
                                (ext, "derived", "desa2020_x_rgpl2026", "foreign residents")):
        s = df.stack().rename("count").reset_index()
        s.columns = ["geo_id", "source_category", "count"]
        s = s[s["count"] > 0].copy()
        s["tier"], s["source_id"], s["note"] = tier, src, note
        rows.append(s)
    out = pd.concat(rows, ignore_index=True)
    out["geo_level"] = "province"
    out["geo_name"] = out["geo_id"].map(lut["name"])
    out["year"] = 2026
    chk = out.groupby("geo_id")["count"].sum()
    if (chk.reindex(lut.index) != lut["total_2026"]).any():
        raise SystemExit("a province does not sum to its RGPL 2026 total")
    t = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"every province sums to RGPL 2026; total {int(t.sum()):,}")
    for k, v in t.head(22).items():
        print(f"   {v:>9,}  {v / t.sum():6.2%}  {k}")
    gab = nat.sum().sort_values(ascending=False) / GABONESE_2026
    print("Gabonese only: " + ", ".join(f"{k} {v:.1%}" for k, v in gab.head(8).items()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out[["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "source_id",
         "year", "note"]].to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} rows")


if __name__ == "__main__":
    main()
