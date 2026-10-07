"""Bhutan: mother tongue by dzongkhag from the 2015 Gross National Happiness survey, on the 2017
census's people; non-Bhutanese by UN DESA origin through each origin's home mix.

    python sources/bt_gnh.py          -> data/normalized/bt.csv

No census asks language (PHCB 2005 and 2017: literacy in Dzongkha, English, Lhotshamkha only;
MICS 2010 has no language variable, its DDI checked 2026-10-05). The Centre for Bhutan & GNH
Studies' 2015 survey asked mother tongue and published it per dzongkhag, weighted: *A Compass
Towards a Just and Harmonious Society* (2016), Table A1.5, pdf pp.300-301. religiondots already
downloaded and read that PDF for its Hindu placement (../religiondots/sources/bt.py); it is read
here from there, read-only, as are religiondots' dzongkhag lookup (2017 census Bhutanese and
non-Bhutanese, Tables 2.6 and 2.8) and the UN DESA migrant-stock workbook.

  Bhutanese_d x A1.5 share_d          rows `modelled`, labels as printed
  non-Bhutanese_d x DESA 2020 mix     rows `modelled`, origin_mix.mix(iso, "bt") node ids
Each dzongkhag's rows sum to its census population exactly (largest remainder, asserted).
Record: sources/bt.md.
"""
import io
import os
import re
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from rdlink import RD  # noqa: E402
from origin_mix import mix  # noqa: E402

GNH2015 = RD / "data" / "raw" / "bt" / "gnh_2015_compass_report.pdf"
LOOKUP = RD / "data" / "geo" / "bt" / "bt_lookup.csv"
DESA = RD / "data" / "raw" / "mr" / "undesa_pd_2024_ims_stock_by_sex_destination_and_origin.xlsx"
OUT = ROOT / "data" / "normalized" / "bt.csv"

A15_PAGES = (300, 301)
A15_COLUMNS = ["Dzongkha", "Cho-cha nga-chakha", "Tshangla", "Bumthangkha", "Khengkha", "Kurtop",
               "Nyenkha", "Dzala", "Dakpa", "Chali kha", "Monpakha", "Brokpa", "Lakha", "Bokha",
               "Nepali", "Lhokpu", "Gongduk", "Lepcha", "Layap", "English", "Others", "Total"]
# the table's own Bhutan row, pinned: a witness on the read
NATIONAL = {"Dzongkha": 21.13, "Tshangla": 33.72, "Nepali": 18.69, "Khengkha": 8.05,
            "Others": 4.28}

DESA_YEAR = "2020.0"
DESA_ORIGINS = {"India": "IN", "Nepal": "NP", "China": "CN", "Japan": "JP", "Republic of Korea": "KR",
                "Bangladesh": "BD", "Pakistan": "PK", "Sri Lanka": "LK", "Myanmar": "MM",
                "Philippines": "PH", "Singapore": "SG", "Thailand": "TH", "Denmark": "DK",
                "Sweden": "SE", "United Kingdom": "GB", "Italy": "IT", "France": "FR",
                "Germany": "DE", "Netherlands": "NL", "Switzerland": "CH", "Canada": "CA",
                "United States of America": "US", "Australia": "AU", "New Zealand": "NZ"}
DESA_WORLD_2020 = 53_612
# Indians in Bhutan are mostly labourers from the four nearest states, not a cross-section of
# India (BhutanWiki, "Foreign Workers in Bhutan", read 2026-10-05: "Most Indian workers in Bhutan
# come from the Indian states of West Bengal, Assam, Bihar, and Jharkhand"). Their mix is those
# states' drawn counts on this map (2011 state codes), pooled, 1% cut, rescaled: origin_mix's own
# home-mix rule applied to four states instead of all India.
IN_STATES = {"19": "West Bengal", "18": "Assam", "10": "Bihar", "20": "Jharkhand"}
MIN_SHARE = 0.01


def pdf_lines(path, pages):
    import fitz

    with open(path, "rb") as fh:
        if b"%%EOF" not in fh.read()[-4096:]:
            raise SystemExit(f"{path} has no %%EOF trailer; the download is truncated")
    doc = fitz.open(path)
    out = []
    for p in pages:
        out += [ln.strip() for ln in doc[p - 1].get_text().splitlines() if ln.strip()]
    return out


def read_a15(names):
    lines = pdf_lines(GNH2015, A15_PAGES)
    if not any(" ".join(ln.split()).startswith("Table A1.5: Distribution of Dzongkhag population by "
                                               "mother tongue") for ln in lines):
        raise SystemExit("Table A1.5 is not on its pages")
    out = {}
    for name in list(names) + ["Bhutan"]:
        hits = [i for i, ln in enumerate(lines) if ln == name]
        if len(hits) != 1:
            raise SystemExit(f"A1.5: {name!r} occurs {len(hits)} times")
        vals, j = [], hits[0] + 1
        while len(vals) < len(A15_COLUMNS):
            toks = lines[j].split()
            if not all(re.fullmatch(r"[0-9]+(\.[0-9]+)?", t) for t in toks):
                raise SystemExit(f"A1.5: {name} reached {lines[j]!r} after {len(vals)} numbers")
            vals += [float(t) for t in toks]
            j += 1
        if len(vals) != len(A15_COLUMNS):
            raise SystemExit(f"A1.5: {name} has {len(vals)} numbers")
        row = dict(zip(A15_COLUMNS, vals))
        if row["Total"] != 100 or abs(sum(vals[:-1]) - 100) > 0.1:
            raise SystemExit(f"A1.5: {name}'s row sums to {sum(vals[:-1]):.2f}")
        out[name] = row
    bad = {k: (out["Bhutan"][k], v) for k, v in NATIONAL.items() if out["Bhutan"][k] != v}
    if bad:
        raise SystemExit(f"A1.5's Bhutan row moved: {bad}")
    print("GNH 2015 Table A1.5: 20 dzongkhags and Bhutan read, every row sums to 100 (+-0.1)")
    return out


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
    ycol = cols.index(DESA_YEAR)                     # the first 2020 column: both sexes
    body = df.iloc[hdr + 1:]
    m = body[body[dcol].astype(str).str.strip().str.rstrip("*").str.strip() == "Bhutan"]
    name = m[ocol].astype(str).str.strip().str.rstrip("*").str.strip()
    world = int(pd.to_numeric(m.loc[name == "World", ycol]).iloc[0])
    if world != DESA_WORLD_2020:
        raise SystemExit(f"DESA 2020 world stock for Bhutan is {world:,}")
    code = pd.to_numeric(m[ccol], errors="coerce")
    ctry = m[(code < 900) | (name == "Others")]
    stock = dict(zip(ctry[ocol].astype(str).str.strip().str.rstrip("*").str.strip(),
                     pd.to_numeric(ctry[ycol]).astype(int)))
    if set(stock) != set(DESA_ORIGINS) | {"Others"}:
        raise SystemExit(f"DESA's origins for Bhutan changed: {sorted(set(stock) ^ set(DESA_ORIGINS))}")
    if sum(stock.values()) != world:
        raise SystemExit(f"DESA's origins sum to {sum(stock.values()):,}, world {world:,}")
    named = {DESA_ORIGINS[k]: v for k, v in stock.items() if k in DESA_ORIGINS}
    print(f"UN DESA 2020: {world:,} migrants in Bhutan; named origins {sum(named.values()):,} "
          f"(India {named['IN']:,}, Nepal {named['NP']:,}, China {named['CN']:,}); "
          f"`Others` {stock['Others']:,} taken to be like them")
    return named


def india_mix():
    sys.path.insert(0, str(ROOT / "taxonomy"))
    from countries import load_one
    df = load_one("in")["counts"]()
    df = df[df["unit"].astype(str).str[:2].isin(IN_STATES)]
    s = df.groupby("node")["count"].sum()
    s = s[s > 0] / s.sum()
    s = s[s >= MIN_SHARE]
    s = s / s.sum()
    print("Indians: " + ", ".join(f"{k.split('.')[-1]} {v:.1%}" for k, v in
                                  s.sort_values(ascending=False).items()))
    return s.to_dict()


def round_rows(m, totals):
    """Largest remainder within each row, so every row sums to its integer total."""
    out = {}
    for g, row in m.iterrows():
        fl = row.apply(int)
        short = int(totals[g]) - int(fl.sum())
        order = (row - fl).sort_values(ascending=False).index[:short]
        fl[order] += 1
        out[g] = fl
    return pd.DataFrame(out).T


def main():
    lut = pd.read_csv(LOOKUP, dtype={"geo_id": str}).set_index("geo_id")
    if len(lut) != 20 or (lut["pop"] != lut["bhutanese"] + lut["non_bhutanese"]).any():
        raise SystemExit(f"{LOOKUP}: {len(lut)} dzongkhags or pop != Bhutanese + non-Bhutanese")
    a15 = read_a15(lut["gnh_name"])
    labels = A15_COLUMNS[:-1]

    share = pd.DataFrame({g: {k: a15[lut.loc[g, "gnh_name"]][k] for k in labels}
                          for g in lut.index}).T
    share = share.div(share.sum(axis=1), axis=0)     # rows sum to 100 +-0.1 as printed
    nat = round_rows(share.mul(lut["bhutanese"], axis=0), lut["bhutanese"])

    named = desa_origins()
    tot = float(sum(named.values()))
    comp = {}
    for iso, w in named.items():
        for node, s in (india_mix() if iso == "IN" else mix(iso, "bt")).items():
            comp[node] = comp.get(node, 0.0) + (w / tot) * s
    fshare = pd.DataFrame({g: comp for g in lut.index}).T
    ext = round_rows(fshare.mul(lut["non_bhutanese"], axis=0), lut["non_bhutanese"])

    rows = []
    for part, df, src in (("Bhutanese", nat, "gnh2015_a15_x_phcb2017"),
                          ("non-Bhutanese", ext, "phcb2017_non_bhutanese_x_desa2020")):
        s = df.stack().rename("count").reset_index()
        s.columns = ["geo_id", "source_category", "count"]
        s["source_id"] = src
        s["note"] = part
        rows.append(s[s["count"] > 0])
    out = pd.concat(rows, ignore_index=True)
    out["geo_level"] = "dzongkhag"
    out["geo_name"] = out["geo_id"].map(lut["name"])
    out["tier"] = "modelled"
    out["year"] = 2017
    chk = out.groupby("geo_id")["count"].sum()
    if not chk.equals(lut["pop"].astype(chk.dtype).reindex(chk.index)) or len(chk) != 20:
        raise SystemExit("a dzongkhag's rows do not sum to its 2017 census population")
    print(f"every dzongkhag sums to its 2017 census count; total {int(chk.sum()):,} "
          f"(Bhutanese {int(nat.values.sum()):,}, non-Bhutanese {int(ext.values.sum()):,})")
    t = out.groupby("source_category")["count"].sum().sort_values(ascending=False)
    for k, v in t.head(25).items():
        print(f"   {v:>8,}  {v / t.sum():6.2%}  {k}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out[["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "source_id",
         "year", "note"]].to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} rows")


if __name__ == "__main__":
    main()
