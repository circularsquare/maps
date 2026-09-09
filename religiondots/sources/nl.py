"""The Netherlands — CBS's own religion table for 403 gemeenten, which nobody had opened.

Writes data/normalized/nl.csv.

Usage:
    python sources/nl.py --fetch     # three CBS files, about 0.3 MB
    python sources/nl.py             # rebuild from data/raw/nl/

WHAT THIS IS. `Religie en kerkbezoek naar gemeente 2010-2014` is a CBS maatwerk table:
nine named denominations and the total religious share, as percentages of the population
aged 18 and over, for every one of the 403 gemeenten of the 2014 classification. It is a
pool of five years of the Enquete Beroepsbevolking, 460,000 adults, which is why a country
of 17 million can be cut 403 ways at all. §11k had closed the Netherlands on StatLine
`82904NED`, which is the same question asked of a much smaller survey and published
nationally; the maatwerk shelf is a different shelf and was never looked at.

THE CATEGORY LIST IS THE POINT. Katholiek, Hervormd, Gereformeerd, PKN, Islam, Joods,
Hindoe, Boeddhist, Anders. Three separate Reformed answers is not a quirk of the form: it is
what a Dutch respondent means, and the three have different maps. Staphorst is 47.4%
hervormd, Urk 52.3% gereformeerd and Dongeradeel 32.4% PKN, and no European survey
instrument on this map can see any of that. `rlgdnanl` in the European Social Survey would
have given a similar list at NUTS 2, twelve provinces; this gives it at 42,000 people a unit.

THE VINTAGE IS 2010-2014 AND IT IS NOT CURRENT. The religion question left the EBB after
2015 and CBS's live series, `Sociale samenhang en welzijn`, publishes at COROP with four
categories instead of nine. The Netherlands has secularised measurably since: this pool puts
52.8% of the country in some denomination and the 2021/2025 SSW figure is 42.9%. That is a
decade of real change rather than a discrepancy, and build() prints the province-by-province
comparison so the size of it is in the repo rather than in a note. It is not used to adjust
anything: the two are different instruments with different age bases and CBS treats them as
different series, so re-levelling one onto the other would invent a number neither published.

THE SHARES ARE OF ADULTS AND ARE APPLIED TO EVERYBODY. CBS asked people aged 18 and over,
and 20.6% of the Netherlands was under 18 in 2014. [[feedback_leave_children_out]]: scale the
shares up rather than leave 3.5 million people off the map, and say so in note_public.

NINE GEMEENTEN ARE SUPPRESSED, which is the whole of the hole. The table's own footnote is
`minimaal 150 waarnemingen per gemeente`, so four Wadden islands (Ameland, Schiermonnikoog,
Terschelling, Vlieland) and five small mainland gemeenten (Rozendaal, Renswoude,
Graft-De Rijp, Schermer, Zeevang) print no figures at all. 35,169 people, 0.209% of the
country. They are not filled in from their province: spec §3.5's line is that a hole is
marked rather than filled, and CBS withheld these because it does not know.
"""

import argparse
import json
import os
import sys
import urllib.request

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "taxonomy"))
RAW = os.path.join(ROOT, "data", "raw", "nl")
OUT = os.path.join(ROOT, "data", "normalized", "nl.csv")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}

# --- the source ---------------------------------------------------------------------------
XLS_URL = ("https://www.cbs.nl/-/media/imported/documents/2015/20/"
           "religie-en-kerkbezoek-naar-gemeente-2010-2014.xls?sc_lang=nl-nl")
XLS = os.path.join(RAW, "religie_kerkbezoek_gemeente_2010_2014.xls")

# The live series, for the comparison at the end. Not drawn.
SSW_URL = "https://www.cbs.nl/-/media/_excel/2026/11/religie_2025_tabellen.xlsx"
SSW = os.path.join(RAW, "religie_2025_tabellen.xlsx")

# CBS StatLine `Regionale kerncijfers Nederland`, population on 1 January 2014, per gemeente.
# The table carries every gemeentecode CBS has ever issued and leaves the dead ones null, so
# the 403 live rows fall out of it without a classification file.
POP_URL = ("https://opendata.cbs.nl/ODataApi/odata/70072ned/TypedDataSet"
           "?$format=json&$filter=Perioden%20eq%20%272014JJ00%27"
           "&$select=RegioS,Perioden,TotaleBevolking_1")
POP = os.path.join(RAW, "pop_gemeente_2014.json")

# --- what the table is -------------------------------------------------------------------
# Column order in the sheet, left to right, after province / gemeente / gemeentecode.
CATS = ["Katholiek", "Hervormd", "Gereformeerd", "PKN", "Islam", "Joods", "Hindoe",
        "Boeddhist", "Anders"]
NO_RELIGION = "Geen"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

N_UNITS = 403
N_DRAWN = 394
N_SUPPRESSED = 9
POP_2014 = 16_829_289          # CBS's own national total on 1 January 2014
SUPPRESSED_POP = 35_169
RESPONDENTS = 460_000          # the sheet's own Toelichting, rounded as CBS rounds it

SUPPRESSED = {
    "GM0060": "Ameland", "GM0088": "Schiermonnikoog", "GM0093": "Terschelling",
    "GM0096": "Vlieland", "GM0277": "Rozendaal", "GM0339": "Renswoude",
    "GM0365": "Graft-De Rijp", "GM0458": "Schermer", "GM0478": "Zeevang",
}


# =======================================================================================
# fetch
# =======================================================================================

def _get(url, dest, note=""):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"  already on disk: {os.path.basename(dest)} "
              f"({os.path.getsize(dest):,} bytes)")
        return
    print("  GET", os.path.basename(dest), note)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=900) as r:
        body = r.read()
    with open(dest, "wb") as fh:
        fh.write(body)
    print(f"    {len(body):,} bytes")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    print("CBS maatwerk, religion by gemeente 2010/2014…")
    _get(XLS_URL, XLS)
    print("CBS population by gemeente, 1 January 2014…")
    _get(POP_URL, POP)
    print("CBS SSW religion by region 2021/2025, for the comparison only…")
    _get(SSW_URL, SSW)


# =======================================================================================
# the table
# =======================================================================================

def _table():
    """The maatwerk sheet as a tidy frame, one row per gemeente.

    THE SHEET HAS NO USABLE HEADER ROW. Its captions are spread over rows 2 to 5 with a
    `w.v.` spanning cell and a blank spacer column between the total and the breakdown, so
    the columns are taken by POSITION and then checked against the caption text below. A
    header-name parse would break on the spacer and a bare positional read would break
    silently if CBS ever re-ordered the sheet; doing both is the cheap insurance.
    """
    d = pd.read_excel(XLS, "Tabel", header=None)
    title = str(d.iloc[0, 0])
    if "403 gemeenten" not in title:
        sys.exit(f"!! sheet title is not the 403-gemeente table: {title!r}")
    caption = [str(x).strip() for x in d.iloc[4, 6:15]]
    if caption != CATS:
        sys.exit(f"!! the nine denomination captions moved: {caption} != {CATS}")
    if "Kerkelijke gezindte" not in str(d.iloc[2, 4]):
        sys.exit(f"!! column 4 is not the religious total: {d.iloc[2, 4]!r}")

    b = d.iloc[7:7 + N_UNITS, [0, 1, 2, 4] + list(range(6, 15))].copy()
    b.columns = ["prov", "name", "code", "total_rel"] + CATS
    b["unit"] = b["code"].astype(int).map(lambda c: "GM%04d" % c)
    b["name"] = b["name"].astype(str).str.strip()
    b["prov"] = b["prov"].astype(str).str.strip()
    for c in ["total_rel"] + CATS:
        b[c] = pd.to_numeric(b[c], errors="coerce")
    if len(b) != N_UNITS or b["unit"].duplicated().any():
        sys.exit(f"!! {len(b)} rows, {b['unit'].duplicated().sum()} duplicated codes")

    blank = b[b["total_rel"].isna()]
    if set(blank["unit"]) != set(SUPPRESSED):
        sys.exit(f"!! the suppressed gemeenten moved: {sorted(blank['unit'])}")
    if b[CATS].isna().sum().sum() != N_SUPPRESSED * len(CATS):
        sys.exit("!! a gemeente is blank in some columns and not others")
    return b


def _population():
    """{gemeentecode -> population on 1 January 2014}, CBS's own."""
    rows = json.load(open(POP, encoding="utf-8"))["value"]
    p = {r["RegioS"].strip(): r["TotaleBevolking_1"] for r in rows
         if r["RegioS"].strip().startswith("GM") and r["TotaleBevolking_1"]}
    if len(p) != N_UNITS:
        sys.exit(f"!! {len(p)} gemeenten have a 2014 population, expected {N_UNITS}")
    if abs(sum(p.values()) - POP_2014) > 1:
        sys.exit(f"!! population sums to {sum(p.values()):,.0f}, not {POP_2014:,}")
    return p


def _ssw():
    """CBS `Religie naar regio, 2021/2025`, Tabel 1: religious share per province, now."""
    t = pd.read_excel(SSW, "Tabel 1", header=None)
    out = {}
    for _, r in t.iloc[7:19].iterrows():
        if isinstance(r[0], str) and r[0].strip():
            out[r[0].strip()] = float(r[2])
    if len(out) != 12:
        sys.exit(f"!! SSW Tabel 1 gave {len(out)} provinces")
    return out


# =======================================================================================
# build
# =======================================================================================

def build():
    import nl2014

    b = _table()
    pop = _population()
    b["pop"] = b["unit"].map(pop)
    print(f"CBS maatwerk: {len(b)} gemeenten, {b['pop'].sum():,.0f} people")

    drawn = b[b["total_rel"].notna()].copy()
    print(f"  {len(drawn)} with figures, {N_SUPPRESSED} suppressed "
          f"({SUPPRESSED_POP:,} people, "
          f"{100 * SUPPRESSED_POP / b['pop'].sum():.3f}% of the country): "
          + ", ".join(sorted(SUPPRESSED.values())))
    if len(drawn) != N_DRAWN:
        sys.exit(f"!! {len(drawn)} drawn gemeenten, expected {N_DRAWN}")
    if abs(b.loc[b['unit'].isin(SUPPRESSED), 'pop'].sum() - SUPPRESSED_POP) > 1:
        sys.exit("!! the suppressed population moved")

    # THE NINE PARTS ARE ROUNDED TO A TENTH AND THE TOTAL IS ROUNDED SEPARATELY, so they do
    # not add up: the gap runs to half a point in the worst gemeente. Both are CBS's own
    # published numbers, so neither is thrown away — `Geen` is 100 minus the PUBLISHED total
    # and the nine parts are scaled to fill exactly that total. The alternative, taking
    # `Geen` as 100 minus the sum of the parts, would put the whole rounding error into the
    # largest category in the country.
    parts = drawn[CATS].sum(axis=1)
    resid = drawn["total_rel"] - parts
    print(f"  rounding: parts vs published total, worst {resid.abs().max():.2f} points "
          f"({drawn.loc[resid.abs().idxmax(), 'name']}), median "
          f"{resid.abs().median():.2f}")
    if resid.abs().max() > 1.0:
        sys.exit("!! the nine parts miss the published total by more than a point somewhere")
    scale = drawn["total_rel"] / parts.replace(0, pd.NA)

    rows = []
    for (_, r), s in zip(drawn.iterrows(), scale):
        n = float(r["pop"])
        for c in CATS:
            v = float(r[c]) * float(s)
            if v <= 0:
                continue
            rows.append((r["unit"], r["name"], c, v * n / 100.0))
        rows.append((r["unit"], r["name"], NO_RELIGION,
                     (100.0 - float(r["total_rel"])) * n / 100.0))

    df = pd.DataFrame(rows, columns=["geo_id", "geo_name", "source_category", "count"])
    df["geo_level"] = "gemeente"
    df["basis"] = "self_id"
    df["year"] = 2014
    df["source_id"] = "cbs_maatwerk_2015_20"
    df["note"] = "CBS EBB 2010/2014, 18+ share applied to the whole gemeente"
    df = df[COLUMNS]

    unknown = sorted(set(df["source_category"]) - set(nl2014.MAP))
    if unknown:
        sys.exit(f"!! source categories with no mapping: {unknown}")
    vanished = sorted(set(nl2014.MAP) - set(df["source_category"]))
    if vanished:
        sys.exit(f"!! nl2014.MAP categories that no gemeente produced: {vanished}")

    total = df["count"].sum()
    if abs(total - (POP_2014 - SUPPRESSED_POP)) > 2:
        sys.exit(f"!! drawn {total:,.0f}, expected {POP_2014 - SUPPRESSED_POP:,}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(df):,} rows, "
          f"{df['source_category'].nunique()} source categories)")
    print(f"drawn {total:,.0f} of {POP_2014:,} — {100 * total / POP_2014:.2f}%")

    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print("\nnational shares, adults' answers applied to everybody:")
    for k, v in nat.items():
        print(f"  {100 * v / total:>6.2f}%  {v:>12,.0f}  {k}")

    # ---- where each answer is strongest, which is the check that the geography is real
    share = (df.pivot_table(index="geo_id", columns="source_category", values="count",
                            aggfunc="sum", fill_value=0.0))
    share = share.div(share.sum(axis=1), axis=0)
    names = drawn.set_index("unit")["name"]
    print("\nstrongest gemeente for each answer:")
    for c in CATS + [NO_RELIGION]:
        top = share[c].nlargest(3)
        print("  %-13s %s" % (c, ", ".join(
            f"{names[u]} {100 * v:.1f}%" for u, v in top.items())))

    # ---- THE DECADE, printed rather than asserted (see the module docstring)
    ssw = _ssw()
    prov_here = (drawn.assign(rel=drawn["total_rel"] * drawn["pop"])
                 .groupby("prov")[["rel", "pop"]].sum())
    prov_here["pct"] = prov_here["rel"] / prov_here["pop"]
    ren = {"Friesland": "Fryslân"}
    print("\nEBB 2010/2014 vs SSW 2021/2025, share in some denomination:")
    print("  (18+ against 15+, two different CBS surveys; the change is real, the levels "
          "are not\n   strictly comparable and nothing here is re-levelled onto the other)")
    for p, v in prov_here["pct"].sort_values(ascending=False).items():
        now = ssw[ren.get(p, p)]
        print(f"  {p:<15} {v:5.1f}%  ->  {now:5.1f}%   {now - v:+5.1f}")
    nat_then = float((drawn["total_rel"] * drawn["pop"]).sum() / drawn["pop"].sum())
    print(f"  {'Netherlands':<15} {nat_then:5.1f}%  ->   42.9%   "
          f"{42.9 - nat_then:+5.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    build()
