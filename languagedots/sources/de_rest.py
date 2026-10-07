"""Germany: the Mikrozensus' three "another language of Europe / Asia / Africa" rows, split into
languages by Zensus 2022 citizenship. -> data/normalized/de_rest.csv

    python sources/de_rest.py --fetch    download the Land-level citizenship table if missing
    python sources/de_rest.py

Anita, 2026-10-06, on the grey wedge in Munich: "is there anything we can do to show more?"

WHAT IT SPLITS. Destatis 12211-40 names 32 languages and leaves four remainders: "Eine andere in
Europa / Asien / Afrika gesprochene Sprache" and "Eine sonstige Sprache". sources/de_mz.py carries
them per Land; countries/de.py drew the first, second and fourth on `other` and the third on
`africa_other` (1.27% + 0.33% of Germany). This file splits the three continental ones. Each
Land's count stays the Mikrozensus' (derived, as before); only which languages make it up is
borrowed. "Eine sonstige Sprache" (359k) stays on `other`: nothing says what it holds (it is not
the Americas, whose languages Spanish, Portuguese and English are named).

THE RULE, per Land and remainder R (Europe, Asia, Africa):
  share of language L in R  ∝  Σ_k  citizens_k(Land) × mix_k(L)
  over every foreign citizenship k of R's continent (the table's own code ranges: LAND1xx
  Europe, 2xx Africa, 4xx Asia; Turkey counted with Asia for this, its unnamed language being
  Zazaki of eastern Anatolia), mix_k = origin_mix.mix(k, "de"), and L any language the
  Mikrozensus does NOT name: someone whose language is Polish or Levantine Arabic answered
  Polish or Arabic, so only the unnamed part of a mix can be in a remainder. Dropped from every
  mix: the 32 named languages and German, everything under them (Arabic's varieties, Chinese's,
  Kurdish's, Dari as Persian), any group node that contains a named language (Afghanistan's
  unnamed Iranian), and `other`.
  So Indians in Germany count towards Asia's remainder with the non-Hindi, non-Urdu two thirds
  of India's mix (Telugu, Tamil, Bengali...), Belarusians with their Belarusian quarter (their
  Russian is named), Nigerians with all of Nigeria's mix but its English.

SOURCE: Zensus 2022 (15 May 2022) database table 1000A-1023 "Personen: Staatsangehoerigkeit
(Laender)", 203 citizenships by Land, the table's default layout (de_place.py fetches the same
table by Gemeinde for placement; by Land fewer small cells are suppressed). Datenlizenz
Deutschland Namensnennung 2.0.

WHAT IT ASSUMES: that a continent's unnamed-language speakers are in the proportions of its
foreign citizens' unnamed languages. Naturalised speakers are not counted by citizenship (most of
Germany's Aramaic and Assyrian speakers, its Sinti and Sorbs, are German citizens), so those
languages are missing from the split and their people go to the languages that are in it.
sources/de.md, "The remainders split by citizenship".

CHECKS (printed; the build fails on the first two): every citizenship of 300+ people in the three
code ranges has an ISO code here; every Land's shares sum to 1 for each remainder; the candidate
pool beside the remainder it splits, nationally (a pool far below its remainder means naturalised
or uncounted speakers dominate it).
"""
import io
import json
import os
import sys
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "2")
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "taxonomy")):
    if p not in sys.path:
        sys.path.insert(0, p)
RAW = os.path.join(ROOT, "data", "raw", "de")
NORM = os.path.join(ROOT, "data", "normalized", "de.csv")
OUT = os.path.join(ROOT, "data", "normalized", "de_rest.csv")
ZDB_FILE_LAND = "zensus2022_1000A-1023_laender.zip"

REMAINDERS = {   # Mikrozensus label -> the table's code-range digit
    "Eine andere in Europa gesprochene Sprache": "1",
    "Eine andere in Afrika gesprochene Sprache": "2",
    "Eine andere in Asien gesprochene Sprache": "4",
}
CONTINENT_OVERRIDE = {"Türkei": "4"}
# Unnamed languages a citizenship's speakers would not report as such in Germany: the speaker
# would have answered the named national language. "*" drops the whole unnamed part. Uncited
# judgements (sources/de.md):
#   Italy: the map's Neapolitan, Sicilian, Venetian, Lombard... are what Italy's own map draws
#     for regional speech; an Italian in Germany answers "Italienisch" (20% of Italy's mix, 116k
#     of pool, would otherwise be 30% of Europe's remainder);
#   Netherlands: Westphalian, Limburgish and Frisian beside Dutch, likewise;
#   Moldova: Moldovan is Romanian, which is named;
#   Kazakhstan: its citizens in Germany are mostly Russian-speaking (Aussiedler families), not
#     at Kazakhstan's home mix of 75% Kazakh; de_place.py already places Russian by them.
EXCLUDE = {"Italien": "*", "Niederlande": "*", "Kasachstan": "*",
           "Moldau, Republik": {"indoeuropean.romance.moldovan"}}

# The table's citizenship labels (code ranges 1, 2, 4; every one of 300+ people) -> ISO 3166.
ISO = {
    "Albanien": "AL", "Bosnien und Herzegowina": "BA", "Belgien": "BE", "Bulgarien": "BG",
    "Dänemark": "DK", "Estland": "EE", "Finnland": "FI", "Frankreich": "FR", "Kroatien": "HR",
    "Slowenien": "SI", "Griechenland": "GR", "Irland": "IE", "Island": "IS", "Italien": "IT",
    "Lettland": "LV", "Montenegro": "ME", "Litauen": "LT", "Luxemburg": "LU",
    "Nordmazedonien (bis 2019: Mazedonien)": "MK", "Malta": "MT", "Moldau, Republik": "MD",
    "Niederlande": "NL", "Norwegen": "NO", "Kosovo": "XK", "Österreich": "AT", "Polen": "PL",
    "Portugal": "PT", "Rumänien": "RO", "Slowakei": "SK", "Schweden": "SE", "Schweiz": "CH",
    "Russische Föderation": "RU", "Spanien": "ES", "Türkei": "TR",
    "Tschechische Republik": "CZ", "Ungarn": "HU", "Ukraine": "UA",
    "Vereinigtes Königreich": "GB", "Vereinigtes Königreich/ Britische Überseegebiete": "GB",
    "Belarus": "BY", "Serbien": "RS", "Zypern": "CY",
    "Algerien": "DZ", "Angola": "AO", "Eritrea": "ER", "Äthiopien": "ET", "Benin": "BJ",
    "Côte d'Ivoire": "CI", "Nigeria": "NG", "Simbabwe": "ZW", "Gabun": "GA", "Gambia": "GM",
    "Ghana": "GH", "Mauretanien": "MR", "Kenia": "KE", "Kongo, Republik": "CG",
    "Kongo, Demokratische Republik": "CD", "Liberia": "LR", "Libyen": "LY",
    "Madagaskar": "MG", "Mali": "ML", "Marokko": "MA", "Mauritius": "MU", "Mosambik": "MZ",
    "Niger": "NE", "Sambia": "ZM", "Burkina Faso": "BF", "Guinea-Bissau": "GW",
    "Guinea": "GN", "Kamerun": "CM", "Südafrika": "ZA", "Ruanda": "RW", "Namibia": "NA",
    "Senegal": "SN", "Sierra Leone": "SL", "Somalia": "SO", "Sudan": "SD", "Südsudan": "SS",
    "Tansania": "TZ", "Togo": "TG", "Tschad": "TD", "Tunesien": "TN", "Uganda": "UG",
    "Ägypten": "EG", "Burundi": "BI", "China (Hongkong)": "HK", "Jemen": "YE",
    "Armenien": "AM", "Afghanistan": "AF", "Bahrain": "BH", "Aserbaidschan": "AZ",
    "Myanmar": "MM", "Georgien": "GE", "Sri Lanka": "LK", "Vietnam": "VN",
    "Korea, Demokratische Volksrepublik": "KP", "Indien": "IN", "Indonesien": "ID",
    "Irak": "IQ", "Iran": "IR", "Israel": "IL", "Japan": "JP", "Kasachstan": "KZ",
    "Jordanien": "JO", "Kambodscha": "KH", "Kuwait": "KW", "Laos": "LA", "Kirgisistan": "KG",
    "Libanon": "LB", "Mongolei": "MN", "Nepal": "NP", "Palästinensische Gebiete": "PS",
    "Bangladesch": "BD", "Pakistan": "PK", "Philippinen": "PH", "Taiwan": "TW",
    "Korea, Republik": "KR", "Vereinigte Arabische Emirate": "AE", "Tadschikistan": "TJ",
    "Turkmenistan": "TM", "Saudi-Arabien": "SA", "Singapur": "SG", "Syrien": "SY",
    "Thailand": "TH", "Usbekistan": "UZ", "China": "CN", "Malaysia": "MY",
}
MIN_PEOPLE = 300
MIN_NAT = 0.005


def fetch():
    import requests
    import de_place as dp
    path = os.path.join(RAW, ZDB_FILE_LAND)
    if os.path.exists(path):
        return
    h = {"User-Agent": dp.UA, "Accept": "*/*",
         "Referer": "https://ergebnisse.zensus2022.de/datenbank/online/"}
    st = requests.get(f"{dp.ZDB}/tables/{dp.ZDB_TABLE}/structure", headers=h, timeout=300)
    st.raise_for_status()
    state = st.json()["initialState"]
    col = state["tableStructure"]["colTitle"]
    assert state["variableBlocks"][col[1]["blockCode"]]["mainVariable"] == "GEOBL1", col
    h["Content-Type"] = "application/json"
    r = requests.post(f"{dp.ZDB}/tables/{dp.ZDB_TABLE}/download/ffcsv/de", headers=h,
                      data=json.dumps(state), timeout=900)
    r.raise_for_status()
    with open(path + ".part", "wb") as f:
        f.write(r.content)
    os.replace(path + ".part", path)
    print(f"  fetched {ZDB_FILE_LAND} ({len(r.content):,} bytes)")


def read_land_table():
    """-> (DataFrame index Land code '01'..'16', columns citizenship label), national Series."""
    with zipfile.ZipFile(os.path.join(RAW, ZDB_FILE_LAND)) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(name) as fh:
            df = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8-sig"), sep=";", dtype=str)
    df = df[df["value_unit"] == "Anzahl"]
    df = df[df["2_variable_attribute_code"].str.startswith("LAND", na=False)]
    v = df["value"].str.strip()
    num = v.str.fullmatch(r"\d+")
    df["n"] = pd.to_numeric(v.where(num, "0"))
    df["code"] = df["2_variable_attribute_code"]
    df["label"] = df["2_variable_attribute_label"]
    land = df[df["1_variable_code"] == "GEOBL1"]
    nat = df[df["1_variable_code"] == "GEODL1"].groupby("label")["n"].sum()
    wide = land.pivot_table(index="1_variable_attribute_code", columns="label", values="n",
                            aggfunc="sum", fill_value=0)
    codes = df.drop_duplicates("label").set_index("label")["code"]
    if len(wide) != 16:
        raise SystemExit(f"{ZDB_FILE_LAND}: {len(wide)} Laender")
    foreign = [c for c in wide.columns if codes[c] != "LAND000"]
    gap = (wide[foreign].sum() - nat.reindex(foreign).fillna(0)).abs()
    print(f"  table by Land: {wide.shape[1]} citizenships, {int((~num).sum())} cells not a "
          f"number (suppressed or zero); foreign citizens {int(wide[foreign].sum().sum()):,} "
          f"in the Laender, {int(nat.reindex(foreign).sum()):,} nationally "
          f"(largest gap {int(gap.max()):,}, {gap.idxmax()})")
    return wide, codes, nat


def named_nodes():
    """The Mikrozensus' 32 named languages and German, as tree nodes."""
    import de2023
    return {n for lab, n in de2023.NAMES.items() if n not in ("other", "africa_other")}


def is_named(node, named):
    if node == "other" or node.startswith("other."):
        return True                          # dropped: says nothing
    for n in named:
        if node == n or node.startswith(n + "."):
            return True                      # a variety of a named language
        if n.startswith(node + "."):
            return True                      # a group holding a named language
    leaf = node.split(".")[-1]
    return (leaf.endswith("_arabic") or leaf in ("darija", "hassaniya", "dari")
            or node.startswith("sinotibetan.sinitic"))


def main():
    import origin_mix
    if not os.path.exists(os.path.join(RAW, ZDB_FILE_LAND)):
        raise SystemExit(f"{ZDB_FILE_LAND} missing: run with --fetch")
    wide, codes, _ = read_land_table()
    tot = wide.sum()
    named = named_nodes()

    cand = [lab for lab in wide.columns if str(codes[lab])[4:5] in "124"]
    missing = sorted(lab for lab in cand if lab not in ISO and tot[lab] >= MIN_PEOPLE)
    if missing:
        raise SystemExit(f"citizenships without an ISO code: {missing}")

    # per citizenship: its unnamed languages
    unnamed = {}
    for lab in cand:
        if lab not in ISO:
            continue
        m = origin_mix.mix(ISO[lab], "de")
        ex = EXCLUDE.get(lab, set())
        if ex == "*":
            continue
        u = {n: s for n, s in m.items() if not is_named(n, named) and n not in ex}
        if u:
            unnamed[lab] = u

    mz = pd.read_csv(NORM, dtype={"geo_id": str})
    mz = mz[(mz["geo_level"] == "land") & mz["source_category"].isin(REMAINDERS)]
    rem = mz.groupby(["geo_id", "source_category"])["count"].sum()

    rows = []
    for rlab, digit in REMAINDERS.items():
        w = {}                                   # (land, node) -> weight
        for land in wide.index:
            for lab, u in unnamed.items():
                cont = CONTINENT_OVERRIDE.get(lab, str(codes[lab])[4])
                if cont != digit:
                    continue
                for n, s in u.items():
                    w[(land, n)] = w.get((land, n), 0.0) + wide.at[land, lab] * s
        w = pd.Series(w)
        w = w[w > 0]
        p = w.groupby(level=1).sum().sort_values(ascending=False)
        # a language under MIN_NAT of the remainder's pool nationally is dropped: a 1% language
        # of one origin's home mix is guesswork at this size, and would draw only a ring
        # and its people stay on the remainder's own node, unnamed, rather than going to the
        # big languages
        keep = p.index[p / p.sum() >= MIN_NAT]
        rest_node = "africa_other" if digit == "2" else "other"
        ws = w.groupby(level=0).sum()
        dropped = w[~w.index.get_level_values(1).isin(keep)].groupby(level=0).sum()
        w = w[w.index.get_level_values(1).isin(keep)]
        for land, v in dropped.items():
            w[(land, rest_node)] = w.get((land, rest_node), 0.0) + v
        missing = sorted(set(wide.index) - set(ws.index))
        if missing:
            raise SystemExit(f"{rlab}: no candidate citizens in {missing}")
        for (land, n), v in w.items():
            rows.append(dict(geo_id=land, remainder=rlab, node=n, weight=v, share=v / ws[land]))
        r_nat = float(rem.xs(rlab, level="source_category").sum())
        print(f"\n  {rlab}: Mikrozensus {r_nat:,.0f}; candidate citizens' unnamed languages "
              f"{p.sum():,.0f} ({p.sum() / r_nat:.0%} of it), {len(p)} languages, {len(keep)} "
              f"kept at {MIN_NAT:.1%} or more ({p[keep].sum() / p.sum():.1%} of the pool; the rest stays on {rest_node})")
        for n, v in p.head(15).items():
            print(f"     {v / p.sum():6.1%}  {n}")

    out = pd.DataFrame(rows)
    chk = out.groupby(["geo_id", "remainder"])["share"].sum()
    if (chk.sub(1).abs() > 1e-9).any():
        raise SystemExit("shares do not sum to 1")
    out.to_csv(OUT, index=False)
    print(f"\n  wrote {OUT}: {len(out):,} rows, {out['node'].nunique()} languages")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    main()
