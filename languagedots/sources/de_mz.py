"""Germany: Mikrozensus 2023, language spoken predominantly at home, by Land.

    python sources/de_mz.py --fetch    download the three workbooks into data/raw/de/ if missing
    python sources/de_mz.py            normalise from data/raw/de/

-> data/normalized/de.csv, two levels, alternatives (never summed):
   `land`        what is drawn: 16 Laender x language, with tier and part
   `land_group`  the published figures it is built from (IntMK J2 groups + A1a totals), kept
                 for the checks and the record

THE QUESTION. Germany's census (Zensus 2022) asks nothing about language. The Mikrozensus, the
1% household sample survey, has asked since 2017 "Welche Sprache sprechen Sie vorwiegend zu
Hause?" (which language do you mainly speak at home). It is not a census; figures are survey
extrapolations, rounded to the thousand, and cover people in private households only.

THE TABLES. Destatis publishes the language item NATIONALLY only (Statistischer Bericht
"Mikrozensus - Bevoelkerung nach Einwanderungsgeschichte", table 12211-40: 32 languages; GENESIS
has no language table at all, checked 2026-10-05 through its open API). The one Land-level
series is the Laender's own Integrationsmonitoring (IntMK, www.integrationsmonitoring-laender.de),
indicator J2: people WITH Migrationsgeschichte by predominant home language in eight groups
(German, west-European = English/French/Italian/Spanish/Dutch, Polish, Russian, Turkish, other
European, Arabic, other), 2023 first results, per Land. Indicator A1a of the same release gives
each Land's population with and without Migrationsgeschichte.

HOW THE DRAWN ROWS ARE BUILT (the record is sources/de.md):
  measured  German, Polish, Russian, Turkish, Arabic among people with Migrationsgeschichte,
            straight from J2.
  derived   (a) J2's three multi-language groups split into Destatis' named languages by the
            national 2023 shares among everyone who is not "ohne Einwanderungsgeschichte"
            (table 12211-40, Endergebnisse 2023);
            (b) J2's few suppressed cells ("/", under 71 sample cases), filled from the row total;
            (c) people WITHOUT Migrationsgeschichte (A1a), by the national 2023 language mix of
            people "ohne Einwanderungsgeschichte" (98.8% German).

CHECKS: J2's Land totals equal A1a's; the 16 Laender sum to J2's and A1a's Deutschland rows;
Destatis' 32 languages partition "vorwiegend nicht-deutsch"; every Destatis language sits in
exactly one J2 group; J2's national groups against Destatis' (consistency of the grouping);
the borrowed within-group shares against the 2024 report (drift); drawn rows sum to A1a.
"""

import io
import os
import sys
import zipfile

import openpyxl
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "de")
OUT = os.path.join(ROOT, "data", "normalized", "de.csv")

YEAR = 2023
SOURCE_ID = "de_mz2023_intmk_j2"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")

FILES = {
    "j2": ("intmk_j2_2017_2023.xlsx",
           "https://www.integrationsmonitoring-laender.de/documents/"
           "j2-2017-2023-1743420780_1743531218.xlsx"),
    "a1a": ("intmk_a1a_2011_2023.xlsx",
            "https://www.integrationsmonitoring-laender.de/documents/"
            "a1a-2011-2023-1743417718_1743504177.xlsx"),
    # Statistischer Bericht, Endergebnisse 2023 (19.05.2025), and 2024 (04.02.2026) for drift
    "mz2023": ("mz2023_einwanderung.zip",
               "https://www.statistischebibliothek.de/mir/servlets/MCRZipServlet/"
               "DEHeft_derivate_00093474"),
    "mz2024": ("mz2024_einwanderung.zip",
               "https://www.statistischebibliothek.de/mir/servlets/MCRZipServlet/"
               "DEHeft_derivate_00099265"),
}

LAND = {  # AGS Land code, the first two digits of religiondots' grid `ars`
    "Schleswig-Holstein": "01", "Hamburg": "02", "Niedersachsen": "03", "Bremen": "04",
    "Nordrhein-Westfalen": "05", "Hessen": "06", "Rheinland-Pfalz": "07",
    "Baden-Württemberg": "08", "Bayern": "09", "Saarland": "10", "Berlin": "11",
    "Brandenburg": "12", "Mecklenburg-Vorpommern": "13", "Sachsen": "14",
    "Sachsen-Anhalt": "15", "Thüringen": "16",
}

# J2's columns, in order after Land | Jahr | insgesamt
J2_GROUPS = ["deutsch", "west-europäische Sprache", "polnisch", "russisch", "türkisch",
             "sonstige europäische Sprache", "arabisch", "sonstige Sprache"]

# Each J2 group -> the Destatis 12211-40 labels it holds. J2 footnote 1: west-European is
# English, French, Italian, Spanish and (from 2021) Dutch. Kurdish, Persian and the rest of
# Asia and Africa are "sonstige Sprache"; the grouping is checked against the national figures.
GROUP_LANGS = {
    "deutsch": ["Deutsch"],
    "west-europäische Sprache": ["Englisch", "Französisch", "Italienisch", "Spanisch",
                                 "Niederländisch"],
    "polnisch": ["Polnisch"],
    "russisch": ["Russisch"],
    "türkisch": ["Türkisch"],
    "sonstige europäische Sprache": ["Albanisch", "Bosnisch", "Bulgarisch", "Dänisch",
                                     "Griechisch", "Kroatisch", "Mazedonisch", "Portugiesisch",
                                     "Rumänisch", "Serbisch", "Ukrainisch", "Ungarisch",
                                     "Eine andere in Europa gesprochene Sprache"],
    "arabisch": ["Arabisch"],
    "sonstige Sprache": ["Chinesisch", "Hindi", "Kurdisch", "Paschtu", "Persisch", "Urdu",
                         "Vietnamesisch", "Eine andere in Afrika gesprochene Sprache",
                         "Eine andere in Asien gesprochene Sprache", "Eine sonstige Sprache"],
}
MEASURED_GROUPS = {"deutsch", "polnisch", "russisch", "türkisch", "arabisch"}

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count", "tier", "part",
           "year", "source_id"]


def fetch():
    import requests
    os.makedirs(RAW, exist_ok=True)
    for name, url in FILES.values():
        path = os.path.join(RAW, name)
        if os.path.exists(path):
            continue
        r = requests.get(url, headers={"User-Agent": UA}, timeout=300)
        r.raise_for_status()
        with open(path + ".part", "wb") as f:
            f.write(r.content)
        os.replace(path + ".part", path)
        print(f"  fetched {name} ({len(r.content):,} bytes)")


def num(v):
    """J2/A1a cells: a number, or '/' (under 71 cases) or '.' (nothing) -> None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().strip("()")
    try:
        return float(s)
    except ValueError:
        return None


def read_j2():
    wb = openpyxl.load_workbook(os.path.join(RAW, FILES["j2"][0]), read_only=True,
                                data_only=True)
    ws = wb.worksheets[0]
    out = {}
    for row in ws.iter_rows(values_only=True):
        if not row or row[1] != YEAR or row[0] not in list(LAND) + ["Deutschland"]:
            continue
        vals = [num(v) for v in row[2:2 + 1 + len(J2_GROUPS)]]
        out[row[0]] = dict(zip(["insgesamt"] + J2_GROUPS, vals))
    assert len(out) == 17, f"J2 {YEAR}: {len(out)} rows, expected 16 Laender + Deutschland"
    return out


def read_a1a():
    wb = openpyxl.load_workbook(os.path.join(RAW, FILES["a1a"][0]), read_only=True,
                                data_only=True)
    ws = wb.worksheets[0]
    out = {}
    for row in ws.iter_rows(values_only=True):
        if (row and row[1] == YEAR and row[2] == "Insgesamt" and row[3] == "Zusammen"
                and row[0] in list(LAND) + ["Deutschland"]):
            v = [num(x) for x in row[4:12]]
            # insgesamt | mit | Deutsche | Auslaender | aus der EU | 1. Gen | 2. Gen | ohne
            tot, mit, ohne = v[0], v[1], v[7]
            assert abs(tot - mit - ohne) < 0.25, (row[0], tot, mit, ohne)
            out[row[0]] = dict(insgesamt=tot, mit=mit, ohne=ohne)
    assert len(out) == 17, f"A1a {YEAR}: {len(out)} rows"
    return out


def read_destatis(key):
    """12211-40 national, Bevoelkerung insgesamt x Bevoelkerungsgruppe -> {group: {label: n}}."""
    z = zipfile.ZipFile(os.path.join(RAW, FILES[key][0]))
    name = [n for n in z.namelist() if n.endswith(".xlsx")][0]
    wb = openpyxl.load_workbook(io.BytesIO(z.read(name)), read_only=True, data_only=True)
    rows = list(wb["csv-12211-40"].iter_rows(values_only=True))
    hdr = list(rows[0])
    vcol = [i for i, h in enumerate(hdr) if h and str(h).startswith("Bevoelkerung")][0]
    out = {}
    for r in rows[1:]:
        if r[hdr.index("VALUE_2")] != "Bevölkerung insgesamt":
            continue
        if r[hdr.index("VARIABLE_3")] != "Bevölkerungsgruppe":
            continue
        grp = r[hdr.index("VALUE_3")]
        # the 2023 and 2024 workbooks spell two labels differently
        lab = " ".join(str(r[hdr.index("VALUE_1")]).replace("vorwiegend:", "")
                       .replace("Nachrichtlich:", "").split())
        if lab == "vorwiegend Albanisch":
            lab = "Albanisch"
        out.setdefault(grp, {})[lab] = num(r[vcol])
    for grp, d in out.items():
        d["Deutsch"] = (d["nur Deutsch"] or 0) + (d["vorwiegend deutsch"] or 0)
    return out


def destatis_checks(nat, tag):
    langs = [l for g in GROUP_LANGS.values() for l in g if l != "Deutsch"]
    assert len(langs) == len(set(langs)) == 32, f"{tag}: {len(langs)} languages in the groups"
    allrow = nat["Nach Einwanderungsgeschichte zusammen"]
    printed = {k for k in allrow if k not in (
        "Zu Hause vorwiegend gesprochene Sprache Insgesamt", "nur Deutsch", "vorwiegend deutsch",
        "vorwiegend nicht-deutsch", "Deutsch", "Deutsch und mindestens eine andere Sprache",
        "kein Deutsch, nur andere Sprache(n)")}
    assert printed == set(langs), (f"{tag}: Destatis labels differ from the grouping: "
                                   f"{sorted(printed ^ set(langs))}")
    s = sum(allrow[l] for l in langs)
    nd = allrow["vorwiegend nicht-deutsch"]
    tot = allrow["Zu Hause vorwiegend gesprochene Sprache Insgesamt"]
    assert abs(s - nd) <= 20, f"{tag}: 32 languages sum to {s}, not {nd}"
    assert abs(allrow["Deutsch"] + nd - tot) <= 3, f"{tag}: German + other != total"
    print(f"  {tag}: 32 languages sum to {s:,.0f}k against `vorwiegend nicht-deutsch` "
          f"{nd:,.0f}k; total {tot:,.0f}k")


def shares(nat):
    """Within-group shares among everyone not `ohne Einwanderungsgeschichte`, and the
    language mix of `ohne` itself (suppressed '/' cells count 0; the shown non-German
    languages are scaled to the row's `vorwiegend nicht-deutsch`)."""
    allrow, ohne = nat["Nach Einwanderungsgeschichte zusammen"], nat["Ohne Einwanderungsgeschichte"]
    within = {}
    for g, langs in GROUP_LANGS.items():
        base = {l: (allrow[l] or 0) - (ohne.get(l) or 0) for l in langs}
        t = sum(base.values())
        within[g] = {l: v / t for l, v in base.items()}
    tot = ohne["Zu Hause vorwiegend gesprochene Sprache Insgesamt"]
    langs = [l for g in GROUP_LANGS.values() for l in g if l != "Deutsch"]
    shown = {l: ohne.get(l) or 0 for l in langs}
    k = ohne["vorwiegend nicht-deutsch"] / sum(shown.values())
    mix = {l: v * k / tot for l, v in shown.items() if v}
    mix["Deutsch"] = ohne["Deutsch"] / tot
    assert abs(sum(mix.values()) - 1) < 0.002, sum(mix.values())
    return within, mix, k


def main():
    if "--fetch" in sys.argv:
        fetch()
    j2, a1a = read_j2(), read_a1a()
    nat = read_destatis("mz2023")
    destatis_checks(nat, "Destatis 2023")
    within, mix, k = shares(nat)
    print(f"  `ohne Einwanderungsgeschichte` 2023: {100 * mix['Deutsch']:.2f}% German; its "
          f"shown non-German cells scaled by {k:.3f} to the row's total")

    # --- J2 and A1a agree, and the Laender add up --------------------------------------
    for land in LAND:
        assert abs(j2[land]["insgesamt"] - a1a[land]["mit"]) < 0.15, land
    for col in ["insgesamt"] + J2_GROUPS:
        vals = [j2[l][col] for l in LAND if j2[l][col] is not None]
        if all(j2[l][col] is not None for l in LAND):
            assert abs(sum(vals) - j2["Deutschland"][col]) < 1.0, (col, sum(vals))
    for col in ("insgesamt", "mit", "ohne"):
        s = sum(a1a[l][col] for l in LAND)
        assert abs(s - a1a["Deutschland"][col]) < 1.0, (col, s, a1a["Deutschland"][col])
    print(f"  J2 Land totals = A1a `mit` in all 16; Laender sum to Deutschland "
          f"({a1a['Deutschland']['insgesamt']:,.1f}k, {a1a['Deutschland']['mit']:,.1f}k with "
          f"Migrationsgeschichte)")

    # --- the J2 grouping against Destatis, nationally ----------------------------------
    allrow, ohne = nat["Nach Einwanderungsgeschichte zusammen"], nat["Ohne Einwanderungsgeschichte"]
    dn = {g: sum((allrow[l] or 0) - (ohne.get(l) or 0) for l in ls)
          for g, ls in GROUP_LANGS.items()}
    jn = j2["Deutschland"]
    dnn = sum(v for g, v in dn.items() if g != "deutsch")
    jnn = sum(jn[g] for g in J2_GROUPS if g != "deutsch")
    print("  J2 national non-German mix vs Destatis (not `ohne`), share of non-German:")
    worst = 0
    for g in J2_GROUPS[1:]:
        a, b = 100 * jn[g] / jnn, 100 * dn[g] / dnn
        worst = max(worst, abs(a - b))
        print(f"    {g:30s} J2 {a:5.1f}%  Destatis {b:5.1f}%")
    assert worst < 3.0, f"J2 grouping disagrees with Destatis by {worst:.1f} points"

    # --- drift of the borrowed shapes, 2023 -> 2024 -------------------------------------
    p24 = os.path.join(RAW, FILES["mz2024"][0])
    if os.path.exists(p24):
        n24 = read_destatis("mz2024")
        destatis_checks(n24, "Destatis 2024")
        w24, mix24, _ = shares(n24)
        drift = max(abs(w24[g][l] - within[g][l]) for g in within for l in within[g])
        print(f"  within-group shares 2023 -> 2024: largest move {100 * drift:.1f} points; "
              f"`ohne` German {100 * mix['Deutsch']:.2f}% -> {100 * mix24['Deutsch']:.2f}%")
        assert drift < 0.06, f"borrowed within-group shares moved {drift:.3f} in a year"

    # --- build ---------------------------------------------------------------------------
    rows = []

    def add(land, level, cat, n, tier, part):
        rows.append(dict(geo_id=LAND[land], geo_level=level, geo_name=land,
                         source_category=cat, count=round(n * 1000, 1), tier=tier, part=part,
                         year=YEAR, source_id=SOURCE_ID))

    filled = 0
    for land in LAND:
        r = dict(j2[land])
        missing = [g for g in J2_GROUPS if r[g] is None]
        assert len(missing) <= 1, (land, missing)
        for g in J2_GROUPS:
            add(land, "land_group", f"J2: {g}", r[g] or 0, "measured", "published")
        add(land, "land_group", "A1a: ohne Migrationsgeschichte", a1a[land]["ohne"],
            "measured", "published")
        if missing:
            rest = r["insgesamt"] - sum(r[g] for g in J2_GROUPS if r[g] is not None)
            r[missing[0]] = max(rest, 0.0)
            filled += r[missing[0]]
            print(f"  {land}: suppressed `{missing[0]}` filled from the row total, "
                  f"{1000 * r[missing[0]]:,.0f}")
        for g in J2_GROUPS:
            v = r[g]
            if not v:
                continue
            if g in missing:
                for l, s in within[g].items():
                    add(land, "land", l, v * s, "derived", "suppressed cell, from the row total")
            elif g in MEASURED_GROUPS:
                add(land, "land", GROUP_LANGS[g][0], v, "measured", "with Migrationsgeschichte")
            else:
                for l, s in within[g].items():
                    add(land, "land", l, v * s, "derived",
                        f"with Migrationsgeschichte, {g} split by national shares")
        for l, s in mix.items():
            add(land, "land", l, a1a[land]["ohne"] * s, "derived",
                "without Migrationsgeschichte, national mix")

    df = pd.DataFrame(rows, columns=COLUMNS)
    drawn = df[df.geo_level == "land"]
    per = drawn.groupby("geo_name")["count"].sum()
    for land in LAND:
        assert abs(per[land] - 1000 * a1a[land]["insgesamt"]) < 300, (land, per[land])
    tot = drawn["count"].sum()
    meas = drawn.loc[drawn.tier == "measured", "count"].sum()
    print(f"  drawn: {tot:,.0f} people in 16 Laender ({1000 * a1a['Deutschland']['insgesamt']:,.0f} "
          f"in A1a); measured {meas:,.0f} ({100 * meas / tot:.1f}%), suppressed cells "
          f"filled {1000 * filled:,.0f}")
    top = drawn.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print("  national, largest:", ", ".join(f"{k} {v / 1e6:.2f}M" for k, v in top.head(10).items()))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT + ".tmp", index=False, encoding="utf-8")
    os.replace(OUT + ".tmp", OUT)
    print(f"  wrote {OUT}: {len(df):,} rows")


if __name__ == "__main__":
    main()
