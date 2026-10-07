"""Portugal: build data/normalized/pt.csv from Censos 2021 nationality by freguesia.

    python sources/pt_censos.py --fetch     download INE indicator 0012294 into data/raw/pt/
    python sources/pt_censos.py             build data/normalized/pt.csv, printing the checks

Portugal's census asks no language. Under Anita's 2026-10-05 ruling for rich countries with no
language question (AGENT_BRIEF Â§2) the map draws Portuguese, plus Mirandese from a published
speaker estimate, plus immigrant languages proxied by nationality. Every row is `derived`.

Per freguesia (3,092):
  1. foreign nationals by nationality: INE Censos 2021, indicator 0012294 ("PopulaÃ§Ã£o residente
     por Local de residÃªncia, Sexo, Grupo etÃ¡rio e Nacionalidade", NUTS 2024), all ages, both
     sexes. 52 named nationalities plus five continent remainders, stateless and "Outros".
  2. each continent remainder split into countries by Eurostat's census 2021 citizenship table
     (cens_21ctz_r3, the same census; religiondots' raw copy, read-only) for the freguesia's
     NUTS 3 (GISCO's LAU 2021 -> NUTS 2021 file), counting only countries INE does not name;
     national shares where the NUTS 3 has none.
  3. each nationality on its country's main first language (COUNTRY_LANG, France's list with
     Portugal's overrides below), less the share of its origin region who speak only the host
     language at home, from France's Trajectoires et Origines 2 (fr_build.TEO_FRENCH): ICOT 2023,
     Portugal's own survey, publishes home language only nationally in three broad groups.
     That share goes onto Portuguese.
  4. Mirandese: 1,500 regular users in Miranda do Douro municipality (Costas, Universidade de
     Vigo, 2020 survey), on its rural freguesias; the same rate in Vimioso's Angueira and Vilar
     Seco, the two villages outside it the sources name.
  5. Portuguese: everyone else.
The record is sources/pt.md.
"""
import json
import os
import sys
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fr_build  # noqa: E402  (country -> language list and the TeO2 retention shares)

RAW = os.path.join(ROOT, "data", "raw", "pt")
OUT = os.path.join(ROOT, "data", "normalized", "pt.csv")
RD = os.path.join(os.path.dirname(ROOT), "religiondots")
EU_CTZ = os.path.join(RD, "data", "raw", "fr", "cens_21ctz_r3_fr.json")   # all of Europe
LAU_NUTS = os.path.join(RD, "data", "geo", "lau2021", "EU-27-LAU-2021-NUTS-2021.xlsx")

VARCD = "0012294"
API = ("https://www.ine.pt/ine/json_indicador/pindica.jsp"
       f"?op=2&varcd={VARCD}&Dim1=S7A2021&Dim3=T&Dim4=T&lang=PT")
CUBE = os.path.join(RAW, f"{VARCD}.json")
SOURCE_ID = "pt_censos2021_nationality_x_surveys"
YEAR = 2021
POP_TOTAL = 10_343_066          # Censos 2021, religiondots' sources/pt.py agrees

# ---------------------------------------------------------------------------------------------
# nationality -> language. France's list (fr_build.COUNTRY_LANG, Eurostat ISO2 codes) with
# Portugal's overrides; sources/pt.md gives the reasons.
# ---------------------------------------------------------------------------------------------
COUNTRY_LANG = dict(fr_build.COUNTRY_LANG)
COUNTRY_LANG.update({
    # Lusophone Africa: Angola, Mozambique, Sao Tome on Portuguese (the main home language of
    # their cities and of their emigrants; Angola's 2014 census: 71% speak Portuguese at home);
    # Cape Verde and Guinea-Bissau on their creoles, as France has them
    "MZ": "Portuguese", "AO": "Portuguese", "ST": "Portuguese",
    "CV": "Kabuverdianu", "GW": "Guinea-Bissau Kriol",
    # India: the Alentejo farm workforce that drives Indian immigration is Punjabi (Odemira)
    "IN": "Punjabi",
    # splits as Spain has them (es2021.ORIGIN)
    "UA": {"Ukrainian": 0.7, "Russian": 0.3},
    "CH": {"German": 0.65, "French": 0.25, "Italian": 0.1},
    "BE": {"Dutch": 0.6, "French": 0.4},
    "CA": {"English": 0.75, "French": 0.25},
})
COUNTRY_LANG["FR"] = "French"                 # France's own list has no France
INE_TO_ISO = {"GB": "UK", "GR": "EL"}       # INE prints ISO; Eurostat writes UK and EL
# Aggregates, not leaves. The labels mislead: "1OP Outros paÃ­ses - Europa" is all of non-EU
# Europe (60,456 = Eurostat's EUR_NEU), the named non-EU countries included, and "ZZ Outros" is
# the remainder of non-EU Europe after them (2,311); the other continents' "nOP" ARE their
# remainders. Every EU-27 country is named. "EST" (foreign) leaves out the stateless, so
# T = PT + EST + APA. main() asserts all of this on every unit.
NOT_COUNTRIES = {"T", "EST", "PT", "1", "2", "3", "4", "5", "1XG", "1OP"}
REMAINDERS = {"ZZ": "EUR", "2OP": "AFR", "3OP": "AME", "4OP": "ASI", "5OP": "OCE"}
OTHER = {"APA"}               # stateless: on `other`
NON_EU_NAMED = ("GB", "UA", "MD", "RU", "CH")

# Mirandese (sources/pt.md). Costas (Universidade de Vigo), "Presente e Futuro da Lingua
# Mirandesa", fieldwork 2020: about 3,500 people know it, about 1,500 use it regularly, in
# Miranda do Douro municipality; family use only in its rural (northern) freguesias.
MIRANDESE_USERS = 1_500
MIRANDA_MUNI = "Miranda do Douro"
VIMIOSO_VILLAGES = ("União das freguesias de Caçarelhos e Angueira", "Vilar Seco")  # Angueira merged 2013


def fetch():
    import requests
    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(CUBE) and os.path.getsize(CUBE) > 50_000_000:
        print("already have", CUBE)
        return
    print("GET", API)
    r = requests.get(API, timeout=900,
                     headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    r.raise_for_status()
    doc = r.json()
    # INE answers errors with HTTP 200 and a `Sucesso: Falso` envelope: assert the shape
    if not isinstance(doc, list) or "Dados" not in doc[0] or "2021" not in doc[0]["Dados"]:
        raise SystemExit(f"unexpected response shape: {str(doc)[:300]}")
    with open(CUBE, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False)
    print(f"  {os.path.getsize(CUBE):,} bytes")


def census():
    """{(geocod, nationality code): count}, names, and {code: label}."""
    doc = json.load(open(CUBE, encoding="utf-8"))
    cells, names, labels = {}, {}, {}
    for x in doc[0]["Dados"]["2021"]:
        if x["dim_3"] != "T" or x["dim_4"] != "T":
            continue
        v = x.get("valor")
        if v in (None, ""):
            raise SystemExit(f"{x['geocod']}/{x['dim_5']}: empty cell; INE has published none")
        cells[(x["geocod"], x["dim_5"])] = int(v)
        names[x["geocod"]] = x["geodsg"].strip()
        labels[x["dim_5"]] = x["dim_5_t"]
    return cells, names, labels


def eurostat():
    """NUTS 3 (2021) x ISO2 foreign citizens for Portugal, and each citizenship's block."""
    d = json.load(open(EU_CTZ, encoding="utf-8"))
    dims, sizes = d["id"], d["size"]
    cats = {x: list(d["dimension"][x]["category"]["index"]) for x in dims}
    blocks, cur = {}, None
    starts = {"BE": "EU", "IS": "EUR", "AO": "AFR", "CA": "AME", "KZ": "ASI", "AU": "OCE"}
    for c in cats["citizen"]:
        cur = starts.get(c, cur)
        if len(c) == 2 and c != "PT":
            blocks[c] = cur
    out = defaultdict(dict)
    for k, v in d["value"].items():
        i, idx = int(k), []
        for s in reversed(sizes):
            idx.append(i % s)
            i //= s
        idx = list(reversed(idx))
        rec = {x: cats[x][idx[j]] for j, x in enumerate(dims)}
        g = rec["geo"]
        if rec["citizen"] in blocks and g.startswith("PT") and (len(g) == 5 or g == "PT"):
            out[g][rec["citizen"]] = v
    missing = sorted(set(blocks) - set(COUNTRY_LANG))
    if missing:
        sys.exit(f"!! Eurostat citizenships with no language: {missing}")
    return out, blocks


def lau_nuts():
    import openpyxl
    wb = openpyxl.load_workbook(LAU_NUTS, read_only=True)
    m = {}
    for i, r in enumerate(wb["PT"].iter_rows(values_only=True)):
        if i and r[1]:
            m[str(r[1])] = r[0]
    return m


def lang_shares(iso):
    """The shared origin table (sources/origin_mix.py, 2026-10-05), Portuguese as a label.
    COUNTRY_LANG's Portugal overrides above (Lusophone Africa on Portuguese, India on Punjabi,
    the Spain-style splits) had no figure behind them and went back to the home mixes."""
    from origin_mix import mix
    return [("Portuguese" if n == "indoeuropean.romance.portuguese" else n, s)
            for n, s in mix(iso, "pt").items()]


def main():
    cells, names, labels = census()
    freg = sorted(g for g in names if len(g) == 6)
    assert len(freg) == 3_092, len(freg)
    muni = {g: names[g] for g in names if len(g) == 4}
    assert len(muni) == 308, len(muni)
    nat_codes = sorted({c for _, c in cells})

    # --- checks on the census table --------------------------------------------------------
    pt_tot = cells[("PT", "T")]
    print(f"Portugal {pt_tot:,} residents, {cells[('PT', 'EST')]:,} foreign nationals")
    assert pt_tot == POP_TOTAL, pt_tot
    leaves = [c for c in nat_codes if c not in NOT_COUNTRIES]
    bad = 0
    for g in names:
        t, est, pt = cells[(g, "T")], cells[(g, "EST")], cells[(g, "PT")]
        apa = cells.get((g, "APA"), 0)
        s = sum(cells.get((g, c), 0) for c in leaves)
        zz = cells.get((g, "1OP"), 0) - sum(cells.get((g, c), 0) for c in NON_EU_NAMED)
        if pt + est + apa != t or s != t - pt or zz != cells.get((g, "ZZ"), 0):
            bad += 1
            if bad < 5:
                print(f"  !! {g} {names[g]}: T {t} PT {pt} EST {est} leaves {s}")
    print(f"  Portuguese + foreign + stateless = total, nationalities = total - Portuguese, "
          f"ZZ = non-EU Europe less its five named, on every unit: "
          f"{'ok' if not bad else f'{bad} units off'}")
    if bad:
        sys.exit("!! the nationality table does not partition")
    for c in nat_codes:
        tot = sum(cells.get((g, c), 0) for g in freg)
        if tot != cells[("PT", c)]:
            sys.exit(f"!! freguesias sum to {tot} for {c}, Portugal {cells[('PT', c)]}")
    print("  3,092 freguesias sum to Portugal for every nationality: ok")

    # --- remainders -> countries ---------------------------------------------------------
    eu, blocks = eurostat()
    nuts = lau_nuts()
    missing = [g for g in freg if g not in nuts]
    assert not missing, f"{len(missing)} freguesias without a NUTS 3: {missing[:5]}"
    named = {INE_TO_ISO.get(c, c) for c in leaves if len(c) == 2 and c not in OTHER and c not in REMAINDERS}
    for c in named:
        if c not in COUNTRY_LANG:
            sys.exit(f"!! INE nationality {c} has no language")
    # INE's named country totals against Eurostat's, nationally (same census)
    worst = max(abs(cells[("PT", c)] - eu["PT"].get(INE_TO_ISO.get(c, c), 0))
                for c in leaves if len(c) == 2 and c not in OTHER and c not in REMAINDERS)
    print(f"  INE named nationalities vs Eurostat cens_21ctz, largest gap {worst:,} people")

    def rem_countries(code):
        want = set(REMAINDERS[code].split("+"))
        return [c for c, b in blocks.items() if b in want and c not in named]

    def split(code, n3):
        cs = rem_countries(code)
        w = {c: eu.get(n3, {}).get(c, 0) for c in cs}
        if sum(w.values()) == 0:
            w = {c: eu["PT"].get(c, 0) for c in cs}
        s = sum(w.values())
        return {c: v / s for c, v in w.items() if v} if s else {}

    # --- per freguesia rows ---------------------------------------------------------------
    rows = []
    moved = defaultdict(float)
    lang_before = defaultdict(float)
    for g in freg:
        n3 = nuts[g]
        tot = cells[(g, "T")]
        langs = defaultdict(float)
        by_iso = defaultdict(float)
        for c in leaves:
            n = cells.get((g, c), 0)
            if not n:
                continue
            if c in OTHER:
                langs["Other"] += n
            elif c in REMAINDERS:
                sh = split(c, n3)
                if not sh:
                    langs["Other"] += n
                for iso, f in sh.items():
                    by_iso[iso] += n * f
            else:
                by_iso[INE_TO_ISO.get(c, c)] += n
        for iso, n in by_iso.items():
            keep = 1 - fr_build.TEO_FRENCH[fr_build.teo_region(iso, blocks.get(iso, "EUR"))]
            for lang, f in lang_shares(iso):
                lang_before[lang] += n * f
                if lang == "Portuguese":
                    langs[lang] += n * f
                    continue
                langs[lang] += n * f * keep
                langs["Portuguese"] += n * f * (1 - keep)
                moved[lang] += n * f * (1 - keep)
        portuguese_nat = cells[(g, "PT")]
        langs["Portuguese"] += portuguese_nat
        for lang, n in langs.items():
            if n > 0:
                rows.append({"geo_id": g, "geo_level": "freguesia", "geo_name": names[g],
                             "source_category": lang, "count": n, "tier": "derived",
                             "year": YEAR, "source_id": SOURCE_ID})

    df = pd.DataFrame(rows)

    # --- Mirandese ------------------------------------------------------------------------
    mcode = [m for m, n in muni.items() if n == MIRANDA_MUNI]
    assert len(mcode) == 1, mcode
    mcode = mcode[0]
    mir_fregs = [g for g in freg if g.startswith(mcode)]
    town = [g for g in mir_fregs if names[g] == MIRANDA_MUNI]
    assert len(town) == 1, [names[g] for g in mir_fregs]
    rural = [g for g in mir_fregs if g not in town]
    rural_pop = sum(cells[(g, "T")] for g in rural)
    muni_pop = cells[(mcode, "T")]
    rate = MIRANDESE_USERS / rural_pop
    print(f"  Mirandese: Miranda do Douro {muni_pop:,} people, {len(rural)} rural freguesias "
          f"{rural_pop:,}; 1,500 users = {rate:.1%} of them")
    vim = [g for g in freg if g.startswith(next(m for m, n in muni.items() if n == "Vimioso"))]
    vill = [g for g in vim if names[g] in VIMIOSO_VILLAGES]
    assert len(vill) == 2, [names[g] for g in vim]
    mir = {g: MIRANDESE_USERS * cells[(g, "T")] / rural_pop for g in rural}
    mir.update({g: rate * cells[(g, "T")] for g in vill})
    for g, n in mir.items():
        sel = (df["geo_id"] == g) & (df["source_category"] == "Portuguese")
        assert sel.sum() == 1 and df.loc[sel, "count"].iloc[0] > n
        df.loc[sel, "count"] -= n
        df = pd.concat([df, pd.DataFrame([{
            "geo_id": g, "geo_level": "freguesia", "geo_name": names[g],
            "source_category": "Mirandese", "count": n, "tier": "derived", "year": YEAR,
            "source_id": SOURCE_ID}])], ignore_index=True)
    print(f"  Mirandese drawn {sum(mir.values()):,.0f} on {len(mir)} freguesias: "
          + ", ".join(names[g] for g in mir))

    # --- checks on the result -------------------------------------------------------------
    per = df.groupby("geo_id")["count"].sum()
    off = max(abs(per[g] - cells[(g, "T")]) for g in freg)
    print(f"  rows sum to each freguesia's census total: largest gap {off:.6f}")
    assert off < 1e-6
    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"  total {nat.sum():,.0f}; Portuguese {nat['Portuguese'] / nat.sum():.2%}")
    print("  top languages after retention:")
    for k, v in nat.head(20).items():
        print(f"    {k:<24} {v:>12,.0f}   (before retention {lang_before.get(k, 0):>9,.0f})")
    print(f"  moved onto Portuguese by retention: {sum(moved.values()):,.0f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print("wrote", OUT, len(df), "rows")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        main()
