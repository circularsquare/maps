"""Ukraine, All-Ukrainian Population Census 2001: native language by raion and city -> data/normalized/ua.csv.

    python sources/ua_c01.py [--fetch]

THE QUESTION. "Рідна мова", native language: one answer per person, the language the person
names as native. In the ex-USSR tradition it leans towards identity rather than use (Ukrstat's
own summary page: 14.8% of Ukrainians named Russian; many more speak it at home).

TWO TABLES OF THE SAME CENSUS, because neither is enough alone.

  A. Ukrstat's "Розподіл населення регіонів України за рідною мовою у розрізі адміністративно-
     територіальних одиниць" (2001.ukrcensus.gov.ua/i/u/popul_adm_00.zip, linked from
     /results/nationality_population/): one .xls per region, every raion, city, town, village
     council and village, with the share (%, two decimals) naming each of 17 languages:
     Ukrainian, Russian, Belarusian, Bulgarian, Armenian, Gagauz, Crimean Tatar, Moldovan,
     German, Polish, Romani ("циганську"), Romanian, Slovak, Hungarian, Karaim, "Jewish"
     ("єврейську") and Greek. Shares only, no counts; "-" means nobody, "0.00" fewer than
     0.005%. Everything else (other languages and "not stated") is the unprinted remainder.

  B. The U.S. Census Bureau's "Ukraine Subnational Population and Housing Data Tables with
     Administrative Boundaries" (HDX, CC BY), transcribing Ukrstat's census database ("Distribution
     of the population by nationality and native language, ... oblast"): the 2001 population per
     unit, the 2001 boundaries keyed identically, and two attribute tables.
       `Language`: counts for Ukrainian, Belarusian, Crimean Tatar, Moldovan, Russian, Romanian,
       Slovak, Hungarian, "Other language", "Unstated". Ukrstat's database blanks cells of 1-9
       people, and the Bureau marks some of its sums -999 (1,136 cells). ONLY THE UKRAINIAN AND
       RUSSIAN COLUMNS ARE USED: the other six fall short of A in 802 cells not marked -999
       (Biliaivskyi raion: 0 Belarusian against A's 199; nationally 16,921 against 56,200), a
       suppressed component silently left out of the sum. Its `Other language` is not the
       remainder either (the Meskhetian Turks' Turkish in Chaplynskyi raion is in no column).
       `Nationality-Language`: nationality x native language ("own nationality's language",
       Ukrainian, Russian, ... other, not stated). Used for two things below.

HOW A UNIT'S COUNTS ARE BUILT (`build_unit`). T = the unit's 2001 population (B).
  1. Ukrainian and Russian: B's exact count where it is not -999 (`measured`); where it is,
     T x A's share (`derived`).
  2. The other 15 languages: T x A's share (`derived`).
  3. Not stated: B's `Unstated` where not -999 (not drawn; it is the gap). A suppressed cell
     holds 1-9 people and is taken as 0.
  4. The remainder r = T - all the above holds every other native language. Out of it come the
     people of eight nationalities who named their own nationality's language, from B's
     `Nationality-Language` (NL_NTV_<x>, each a census count; -999 taken as 0): Tatars (Tatar),
     Azerbaijanis, Georgians, Turks (Turkish), Arabs (Arabic), Vietnamese, Uzbeks, Koreans,
     the eight whose own language is unambiguous and who are at least 1,800 nationally. Each is
     capped at what is left of r. What is still left is `other` (`derived`): it holds the other
     nationalities' own languages and anyone naming a language of a nationality not their own
     outside the 17 (a Gagauz who named Bulgarian is in A's Bulgarian already).
  5. Rounding: A's shares carry two decimals, so a share-based count is off by up to 0.005% of
     T. If r comes out negative the shortfall comes off not-stated first, then off the largest
     share-based language, so every unit sums to T exactly. Printed.

UNITS. 661 raions and cities of oblast significance (B's ADM2, 2001 boundaries), Sevastopol as
one unit (A splits it into city districts B has no polygons for), and Kyiv's 10 districts: B has
Kyiv's languages only for the whole city, A has shares per district, so Kyiv is built as one unit
by steps 1-5 and then each language is shared over the districts by A's share x B's district
population (other, not stated and the eight own-language counts by each district's unprinted
remainder). Every Kyiv district row is `derived`.

JOIN, A to B. Per region file, A's rows at the raion/city level (label in the sheet's second
column) are joined to B's ADM2 rows on a transliteration of the Ukrainian name against B's
`NSO_NAME` (Ukraine's national romanisation), with the pins below (B's own misprint for
Volodymyr-Volynskyi raion; three romanisation variants, one in Kyiv). The witness neither name decides: every
Ukrainian and Russian count in B must agree with A's share (check 4).

CHECKS (all must pass, or nothing is written):
  1. B's Language sheet holds only the -999 sentinel; units sum to 48,240,902, the census figure
  2. A: 27 region files; raion/city rows per region equal B's ADM2 count (+ Prypiat, emptied)
  3. the join is one to one in every region, with no name left over on either side
  4. every B count of Ukrainian and Russian (661 units, Kyiv, Sevastopol) agrees with A's share
     within 0.01 points; those beyond A's rounding (0.005) are printed (one, by 2 people)
  5. Kyiv's district populations sum to the city's; A's city row agrees with B's city counts
  6. every unit's rows sum to its population; the country to 48,240,902
  7. printed, a witness from the other table: per language, A's count against B's people of the
     matching nationality naming their own language (A's must be the larger, near it)
"""
import argparse
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "ua"
OUT = HERE / "data" / "normalized" / "ua.csv"
XLSX = RAW / "ukraine_uscb_201905.xlsx"
GDB = RAW / "ukraine.gdb.zip"
ZIP = RAW / "popul_adm_00.zip"
URLS = {
    XLSX: ("https://data.humdata.org/dataset/45c19756-9d0b-4673-9de4-fc47168c369a/resource/"
           "7f1ace67-fd49-4d2a-8b25-cc0561b5145b/download/ukraine_uscb_201905.xlsx"),
    GDB: ("https://data.humdata.org/dataset/45c19756-9d0b-4673-9de4-fc47168c369a/resource/"
          "134a344c-0d1c-4055-a5f3-1336f25b5655/download/ukraine.gdb.zip"),
    ZIP: "http://2001.ukrcensus.gov.ua/i/u/popul_adm_00.zip",
}
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

NATIONAL = 48_240_902
# A's 17 column headers, as printed (accusative: "named as native ... language")
A_LANGS = ["українську", "російську", "білоруську", "болгарську", "вірменську", "гагаузьку",
           "кримсько-татарську", "молдовську", "німецьку", "польську", "циганську", "румунську",
           "словацьку", "угорську", "караїмську", "єврейську", "грецьку"]
# B's exact columns and A's header for the same language
B_COLS = {"LNG_UKR": "українську", "LNG_RUS": "російську"}
# B's other six language columns fall short of A wherever part of the language was suppressed
# without a -999 (Biliaivskyi raion: B 0 Belarusian, A 0.19% = 199 people); printed, not used
B_SHORT = {"LNG_BEL": "білоруську", "LNG_CRH": "кримсько-татарську", "LNG_MOL": "молдовську",
           "LNG_RON": "румунську", "LNG_SLK": "словацьку", "LNG_HUN": "угорську"}
NOT_STATED = "Did Not Indicate"           # B's original field name for LNG_UNST
REMAINDER = "other native languages (remainder)"
# step 4: nationality code -> B's original field name ("Native Language, <people>")
OWN = {"TAT": "Native Language, Tatars", "AZE": "Native Language, Azerbaijanians",
       "GEO": "Native Language, Georgians", "TUR": "Native Language, Turks",
       "ARA": "Native Language, Arabs", "VNM": "Native Language, Vietnamiens",
       "UZB": "Native Language, Uzbeks", "KOR": "Native Language, Koreans"}
# witness 7: A's language -> the nationality whose own language it is
WITNESS = {"болгарську": "BGR", "вірменську": "ARM", "гагаузьку": "GGZ", "німецьку": "DEU",
           "польську": "POL", "циганську": "ROM", "караїмську": "KRM", "єврейську": "JEW",
           "грецьку": "GRC", "білоруську": "BLR", "молдовську": "MDA", "румунську": "ROU",
           "угорську": "HUN", "словацьку": "SVK", "кримсько-татарську": "CRH"}

# JOIN PINS: B GEO_MATCH -> A's label in that region
PINS = {
    # B's NSO_NAME for the raion repeats the city's ("M. VOLODYMYR-VOLYNSKYI"); AREA_NAME and
    # USCBCMNT ("Vladimir-Volynskiy Rayon") say raion, and A's raion row has its exact shares
    "UKR_24_20": "ВОЛОДИМИР-ВОЛИНСЬКИЙ РАЙОН",
    "UKR_07_32": "СЕЛИДОВЕ (міськрада)",      # B romanises SELIDOVE
    "UKR_07_35": "СЛАВ'ЯНСЬК (міськрада)",    # B: SLOVIANSK (the 2016 spelling); A: Слав'янськ
}
KYIV_PINS = {"UKR_16_07": "ПОДІЛЬСЬКИЙ район"}   # B: PODOLSKYI RAION
EMPTY_IN_A = {"ПРИП'ЯТЬ (міськрада)"}         # Prypiat: a row of dashes, no B unit
KYIV, SEV = "UKR_16", "UKR_02"
SEV_UNIT = "UKR_02_01"                       # B's one ADM2 polygon for Sevastopol


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for path, url in URLS.items():
        data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300).read()
        if data[:2] != b"PK" or len(data) < 1_000_000:
            raise SystemExit(f"{path.name}: not a zip/xlsx ({len(data):,} bytes)")
        path.write_bytes(data)
        print(f"wrote {path} ({len(data):,} bytes)")


# ---- transliteration (Ukraine's national system, 2010) ----
_T = {"А": "A", "Б": "B", "В": "V", "Г": "H", "Ґ": "G", "Д": "D", "Е": "E", "Є": "IE", "Ж": "ZH",
      "З": "Z", "И": "Y", "І": "I", "Ї": "I", "Й": "I", "К": "K", "Л": "L", "М": "M", "Н": "N",
      "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U", "Ф": "F", "Х": "KH", "Ц": "TS",
      "Ч": "CH", "Ш": "SH", "Щ": "SHCH", "Ь": "", "Ю": "IU", "Я": "IA"}
_INITIAL = {"Є": "YE", "Ї": "YI", "Й": "Y", "Ю": "YU", "Я": "YA"}


def translit(s):
    s = s.upper().replace("'", "").replace("’", "").replace("ЗГ", "ZGH")
    out = []
    for i, ch in enumerate(s):
        initial = i == 0 or not s[i - 1].isalpha()
        out.append(_INITIAL[ch] if initial and ch in _INITIAL else _T.get(ch, ch))
    return "".join(out)


def name_key(s):
    """Fold a romanised unit name: drop "M. " and "(MISKRADA)" and B's "(... OBLAST)" qualifier."""
    s = s.upper().strip()
    s = s.replace("(MISKRADA)", "") if "MISKRADA" in s else re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"^M\. ", "", s.strip())
    return re.sub(r"[^A-Z]", "", s)


def share(v):
    """A's cell: '-' is nobody, a number is a percentage."""
    if isinstance(v, str):
        v = v.strip()
        # "-" is nobody; a few cells carry another dash (U+05BE, U+2013) for the same thing
        if v in ("", "-", "־", "–", "—", "−"):
            return 0.0
        return float(v.replace(",", "."))
    return float(v)


def read_a():
    import xlrd
    rows = []
    z = zipfile.ZipFile(ZIP)
    files = z.infolist()
    for info in files:
        fname = info.filename.encode("cp437").decode("cp866", "replace")
        s = xlrd.open_workbook(file_contents=z.read(info)).sheets()[0]
        c0 = None
        for r in range(12):
            v = [str(x).strip() for x in s.row_values(r)]
            if "українську" in v:
                c0, hr = v.index("українську"), r
                if v[c0:c0 + 17] != A_LANGS:
                    raise SystemExit(f"{fname}: header {v[c0:c0 + 17]}")
                break
        if c0 is None:
            raise SystemExit(f"{fname}: no header row")
        for r in range(hr + 2, s.nrows):
            v = s.row_values(r)
            labs = [(j, str(v[j]).strip()) for j in range(c0) if str(v[j]).strip()]
            if not labs:
                continue
            level, label = labs[0]
            if label.startswith("*") or label == "у тому числі:":
                continue
            rec = dict(file=fname, row=r, level=level, label=label)
            for k, h in enumerate(A_LANGS):
                rec[h] = share(v[c0 + k])
            rec["blank"] = all(rec[h] == 0 for h in A_LANGS)
            rows.append(rec)
    return len(files), pd.DataFrame(rows)


def read_b(sheet):
    df = pd.read_excel(XLSX, sheet_name=sheet, header=0, skiprows=[1])
    for c in df.columns:
        if c.startswith(("LNG_", "NL_", "B")) and c not in ("BTOTL_",):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def apportion(total, weights):
    """Largest-remainder split of an integer total by weights (equal shares if all are 0)."""
    w = pd.Series(weights, dtype=float).clip(lower=0)
    if w.sum() <= 0:
        w = pd.Series(1.0, index=w.index)
    raw = w / w.sum() * total
    base = raw.astype("int64")
    left = int(total - base.sum())
    if left:
        base[(raw - base).sort_values(ascending=False).index[:left]] += 1
    return base


def build_unit(T, a, b, nl):
    """Steps 1-5. a: A's shares (dict by header); b: B's Language row; nl: B's N-L row.
    Returns {source_category: (count, tier)} summing to T exactly, plus a log dict."""
    out, log = {}, {"capped": 0, "short": 0}
    exact = {h: int(b[c]) for c, h in B_COLS.items() if b[c] >= 0}
    for h in A_LANGS:
        if h in exact:
            out[h] = [exact[h], "measured"]
        else:
            out[h] = [int(round(a[h] * T / 100)), "derived"]
    unst = int(b["LNG_UNST"]) if b["LNG_UNST"] >= 0 else 0
    r = T - sum(v[0] for v in out.values()) - unst
    for x, cat in OWN.items():
        n = nl[f"NL_NTV_{x}"]
        n = int(n) if pd.notna(n) and n > 0 else 0
        take = min(n, max(r, 0))
        log["capped"] += n - take
        out[cat] = [take, "measured" if take == n else "derived"]
        r -= take
    if r < 0:
        log["short"] = -r
        cut = min(unst, -r)
        unst -= cut
        r += cut
        while r < 0:
            # off the largest share-based language
            h = max((k for k in A_LANGS if out[k][1] == "derived"), key=lambda k: out[k][0])
            cut = min(out[h][0], -r)
            out[h][0] -= cut
            r += cut
    out[REMAINDER] = [r, "derived"]
    out[NOT_STATED] = [unst, "measured"]
    assert sum(v[0] for v in out.values()) == T
    return out, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch or not all(p.exists() for p in URLS):
        fetch()

    ok = True

    def report(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    print("Ukraine, 2001 census native language (Ukrstat by unit + USCB counts)\n")
    lang = read_b("Language")
    nl = read_b("Nationality-Language").set_index("GEO_MATCH")
    age = read_b("Age-Sex 2001").set_index("GEO_MATCH")["BTOTL"]
    lcols = [c for c in lang.columns if c.startswith("LNG_")]
    neg = sorted(set(lang[lcols].stack()[lambda s: s < 0]))
    report(neg == [-999], f"B's Language sheet: the only negative value is -999 ({neg})")
    units = lang[lang["LNG_TPOP"].notna() & (lang["ADM_LEVEL"] > 0)]
    adm2 = units[units["ADM_LEVEL"] == 2].copy()
    whole = units[units["ADM_LEVEL"] == 1].set_index("GEO_MATCH")
    report(len(adm2) == 661 and sorted(whole.index) == [SEV, KYIV],
           f"B: {len(adm2)} ADM2 units with data, and whole-city rows for {sorted(whole.index)}")
    tot = int(units["LNG_TPOP"].sum())
    report(tot == NATIONAL, f"B's units sum to {tot:,} (census {NATIONAL:,})")

    # ---- A ----
    nfiles, A = read_a()
    region = A[A["level"] == 0].groupby("file")["label"].first()
    rk = {f: name_key(translit(l)) for f, l in region.items()}
    fix = {"KYYIVSKAOBLAST": "KYIVSKAOBLAST", "MYKOLAYIVSKAOBLAST": "MYKOLAIVSKAOBLAST"}
    adm2["rk"] = adm2["ADM1_NAME"].map(lambda s: fix.get(re.sub(r"[^A-Z]", "", s.upper()),
                                                         re.sub(r"[^A-Z]", "", s.upper())))
    kyiv_f = [f for f in rk if rk[f] == "KYIV"]
    sev_f = [f for f in rk if rk[f].startswith("SEVASTOPOL")]
    obl = {f for f in rk if f not in kyiv_f + sev_f}
    report(nfiles == 27 and len(kyiv_f) == 1 and len(sev_f) == 1
           and set(rk[f] for f in obl) == set(adm2["rk"]),
           f"A: {nfiles} region files; 25 regions name-match B's ADM1, plus Kyiv and Sevastopol")
    A1 = A[A["file"].isin(obl) & (A["level"] == 1)].copy()
    A1["rk"] = A1["file"].map(rk)
    dropped = A1[A1["label"].isin(EMPTY_IN_A)]
    report(len(dropped) == 1 and bool(dropped["blank"].all()),
           f"A's only raion/city row with no B unit is Prypiat, and it is all dashes")
    A1 = A1[~A1["label"].isin(EMPTY_IN_A)]
    per_a, per_b = A1.groupby("rk").size(), adm2.groupby("rk").size()
    report(per_a.equals(per_b.reindex(per_a.index)) and len(per_a) == 25,
           "raion/city rows per region equal B's ADM2 count in all 25 regions")

    # ---- join ----
    A1["k"] = A1["label"].map(lambda s: name_key(translit(s)))
    adm2["k"] = adm2["NSO_NAME"].astype(str).map(name_key)
    for g, lab in PINS.items():
        hit = A1[A1["label"] == lab]
        if len(hit) != 1:
            raise SystemExit(f"pin {g}: {lab} found {len(hit)} times in A")
        adm2.loc[adm2["GEO_MATCH"] == g, "k"] = "PIN" + g
        A1.loc[hit.index, "k"] = "PIN" + g
    dup = int(adm2.duplicated(["rk", "k"]).sum() + A1.duplicated(["rk", "k"]).sum())
    j = adm2.merge(A1, on=["rk", "k"], how="outer", indicator=True)
    lost = j[j["_merge"] != "both"]
    report(dup == 0 and lost.empty, f"join one to one: {int((j['_merge'] == 'both').sum())} pairs, "
                                    f"{dup} duplicate keys, {len(lost)} left over")
    if not lost.empty:
        print(lost[["rk", "k", "NSO_NAME", "label"]].to_string())
    j = j[j["_merge"] == "both"].copy()

    # whole-city rows: Kyiv and Sevastopol from their own files' first (level 0) row
    def city_row(f):
        r = A[(A["file"] == f) & (A["level"] == 0)].iloc[0]
        return r
    kyiv_a, sev_a = city_row(kyiv_f[0]), city_row(sev_f[0])
    kyiv_d = A[(A["file"] == kyiv_f[0]) & (A["level"] == 0)].iloc[1:]

    # ---- witness 4 ----
    bad, edge, short = [], [], []

    def witness(geo, b, a):
        for c, h in B_SHORT.items():
            if b[c] >= 0 and 100 * b[c] / b["LNG_TPOP"] < a[h] - 0.005 - 1e-9:
                short.append(c)
        for c, h in B_COLS.items():
            if b[c] >= 0:
                d = abs(a[h] - 100 * b[c] / b["LNG_TPOP"])
                if d > 0.005 + 1e-9:
                    edge.append((geo, h, a[h], round(100 * b[c] / b["LNG_TPOP"], 4)))
                if d > 0.01 + 1e-9:
                    bad.append((geo, h, a[h], round(100 * b[c] / b["LNG_TPOP"], 4)))
    n_exact = 0
    for _, r in j.iterrows():
        witness(r["GEO_MATCH"], r, r)
        n_exact += sum(r[c] >= 0 for c in B_COLS)
    witness(KYIV, whole.loc[KYIV], kyiv_a)
    witness(SEV, whole.loc[SEV], sev_a)
    report(not bad, f"{n_exact:,} B counts of Ukrainian and Russian agree with A's share within "
                    f"0.01 points ({len(bad)} do not{': ' + str(bad[:4]) if bad else ''}); "
                    f"{len(edge)} differ by more than A's rounding: {edge[:3]}")
    print(f"  ..  B's other six language columns fall short of A's share in "
          f"{len(short)} cells ({pd.Series(short).value_counts().to_dict()}); not used")

    # ---- Kyiv districts ----
    kd = adm2.iloc[:0]
    kb = lang[lang["GEO_MATCH"].str.match(rf"{KYIV}_\d\d$")].copy()
    kb["pop"] = kb["GEO_MATCH"].map(age).astype("int64")
    kb["k"] = kb["NSO_NAME"].astype(str).map(name_key)
    kyiv_d = kyiv_d.assign(k=kyiv_d["label"].map(lambda s: name_key(translit(s))))
    for g, lab in KYIV_PINS.items():
        kb.loc[kb["GEO_MATCH"] == g, "k"] = "PIN" + g
        kyiv_d.loc[kyiv_d["label"] == lab, "k"] = "PIN" + g
    kj = kb.merge(kyiv_d, on="k", how="inner")
    report(len(kj) == 10 and len(kb) == 10 and int(kb["pop"].sum()) == int(whole.loc[KYIV, "LNG_TPOP"]),
           f"Kyiv: 10 districts joined by name; their populations sum to the city's "
           f"({int(kb['pop'].sum()):,})")

    # ---- build ----
    recs, logs = [], []

    def emit(geo, name, oblast, built, tier_override=None):
        for cat, (n, tier) in built.items():
            if n > 0:
                recs.append((geo, name, oblast, cat, int(n), tier_override or tier))

    for _, r in j.iterrows():
        built, log = build_unit(int(r["LNG_TPOP"]), r, r, nl.loc[r["GEO_MATCH"]])
        logs.append(log)
        emit(r["GEO_MATCH"], r["AREA_NAME"], r["ADM1_NAME"], built)
    built, log = build_unit(int(whole.loc[SEV, "LNG_TPOP"]), sev_a, whole.loc[SEV], nl.loc[SEV])
    logs.append(log)
    emit(SEV_UNIT, "MISTO SEVASTOPOL’", "MISTO SEVASTOPOL’", built)
    kyiv, log = build_unit(int(whole.loc[KYIV, "LNG_TPOP"]), kyiv_a, whole.loc[KYIV], nl.loc[KYIV])
    logs.append(log)
    kj = kj.set_index("GEO_MATCH")
    rem_w = (kj["pop"] - sum(kj[h] * kj["pop"] / 100 for h in A_LANGS)).clip(lower=0)
    for cat, (n, _) in kyiv.items():
        w = kj[cat] * kj["pop"] if cat in A_LANGS else rem_w
        for g, v in apportion(n, w).items():
            if v > 0:
                recs.append((g, kj.loc[g, "AREA_NAME"], "MISTO KYYIV", cat, int(v), "derived"))

    out = pd.DataFrame(recs, columns=["geo_id", "geo_name", "oblast", "source_category", "count", "tier"])
    out.insert(1, "geo_level", "unit")
    capped = sum(l["capped"] for l in logs)
    short = [l["short"] for l in logs if l["short"]]
    print(f"  ..  own-language counts capped by the remainder: {capped:,} people; "
          f"{len(short)} units where A's rounding left the remainder short (largest {max(short or [0])})")

    # ---- check 6 ----
    pop = dict(zip(j["GEO_MATCH"], j["LNG_TPOP"].astype("int64")))
    pop[SEV_UNIT] = int(whole.loc[SEV, "LNG_TPOP"])
    s = out.groupby("geo_id")["count"].sum()
    kyiv_ids = list(kj.index)
    off = [(g, int(s.get(g, 0)), p) for g, p in pop.items() if int(s.get(g, 0)) != p]
    report(not off, f"every unit's rows sum to its population ({len(off)} differ)")
    kt = int(s[kyiv_ids].sum())
    report(kt == int(whole.loc[KYIV, "LNG_TPOP"]), f"Kyiv's districts sum to the city ({kt:,})")
    kdev = (s[kyiv_ids] - kj["pop"]).abs().max()
    print(f"  ..  Kyiv district totals differ from their census populations by at most {kdev:,}")
    report(int(out["count"].sum()) == NATIONAL, f"the country sums to {int(out['count'].sum()):,}")

    # ---- witness 7 ----
    print("\n  witness: A's count against B's people of the nationality naming its own language")
    nat = out.groupby("source_category")["count"].sum()
    for h, x in WITNESS.items():
        own = nl[f"NL_NTV_{x}"]
        own = int(own[(nl["ADM_LEVEL"] > 0) & (own > 0) & nl.index.isin(list(pop) + [KYIV, SEV])].sum())
        print(f"    {h:<20} {int(nat.get(h, 0)):>10,}  own-nationality speakers {own:>10,}  "
              f"ratio {nat.get(h, 0) / own if own else float('nan'):.2f}")
    print("\n  national totals:")
    for cat, v in nat.sort_values(ascending=False).items():
        print(f"    {cat:<40} {v:>12,}  {100 * v / NATIONAL:6.2f}%")

    if not ok:
        raise SystemExit("\nreconciliation FAILED; nothing written")
    out = out.sort_values(["geo_id", "count"], ascending=[True, False])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"\nwrote {OUT}: {len(out):,} rows, {out['geo_id'].nunique()} units, "
          f"{out['count'].sum():,} people")


if __name__ == "__main__":
    main()
