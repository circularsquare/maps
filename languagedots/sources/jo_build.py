"""Jordan: Jordanians by Arab Barometer first language (and ethnic group, for Circassian), everyone
else by the 2015 census nationality, per governorate, applied to the Department of Statistics'
end-2025 governorate populations -> data/normalized/jo.csv (node ids, as sources/kw_build.py).

    python sources/jo_build.py [--fetch]

NO CENSUS ASKS LANGUAGE. The 2015 census asked nationality (2026's is in the field now). So:

  * NATIONALITY SHARES per governorate from the 2015 census (DoS, Population and Housing Census
    2015): Table 3.1 (Jordanians and non-Jordanians by governorate) and Table 8.1 (non-Jordanians
    by nationality and governorate, 2,918,125). Each governorate's 2015 shares are applied to its
    end-2025 population (religiondots' jo_lookup.csv, 11,937,000). Syrians were 1,265,514 in
    2015; UNHCR registered about 0.5M in 2025-26, and returns since December 2024 are not
    reflected (sources/jo.md).
  * JORDANIANS: Arab Barometer II (2011), III (2013), IV (2016) first language, pooled per
    governorate, one respondent one vote (sources/ab_firstlang.py), the Iraq rule. Circassian
    from Arab Barometer VII (2021-22) ethnic group, 7 of 2,398 "Circassian", times the share of
    Circassian pupils who speak Circassian at home alone or with Arabic, 21.5% (Rannut 2009,
    Journal of Multilingual and Multicultural Development 30:4: 6.5% Circassian only, 15% both);
    VII's other answers (Arab 2,389, Turkmen 1, Other 1) as Arabic / other.
  * NON-JORDANIANS: each nationality on its home language or home mix (sources/gulf_mix.py's
    origin_mix, the Saudi method, sources/sa.md): Syrians, Palestinians (Gaza origin, no
    Jordanian nationality), Lebanese on Levantine Arabic; Egyptians on Egyptian Arabic; Iraqis
    on Iraqi Arabic; Yemenis on Yemeni Arabic; India and Sri Lanka at their drawn mixes.
    Nationalities under 500 in the whole country, and the census's "Other ..." rows, on `other`.

CHECKS: Table 8.1's per-governorate group totals equal the sum of their rows; its national
section equals the sum of the twelve governorates per nationality; Table 3.1's non-Jordanians per
governorate equal Table 8.1's; output sums to jo_lookup.csv's population.
"""
import csv
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(HERE), str(HERE / "sources"), str(HERE / "taxonomy")]
from rdlink import RD_GEO  # noqa: E402
import ab_firstlang as ab  # noqa: E402

RAW = HERE / "data" / "raw" / "jo"
OUT = HERE / "data" / "normalized" / "jo.csv"
LOOKUP = RD_GEO / "jo" / "jo_lookup.csv"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
URLS = {"Persons_3.1.pdf": "https://dosweb.dos.gov.jo/DataBank/Census2015/Persons/Persons_3.1.pdf",
        "Non-jordanian_8.1.pdf": ("https://dosweb.dos.gov.jo/DataBank/Census2015/Non-Jordanians/"
                                  "Non-jordanian_8.1.pdf")}
TOTAL = 11_937_000
NONJ_2015 = 2_918_125
POP_2015 = 9_531_712
SMALL = 500

GOV = {"Amman": "JO11", "Balqa": "JO12", "Zarqa": "JO13", "Madaba": "JO14", "Irbid": "JO21",
       "Mafraq": "JO22", "Jerash": "JO23", "Ajloun": "JO24", "Karak": "JO31",
       "Tafielah": "JO32", "Tafilah": "JO32", "Ma'an": "JO33", "Aqaba": "JO34"}

# Table 8.1 nationality -> ISO 3166 alpha-2 (those with SMALL or more people nationally)
ISO = {"Syria": "SY", "Palestine": "PS", "Egypt": "EG", "Iraq": "IQ", "Yemen": "YE",
       "Libya": "LY", "Bangladesh": "BD", "Philippines": "PH", "Saudi Arabia": "SA",
       "Sudan": "SD", "India": "IN", "Sri Lanka": "LK", "Indonesia": "ID", "Lebanon": "LB",
       "Pakistan": "PK", "United States of America": "US", "Kenya": "KE", "Britain": "UK",
       "Canada": "CA", "Ukraine": "UA", "Oman": "OM", "Kuwait": "KW", "Russia": "RU",
       "Turkey": "TR", "China": "CN", "Ethiopia": "ET", "Morocco": "MA", "France": "FR",
       "Australia": "AU", "Germany": "DE", "Malaysia": "MY", "Tunisia": "TN", "Spain": "ES",
       "Italy": "IT", "Sweden": "SE", "Bahrain": "BH", "Thailand": "TH", "South Korea": "KR",
       "Algeria": "DZ", "Nigeria": "NG", "Holland": "NL", "United Arab Emirates": "AE",
       "Romania": "RO", "Burma - Myanmar": "MM", "Qatar": "QA", "Brazil": "BR",
       "Norway": "NO", "Iran": "IR"}
# drawn on this map and taken at their drawn mix whatever their size (gulf_mix's 20,000 floor
# would put Sri Lankans on Tamil and Canadians on French via COUNTRY_LANG)
FORCE_MIX = {"LK", "IN"}
FORCE_LANG = {"CA": "indoeuropean.germanic.english", "BR": "indoeuropean.romance.portuguese"}

# Arab Barometer region labels -> unit
REGION = {"3501. Capital": "JO11", "3502. Balqa": "JO12", "3503. az-Zarqa": "JO13",
          "3504. Madaba": "JO14", "3505. Irbid": "JO21", "3506. Mafraq": "JO22",
          "3507. Jerash": "JO23", "3508. Ajloun": "JO24", "3509. Karak": "JO31",
          "3510. Tafila": "JO32", "3511. Maan": "JO33", "3512. Aqaba": "JO34",
          "The capital": "JO11", "Amman": "JO11", "Balqa": "JO12", "Zarqa": "JO13",
          "Madaba": "JO14", "Irbid": "JO21", "Mafraq": "JO22", "Jerash": "JO23",
          "Ajloun": "JO24", "al Karak": "JO31", "Karak": "JO31", "Tafilah": "JO32",
          "Ma'an": "JO33", "Aqaba": "JO34",
          "العاصمة": "JO11", "البلقاء": "JO12", "الزرقاء": "JO13", "مادبا": "JO14",
          "اربد": "JO21", "المفرق": "JO22", "جرش": "JO23", "عجلون": "JO24", "الكرك": "JO31",
          "الطفيلة": "JO32", "معان": "JO33", "العقبة": "JO34"}
AB_N = {"ABII": 1188, "ABIII": 1795, "ABIV": 1500, "ABVII": 2399}

LEVANTINE = "afroasiatic.levantine_arabic"
CIRCASSIAN = "abkhazadyghe.circassian"
CIRCASSIAN_HOME = 0.215     # Rannut 2009: 6.5% Circassian only + 15% Circassian and Arabic
# survey answer -> node (Jordanians)
ANSWER = {"1. Arabic": LEVANTINE, "Arabic": LEVANTINE, "Arab": LEVANTINE,
          "2. English": "indoeuropean.germanic.english", "English": "indoeuropean.germanic.english",
          "3. French": "indoeuropean.romance.french", "German": "indoeuropean.germanic.continental.german",
          # four answers, two in Jerash, two in Mafraq: no Bosnian or Serbian nationals live in
          # either (Table 8.1), Jerash is a Circassian settlement; not readable as printed
          "Serbo-Croatian": "other",
          "Turkmen": "other", "Other": "other", "Circassian": CIRCASSIAN}


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings()
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        r = requests.get(url, headers={"User-Agent": UA}, timeout=120, verify=False)
        r.raise_for_status()
        tmp = (RAW / name).with_suffix(".part")
        tmp.write_bytes(r.content)
        tmp.replace(RAW / name)
        print(f"  {name}: {len(r.content):,} bytes")


def _lines(pdf):
    import fitz
    for page in fitz.open(pdf):
        for line in page.get_text().splitlines():
            if line.strip():
                yield line.strip()


def table_81():
    """{section: {nationality: total}}; section 'Jordan' is the national one."""
    lines = list(_lines(RAW / "Non-jordanian_8.1.pdf"))
    out, sec, group = {}, None, []
    j = 0
    while j < len(lines):
        ln = lines[j]
        nums = lines[j + 1:j + 10]
        if (re.fullmatch(r"[A-Za-z][A-Za-z .,'&/()-]*", ln) and len(nums) == 9
                and all(re.fullmatch(r"\d+", x) for x in nums)):
            v = [int(x) for x in nums]
            if v[2] != v[0] + v[1] or v[5] != v[3] + v[4] or v[8] != v[2] + v[5]:
                raise SystemExit(f"8.1 {sec} {ln}: row does not add up {v}")
            if ln == "Total":
                if group and sum(group) != v[8]:
                    raise SystemExit(f"8.1 {sec}: group total {v[8]} != {sum(group)}")
                if not group and v[8] not in (sum(out[sec].values()), NONJ_2015):
                    # two Totals in a row: the second is the section's grand total
                    raise SystemExit(f"8.1 {sec}: grand total {v[8]} != {sum(out[sec].values())}")
                group = []
            else:
                if ln == "Countries":            # "Other Central American" / "Countries"
                    ln = "Other Central American Countries"
                key = ln
                while key in out[sec]:           # Zambia is printed twice
                    key += "*"
                out[sec][key] = v[8]
                group.append(v[8])
            j += 10
            continue
        if ln == "Jordan" and sec is None:
            sec = "Jordan"
            out[sec] = {}
        elif ln in GOV:
            sec = ln
            out[sec] = {}
            group = []
        j += 1
    return out


def table_31():
    """{governorate: (non-Jordanian, total inside)} for the 12 governorates, 2015."""
    lines = list(_lines(RAW / "Persons_3.1.pdf"))
    out = {}
    for i, ln in enumerate(lines):
        if ln in GOV and ln not in out:
            k = i + 1
            while k < len(lines) and not (
                    lines[k] == "Total"
                    and all(re.fullmatch(r"\d+", x) for x in lines[k + 1:k + 13])):
                k += 1
            v = [int(x) for x in lines[k + 1:k + 13]]
            out[ln] = (v[5], v[11])
    return out


def main():
    if "--fetch" in sys.argv or not all((RAW / n).exists() for n in URLS):
        fetch()
    import gulf_mix as gm

    pop, names = {}, {}
    with open(LOOKUP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pop[r["unit"]] = int(r["pop"])
            names[r["unit"]] = r["name"]
    if len(pop) != 12 or sum(pop.values()) != TOTAL:
        raise SystemExit(f"jo_lookup.csv: {len(pop)} units, {sum(pop.values()):,}")

    t81, t31 = table_81(), table_31()
    govs = [g for g in t81 if g != "Jordan"]
    if len(govs) != 12 or len(t31) != 12:
        raise SystemExit(f"8.1 sections {govs}, 3.1 governorates {sorted(t31)}")
    nat = t81["Jordan"]
    if sum(nat.values()) != NONJ_2015:
        raise SystemExit(f"8.1 national rows sum to {sum(nat.values()):,}")
    for k, v in nat.items():
        s = sum(t81[g].get(k, 0) for g in govs)
        if s != v:
            raise SystemExit(f"8.1 {k}: governorates sum {s}, national {v}")
    if sum(t31[g][1] for g in govs) != POP_2015:
        raise SystemExit("3.1 governorates do not sum to 9,531,712")
    for g in govs:
        if sum(t81[g].values()) != t31[g][0]:
            raise SystemExit(f"{g}: 8.1 sums {sum(t81[g].values())}, 3.1 non-Jordanians {t31[g][0]}")
    print(f"  census 2015: {POP_2015:,} people, {NONJ_2015:,} non-Jordanians in 12 governorates; "
          "8.1 and 3.1 agree per governorate")

    # nationality -> mix
    def key(k):
        k = k.rstrip("*")
        if nat.get(k, 0) < SMALL or k not in ISO:
            return "node:other"
        return ISO[k]
    mixes = {"node:other": {"other": 1.0}}
    for k in nat:
        iso = key(k)
        if iso in mixes:
            continue
        if iso in FORCE_LANG:
            mixes[iso] = {FORCE_LANG[iso]: 1.0}
        elif iso in FORCE_MIX:
            mixes[iso] = gm.home_mix(iso)
        else:
            mixes[iso] = gm.origin_mix(iso, "JO", nat[k.rstrip("*")])
    small = sum(v for k, v in nat.items() if key(k) == "node:other")
    print(f"  {small:,} non-Jordanians in nationalities under {SMALL:,} or 'Other' rows -> other")

    # Jordanians: pooled survey answers per governorate
    pooled = {u: {} for u in pop}
    for wave in ("ABII", "ABIII", "ABIV"):
        for u, v in ab.answers("Jordan", wave, REGION, AB_N[wave]).items():
            for a, n in v.items():
                if a not in ANSWER:
                    raise SystemExit(f"{wave}: answer {a!r} not mapped")
                pooled[u][ANSWER[a]] = pooled[u].get(ANSWER[a], 0) + n
    for u, v in ab.answers("Jordan", "ABVII", REGION, AB_N["ABVII"]).items():
        for a, n in v.items():
            if a not in ANSWER:
                raise SystemExit(f"ABVII: answer {a!r} not mapped")
            if a == "Circassian":
                pooled[u][CIRCASSIAN] = pooled[u].get(CIRCASSIAN, 0) + n * CIRCASSIAN_HOME
                pooled[u][LEVANTINE] = pooled[u].get(LEVANTINE, 0) + n * (1 - CIRCASSIAN_HOME)
            else:
                pooled[u][ANSWER[a]] = pooled[u].get(ANSWER[a], 0) + n

    rows = []
    for g in govs:
        u = GOV[g]
        nonj, tot15 = t31[g]
        parts = {"jordanian": (tot15 - nonj) / tot15}
        for k, v in t81[g].items():
            iso = key(k)
            parts[iso] = parts.get(iso, 0) + v / tot15
        share = {}
        jv = pooled[u]
        jt = sum(jv.values())
        for n, x in jv.items():
            share.setdefault(("modelled", "Jordanian", n), 0)
            share[("modelled", "Jordanian", n)] += parts["jordanian"] * x / jt
        for iso, p in parts.items():
            if iso == "jordanian":
                continue
            for n, x in mixes[iso].items():
                share.setdefault(("derived", "non-Jordanian", n), 0)
                share[("derived", "non-Jordanian", n)] += p * x
        cnt = ab.shares_to_counts(share, pop[u])
        for (tier, origin, n), c in sorted(cnt.items(), key=lambda kv: -kv[1]):
            if c:
                rows.append(dict(geo_id=u, geo_level="governorate", geo_name=names[u],
                                 source_category=n, count=c, tier=tier, source_id=origin,
                                 year=2015, note=f"{origin}; DoS end-2025 population {pop[u]}"))
        print(f"  {names[u]:<8} Jordanian {parts['jordanian']:.1%}, survey n {jt:.0f}: "
              + ", ".join(f"{n.split('.')[-1]} {x / jt:.2%}" for n, x in
                          sorted(jv.items(), key=lambda kv: -kv[1])[:4]))
    ab.report(rows, TOTAL)
    ab.write_csv(OUT, rows)
    print(f"  wrote {OUT.relative_to(HERE)}")


if __name__ == "__main__":
    main()
