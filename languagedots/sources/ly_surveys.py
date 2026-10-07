"""Libya: first / home language by district from three survey rounds, pooled, as shares applied
to the Bureau of Statistics and Census's 2020 estimate of Libyans by district -> data/normalized/ly.csv.

    python sources/ly_surveys.py [--fetch]

NO CENSUS ASKS (2006 was the last, no language item; the coverage sweep found no tabulation).
Open sources with a language question and a district code:

  World Values Survey (online analysis tool, no registration; unweighted counts):
    wave 7, 2022   Q272 language at home x N_REGION_ISO        1,196
    wave 6, 2014   V247 language at home x V256 region          2,131
  Arab Barometer III, 2014: q1019_1 first language x q1 district, 1,247 (religiondots' .sav,
    read-only)
  Arab Barometer VI-3 (2021) and VII (2022) ask ethnicity only (Q1012B: Arab, Amazigh, Tuareg,
    Toubou, Kouloughli): used as the CHECK on the pooled Berber share, and as the only source
    for Tebu (no language card printed a Tebu answer).

POOL. Per district, every usable respondent counts once, unweighted; "No answer", "Don't know"
and missing left out; shares x the district's Libyans (religiondots' ly_lookup.csv `pop`, BSC
2020, 6,872,674), largest remainder.

TEBU. The language cards had no Tebu (Tedaga) answer; the surveys' Tebu speakers answered
Arabic or Other. Arab Barometer VI-3 + VII ethnicity is pooled per district for the share who
are Toubou (13 answers) and, read as Tedaga under the ethnicity ruling, moved off that
district's Arabic. Tuareg: the language cards had "Tahaggart Tamahaq" (WVS 6, 3) and Berber; the
Tuareg ethnic answers (40) are not added on top, they check that Ghat and Ubari carry Berber.

CHECKS: WVS percent x N integral and columns summing to N (ir_wvs.counts); Arab Barometer
Libya rows per wave; units = religiondots' 22 districts; population = 6,872,674; pooled Berber
share per district against Arab Barometer VI-3 + VII's Amazigh + Tuareg share (Pearson).
"""
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sources"))
from rdlink import RD, RD_GEO  # noqa: E402
from ir_wvs import counts as wvs_counts  # noqa: E402

RAW = HERE / "data" / "raw" / "ly"
OUT = HERE / "data" / "normalized" / "ly.csv"
ARB = RD / "data" / "raw" / "arabbarometer"
LOOKUP = RD_GEO / "ly" / "ly_lookup.csv"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
BASE = "https://www.worldvaluessurvey.org/"
SOURCE_ID = "wvs_w6_w7_arabbarometer_iii"
TOTAL = 6_872_674

# file -> (wave id, sample id, question MAIDX, cross 1). Libya is AMIDS 434.
PAGES = {
    "w7_q272_region.html": ("1562", "3776", "C_Q272", "2437884"),
    "w7_intlang_region.html": ("1562", "3776", "A_S", "2437884"),
    "w6_v247_region.html": ("1", "2228", "007_003", "43783"),
    "w6_v254_region.html": ("1", "2228", "011_022", "43783"),
    "w6_v257_region.html": ("1", "2228", "010_008", "43783"),
}

# every district spelling the surveys print -> religiondots unit (ly_lookup.csv)
GOV = {
    # WVS wave 7 (ISO 3166-2)
    "LY-BU Al Butnan": "LY0104", "LY-JA Al Jabal al Akhdar": "LY0106",
    "LY-JG Al Jabal al Gharbi": "LY0216", "LY-JI Al Jafarah": "LY0212", "LY-JU Al Jufrah": "LY0317",
    "LY-KF Al Kufrah": "LY0107", "LY-MJ Al Marj": "LY0102", "LY-MB Al Marqab": "LY0210",
    "LY-WA Al Wahat": "LY0105", "LY-NQ An Nuqat al Khams": "LY0215", "LY-ZA Az Zawiyah": "LY0213",
    "LY-BA Banghazi": "LY0103", "LY-DR Darnah": "LY0101", "LY-GT Ghat": "LY0321",
    "LY-MI Misratah": "LY0214", "LY-MQ Murzuq": "LY0322", "LY-NL Nalut": "LY0209",
    "LY-SB Sabha": "LY0319", "LY-SR Surt": "LY0208", "LY-TB Tarabulus": "LY0211",
    "LY-WD Wadi Al Hayat": "LY0320", "LY-WS Wadi ash Shati": "LY0318",
    # WVS wave 6
    "LY: Al Butnan": "LY0104", "LY: Darnah": "LY0101", "LY: Al Jabal al Gharbi": "LY0216",
    "LY: Al-Marj": "LY0102", "LY: Banghazi": "LY0103", "LY: Al Wahat": "LY0105",
    "LY: Al Kufrah": "LY0107", "LY: Surt": "LY0208", "LY: Al Jufrah": "LY0317",
    "LY: Misratah": "LY0214", "LY: Al Marqab": "LY0210", "LY: Tarabulus/Tripoli": "LY0211",
    "LY: Al Jafarah": "LY0212", "LY:  Az Zawiyah": "LY0213", "LY: An Nuqat al Khams": "LY0215",
    "LY: Al Jabal al Akhdar": "LY0106", "LY: Nalut": "LY0209", "LY: Sabha": "LY0319",
    "LY: Wadi ash Shati": "LY0318", "LY: Murzuk": "LY0322", "LY: Wadi Al Hayat": "LY0320",
    "LY: Ghat": "LY0321",
    # Arab Barometer III ("Bahariya" is the only eastern name left once Butnan is Tobruk: Al Wahat,
    # Ajdabiya; its 39 interviews match Ajdabiya's size against Marj's 40)
    "Butnan": "LY0104", "Darnah": "LY0101", "Jabal Akhdar": "LY0106", "Benghazi": "LY0103",
    "Marj": "LY0102", "Bahariya": "LY0105", "Kufra": "LY0107", "Murqub": "LY0210",
    "Misurata": "LY0214", "Tripoli": "LY0211", "Ajafarh": "LY0212", "Zawiya": "LY0213",
    "Nuqat al Khams": "LY0215", "Jabal al Gharbi": "LY0216", "Nalut": "LY0209", "Ghat": "LY0321",
    "Sabha": "LY0319", "Murzuk": "LY0322", "Wadi al Hayaa": "LY0320", "Wadi al Shatii": "LY0318",
    "Jufra": "LY0317", "Sirte": "LY0208",
    # Arab Barometer VI-3 and VII (the ethnicity check only)
    "Al-Wahat": "LY0105", "Al-Mergheb": "LY0210", "Zwara": "LY0215", "Jafara": "LY0212",
    "Misrata": "LY0214", "Zawia": "LY0213", "Sirt": "LY0208", "Al-Gabal al-Akhdar": "LY0106",
    "Al-Jofra": "LY0317", "Ejdabia": "LY0105", "Al-Marj": "LY0102",
    "Al-Gabal al-Gharbi": "LY0216", "Derna": "LY0101", "Tobruk": "LY0104", "Al-Kufra": "LY0107",
    "Wadi al-Haya": "LY0320", "Wadi Shati": "LY0318", "Murzuk": "LY0322", "Azzawya": "LY0213",
    "Sebha": "LY0319", "Aljfara": "LY0212", "Wadi Ashshati": "LY0318",
    "Al Jabal Al Gharbi": "LY0216", "Al Jabal Al Akhdar": "LY0106", "Aljufra": "LY0317",
    "Almargeb": "LY0210", "Almarj": "LY0102", "Ubari": "LY0320", "Alkufra": "LY0107",
    "Murzuq": "LY0322",
}
# GHAT_OTHER: wave 7's card had no Tamahaq answer; Ghat's 4 "Other" of 10 are read as Tamahaq
# (wave 6's card had it, and Ghat answered it 31%).

# Nafusi: Ethnologue (27th ed., 2024) 300,000 speakers in 2020, via Wikipedia's Nafusi page. The
# surveys reach Nalut, Zuwara and Tripoli; the shortfall goes to Jabal al Gharbi (Yafran, Kikla,
# al-Qalaa), whose three surveys sampled Gharyan and Zintan, capped at GHARBI_CAP of it.
NAFUSI_TOTAL = 300_000
GHARBI_CAP = 0.35
# Tuareg (Tamahaq) and Tebu (Tedaga): Wikipedia's infoboxes, Tuareg in Libya 100,000-250,000
# ("1.5% of its total population"), Toubou 50,000-85,000 (Shoup). The low ends, since this base
# counts Libyan citizens only, placed by district on the places Wikipedia and the surveys name
# (Ghat; Ubari = Wadi al Hayat; Murzuq and Qatrun; Kufra; Sabha). The split is this build's call.
SOUTH = {
    "Tamahaq": {"LY0321": 20_000, "LY0320": 50_000, "LY0319": 15_000, "LY0322": 10_000,
                "LY0318": 5_000},
    "Tedaga": {"LY0322": 20_000, "LY0107": 15_000, "LY0319": 8_000, "LY0320": 7_000},
}

LABEL = {
    "Arabic": "Arabic", "Berber; Amazigh;Tamaziɣt": "Berber", "Amazigh": "Berber",
    "Tahaggart Tamahaq": "Tamahaq", "Other": "Other", "English": "English",
    "Toubou": "Tedaga",
}
NOT_DRAWN = {"No answer", "Don´t know", "Don't know", "Missing", "Missing; Not available",
             "Refused to answer", "Don’t know"}
AB_N = {"ABIII": 1247, "ABVI3": 1002, "ABVII": 2505}


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings()
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (wave, said, maidx, x1) in PAGES.items():
        s = requests.Session()
        s.verify = False
        s.headers["User-Agent"] = UA
        s.get(BASE + "WVSOnline.jsp", timeout=60)
        s.get(BASE + "AJOnline.jsp?WAVE=&COUNTRY=", timeout=60)
        form = {"ulthost": "WVS", "CMSID": "", "WAVE": wave, "MAIDX": "", "SAIDS": said,
                "AMIDS": "434", "SATITULOS": "Libya", "COUNTRY": "", "CRUCEX": ""}
        s.post(BASE + "AJOnlineCountries.jsp", data=form, timeout=60)
        s.post(BASE + "AJOnlineIndex.jsp", data=form, timeout=60)
        form["MAIDX"] = maidx
        s.post(BASE + "AJOnlineQtn.jsp", data=form, timeout=120)
        form2 = {"ulthost": "WVS", "CMSID": "", "WAVE": wave, "SAIDS": said, "SATITULOS": "Libya",
                 "AMIDS": "434", "MAIDX": maidx, "MACRUCE1": x1, "MACRUCE2": "",
                 "CRUCES_ROTARXY": "", "CRUCE_TYPE": "TAB", "AJArchive": "WVS Data Archive"}
        r = s.post(BASE + "AJOnlineQtn.jsp", data=form2, timeout=120)
        r.raise_for_status()
        if "JDSTableCellHeader" not in r.text:
            raise SystemExit(f"{name}: no crosstab in the response")
        tmp = (RAW / name).with_suffix(".part")
        tmp.write_text(r.text, encoding="utf-8")
        tmp.replace(RAW / name)
        print(f"  {name}: {len(r.text):,} chars")


def wvs(name):
    """{region: {answer: n}}. Wave 7's pages are unweighted counts (checked integral by
    ir_wvs.counts). Wave 6's Libya pages print WEIGHTED column percentages (0.5% of Derna's 60 is
    0.3 of a person), so there n = percent x N, fractional, and the column still sums to N."""
    if name.startswith("w7"):
        out = {}
        for (filt, col), v in wvs_counts(RAW / name).items():
            if filt is not None or col == "TOTAL":
                continue
            out[col] = v
        return out
    from ir_wvs import tables
    out = {}
    for filt, cols, data, ns in tables(RAW / name):
        if filt is not None:
            continue
        for j, col in enumerate(cols):
            if col == "TOTAL":
                continue
            got = {}
            for lab, cells in data.items():
                c = cells[j]
                if c in ("-", ""):
                    continue
                x = float(c.rstrip("%")) * ns[j] / 100.0
                if x > 0:
                    got[lab] = x
            if abs(sum(got.values()) - ns[j]) > 0.02 * ns[j] + 0.5:
                raise SystemExit(f"{name} {col}: {sum(got.values()):.1f} against N {ns[j]}")
            out[col] = got
    return out


def arab_barometer():
    import pyreadstat
    spec = {"ABIII": ("ABIII_English.sav", "q1019_1"),
            "ABVI3": ("Arab_Barometer_Wave_6_Part_3_ENG_RELEASE.sav", "Q1012B"),
            "ABVII": ("AB7_ENG_Release_Version6.sav", "Q1012B")}
    out = {}
    for wave, (fname, q) in spec.items():
        df, _ = pyreadstat.read_sav(str(ARB / fname), apply_value_formats=True)
        ccol = next(c for c in df.columns if c.lower() == "country")
        gcol = next(c for c in df.columns if c.lower() == "q1")
        sub = df[df[ccol].astype(str).str.contains("Libya", case=False)]
        if len(sub) != AB_N[wave]:
            raise SystemExit(f"Arab Barometer {wave}: {len(sub)} Libyan rows, expected {AB_N[wave]}")
        got = {}
        for g, a in zip(sub[gcol].astype(str), sub[q].astype(str)):
            got.setdefault(g, {}).setdefault(a, 0)
            got[g][a] += 1
        out[wave] = got
    return out


def pearson(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    return (sum((x - ma) * (y - mb) for x, y in zip(a, b))
            / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5)


def main():
    if "--fetch" in sys.argv:
        fetch()
    for name in PAGES:
        if not (RAW / name).exists():
            raise SystemExit(f"missing {RAW / name}; run with --fetch")
    if "--dump" in sys.argv:
        for name in PAGES:
            print(name, wvs(name))
        for w, v in arab_barometer().items():
            print(w, v)
        return

    pop, names = {}, {}
    with open(LOOKUP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pop[r["unit"]] = int(r["pop"])
            names[r["unit"]] = r["name"]
    if len(pop) != 22 or sum(pop.values()) != TOTAL:
        raise SystemExit(f"ly_lookup.csv: {len(pop)} units, {sum(pop.values()):,} people")

    pooled = {u: {} for u in pop}
    nsrc = {u: {} for u in pop}

    def add(src, region, answer, n):
        if region not in GOV:
            raise SystemExit(f"{src}: district {region!r} not in GOV")
        u = GOV[region]
        if answer in NOT_DRAWN or not n:
            return
        if src == "WVS7" and u == "LY0321" and answer == "Other":
            answer = "Tahaggart Tamahaq"      # see GHAT_OTHER
        if answer not in LABEL:
            raise SystemExit(f"{src}: answer {answer!r} has no LABEL")
        lab = LABEL[answer]
        pooled[u][lab] = pooled[u].get(lab, 0) + n
        nsrc[u][src] = nsrc[u].get(src, 0) + n

    for region, v in wvs("w7_q272_region.html").items():
        for a, n in v.items():
            add("WVS7", region, a, n)
    for region, v in wvs("w6_v247_region.html").items():
        for a, n in v.items():
            add("WVS6", region, a, n)
    ab = arab_barometer()
    for region, v in ab["ABIII"].items():
        for a, n in v.items():
            add("ABIII", region, a, n)
    # interview language: every WVS interview in Libya was in Arabic (the Sudan trap)
    for name in ("w7_intlang_region.html", "w6_v257_region.html"):
        langs = {a for v in wvs(name).values() for a in v}
        print(f"  {name}: interview languages {sorted(langs)}")

    missing = [names[u] for u in pop if not pooled[u]]
    if missing:
        raise SystemExit(f"districts with no answers: {missing}")

    # survey shares x population
    est = {}
    for u in pop:
        tot = sum(pooled[u].values())
        est[u] = {lab: n / tot * pop[u] for lab, n in pooled[u].items()}
    berber_pool = sum(v.get("Berber", 0) for v in est.values())
    print(f"  pooled Berber {berber_pool:,.0f} against Ethnologue's {NAFUSI_TOTAL:,} Nafusi")

    # Nafusi shortfall into Jabal al Gharbi (the Nafusa towns no survey reached)
    gharbi = "LY0216"
    top = max(0.0, NAFUSI_TOTAL - berber_pool)
    top = min(top, GHARBI_CAP * pop[gharbi])
    rest = pop[gharbi] - top
    tot = sum(v for k, v in est[gharbi].items())
    est[gharbi] = {k: v / tot * rest for k, v in est[gharbi].items()}
    est[gharbi]["Berber"] = est[gharbi].get("Berber", 0) + top
    print(f"  Jabal al Gharbi: {top:,.0f} Nafusi added ({top / pop[gharbi]:.1%})")

    # the south: cited Tuareg and Tebu totals, carved out; the rest at the pool's other shares
    for u in pop:
        carve = {lab: SOUTH[lab].get(u, 0) for lab in SOUTH}
        if not any(carve.values()):
            continue
        others = {k: v for k, v in est[u].items() if k not in SOUTH}
        rest = pop[u] - sum(carve.values())
        if rest <= 0:
            raise SystemExit(f"{names[u]}: southern carve-outs exceed the population")
        tot = sum(others.values())
        est[u] = {k: v / tot * rest for k, v in others.items()}
        est[u].update({k: v for k, v in carve.items() if v})

    # check: pooled Berber + Tuareg share against Arab Barometer VI-3 + VII ethnicity
    eth = {}
    for wave in ("ABVI3", "ABVII"):
        for region, v in ab[wave].items():
            if region == "Don't know":
                continue
            if region not in GOV:
                raise SystemExit(f"{wave}: district {region!r} not in GOV")
            e = eth.setdefault(GOV[region], [0, 0])
            for a, n in v.items():
                if a in NOT_DRAWN:
                    continue
                e[1] += n
                if a in ("Amazigh", "Amazigh/Berber", "Tourag"):
                    e[0] += n
    us = sorted(u for u in pop if u in eth)
    a = [sum(pooled[u].get(k, 0) for k in ("Berber", "Tamahaq")) / sum(pooled[u].values())
         for u in us]
    b = [eth[u][0] / eth[u][1] for u in us]
    print(f"  pooled Berber+Tamahaq share vs Arab Barometer VI-3+VII Amazigh+Tuareg share, "
          f"{len(us)} districts: Pearson {pearson(a, b):+.3f}")
    for u, x, y in sorted(zip(us, a, b), key=lambda t: -abs(t[1] - t[2]))[:5]:
        print(f"      {names[u]:<20} pool {x:6.1%}  ethnicity {y:6.1%}")

    out, natl = [], {}
    for u in sorted(pop):
        raw = est[u]
        cnt = {lab: int(x) for lab, x in raw.items()}
        for lab in sorted(raw, key=lambda x: raw[x] - cnt[x], reverse=True)[:pop[u] - sum(cnt.values())]:
            cnt[lab] += 1
        assert sum(cnt.values()) == pop[u], names[u]
        srcs = ", ".join(f"{s} {n:g}" for s, n in nsrc[u].items())
        for lab in sorted(cnt, key=cnt.get, reverse=True):
            if not cnt[lab]:
                continue
            natl[lab] = natl.get(lab, 0) + cnt[lab]
            if lab in SOUTH:
                note = f"cited national estimate, placed by district (sources/ly.md): {lab}"
            elif u == gharbi and lab == "Berber":
                note = "pooled answers plus Ethnologue's Nafusi shortfall (sources/ly.md)"
            else:
                note = f"pooled answers ({srcs}); BSC 2020 Libyans {pop[u]}"
            out.append(dict(geo_id=u, geo_level="district", geo_name=names[u],
                            source_category=lab, count=cnt[lab], tier="modelled",
                            source_id=SOURCE_ID, year=2020, note=note))
        print(f"  {names[u]:<20} n {sum(pooled[u].values()):>6.1f}  " + ", ".join(
            f"{lab} {n / pop[u]:.1%}" for lab, n in sorted(cnt.items(), key=lambda kv: -kv[1])
            if n) + f"   [{srcs}]")

    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["geo_id", "geo_level", "geo_name", "source_category",
                                          "count", "tier", "source_id", "year", "note"])
        w.writeheader()
        w.writerows(out)
    tmp.replace(OUT)
    print(f"  wrote {OUT.relative_to(HERE)}: {len(out)} rows, {sum(natl.values()):,} people")
    for lab, n in sorted(natl.items(), key=lambda kv: -kv[1]):
        print(f"      {lab:<12} {n:>10,}  {n / TOTAL * 100:5.2f}%")


if __name__ == "__main__":
    main()
