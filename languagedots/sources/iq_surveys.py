"""Iraq: first / home language by governorate from five survey rounds, pooled, as shares applied
to the 2024 census governorate populations -> data/normalized/iq_surveys.csv.

SUPERSEDED 2026-10-09 by sources/iq_mics6.py (MICS6 2018 microdata), which writes iq.csv. Kept as
the comparison: it put Kirkuk at 71% Arabic and 11% Turkmen against MICS6's 37% and 31%.

    python sources/iq_surveys.py [--fetch]

NO CENSUS ASKS. The 2024 census (the first since 1997) left out language, ethnicity and ancestry
on purpose (the disputed territories); 1997 asked language but skipped the three Kurdistan
governorates and its microdata is IPUMS-gated; 1957 is by liwa. MICS6 2018 asks mother tongue
with Sorani and Badini apart but needs a UNICEF registration. So the open sources are surveys:

  World Values Survey (online analysis tool, no registration; unweighted counts):
    wave 7, 2018   Q272 language at home x N_REGION_ISO     9 governorates, 1,200
    wave 6, 2013   V247 language at home x V256              9 governorates, 1,200
                   (Arabic / Other only; Other split by V254 ethnic group, see below)
    wave 4, 2004   V219 language at home x V243 region     16 governorates, 2,325
                   (Kirkuk left out, see below)
    wave 5, 2006   V222: fetched, NOT USED. Its codes are scrambled for Iraq: 20-25% "Kurdish" in
                   Basra and Muthanna, 57% Arabic in Sulaymaniyah, 127 of 529 Baghdad no answer.
  Arab Barometer (religiondots' .sav files, read-only; unweighted counts):
    wave II, 2011  q10191 first language                    10 governorates, 1,232
    wave III, 2013 q1019_1 first language                    9 governorates, 1,215
    waves VI-3 (2020-21) and VII (2022) ask ethnicity only (Q1012B); used for Duhok alone, the
    one governorate no language question reached (only WVS wave 5, unusable). Kurd and Yazidi
    are read as Kurdish, Assyrian as Assyrian Neo-Aramaic (AGENT_BRIEF section 2, ethnicity).

POOL. Per governorate, every usable respondent counts once, unweighted; "No answer" and missing
left out; shares x the 2024 census population (religiondots' iq_lookup.csv, 46,118,793),
largest remainder so each governorate sums exactly.

  * WVS wave 6's card held only Arabic and Other. In Erbil and Sulaymaniyah every Other was
    interviewed in Kurdish (V257, asserted). Each governorate's Other is shared across that
    governorate's non-Arab ethnic answers (V254: Kurdish, Turk) in proportion; Nineveh's 59
    Other are exactly its 59 ethnic Turks (Tal Afar).
  * WVS wave 4's Kirkuk column is left out: 59 of 114 answered Other, more than its 34 ethnic
    Turks (V242 x region), so the Other mixes Turkmen with Arabs and Kurds and cannot be read.
    Kirkuk has three other rounds.

INTERVIEW LANGUAGE (the Sudan trap). WVS wave 7's S_INTLANGUAGE prints Arabic for all 1,200,
including the 167 in Erbil and Sulaymaniyah who all answered Kurdish at home, so that field is
not informative; wave 6's V257 has Kurdish interviews exactly where Kurdish speakers are. Arab
Barometer files carry no interview-language field for Iraq. Every survey here reached the
Kurdistan Region and found it 96-100% Kurdish, unlike Sudan's surveys.

CHECKS: WVS percent x N integral and columns summing to N (ir_wvs.counts); Arab Barometer Iraq
row counts per wave; wave 6 Other = Kurdish-language interviews in Erbil and Sulaymaniyah; units
= religiondots' 18 governorates; population = 46,118,793; national Kurdish share inside the
15-20% "Kurdish" of the usual ethnic estimates (CIA World Factbook); Spearman of the pooled
Kurdish share against Arab Barometer VI-3 + VII's Kurd share (a different question, other years)
over the 17 governorates both reach.
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
from ir_wvs import counts as wvs_counts  # noqa: E402  (the same online-tool page parser)

RAW = HERE / "data" / "raw" / "iq"
OUT = HERE / "data" / "normalized" / "iq_surveys.csv"
ARB = RD / "data" / "raw" / "arabbarometer"
LOOKUP = RD_GEO / "iq" / "iq_lookup.csv"          # religiondots, read-only: 2024 census
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
BASE = "https://www.worldvaluessurvey.org/"
SOURCE_ID = "wvs_w4_w6_w7_arabbarometer_ii_iii"
CENSUS_TOTAL = 46_118_793

# file -> (wave id, sample id, question MAIDX, cross 1, cross 2). Iraq is AMIDS 368.
PAGES = {
    "w7_q272_region.html": ("1562", "3323", "C_Q272", "2437884", ""),
    "w7_intlang_region.html": ("1562", "3323", "A_S", "2437884", ""),
    "w6_v247_region.html": ("1", "2229", "007_003", "43783", ""),
    "w6_v254_region.html": ("1", "2229", "011_022", "43783", ""),
    "w6_v257_region.html": ("1", "2229", "010_008", "43783", ""),
    "w5_v222_region.html": ("2", "462", "007_003", "1512", ""),
    "w4_v219_region.html": ("12", "481", "G_016", "49506", ""),
    "w4_x051_region.html": ("12", "481", "X_051", "49506", ""),
}

# every governorate spelling the surveys print -> religiondots unit (iq_lookup.csv)
GOV = {
    # WVS wave 7 (ISO 3166-2)
    "IQ-AN Al Anbar": "IQG01", "IQ-AR Arbil": "IQG11", "IQ-BA Al Basrah": "IQG02",
    "IQ-BB Babil": "IQG07", "IQ-BG Baghdad": "IQG08", "IQ-DQ Dhi Qar": "IQG17",
    "IQ-KI Kirkuk": "IQG13", "IQ-NI Ninawá": "IQG15", "IQ-SU As Sulaymaniyah": "IQG06",
    # WVS waves 4-6
    "IQ: Al Anbar": "IQG01", "IQ: Al Basrah": "IQG02", "IQ: Al Muthanná": "IQG03",
    "IQ: An Najaf": "IQG04", "IQ: Al Qadisiyah": "IQG05", "IQ: As Sulaymaniyah (Slêmanî)": "IQG06",
    "IQ: Babil": "IQG07", "IQ: Baghdad": "IQG08", "IQ: Dahuk (Dihok)": "IQG09",
    "IQ: Diyalá": "IQG10", "IQ: Arbil (Hewlêr)": "IQG11", "IQ: Karbala": "IQG12",
    "IQ: Kirkuk": "IQG13", "IQ: Maysan": "IQG14", "IQ: Ninawá": "IQG15",
    "IQ: Salah ad Din": "IQG16", "IQ: Dhi Qar": "IQG17", "IQ: Wasit": "IQG18",
    # Arab Barometer II
    "3001. Salah al-Din": "IQG16", "3002. Basra": "IQG02", "3003. Babylon": "IQG07",
    "3004. Nineveh": "IQG15", "3005. Dhi Qar": "IQG17", "3006. Diyala": "IQG10",
    "3007. Sulaymaniyah": "IQG06", "3008. Najaf": "IQG04", "3009. Irbil": "IQG11",
    "3010. Baghdad": "IQG08",
    # Arab Barometer III
    "Babylon": "IQG07", "Baghdad": "IQG08", "Basra": "IQG02", "Diyala": "IQG10",
    "Erbil": "IQG11", "Kirkuk": "IQG13", "Nineveh": "IQG15", "Sulaymaniyah": "IQG06",
    "al-Qādisiyyah": "IQG05",
    # Arab Barometer VI-3 / VII (only Duhok is used for language; the rest feed the check)
    "Anbar": "IQG01", "Babel": "IQG07", "Diwaniyah": "IQG05", "Qadisiyah": "IQG05",
    "Dohuk": "IQG09", "Karbala": "IQG12", "Missan": "IQG14", "Maysan": "IQG14",
    "Muthanna": "IQG03", "Muthana": "IQG03", "Najaf": "IQG04", "Ninewa": "IQG15",
    "Salahaddin": "IQG16", "Thi-Qar": "IQG17", "Dhi Qar": "IQG17", "Wasit": "IQG18",
}

# answer as printed -> the label written to iq.csv (taxonomy/iq2018.py maps these)
LABEL = {
    "Arabic": "Arabic", "Kurdish; Yezidi": "Kurdish", "Kurdish": "Kurdish",
    "Turkmen": "Turkmen", "Assyrian Neo-Aramaic": "Assyrian Neo-Aramaic",
    "Assyrian": "Assyrian Neo-Aramaic", "Shabaki": "Shabaki", "Other": "Other",
    # Arab Barometer II prints the code inside the label
    "1. Arabic": "Arabic", "11. Kurdish": "Kurdish", "10. Turkmen": "Turkmen",
    "12. Shabaki": "Shabaki",
    # ethnic answers (wave 6 Other split; Arab Barometer VI-3 / VII for Duhok)
    "IQ: Kurdish": "Kurdish", "IQ: Turk": "Turkmen", "Kurd": "Kurdish",
    "Yazidi": "Kurdish", "Arab": "Arabic",
}
NOT_DRAWN = {"No answer", "Missing; Not available", "Don't know", "0. missing", "Missing",
             "Refuse", "Refused to answer", "Don’t know"}
AB_N = {"ABII": 1234, "ABIII": 1215, "ABVI3": 1016, "ABVII": 2460}
KURDISH_ESTIMATE = (15.0, 20.0)   # "Kurdish" share of Iraqis in the usual ethnic estimates


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings()
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (wave, said, maidx, x1, x2) in PAGES.items():
        s = requests.Session()
        s.verify = False                     # the site omits an intermediate certificate
        s.headers["User-Agent"] = UA
        s.get(BASE + "WVSOnline.jsp", timeout=60)
        s.get(BASE + "AJOnline.jsp?WAVE=&COUNTRY=", timeout=60)
        form = {"ulthost": "WVS", "CMSID": "", "WAVE": wave, "MAIDX": "", "SAIDS": said,
                "AMIDS": "368", "SATITULOS": "Iraq", "COUNTRY": "", "CRUCEX": ""}
        s.post(BASE + "AJOnlineCountries.jsp", data=form, timeout=60)
        s.post(BASE + "AJOnlineIndex.jsp", data=form, timeout=60)
        form["MAIDX"] = maidx
        s.post(BASE + "AJOnlineQtn.jsp", data=form, timeout=120)
        form2 = {"ulthost": "WVS", "CMSID": "", "WAVE": wave, "SAIDS": said, "SATITULOS": "Iraq",
                 "AMIDS": "368", "MAIDX": maidx, "MACRUCE1": x1, "MACRUCE2": x2,
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
    """{unit: {printed answer: n}} for one WVS page (two-way tables only)."""
    out = {}
    for (filt, col), v in wvs_counts(RAW / name).items():
        if filt is not None or col == "TOTAL":
            continue
        if col not in GOV:
            raise SystemExit(f"{name}: governorate {col!r} not in GOV")
        out[GOV[col]] = v
    return out


def arab_barometer():
    """{wave: {unit: {answer: n}}} from the .sav files, Iraq rows, unweighted."""
    import pyreadstat
    spec = {"ABII": ("ABII_English.sav", "q10191"),
            "ABIII": ("ABIII_English.sav", "q1019_1"),
            "ABVI3": ("Arab_Barometer_Wave_6_Part_3_ENG_RELEASE.sav", "Q1012B"),
            "ABVII": ("AB7_ENG_Release_Version6.sav", "Q1012B")}
    out = {}
    for wave, (fname, q) in spec.items():
        df, _ = pyreadstat.read_sav(str(ARB / fname), apply_value_formats=True)
        ccol = next(c for c in df.columns if c.lower() == "country")
        gcol = next(c for c in df.columns if c.lower() == "q1")
        iq = df[df[ccol].astype(str).str.contains("Iraq", case=False)]
        if len(iq) != AB_N[wave]:
            raise SystemExit(f"Arab Barometer {wave}: {len(iq)} Iraqi rows, expected {AB_N[wave]}")
        got = {}
        for g, a in zip(iq[gcol].astype(str), iq[q].astype(str)):
            if g not in GOV:
                raise SystemExit(f"Arab Barometer {wave}: governorate {g!r} not in GOV")
            got.setdefault(GOV[g], {}).setdefault(a, 0)
            got[GOV[g]][a] += 1
        out[wave] = got
    return out


def spearman(a, b):
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    return (sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
            / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5)


def main():
    if "--fetch" in sys.argv:
        fetch()
    for name in PAGES:
        if not (RAW / name).exists():
            raise SystemExit(f"missing {RAW / name}; run with --fetch")

    pop, names = {}, {}
    with open(LOOKUP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pop[r["unit"]] = int(r["pop"])
            names[r["unit"]] = r["name"]
    if len(pop) != 18 or sum(pop.values()) != CENSUS_TOTAL:
        raise SystemExit(f"iq_lookup.csv: {len(pop)} units, {sum(pop.values()):,} people")

    pooled = {u: {} for u in pop}
    nsrc = {u: {} for u in pop}

    def add(unit, src, answer, n):
        if answer in NOT_DRAWN or not n:
            return
        if answer not in LABEL:
            raise SystemExit(f"{src}: answer {answer!r} has no LABEL")
        lab = LABEL[answer]
        pooled[unit][lab] = pooled[unit].get(lab, 0) + n
        nsrc[unit][src] = nsrc[unit].get(src, 0) + n

    # WVS wave 7
    for u, v in wvs("w7_q272_region.html").items():
        for a, n in v.items():
            add(u, "WVS7", a, n)
    intl = wvs("w7_intlang_region.html")
    print("  WVS 7 interview language: " + ", ".join(
        f"{names[u]} {'/'.join(f'{k} {n}' for k, n in v.items())}" for u, v in intl.items()))

    # WVS wave 6: Other shared over the governorate's non-Arab ethnic answers
    w6, w6eth, w6int = (wvs("w6_v247_region.html"), wvs("w6_v254_region.html"),
                        wvs("w6_v257_region.html"))
    for u in ("IQG11", "IQG06"):        # Erbil, Sulaymaniyah
        if w6[u].get("Other", 0) != w6int[u].get("Kurdish; Yezidi", 0):
            raise SystemExit(f"WVS 6 {names[u]}: Other {w6[u].get('Other')} != Kurdish "
                             f"interviews {w6int[u].get('Kurdish; Yezidi')}")
    print("  WVS 6: Other = Kurdish-language interviews in Erbil and Sulaymaniyah")
    for u, v in w6.items():
        for a, n in v.items():
            if a != "Other":
                add(u, "WVS6", a, n)
                continue
            non_arab = {k: m for k, m in w6eth[u].items() if k not in ("IQ: Arab",)}
            tot = sum(non_arab.values())
            if not tot:
                raise SystemExit(f"WVS 6 {names[u]}: Other with no non-Arab ethnic answer")
            for k, m in non_arab.items():
                add(u, "WVS6", k, n * m / tot)
            print(f"    {names[u]}: {n} Other shared over "
                  + ", ".join(f"{k} {m}" for k, m in non_arab.items()))

    # WVS wave 4, Kirkuk left out
    w4, w4eth = wvs("w4_v219_region.html"), wvs("w4_x051_region.html")
    k = "IQG13"
    print(f"  WVS 4 Kirkuk left out: language {w4[k]}, ethnic group {w4eth[k]}")
    for u, v in w4.items():
        if u == k:
            continue
        for a, n in v.items():
            add(u, "WVS4", a, n)

    # Arab Barometer II, III (first language); VI-3, VII ethnicity for Duhok only
    ab = arab_barometer()
    for wave in ("ABII", "ABIII"):
        for u, v in ab[wave].items():
            for a, n in v.items():
                add(u, wave, a, n)
    duhok = "IQG09"
    if nsrc[duhok]:
        raise SystemExit("Duhok has a language source now; revisit the ethnicity fill")
    for wave in ("ABVI3", "ABVII"):
        for a, n in ab[wave].get(duhok, {}).items():
            add(duhok, wave + "-ethnic", a, n)

    missing = [names[u] for u in pop if not pooled[u]]
    if missing:
        raise SystemExit(f"governorates with no answers: {missing}")

    # shares x census population, largest remainder
    out, natl = [], {}
    for u in sorted(pop):
        v = pooled[u]
        tot = sum(v.values())
        raw = {lab: n / tot * pop[u] for lab, n in v.items()}
        cnt = {lab: int(x) for lab, x in raw.items()}
        for lab in sorted(raw, key=lambda x: raw[x] - cnt[x], reverse=True)[:pop[u] - sum(cnt.values())]:
            cnt[lab] += 1
        assert sum(cnt.values()) == pop[u]
        srcs = ", ".join(f"{s} {n:g}" for s, n in nsrc[u].items())
        for lab in sorted(cnt, key=cnt.get, reverse=True):
            if not cnt[lab]:
                continue
            natl[lab] = natl.get(lab, 0) + cnt[lab]
            out.append(dict(geo_id=u, geo_level="governorate", geo_name=names[u],
                            source_category=lab, count=cnt[lab], tier="modelled",
                            source_id=SOURCE_ID, year=2018,
                            note=f"{v[lab]:.1f} of {tot:.1f} pooled answers ({srcs}); "
                                 f"2024 census population {pop[u]}"))
        print(f"  {names[u]:<14} n {tot:>6.1f}  " + ", ".join(
            f"{lab} {n / tot * 100:.1f}" for lab, n in sorted(v.items(), key=lambda kv: -kv[1]))
            + f"   [{srcs}]")

    kshare = natl.get("Kurdish", 0) / CENSUS_TOTAL * 100
    print(f"  national Kurdish {kshare:.1f}% (usual ethnic estimates "
          f"{KURDISH_ESTIMATE[0]:.0f}-{KURDISH_ESTIMATE[1]:.0f}%)")
    if not (KURDISH_ESTIMATE[0] - 3 <= kshare <= KURDISH_ESTIMATE[1] + 3):
        raise SystemExit("national Kurdish share far from the published estimates")

    # independent check: Kurd share in Arab Barometer VI-3 + VII ethnicity
    eth = {}
    for wave in ("ABVI3", "ABVII"):
        for u, v in ab[wave].items():
            for a, n in v.items():
                if a in NOT_DRAWN:
                    continue
                e = eth.setdefault(u, [0, 0])
                e[1] += n
                if a in ("Kurd", "Yazidi"):
                    e[0] += n
    us = [u for u in sorted(pop) if u != duhok]
    a = [pooled[u].get("Kurdish", 0) / sum(pooled[u].values()) for u in us]
    b = [eth[u][0] / eth[u][1] for u in us]
    rho = spearman(a, b)
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    r = (sum((x - ma) * (y - mb) for x, y in zip(a, b))
         / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5)
    # Spearman is held down by the ten southern governorates tied at or near zero on both.
    print(f"  pooled Kurdish share vs Arab Barometer VI-3+VII Kurd share, {len(us)} "
          f"governorates: Pearson {r:+.3f}, Spearman {rho:+.3f}")
    worst = sorted(us, key=lambda u: -abs(a[us.index(u)] - b[us.index(u)]))[:4]
    print("    largest gaps (pooled - ethnicity rounds, points): " + ", ".join(
        f"{names[u]} {(a[us.index(u)] - b[us.index(u)]) * 100:+.1f}" for u in worst))
    if r < 0.95 or rho < 0.6:
        raise SystemExit("pooled Kurdish shares disagree with the ethnicity rounds")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["geo_id", "geo_level", "geo_name", "source_category",
                                          "count", "tier", "source_id", "year", "note"])
        w.writeheader()
        w.writerows(out)
    tmp.replace(OUT)
    print(f"  wrote {OUT.relative_to(HERE)}: {len(out)} rows, {sum(natl.values()):,} people")
    for lab, n in sorted(natl.items(), key=lambda kv: -kv[1]):
        print(f"      {lab:<22} {n:>11,}  {n / CENSUS_TOTAL * 100:5.2f}%")


if __name__ == "__main__":
    main()
