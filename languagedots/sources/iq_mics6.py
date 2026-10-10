"""Iraq: native language by governorate from MICS6 2018 microdata, as shares applied to the 2024
census governorate populations -> data/normalized/iq.csv.

    python sources/iq_mics6.py

SOURCE. Iraq Multiple Indicator Cluster Survey 2018 (MICS6; Central Statistical Organization,
Kurdistan Region Statistics Office, UNICEF), SPSS files from mics.unicef.org (Anita's UNICEF
account, 2026-10-09), unzipped to data/raw/iq/mics6/ (gitignored; the terms allow research use and
no redistribution of the files). 20,521 households in 18 governorates, 131,394 household members.

ITEM. HC1B, "Language of household head" (Arabic / Kurdish / Turkman / Asserian / others), MICS's
standard household mother-tongue item, read as every member's: hl.sav members x hhweight, so each
governorate's shares are of persons. Kurdish is split by HH16, "Native language of the
Respondent" (Arabic / Kurdish Surani / Kurdish Badinani / Turkman / Asserian / others), within the
Kurdish-headed households of each governorate; those whose respondent answered something else
(Baghdad's, mostly Feyli) stay on plain Kurdish.

Why HC1B and not HH16 alone (first build, same day): where the interview was in Arabic, HH16
slides to Arabic. Of households whose head speaks Kurdish, 107 answer HH16 Arabic (Baghdad: 8
Kurdish heads, 0 Kurdish respondents; Diyala 74 against 20); of Turkmen heads, 36. Every Baghdad
interview was in Arabic. WM14 (women 15-49, own native language, wm.sav x wmweight) is printed as
a third reading.

BAGHDAD'S TURKMEN (Anita, 2026-10-09). MICS finds none of 2,153 Baghdad households, but its 180
clusters miss an enclave holding 1% of Baghdad one time in six; Arab Barometer VI-3 + VII's
ethnicity question (religiondots' .sav, through iq_surveys.arab_barometer) finds 2 of 764, and
that share is drawn as Turkmen, taken pro rata from the rest (ETHNIC_FILL).

SHARES. Per governorate, weighted; households with no HH16 (not interviewed, weight 0) drop out.
Shares x the 2024 census population (religiondots' iq_lookup.csv, 46,118,793), largest remainder.

CHECKS. Every hl.sav member matches a household; HH16 vs HC1B per governorate, Kurdish (Sorani +
Badini) and Turkmen and Arabic within 6 points; national Kurdish inside 15-20% +/- 3 (the usual
ethnic estimates, CIA World Factbook); national Turkmen 1-3%.

Superseded: sources/iq_surveys.py (WVS 2004-2018 and Arab Barometer 2011-13 pooled), kept as a
comparison; it now writes data/normalized/iq_surveys.csv.
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
from rdlink import RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "iq" / "mics6"
OUT = HERE / "data" / "normalized" / "iq.csv"
LOOKUP = RD_GEO / "iq" / "iq_lookup.csv"          # religiondots, read-only: 2024 census
SOURCE_ID = "mics6_2018_hc1b"
CENSUS_TOTAL = 46_118_793
N_HH, N_HL = 20_521, 131_394

# MICS HH7 -> religiondots unit
GOV = {"ANBAR": "IQG01", "BASRAH": "IQG02", "MUTHANA": "IQG03", "NAJAF": "IQG04",
       "QADISYAH": "IQG05", "SULAIMANIYA": "IQG06", "BABIL": "IQG07", "BAGHDAD": "IQG08",
       "DUHOK": "IQG09", "DIALA": "IQG10", "ERBIL": "IQG11", "KARBALAH": "IQG12",
       "KIRKUK": "IQG13", "MISAN": "IQG14", "NAINAWA": "IQG15", "SALAHADDIN": "IQG16",
       "THIQAR": "IQG17", "WASIT": "IQG18"}
# HH16 / WM14 answer -> the label written to iq.csv (taxonomy/iq2018.py maps these)
LABEL = {"Arabic": "Arabic", "Kurdish Surani": "Kurdish (Sorani)",
         "Kurdish Badinani": "Kurdish (Badini)", "Turkman": "Turkmen",
         "Asserian": "Assyrian Neo-Aramaic", "others": "Other"}
HEAD = {"ARABIC": "Arabic", "KURDISH": "Kurdish", "TURKMAN": "Turkmen",
        "ASSERIAN": "Assyrian Neo-Aramaic", "OTHERS": "Other"}
KURDISH_ESTIMATE = (15.0, 20.0)
# (unit, label, Arab Barometer Q1012B answers): a group MICS's language items found none of in a
# governorate, drawn from the ethnicity rounds instead (sources/iq.md §0)
ETHNIC_FILL = [("IQG08", "Turkmen", {"Turkmen"})]


def shares(df, col, w, label):
    """{unit: {label: weighted persons}}"""
    out = {}
    d = df[(df[w] > 0) & df[col].notna()]
    for (g, a), s in d.groupby(["HH7", col], observed=True)[w].sum().items():
        if not s:
            continue
        if g not in GOV:
            raise SystemExit(f"governorate {g!r} not in GOV")
        a = str(a)
        if a not in label:
            if a == "NO RESPONSE":
                continue
            raise SystemExit(f"{col}: answer {a!r} has no label")
        u = out.setdefault(GOV[g], {})
        u[label[a]] = u.get(label[a], 0) + s
    return out


def pct(v):
    t = sum(v.values())
    return {k: x / t * 100 for k, x in v.items()}


def main():
    import pyreadstat
    hh, _ = pyreadstat.read_sav(str(RAW / "hh.sav"), apply_value_formats=True,
                                usecols=["HH1", "HH2", "HH7", "HH16", "HC1B", "hhweight"])
    hl, _ = pyreadstat.read_sav(str(RAW / "hl.sav"), usecols=["HH1", "HH2", "HL1", "hhweight"])
    wm, _ = pyreadstat.read_sav(str(RAW / "wm.sav"), apply_value_formats=True,
                                usecols=["HH7", "WM14", "wmweight"])
    if len(hh) != N_HH or len(hl) != N_HL:
        raise SystemExit(f"hh {len(hh)} / hl {len(hl)} rows, expected {N_HH} / {N_HL}")
    persons = hl.merge(hh.drop(columns="hhweight"), on=["HH1", "HH2"], how="left",
                       indicator=True)
    if (persons["_merge"] != "both").any():
        raise SystemExit("hl.sav members with no household in hh.sav")
    print(f"  {len(hh):,} households, {len(hl):,} members, "
          f"{int((hh.hhweight > 0).sum()):,} interviewed")

    resp = shares(persons, "HH16", "hhweight", LABEL)
    head = shares(persons, "HC1B", "hhweight", HEAD)
    women = shares(wm, "WM14", "wmweight", LABEL)
    if set(resp) != set(GOV.values()):
        raise SystemExit("a governorate has no HH16 answers")

    pop, names = {}, {}
    with open(LOOKUP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pop[r["unit"]] = int(r["pop"])
            names[r["unit"]] = r["name"]
    if len(pop) != 18 or sum(pop.values()) != CENSUS_TOTAL:
        raise SystemExit(f"iq_lookup.csv: {len(pop)} units, {sum(pop.values()):,} people")

    print("  per governorate, % of persons: respondent's language (HH16) | head (HC1B) | "
          "women 15-49 (WM14)")
    bad = []
    for u in sorted(pop, key=lambda k: names[k]):
        r, h, w = pct(resp[u]), pct(head.get(u, {})), pct(women.get(u, {}))
        kr = r.get("Kurdish (Sorani)", 0) + r.get("Kurdish (Badini)", 0)
        kw = w.get("Kurdish (Sorani)", 0) + w.get("Kurdish (Badini)", 0)
        for lab, a, b in (("Kurdish", kr, h.get("Kurdish", 0)),
                          ("Turkmen", r.get("Turkmen", 0), h.get("Turkmen", 0)),
                          ("Arabic", r.get("Arabic", 0), h.get("Arabic", 0))):
            if abs(a - b) > 6:
                bad.append(f"{names[u]} {lab} HH16 {a:.1f} vs HC1B {b:.1f}")
        top = sorted(r.items(), key=lambda kv: -kv[1])
        print(f"    {names[u]:<14} " + ", ".join(f"{k} {v:.1f}" for k, v in top if v >= 0.05)
              + f" | Kurdish {h.get('Kurdish', 0):.1f}, Turkmen {h.get('Turkmen', 0):.1f}"
              + f" | Kurdish {kw:.1f}, Turkmen {w.get('Turkmen', 0):.1f}")
    if bad:
        raise SystemExit("HH16 and HC1B disagree: " + "; ".join(bad))

    # Drawn: HC1B, with Kurdish split by HH16 inside the Kurdish-headed households
    drawn = {u: dict(v) for u, v in head.items()}
    kh = persons[(persons["hhweight"] > 0) & (persons["HC1B"].astype(str) == "KURDISH")]
    split = kh.groupby(["HH7", "HH16"], observed=True)["hhweight"].sum()
    print("  Kurdish-headed persons by HH16 (Sorani / Badini / not split, %):")
    for g, u in GOV.items():
        k = drawn[u].pop("Kurdish", 0)
        if not k:
            continue
        s = split.get(g, None)
        tot = s.sum() if s is not None else 0
        so = s.get("Kurdish Surani", 0) / tot if tot else 0
        ba = s.get("Kurdish Badinani", 0) / tot if tot else 0
        for lab, f in (("Kurdish (Sorani)", so), ("Kurdish (Badini)", ba),
                       ("Kurdish", 1 - so - ba)):
            if f > 0:
                drawn[u][lab] = drawn[u].get(lab, 0) + k * f
        print(f"    {names[u]:<14} {so * 100:5.1f} / {ba * 100:5.1f} / {(1 - so - ba) * 100:5.1f}"
              f"   ({k / sum(head[u].values()) * 100:.1f}% of the governorate)")

    # Baghdad's Turkmen from ethnicity (Anita, 2026-10-09): MICS finds none in 2,153 households,
    # but 180 clusters miss an enclave of 1% of Baghdad one time in six; Arab Barometer VI-3 + VII
    # ask ethnicity in Baghdad. Their Turkmen share replaces MICS's zero, taken pro rata from
    # the rest.
    import iq_surveys
    ab = iq_surveys.arab_barometer()
    for u, lab, answers in ETHNIC_FILL:
        n = sum(c for w in ("ABVI3", "ABVII") for a, c in ab[w].get(u, {}).items()
                if a in answers)
        tot = sum(c for w in ("ABVI3", "ABVII") for a, c in ab[w].get(u, {}).items()
                  if a not in iq_surveys.NOT_DRAWN)
        if drawn[u].get(lab, 0):
            raise SystemExit(f"{names[u]}: MICS now has {lab}; revisit ETHNIC_FILL")
        s = n / tot
        rest = sum(drawn[u].values())
        drawn[u] = {k: x * (1 - s) for k, x in drawn[u].items()}
        drawn[u][lab] = rest * s
        print(f"  {names[u]}: {lab} {n} of {tot} Arab Barometer VI-3 + VII ethnic answers "
              f"({s * 100:.2f}%), in place of MICS's zero")

    out, natl = [], {}
    for u in sorted(pop):
        v = drawn[u]
        tot = sum(v.values())
        raw = {lab: n / tot * pop[u] for lab, n in v.items()}
        cnt = {lab: int(x) for lab, x in raw.items()}
        for lab in sorted(raw, key=lambda x: raw[x] - cnt[x], reverse=True)[:pop[u] - sum(cnt.values())]:
            cnt[lab] += 1
        assert sum(cnt.values()) == pop[u]
        n_hh = int(((hh["HH7"] == next(k for k, x in GOV.items() if x == u))
                    & (hh["hhweight"] > 0)).sum())
        for lab in sorted(cnt, key=cnt.get, reverse=True):
            if not cnt[lab]:
                continue
            natl[lab] = natl.get(lab, 0) + cnt[lab]
            out.append(dict(geo_id=u, geo_level="governorate", geo_name=names[u],
                            source_category=lab, count=cnt[lab], tier="modelled",
                            source_id=SOURCE_ID, year=2018,
                            note=f"MICS6 HC1B (Kurdish split by HH16), {v[lab] / tot * 100:.2f}% "
                                 f"of persons in "
                                 f"{n_hh} interviewed households (weighted); 2024 census "
                                 f"population {pop[u]}"))

    k = (natl.get("Kurdish (Sorani)", 0) + natl.get("Kurdish (Badini)", 0)
         + natl.get("Kurdish", 0)) / CENSUS_TOTAL * 100
    t = natl.get("Turkmen", 0) / CENSUS_TOTAL * 100
    print(f"  national Kurdish {k:.1f}% (usual ethnic estimates {KURDISH_ESTIMATE[0]:.0f}-"
          f"{KURDISH_ESTIMATE[1]:.0f}%), Turkmen {t:.2f}%")
    if not (KURDISH_ESTIMATE[0] - 3 <= k <= KURDISH_ESTIMATE[1] + 3):
        raise SystemExit("national Kurdish share far from the published estimates")
    if not 1.0 <= t <= 3.0:
        raise SystemExit("national Turkmen share outside 1-3%")

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
