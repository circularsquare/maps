"""Afghanistan: language of the household head by province (urban and rural apart where both are
sampled well enough), from MICS6 2022-23 microdata, as shares applied to NSIA's 1404 (2025-26)
settled population -> data/normalized/af.csv.

    python sources/af_mics.py

SOURCE. Afghanistan Multiple Indicator Cluster Survey 2022-23 (MICS6; UNICEF with the National
Statistics and Information Authority), SPSS files from mics.unicef.org (Anita's UNICEF account,
2026-10-09), unzipped to data/raw/af/mics_2022/ (gitignored; research use, no redistribution of
the files). 23,338 households sampled in all 34 provinces, 23,213 interviewed, 199,354
household members; 28 clusters per province (34 in Kabul, Herat and Nangarhar, 33 in Kandahar,
25 in Balkh). Fieldwork was under the Taliban administration; every province was reached.

ITEM. HC1B, "Language of household head" (Dari / Pashto / Uzbaki / Turkmani / Nooristani /
Balochi / Pashaie / other), MICS's standard household mother-tongue item, read as every member's:
hl.sav members x hhweight. Not HH16 ("Native language of the Respondent"), which slides towards
the interview language (HH15) the way Iraq's did: of 9,971 Pashto-headed households, 338 answer
HH16 Dari, against 166 the other way; Kabul's Pashto is 37.5% by HC1B, 30.2% by HH16 and 29.7%
by interview language; Herat's 17.1 / 10.5 / 8.6. WM14 (women 15-49, own native language) is
printed as a third reading. The 1,221 interviews held in a language other than Dari or Pashto
(Uzbek, Turkmen, Nuristani areas) show the teams did not force answers into the two.

STRATA. Per province, MICS's urban and rural strata (HH6) are tabulated apart where each has at
least MIN_CLUSTERS clusters and NSIA counts people in both; each stratum's shares then go on
NSIA's urban or rural population, and countries/af.py places each language's urban part on the
province's densest hexes. Elsewhere the province's pooled shares (MICS's own weights) go on its
whole population.

CARVE-OUTS. MICS's list has no Pamiri languages, Kyrgyz, Parachi, Gawar-Bati or Brahui. The
speaker estimates the previous build drew for them (sources/af_mrrd.py MINORITIES, BRAHUI,
sources/af.md §3b) are kept, carved out of the rows their speakers would most likely have
answered: MICS's "other" first, then the province's main related language. Brahui comes out of
Balochi (at most half of it) and "other" only, never out of Pashto, which MICS measures; so it is
drawn at what those rows hold, not at the 200,000 estimate.

SHARES. Weighted persons per (province, stratum, HC1B answer); x NSIA's population of that
province or stratum (religiondots' af_lookup.csv, read only), largest remainder, so every
province sums to NSIA's settled figure (34,935,197). 1.5 million Kuchis have no province there
and stay the gap.

CHECKS. Row counts; every member matches a household; 34 provinces, each >= 25 clusters; per
province HC1B and HH16 within 10 points for every language; national Dari and Pashto within the
Asia Foundation 2006 first-language figures (49 / 40) +/- 10; province totals = NSIA.
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
from rdlink import RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "af" / "mics_2022"
OUT = HERE / "data" / "normalized" / "af.csv"
LOOKUP = RD_GEO / "af" / "af_lookup.csv"          # religiondots, read-only: NSIA 1404
SOURCE_ID = "mics6_2022_hc1b"
SETTLED_1404 = 34_935_197
N_HH, N_IV, N_HL = 23_338, 23_213, 199_354
MIN_CLUSTERS = 5          # per stratum, to tabulate urban and rural apart

# MICS HH7 -> religiondots unit (af_lookup.csv geo_id, NSIA's order)
PROV = {"KABUL": "AF01", "KAPISA": "AF02", "PARWAN": "AF03", "MAIDAN WARDAK": "AF04",
        "LOGAR": "AF05", "NANGARHAR": "AF06", "LAGHMAN": "AF07", "PANJSHER": "AF08",
        "BAGHLAN": "AF09", "BAMYAN": "AF10", "GHAZNI": "AF11", "PAKTIKA": "AF12",
        "PAKTYA": "AF13", "KHOST": "AF14", "KUNARHA": "AF15", "NOORISTAN": "AF16",
        "BADAKHSHAN": "AF17", "TAKHAR": "AF18", "KUNDUZ": "AF19", "SAMANGAN": "AF20",
        "BALKH": "AF21", "SAR-E-PUL": "AF22", "GHOR": "AF23", "DAYKUNDI": "AF24",
        "UROZGAN": "AF25", "ZABUL": "AF26", "KANDAHAR": "AF27", "JAWZJAN": "AF28",
        "FARYAB": "AF29", "HELMAND": "AF30", "BADGHIS": "AF31", "HERAT": "AF32",
        "FARAH": "AF33", "NIMROZ": "AF34"}
# HC1B / HH16 / WM14 answer -> label written to af.csv (taxonomy/af2022.py maps these)
LABEL = {"DARI": "Dari", "PASHTO": "Pashto", "UZBAKI": "Uzbek", "TURKMANI": "Turkmen",
         "NOORISTANI": "Nuristani", "BALOCHI": "Balochi", "PASHAIE": "Pashai",
         "OTHER LANGUAGE": "Other language"}
NOT_DRAWN = {"NO RESPONSE"}
TAF_2006 = {"Dari": 49, "Pashto": 40}

# Speaker estimates for languages MICS does not list (figures and sources as sources/af_mrrd.py
# MINORITIES; sources/af.md §3b). label: (geo_id, speakers, [(row, at most this share of it)])
OTHER = "Other language"
MINORITIES = {
    "Shughni (speaker estimate)": ("AF17", 20_000, [(OTHER, 1), ("Dari", 1)]),
    "Wakhi (speaker estimate)": ("AF17", 17_500, [(OTHER, 1), ("Dari", 1)]),
    "Munji (speaker estimate)": ("AF17", 5_300, [(OTHER, 1), ("Dari", 1)]),
    "Sanglechi (speaker estimate)": ("AF17", 2_200, [(OTHER, 1), ("Dari", 1)]),
    "Ishkashimi (speaker estimate)": ("AF17", 1_500, [(OTHER, 1), ("Dari", 1)]),
    "Kyrgyz (speaker estimate)": ("AF17", 1_500, [(OTHER, 1), ("Uzbek", 1), ("Dari", 1)]),
    "Parachi (speaker estimate)": ("AF02", 3_500, [(OTHER, 1), ("Dari", 1)]),
    "Gawar-Bati (speaker estimate)": ("AF15", 7_500, [(OTHER, 1), ("Pashto", 1)]),
}
# Brahui: 200,000 (Ethnologue via Bashir 2003), split by the southern belt's Kontur population
# (af_mrrd.py BRAHUI_ZONE_POP), taken out of Balochi (at most half) and "other" only.
BRAHUI = 200_000
BRAHUI_ZONE_POP = {"AF34": 41_001, "AF30": 243_832, "AF27": 26_076}
BRAHUI_TAKE = [("Balochi", 0.5), (OTHER, 1)]
CARVE_NOTE = {
    "Brahui (speaker estimate)": "Ethnologue via Bashir (2003), 200,000 in Afghanistan, split by "
                                 "the southern belt's population; drawn at what MICS's Balochi "
                                 "(half) and other answers hold",
}


def weighted(persons, keys):
    d = persons[(persons["hhweight"] > 0) & ~persons["HC1B"].astype(str).isin(NOT_DRAWN)]
    return d.groupby(keys, observed=True)["hhweight"].sum()


def pct(s):
    t = s.sum()
    return {LABEL[str(k)]: v / t * 100 for k, v in s.items() if v > 0}


def lr_round(shares, total):
    """{label: weight} -> {label: int} summing to total, largest remainder."""
    t = sum(shares.values())
    raw = {k: v / t * total for k, v in shares.items()}
    cnt = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - cnt[k], reverse=True)[:total - sum(cnt.values())]:
        cnt[k] += 1
    assert sum(cnt.values()) == total
    return cnt


def carve(cells, g, label, n, take, cap_short=False):
    """Move n people of province g into `label`, out of the rows in `take`, from its rural
    stratum if it has one (the homelands are rural), else the pooled cell."""
    key = (g, "rural") if (g, "rural") in cells else (g, "all")
    c = cells[key]
    left = n
    for src, cap in take:
        k = min(left, int(c.get(src, 0) * cap))
        if k:
            c[src] -= k
            left -= k
    if left and not cap_short:
        raise SystemExit(f"{g} {label}: {left:,} of {n:,} found no row to come out of")
    c[label] = c.get(label, 0) + n - left
    return n - left, key[1]


def main():
    import pyreadstat
    hh, _ = pyreadstat.read_sav(str(RAW / "hh.sav"), apply_value_formats=True,
                                usecols=["HH1", "HH2", "HH6", "HH7", "HH15", "HH16", "HH17",
                                         "HC1B", "hhweight"])
    hl, _ = pyreadstat.read_sav(str(RAW / "hl.sav"), usecols=["HH1", "HH2", "HL1"])
    wm, _ = pyreadstat.read_sav(str(RAW / "wm.sav"), apply_value_formats=True,
                                usecols=["HH7", "WM14", "wmweight"])
    iv = hh[hh["hhweight"] > 0]
    if len(hh) != N_HH or len(iv) != N_IV or len(hl) != N_HL:
        raise SystemExit(f"hh {len(hh)} / interviewed {len(iv)} / hl {len(hl)}, expected "
                         f"{N_HH} / {N_IV} / {N_HL}")
    persons = hl.merge(hh, on=["HH1", "HH2"], how="left", indicator=True)
    if (persons["_merge"] != "both").any():
        raise SystemExit("hl.sav members with no household in hh.sav")
    if set(hh["HH7"].astype(str)) != set(PROV):
        raise SystemExit(f"HH7 provinces differ from PROV: {sorted(set(hh['HH7'].astype(str)) ^ set(PROV))}")
    print(f"  {len(hh):,} households, {len(iv):,} interviewed, {len(hl):,} members")

    # ---- which item: HC1B against HH16, and both against the interview language ----
    a, b = iv["HC1B"].astype(str), iv["HH16"].astype(str)
    p2d = int(((a == "PASHTO") & (b == "DARI")).sum())
    d2p = int(((a == "DARI") & (b == "PASHTO")).sum())
    p2d_dari_iv = int(((a == "PASHTO") & (b == "DARI") & (iv["HH15"].astype(str) == "DARI")).sum())
    print(f"  households: HC1B Pashto / HH16 Dari {p2d} ({p2d_dari_iv} of them interviewed in "
          f"Dari), HC1B Dari / HH16 Pashto {d2p}; HC1B and HH16 agree in "
          f"{(a == b).mean() * 100:.1f}%")
    print(f"  interview language: " + ", ".join(f"{k} {v:,}" for k, v in
                                               iv["HH15"].astype(str).value_counts().items())
          + "; translator used: " + ", ".join(f"{k} {v:,}" for k, v in
                                               iv["HH17"].astype(str).value_counts().items()))
    head = weighted(persons, ["HH7", "HC1B"])
    resp = persons[persons["hhweight"] > 0].groupby(["HH7", "HH16"], observed=True)["hhweight"].sum()
    intv = persons[persons["hhweight"] > 0].groupby(["HH7", "HH15"], observed=True)["hhweight"].sum()
    women = wm[wm["wmweight"] > 0].groupby(["HH7", "WM14"], observed=True)["wmweight"].sum()
    clusters = iv.groupby("HH7", observed=True)["HH1"].nunique()
    ncl = iv.groupby(["HH7", "HH6"], observed=True)["HH1"].nunique()
    print("  per province, % of persons, Dari / Pashto: head HC1B | respondent HH16 | interview "
          "HH15 | women WM14   (clusters urban + rural)")
    bad = []
    for m in sorted(PROV, key=PROV.get):
        if clusters[m] < 25:
            bad.append(f"{m} only {clusters[m]} clusters")
        h = pct(head[m])
        r = {LABEL.get(str(k), str(k)): v for k, v in pct_any(resp[m]).items()}
        i = {str(k).title(): v for k, v in pct_any(intv[m]).items()}
        w = {LABEL.get(str(k), str(k)): v for k, v in pct_any(women[m]).items()}
        for lab in set(h) | set(r):
            if abs(h.get(lab, 0) - r.get(lab, 0)) > 10:
                bad.append(f"{m} {lab} HC1B {h.get(lab, 0):.1f} vs HH16 {r.get(lab, 0):.1f}")
        f = lambda d: f"{d.get('Dari', 0):5.1f} {d.get('Pashto', 0):5.1f}"  # noqa: E731
        print(f"    {m:<14} {f(h)} | {f(r)} | {f(i)} | {f(w)}   "
              f"({ncl.get((m, 'URBAN'), 0)} + {ncl.get((m, 'RURAL'), 0)})")
    if bad:
        raise SystemExit("checks failed: " + "; ".join(bad))

    # ---- shares per province, or per stratum where both are sampled well enough ----
    lut = {}
    with open(LOOKUP, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            lut[r["geo_id"]] = r
    if set(lut) != set(PROV.values()):
        raise SystemExit("af_lookup.csv provinces differ from PROV")
    if sum(int(r["pop"]) for r in lut.values()) != SETTLED_1404:
        raise SystemExit("NSIA 1404 settled population changed")
    hs = weighted(persons, ["HH7", "HH6", "HC1B"])
    cells, base, how = {}, {}, {}
    for m, g in PROV.items():
        r = lut[g]
        pop, urb, rur = int(r["pop"]), int(r["urban"]), int(r["rural"])
        assert urb + rur == pop, g
        nu, nr = ncl.get((m, "URBAN"), 0), ncl.get((m, "RURAL"), 0)
        if urb and rur and nu >= MIN_CLUSTERS and nr >= MIN_CLUSTERS:
            for st, n in (("urban", urb), ("rural", rur)):
                s = hs[m][st.upper()]
                cells[(g, st)] = lr_round({LABEL[str(k)]: v for k, v in s.items() if v > 0}, n)
                base[(g, st)] = n
            how[g] = f"urban ({nu} clusters) and rural ({nr}) apart"
        else:
            cells[(g, "all")] = lr_round({LABEL[str(k)]: v for k, v in head[m].items() if v > 0},
                                         pop)
            base[(g, "all")] = pop
            how[g] = f"pooled ({nu} urban + {nr} rural clusters)"

    # ---- carve-outs: languages MICS does not list ----
    for label, (g, n, take) in MINORITIES.items():
        carve(cells, g, label, n, take)
    zt = sum(BRAHUI_ZONE_POP.values())
    want = {g: round(BRAHUI * p / zt) for g, p in BRAHUI_ZONE_POP.items()}
    want["AF30"] += BRAHUI - sum(want.values())
    got = {}
    for g, n in want.items():
        got[g] = carve(cells, g, "Brahui (speaker estimate)", n, BRAHUI_TAKE, cap_short=True)[0]
    print("  Brahui: " + ", ".join(f"{g} {got[g]:,} of {want[g]:,}" for g in want)
          + f"; {sum(got.values()):,} drawn of the {BRAHUI:,} estimate")

    out, natl = [], {}
    for (g, st), c in sorted(cells.items()):
        assert sum(c.values()) == base[(g, st)], (g, st)
        m = next(k for k, v in PROV.items() if v == g)
        for lab in sorted(c, key=c.get, reverse=True):
            if c[lab] <= 0:
                continue
            natl[lab] = natl.get(lab, 0) + c[lab]
            est = lab.endswith("(speaker estimate)")
            note = (CARVE_NOTE.get(lab, "speaker estimate, sources/af.md §3b") if est else
                    f"MICS6 2022-23 HC1B, {how[g]}, {clusters[m]} clusters; NSIA 1404 "
                    f"{st if st != 'all' else 'settled'} population {base[(g, st)]}")
            out.append(dict(geo_id=g, geo_level="province", geo_name=lut[g]["name"], stratum=st,
                            source_category=lab, count=c[lab], tier="modelled",
                            source_id="af_speaker_estimates" if est else SOURCE_ID,
                            year=2023, note=note))
    tot = sum(natl.values())
    if tot != SETTLED_1404:
        raise SystemExit(f"rows sum to {tot:,}, expected {SETTLED_1404:,}")
    for lab, ref in TAF_2006.items():
        v = natl[lab] / tot * 100
        if abs(v - ref) > 10:
            raise SystemExit(f"national {lab} {v:.1f}%, far from the 2006 first-language {ref}%")
    print("  " + "; ".join(f"{g} {h}" for g, h in sorted(how.items()) if h.startswith("urban")))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    tmp.replace(OUT)
    print(f"  wrote {OUT.relative_to(HERE)}: {len(out)} rows, {tot:,} people")
    for lab, n in sorted(natl.items(), key=lambda kv: -kv[1]):
        print(f"      {lab:<30} {n:>11,}  {n / tot * 100:5.2f}%")


def pct_any(s):
    t = s.sum()
    return {k: v / t * 100 for k, v in s.items() if v > 0}


if __name__ == "__main__":
    main()
