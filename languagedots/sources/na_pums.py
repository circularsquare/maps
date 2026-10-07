"""Namibia 2011 census, main language spoken in the household -> data/normalized/na.csv.

    python sources/na_pums.py [--fetch]

SOURCE. Namibia Statistics Agency, Namibia 2011 Population and Housing Census, Public Use
Microdata Sample (PUMS), NAM_NSA_PHC_2011_V01_PUMS, version 1.0 (June 2013), from the NSA's
National Data Archive (microdata.nsanamibia.com, catalog 9). Access policy "Public use"; the
files are listed openly under the study's related materials (download ids 171 documentation,
172 data). Citation: Namibia Statistics Agency. Namibia 2011 Population and Housing Census
[PUMS dataset]. Version 1.0, Windhoek: NSA, August 2013.

THE QUESTION. H13, asked once per household: "What is the MAIN language spoken in this
household?" Every member of a household is drawn at the household's answer. Thirteen answers,
most of them groups (San, Caprivi, Herero, Kavango, Nama/Damara, Oshiwambo languages), plus
Don't know. Institutions and the special populations (hostels, barracks, prisons, hotels,
travellers; HH_TYPE 2xx and 3xx) were not asked: their H13 is blank.

THE SAMPLE. A 20% simple random sample of households in each constituency x urban/rural stratum,
weight 5; the seven strata under 250 households, and every household of 50 or more people, are
taken whole at weight 1 (documentation p.6). Weighted, the file holds 2,115,377 people against
the census's 2,113,077.

THE FILE. One fixed-width text file, three record types by the first character: 1 person (92
characters), 2 death (22), 3 housing (54). The first 16 characters are the household key:
record type, region (2), constituency (2), "0", urban/rural (1 urban, 2 rural), household type
(3), serial (6). The weight is the last character of every record. The DDI's own start
positions are per file and do not hold for this combined text; H13 is at characters 43-44 of
the housing record, which is asserted below by matching all fifteen of its codes' counts to the
DDI's catStat frequencies exactly.

CONSTITUENCY CODES. The file's codes run 1-12 inside each of the 13 regions of 2011 and are not
labelled ("consult the code book"). They are the NSA codes that COD-AB Namibia's admin-2 pcodes
carry (NA<region><constituency>), with Kavango's 2013 split renumbering three of them: Kavango
(05) constituencies 01 Kahenge, 02 Kapako and 04 Mpungu are COD-AB's NA1401, NA1402, NA1404 in
Kavango West. The witnesses, neither of which the codes decide:
  (a) the documentation names the seven strata taken whole (Arandis rural, Rehoboth Urban East
      rural, Walvis Bay Rural rural, Mpungu urban, Etayi urban, Kalahari urban, Ondobe urban);
      the strata whose households all weigh 1 must be exactly those, through the mapping;
  (b) every constituency COD-AB names "... Urban" is at least 75% urban in the file.

CHECKS (all must pass):
  1. record counts: 94,774 housing, 441,929 person, 4,401 death records (DDI)
  2. H13 counts per code equal the DDI's catStat, all 15 values
  3. every person record joins a housing record
  4. weighted people within 0.5% of the census's 2,113,077
  5. witnesses (a) and (b) above; 107 constituencies, the same per region as COD-AB
  6. against the full count: the regional profiles' Tables 6.16-6.18 (population by main
     language, region, urban and rural) for the six regions whose profiles are on the web,
     every language cell over 2,000 people within 6% of the weighted sample

OUTPUT. data/normalized/na.csv: geo_id (COD-AB pcode), residence (U, R), source_category (the
DDI's label; `Not asked (institutions and special populations)` for the blank), households,
count (weighted people).
"""
import argparse
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "na"
OUT = ROOT / "data" / "normalized" / "na.csv"
COD_ADM2 = ROOT.parent / "religiondots" / "data" / "raw" / "na" / "shp" / "nam_admin2.shp"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
NADA = "https://microdata.nsanamibia.com/index.php"
FILES = {
    "pums_2011.txt": f"{NADA}/catalog/9/download/172",
    "pums_2011_documentation.pdf": f"{NADA}/catalog/9/download/171",
    "pums_2011_ddi.xml": f"{NADA}/metadata/export/9/ddi",
}
# The 2011 regional profiles that are on the web (mirrored at cms.my.na; the NSA's own links
# are gone). Six of thirteen; the check uses whichever are present.
PROFILES = {
    "02": "https://cms.my.na/assets/documents/p19dptss1rt6erfri0a1k3q1mrhm.pdf",  # Erongo
    "04": "https://cms.my.na/assets/documents/p19dptss1r6qqq76pcp1ssp46tn.pdf",   # Karas
    "05": "https://cms.my.na/assets/documents/p19dptss1rcu81nvk1r3c1ipt1r1vp.pdf",  # Kavango
    "08": "https://cms.my.na/assets/documents/p19dptss1r7ao1dp7d3b1oibgvoq.pdf",  # Ohangwena
    "09": "https://cms.my.na/assets/documents/p19dptss1q1dh4jn11mk04hj4c5b.pdf",  # Omaheke
    "13": "https://cms.my.na/assets/documents/p19dptss1r1b6ufvsfb1mh41acvo.pdf",  # Otjozondjupa
}
REGION_NAMES = {"01": "Caprivi", "02": "Erongo", "03": "Hardap", "04": "Karas", "05": "Kavango",
                "06": "Khomas", "07": "Kunene", "08": "Ohangwena", "09": "Omaheke",
                "10": "Omusati", "11": "Oshana", "12": "Oshikoto", "13": "Otjozondjupa"}

# Erongo's profile prints its second row, between San and Herero, as "Erongo languages" in all
# three tables, where every other profile and the DDI have Caprivi languages.
PRINT_SLIPS = {("profile_2011_02.pdf", "Erongo languages"): "Caprivi languages"}

CENSUS_TOTAL = 2_113_077
NOT_ASKED = "Not asked (institutions and special populations)"
WHOLE_STRATA = {  # documentation p.6, strata under 250 households, taken whole
    ("Arandis", "R"), ("Rehoboth Urban East", "R"), ("Walvis Bay Rural", "R"),
    ("Mpungu", "U"), ("Etayi", "U"), ("Kalahari", "U"), ("Ondobe", "U")}
# Kavango's constituencies that the 2013 split put in Kavango West keep their numbers there
KAVANGO_WEST = {"01", "02", "04"}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = dict(FILES)
    jobs.update({f"profile_2011_{r}.pdf": u for r, u in PROFILES.items()})
    for name, url in jobs.items():
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 10_000:
            continue
        print(f"  fetching {name}")
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as fh:
            fh.write(r.read())
        print(f"    {dest.stat().st_size:,} bytes")


def pcode(reg, con):
    if reg == "05" and con in KAVANGO_WEST:
        return f"NA14{con}"
    return f"NA{reg}{con}"


def ddi_h13():
    """H13's labels and catStat frequencies from the DDI, and the file case counts."""
    root = ET.parse(RAW / "pums_2011_ddi.xml").getroot()
    tag = lambda e: e.tag.split("}")[-1]  # noqa: E731
    labels, freq = {}, {}
    for v in root.iter():
        if tag(v) == "var" and v.get("name") == "H13":
            for c in v.iter():
                if tag(c) != "catgry":
                    continue
                val = [x.text.strip() for x in c if tag(x) == "catValu"][0]
                lab = [x.text.strip() for x in c if tag(x) == "labl"]
                st = [x.text.strip() for x in c.iter() if tag(x) == "catStat"]
                code = "  " if val == "Sysmiss" else val.zfill(2)
                labels[code] = lab[0] if lab else NOT_ASKED
                freq[code] = int(st[0])
    cases = {}
    for f in root.iter():
        if tag(f) == "fileDscr":
            name = [x.text.strip() for x in f.iter() if tag(x) == "fileName"][0]
            n = [x.text.strip() for x in f.iter() if tag(x) == "caseQnty"][0]
            cases[name.split(".")[0]] = int(n)
    return labels, freq, cases


def read_pums():
    hh, persons, n = {}, [], defaultdict(int)
    with open(RAW / "pums_2011.txt", encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\r\n")
            t = line[0]
            n[t] += 1
            if t == "3":
                hh[line[1:16]] = (line[42:44], int(line[-1]), int(line[7:10] == "100"))
            elif t == "1":
                persons.append((line[1:16], int(line[-1])))
    return hh, persons, n


def parse_profile(path, sample_split):
    """The annexure's three tables of households and population by main language: the region,
    its urban part and its rural part (numbered 6.12-6.14 or 6.16-6.18 by profile, laid out side
    by side in different orders). Returns {"T"|"U"|"R": {label: population}}.

    Every label/households/population row is read in text order; each label occurs once per
    table, in the same table order as the three `Total` rows, so the k-th occurrence of a label
    belongs to the k-th table. The region table is the one whose total is the sum of the other
    two; which of those is urban is taken from the sample's urban/rural split (`sample_split`,
    {U: people, R: people}), the only use of the sample here. Every table must sum to its total.
    """
    import fitz
    doc = fitz.open(path)
    toks = []
    for page in doc:
        t = page.get_text()
        if "Households and population" in t and re.search(r"main\s+lang", t) and "Annexure" in t:
            toks += [s.strip() for s in t.split("\n") if s.strip()]
    if not toks:
        raise SystemExit(f"{path.name}: no annexure language tables")
    num = re.compile(r"^\d+( \d{3})*$")       # some cells print 17117, most 17 117
    rows = []
    i = 0
    while i < len(toks) - 2:
        if not num.match(toks[i]) and num.match(toks[i + 1]) and num.match(toks[i + 2]):
            lab = re.sub(r"\s+", " ", toks[i])
            lab = PRINT_SLIPS.get((path.name, lab), lab)
            rows.append((lab, int(toks[i + 1].replace(" ", "")),
                         int(toks[i + 2].replace(" ", ""))))
            i += 3
        else:
            i += 1
    tables, hh = [{}, {}, {}], [{}, {}, {}]
    seen = defaultdict(int)
    for lab, nh, pop in rows:
        if lab == "languages":       # a label wrapped onto two lines would read like this
            raise SystemExit(f"{path.name}: a wrapped label")
        k = seen[lab]
        if k > 2:
            raise SystemExit(f"{path.name}: {lab} occurs more than three times")
        tables[k][lab] = pop
        hh[k][lab] = nh
        seen[lab] += 1
    tot = [t.get("Total") for t in tables]
    if None in tot:
        raise SystemExit(f"{path.name}: {tot} totals")
    for t in tables:
        s = sum(v for lab, v in t.items() if lab != "Total")
        if s != t["Total"]:
            raise SystemExit(f"{path.name}: a table sums to {s:,} against its total {t['Total']:,}")
    ti = max(range(3), key=lambda j: tot[j])
    a, b = [j for j in range(3) if j != ti]
    if tot[a] + tot[b] != tot[ti]:
        raise SystemExit(f"{path.name}: urban and rural do not sum to the region: {tot}")
    su = sample_split["U"] / (sample_split["U"] + sample_split["R"])
    ua = tot[a] / tot[ti]
    if abs(ua - su) <= abs(tot[b] / tot[ti] - su):
        u, r = a, b
    else:
        u, r = b, a
    return ({"T": tables[ti], "U": tables[u], "R": tables[r]},
            {"T": hh[ti], "U": hh[u], "R": hh[r]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    ok = True

    def report(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    labels, freq, cases = ddi_h13()
    hh, persons, n = read_pums()
    report(n["3"] == cases["HOUSING_REC"] == 94_774 and n["1"] == cases["PERSON_REC"] == 441_929
           and n["2"] == cases["MORTALITY_REC"] == 4_401,
           f"record counts {n['3']:,} housing, {n['1']:,} person, {n['2']:,} death (DDI {cases})")
    got = defaultdict(int)
    for lang, _, _ in hh.values():
        got[lang] += 1
    report(dict(got) == freq, f"H13 at 43-44 reproduces the DDI's catStat for all {len(freq)} "
                              f"values" + ("" if dict(got) == freq else f": {dict(got)} vs {freq}"))
    # blank H13 is exactly the non-conventional households
    blank_nonconv = all((lang == "  ") == (conv == 0) for lang, _, conv in hh.values())
    report(blank_nonconv, "H13 is blank exactly for institutions and special populations")

    cell = defaultdict(float)
    hcount = defaultdict(int)
    missing = 0
    for key, w in persons:
        if key not in hh:
            missing += 1
            continue
        lang = hh[key][0]
        reg, con, ur = key[0:2], key[2:4], key[5]
        res = "U" if ur == "1" else "R"
        cell[(pcode(reg, con), res, labels[lang])] += w
    for key, (lang, w, _) in hh.items():
        reg, con, ur = key[0:2], key[2:4], key[5]
        hcount[(pcode(reg, con), "U" if ur == "1" else "R", labels[lang])] += w
    report(missing == 0, f"every person record joins a housing record ({missing} do not)")
    total = sum(cell.values())
    report(abs(total / CENSUS_TOTAL - 1) < 0.005,
           f"weighted people {total:,.0f} against the census's {CENSUS_TOTAL:,} "
           f"({total / CENSUS_TOTAL - 1:+.2%})")

    # ---- the constituency witnesses ----
    import pyogrio
    cod = pyogrio.read_dataframe(COD_ADM2, read_geometry=False)
    name = dict(zip(cod["adm2_pcode"], cod["adm2_name"]))
    cons = {k[0] for k in cell}
    report(cons == set(name) and len(cons) == 107,
           f"{len(cons)} constituencies, the same pcodes as COD-AB's {len(name)} "
           f"(missing {sorted(set(name) - cons)}, extra {sorted(cons - set(name))})")
    whole = defaultdict(lambda: [0, 0])     # stratum -> [households at weight 1, all]
    for key, (lang, w, conv) in hh.items():
        reg, con, ur = key[0:2], key[2:4], key[5]
        s = (name.get(pcode(reg, con)), "U" if ur == "1" else "R")
        whole[s][1] += 1
        if w == 1:
            whole[s][0] += 1
    all_one = {s for s, (k1, k) in whole.items() if k1 == k}
    report(all_one == WHOLE_STRATA,
           f"strata taken whole are the documentation's seven: {sorted(all_one)}")
    pop_c = defaultdict(lambda: {"U": 0.0, "R": 0.0})
    for (pc, res, _), v in cell.items():
        pop_c[pc][res] += v
    urban = {pc: d["U"] / (d["U"] + d["R"]) for pc, d in pop_c.items()}
    named_u = {pc: urban[pc] for pc in urban if name[pc].endswith("Urban")
               or "Urban " in name[pc]}
    report(all(v >= 0.75 for v in named_u.values()) and len(named_u) >= 6,
           "constituencies named Urban are at least 75% urban: "
           + ", ".join(f"{name[pc]} {v:.0%}" for pc, v in sorted(named_u.items())))
    named_r = {pc: urban[pc] for pc in urban if "Rural" in name[pc]}
    print("     (not asserted) urban share of those named Rural: "
          + ", ".join(f"{name[pc]} {v:.0%}" for pc, v in sorted(named_r.items())))

    # ---- against the full count, six regional profiles ----
    region_cells = defaultdict(float)
    for key, w in persons:
        lang = hh[key][0]
        reg, ur = key[0:2], key[5]
        region_cells[(reg, "U" if ur == "1" else "R", labels[lang])] += w
    # A 20% sample of households: a cell of h households is drawn from about h/5 of them, so its
    # relative standard error is about sqrt(0.8 / (h/5)) = 2/sqrt(h) (households carry their
    # members together, so this is per household, not per person). Every cell of 100 households
    # or more must sit within 4 standard errors; the largest |z| and the cells past 2.5 print.
    zs, worst_tot = [], 0.0
    for reg in PROFILES:
        p = RAW / f"profile_2011_{reg}.pdf"
        if not p.exists():
            continue
        split = {res: sum(v for (r, rs, lab), v in region_cells.items()
                          if r == reg and rs == res and lab != NOT_ASKED) for res in "UR"}
        prof, prof_hh = parse_profile(p, split)
        for res in ("U", "R"):
            for lab, pop in prof[res].items():
                h = prof_hh[res][lab]
                if lab == "Total" or h < 100:
                    continue
                est = region_cells.get((reg, res, lab), 0.0)
                d = est / pop - 1
                z = d / (2 / h ** 0.5)
                zs.append(abs(z))
                if abs(z) > 2.5:
                    print(f"     {REGION_NAMES[reg]} {res} {lab}: census {pop:,} ({h:,} households), "
                          f"sample {est:,.0f} ({d:+.1%}, z {z:+.1f})")
        # the profiles count people in households only, so the sample's institutions are left out
        for res in ("U", "R"):
            samp = split[res]
            d = samp / prof[res]["Total"] - 1
            worst_tot = max(worst_tot, abs(d))
            print(f"     {REGION_NAMES[reg]:<13}{res}: census (in households) "
                  f"{prof[res]['Total']:>9,}  sample {samp:>11,.0f}  ({d:+.2%})")
    report(zs and max(zs) < 4 and worst_tot < 0.035,
           f"against {len(zs)} full-count region x residence x language cells of 100+ households "
           f"in six regional profiles: largest |z| {max(zs):.1f}, "
           f"{sum(z > 2 for z in zs)} past 2 (about {0.046 * len(zs):.0f} expected by chance); "
           f"urban and rural totals within {worst_tot:.1%}")

    if not ok:
        raise SystemExit("FAILED; nothing written")
    rows = [dict(geo_id=pc, residence=res, source_category=lab,
                 households=hcount.get((pc, res, lab), 0), count=round(v))
            for (pc, res, lab), v in sorted(cell.items())]
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print("\n  national, weighted people:")
    for lab, v in nat.items():
        print(f"    {lab:<52}{v:>11,}  {v / nat.sum():6.2%}")
    print(f"\nwrote {OUT}: {len(df):,} rows, {df['geo_id'].nunique()} constituencies")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
