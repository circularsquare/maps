"""Iran: World Values Survey waves 5 (2005) and 7 (2020), language at home by province, pooled,
as shares applied to the 1395 (2016) census province populations -> data/normalized/ir.csv.

    python sources/ir_wvs.py [--fetch]

NO CENSUS ASKS. Iran's census has no language or ethnicity question (religiondots' sources/ir.md
read the 1395 tables). The open source with a home-language item, province codes and a probability
sample is the World Values Survey: wave 7 (2020, 1,499 adults, `Q272` "language at home",
`N_REGION_ISO` 30 provinces, none in Kohgiluyeh and Boyer-Ahmad) and wave 5 (2005, 2,667 adults,
`V222`, `V257` 30 regions: Tehran still held Alborz, which split off in 2010). Wave 4 (2000) is
in the online tool with its language collapsed to Azerbaijani / Persian / other, so it adds
nothing past Azerbaijani and is not used. sources/ir.md has the sources weighed and rejected.

THE ROUTE. The WVS online analysis tool (worldvaluessurvey.org/WVSOnline.jsp) needs no
registration. Its chain of JSP form posts (AJOnlineCountries -> AJOnlineIndex -> AJOnlineQtn, then
AJOnlineQtn again with MACRUCE1 / MACRUCE2 set) returns a crosstab as HTML: column percentages to one
decimal and each column's N. Counts are percent x N, rounded; the checks below confirm each one
lands within rounding of a whole number and that the columns sum to their N. --fetch saves the
four pages under data/raw/ir/. The WVS file downloads sit behind a form and are not used.

  w7_q272_region.html      wave 7, Q272 x N_REGION_ISO
  w7_q272_region_eth.html  wave 7, Q272 x N_REGION_ISO x Q290 (ethnic group), for "Other" below
  w5_v222_region.html      wave 5, V222 x V257
  w4_v219_region.html      wave 4, V219 x V243 (fetched for the record, not used)

THE POOL. Per province, wave 5 and wave 7 counts are added, unweighted, "No answer" left out,
and the shares applied to the census population. Two exceptions, both because an answer was
missing from one wave's code list:
  * Mazandaran and Golestan use wave 7 only. Wave 5's list had no Mazandarani answer, and its
    Mazandaran sample answered Persian (96.7%); wave 7's "Gilaki" answer took Mazandarani
    speakers (Mazandaran 41 of 41; the ethnic card's "Gilak/Mazani/Shomali" is one group).
  * Wave 5's Tehran (540 interviews, Tehran and Karaj before Alborz split off) is shared between
    Tehran and Alborz in proportion to their 2016 populations.
Kohgiluyeh and Boyer-Ahmad has wave 5 only (25 interviews).

"OTHER" IN WAVE 7. Wave 7's language card had no Balochi or Turkmen answer. 24 of Sistan and
Baluchestan's 25 "Other" answers came from respondents whose ethnic group (Q290) is Baluch; those
are written as `Other (ethnic group Baluch)` and drawn as Balochi. Every other "Other" stays a bare
other (Golestan's 25, of whom 18 gave ethnic group "Other", Turkmen not being on that card
either, are not guessed into Turkmen).

CHECKS: national frequencies against the IHSN catalogue's (wave 7 Q272, wave 5 V222) to the
person; each column's counts sum to its N; every percent x N within 0.08 of a whole number
(column percentages print to 0.1%); the 3-way's language totals by province within the 2-way's
(only the 13 respondents with no ethnic answer drop out); population base = religiondots'
1395 census province totals, 79,926,270.
"""
import csv
import html
import re
import sys
import urllib3
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from rdlink import RD_GEO  # noqa: E402

RAW = HERE / "data" / "raw" / "ir"
OUT = HERE / "data" / "normalized" / "ir.csv"
LOOKUP = RD_GEO / "ir" / "ir_lookup.csv"      # religiondots, read-only: 1395 census totals
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
BASE = "https://www.worldvaluessurvey.org/"
SOURCE_ID = "wvs_w5_2005_w7_2020_online"
CENSUS_TOTAL = 79_926_270

# (file, wave id, sample id, question MAIDX, cross 1, cross 2)
PAGES = {
    "w7": ("w7_q272_region.html", "1562", "3466", "C_Q272", "2437884", ""),
    "w7eth": ("w7_q272_region_eth.html", "1562", "3466", "C_Q272", "2437884", "2415047"),
    "w5": ("w5_v222_region.html", "2", "461", "007_003", "1512", ""),
    "w4": ("w4_v219_region.html", "12", "482", "G_016", "49506", ""),
}

# IHSN catalogue frequencies (catalog.ihsn.org/catalog/11583 V308; /8974 V721), read 2026-10-05.
# Wave 5's online tool relabels the file's codes: "Azari" (458) prints as "Turkish" and the
# file's own "Turkish" (7) as "Turkmen"; "Asirien" as Assyrian Neo-Aramaic.
IHSN_W7 = {"Arabic": 18, "Azerbaijani;  Azeri": 258, "Gilaki": 94, "Kurdish; Yezidi": 87,
           "Lurish; Luri; Bakhtiari": 60, "Persian; Farsi; Dari": 911, "Other": 70,
           "No answer": 1}
IHSN_W5 = {"Arabic": 59, "Armenian; Hayeren": 2, "Assyrian Neo-Aramaic": 1, "Turkish": 458,
           "Balochi": 35, "Gilaki": 77, "Kurdish; Yezidi": 166, "Lurish; Luri; Bakhtiari": 108,
           "Persian; Farsi; Dari": 1729, "Turkmen": 7, "Zoroastrian": 1, "No answer": 24}
NOT_DRAWN = {"No answer", "Missing; Not available", "Don't know"}

# survey column -> religiondots unit (ir_lookup.csv `unit`)
W7_UNITS = {
    "IR-01 Azerbaijan-E Sharqi": "East Azerbaijan", "IR-02 Azerbaijan-E Gharbi": "West Azerbaijan",
    "IR-03 Ardabil": "Ardabil", "IR-04 Esfahan": "Isfahan", "IR-05 Ilam": "Ilam",
    "IR-06 Bushehr": "Bushehr", "IR-07 Tehran": "Tehran",
    "IR-08 Chahar Mahal Va Bakhtiari": "Chaharmahal and Bakhtiari", "IR-10 Khuzestan": "Khuzestan",
    "IR-11 Zanjan": "Zanjan", "IR-12 Semnan": "Semnan",
    "IR-13 Sistan Va Baluchestan": "Sistan and Baluchestan", "IR-14 Fars": "Fars",
    "IR-15 Kerman": "Kerman", "IR-16 Kordestan": "Kurdistan", "IR-17 Kermanshah": "Kermanshah",
    "IR-19 Gilan": "Gilan", "IR-20 Lorestan": "Lorestan", "IR-21 Mazandaran": "Mazandaran",
    "IR-22 Markazi": "Markazi", "IR-23 Hormozgan": "Hormozgan", "IR-24 Hamadan": "Hamadan",
    "IR-25 Yazd": "Yazd", "IR-26 Qom": "Qom", "IR-27 Golestan": "Golestan",
    "IR-28 Qazvin": "Qazvin", "IR-29 Khorasan-E Jonubi": "South Khorasan",
    "IR-30 Razavi Khorasan": "Razavi Khorasan", "IR-31 Khorasan-E Shomali": "North Khorasan",
    "IR-32 Alborz": "Alborz",
}
W5_UNITS = {
    "IR: Gilan": "Gilan", "IR: Mazandaran": "Mazandaran", "IR: Fars": "Fars", "IR: Kerman": "Kerman",
    "IR: West azarbayjan": "West Azerbaijan", "IR: East azarbayjan": "East Azerbaijan",
    "IR: Kermanshah": "Kermanshah", "IR: Sistan and balouchestan": "Sistan and Baluchestan",
    "IR: Isfahan": "Isfahan", "IR: Khozestan": "Khuzestan", "IR: Kordestan": "Kurdistan",
    "IR: Khorasan": "Razavi Khorasan", "IR: Tehran": "Tehran+Alborz",
    "IR: Boyer ahmad": "Kohgiluyeh and Boyer-Ahmad", "IR: Bushehr": "Bushehr",
    "IR: Chaharmahal": "Chaharmahal and Bakhtiari", "IR: Hamadan": "Hamadan",
    "IR: Hormozgan": "Hormozgan", "IR: Ilam": "Ilam", "IR: Lorestan": "Lorestan",
    "IR: Markazi": "Markazi", "IR: Semnan": "Semnan", "IR: Yazd": "Yazd", "IR: Zanjan": "Zanjan",
    "IR: Ghom": "Qom", "IR: Ardabil": "Ardabil", "IR: Ghazvin": "Qazvin", "IR: Golestan": "Golestan",
    "IR: North Khorasan": "North Khorasan", "IR: South Khorasan": "South Khorasan",
}
W7_ONLY = {"Mazandaran", "Golestan"}          # wave 5 had no Mazandarani answer
GILAKI_IS_MAZANI = {"Mazandaran", "Golestan"}  # where wave 7's "Gilaki" answer is Mazandarani
BALUCH_OTHER = "Other (ethnic group Baluch)"

# Values and Attitudes of Iranians, wave 3 (2015, 14,906 face to face, Ministry of Culture's
# Office of National Plans): share speaking a "local or ethnic language or dialect" rather than
# Persian at home, per province, as FDD's map of the microdata prints it (fdd.org, "Iran Is More
# Than Persia", 2021). A check only: its "local" holds Persian dialects too (Yazd 4%).
VA2015_LOCAL = {
    "East Azerbaijan": 98, "West Azerbaijan": 99, "Ardabil": 98, "Isfahan": 13, "Alborz": 28,
    "Ilam": 97, "Bushehr": 29, "Tehran": 17, "Chaharmahal and Bakhtiari": 63,
    "South Khorasan": 10, "Razavi Khorasan": 14, "North Khorasan": 47, "Khuzestan": 62,
    "Zanjan": 85, "Semnan": 22, "Sistan and Baluchestan": 78, "Fars": 30, "Qazvin": 42, "Qom": 25,
    "Kurdistan": 99, "Kerman": 20, "Kermanshah": 70, "Kohgiluyeh and Boyer-Ahmad": 98,
    "Golestan": 56, "Gilan": 69, "Lorestan": 87, "Mazandaran": 61, "Markazi": 33, "Hormozgan": 62,
    "Hamadan": 59, "Yazd": 4,
}
PERSIAN = "Persian; Farsi; Dari"


def fetch():
    import requests
    urllib3.disable_warnings()
    RAW.mkdir(parents=True, exist_ok=True)
    for key, (name, wave, said, maidx, x1, x2) in PAGES.items():
        s = requests.Session()
        s.verify = False                     # the site omits an intermediate certificate
        s.headers["User-Agent"] = UA
        s.get(BASE + "WVSOnline.jsp", timeout=60)
        s.get(BASE + "AJOnline.jsp?WAVE=&COUNTRY=", timeout=60)
        form = {"ulthost": "WVS", "CMSID": "", "WAVE": wave, "MAIDX": "", "SAIDS": said,
                "AMIDS": "364", "SATITULOS": "Iran", "COUNTRY": "", "CRUCEX": ""}
        s.post(BASE + "AJOnlineCountries.jsp", data=form, timeout=60)
        s.post(BASE + "AJOnlineIndex.jsp", data=form, timeout=60)
        form["MAIDX"] = maidx
        s.post(BASE + "AJOnlineQtn.jsp", data=form, timeout=120)
        form2 = {"ulthost": "WVS", "CMSID": "", "WAVE": wave, "SAIDS": said, "SATITULOS": "Iran",
                 "AMIDS": "364", "MAIDX": maidx, "MACRUCE1": x1, "MACRUCE2": x2,
                 "CRUCES_ROTARXY": "", "CRUCE_TYPE": "TAB", "AJArchive": "WVS Data Archive"}
        r = s.post(BASE + "AJOnlineQtn.jsp", data=form2, timeout=120)
        r.raise_for_status()
        if "JDSTableCellHeader" not in r.text:
            raise SystemExit(f"{name}: no crosstab in the response")
        tmp = (RAW / name).with_suffix(".part")
        tmp.write_text(r.text, encoding="utf-8")
        tmp.replace(RAW / name)
        print(f"  {name}: {len(r.text):,} chars")


def tables(path):
    """Yield (filter, columns, {row label: [cells]}, [N per column]) for each printed table."""
    t = path.read_text(encoding="utf-8")
    for m in re.finditer(r"(?s)<table[^>]*>(.*?)</table>", t):
        body = m.group(1)
        if "JDSTable" not in body:
            continue
        rows = []
        for row in re.findall(r"(?s)<tr[^>]*>(.*?)</tr>", body):
            cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).replace("​", "").strip()
                     for c in re.findall(r"(?s)<td[^>]*>(.*?)</td>", row)]
            rows.append(cells)
        if len(rows) < 3:
            continue
        head = rows[0]
        filt = re.search(r"\[(.*)\]", " ".join(head))
        cols = rows[1]
        data, ns = {}, None
        for r in rows[2:]:
            if not r:
                continue
            if r[0].startswith("(N)"):
                ns = [int(x.strip("()").replace(",", "")) for x in r[1:]]
            else:
                data[r[0]] = r[1:]
        has_total = "TOTAL" in head
        if has_total:                       # the first table repeats the national column
            cols = ["TOTAL"] + cols
        yield (filt.group(1) if filt else None), cols, data, ns


def counts(path):
    """{(filter, column): {label: n}} from percent x N, with the rounding checked."""
    out = {}
    for filt, cols, data, ns in tables(path):
        if ns is None or len(ns) != len(cols):
            raise SystemExit(f"{path.name}: columns {cols} against N {ns}")
        for j, col in enumerate(cols):
            if col == "TOTAL":              # 0.0% hides single answers; summed from provinces
                continue
            got = {}
            for lab, cells in data.items():
                c = cells[j]
                if c in ("-", ""):
                    continue
                x = float(c.rstrip("%")) * ns[j] / 100.0
                if abs(x - round(x)) > max(0.08, 0.0005 * ns[j]):
                    raise SystemExit(f"{path.name} {col} {lab}: {c} of {ns[j]} = {x:.3f}")
                if round(x):
                    got[lab] = round(x)
            if sum(got.values()) != ns[j]:
                raise SystemExit(f"{path.name} {col}: counts sum {sum(got.values())} != N {ns[j]}")
            out[(filt, col)] = got
    return out


def by_column(c):
    return {col: v for (filt, col), v in c.items() if filt is None and col != "TOTAL"}


def main():
    if "--fetch" in sys.argv:
        fetch()
    for k, (name, *_rest) in PAGES.items():
        if not (RAW / name).exists():
            raise SystemExit(f"missing {RAW / name}; run with --fetch")

    w7all = counts(RAW / PAGES["w7"][0])
    w5all = counts(RAW / PAGES["w5"][0])
    def national(c):
        tot = {}
        for (filt, col), v in c.items():
            if filt is None:
                for lab, n in v.items():
                    tot[lab] = tot.get(lab, 0) + n
        return tot
    for want, got, tag in ((IHSN_W7, national(w7all), "wave 7"),
                           (IHSN_W5, national(w5all), "wave 5")):
        if got != want:
            raise SystemExit(f"{tag} national counts {got} != IHSN {want}")
        print(f"  {tag}: national counts = IHSN frequencies ({sum(got.values()):,})")
    w7, w5 = by_column(w7all), by_column(w5all)
    if set(w7) != set(W7_UNITS) or set(w5) != set(W5_UNITS):
        raise SystemExit(f"columns: w7 extra {set(w7) ^ set(W7_UNITS)}, "
                         f"w5 extra {set(w5) ^ set(W5_UNITS)}")

    # wave 7 "Other" answered by self-identified Baluch
    eth = counts(RAW / PAGES["w7eth"][0])
    baluch_other = {}
    by_prov = {}
    for (filt, col), v in eth.items():
        if col == "TOTAL" or filt is None:
            continue
        for lab, n in v.items():
            by_prov.setdefault(col, {}).setdefault(lab, 0)
            by_prov[col][lab] += n
        if filt.endswith("Baluch") and v.get("Other"):
            baluch_other[col] = v["Other"]
    for col, v in by_prov.items():
        for lab, n in v.items():
            if n > w7[col].get(lab, 0):
                raise SystemExit(f"3-way {col} {lab} {n} > 2-way {w7[col].get(lab, 0)}")
    dropped = sum(sum(v.values()) for v in w7.values()) - sum(sum(v.values()) for v in by_prov.values())
    print(f"  wave 7 3-way: {dropped} respondents with no ethnic answer drop out (13 expected); "
          f"Baluch 'Other': {baluch_other}")
    if dropped != 13:
        raise SystemExit("3-way drop-out is not the 13 with no ethnic answer")

    # population base
    pop = {}
    with open(LOOKUP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pop[r["unit"]] = int(r["census_pop_2016"])
    if len(pop) != 31 or sum(pop.values()) != CENSUS_TOTAL:
        raise SystemExit(f"ir_lookup.csv: {len(pop)} units, {sum(pop.values()):,} people")
    units = set(W7_UNITS.values()) | {"Kohgiluyeh and Boyer-Ahmad"}
    if units != set(pop):
        raise SystemExit(f"units differ from religiondots: {units ^ set(pop)}")

    # pool
    pooled = {u: {} for u in pop}
    n7 = {u: 0 for u in pop}
    n5 = {u: 0.0 for u in pop}

    def add(unit, lab, n, which):
        if lab in NOT_DRAWN:
            return
        pooled[unit][lab] = pooled[unit].get(lab, 0) + n
        if which == 7:
            n7[unit] += n
        else:
            n5[unit] += n

    for col, v in w7.items():
        u = W7_UNITS[col]
        v = dict(v)
        b = baluch_other.get(col, 0)
        if b:
            v["Other"] -= b
            v[BALUCH_OTHER] = b
        for lab, n in v.items():
            if lab == "Gilaki" and u in GILAKI_IS_MAZANI:
                lab = "Gilaki [Mazandaran, Golestan]"
            add(u, lab, n, 7)
    ta = pop["Tehran"] + pop["Alborz"]
    for col, v in w5.items():
        u = W5_UNITS[col]
        if u in W7_ONLY:
            continue
        for lab, n in v.items():
            lab5 = f"{lab} [2005]" if lab in ("Turkish", "Turkmen", "Zoroastrian") else lab
            if u == "Tehran+Alborz":
                add("Tehran", lab5, n * pop["Tehran"] / ta, 5)
                add("Alborz", lab5, n * pop["Alborz"] / ta, 5)
            else:
                add(u, lab5, n, 5)

    out, natl = [], {}
    for u in sorted(pop):
        v = pooled[u]
        tot = sum(v.values())
        raw = {lab: n / tot * pop[u] for lab, n in v.items()}
        cnt = {lab: int(x) for lab, x in raw.items()}
        short = pop[u] - sum(cnt.values())
        for lab in sorted(raw, key=lambda k: raw[k] - cnt[k], reverse=True)[:short]:
            cnt[lab] += 1
        assert sum(cnt.values()) == pop[u]
        for lab in sorted(cnt, key=cnt.get, reverse=True):
            natl[lab] = natl.get(lab, 0) + cnt[lab]
            out.append(dict(geo_id=u, geo_level="province", geo_name=u, source_category=lab,
                            count=cnt[lab], tier="modelled", source_id=SOURCE_ID, year=2020,
                            note=f"{v[lab]:.1f} of {tot:.1f} pooled answers (wave 7: {n7[u]}, "
                                 f"wave 5: {n5[u]:.1f}); 1395 census population {pop[u]}"))
        pers = v.get(PERSIAN, 0) / tot * 100
        print(f"  {u:<27} n7 {n7[u]:>4} n5 {n5[u]:>6.1f}  Persian {pers:5.1f}  "
              f"V&A-2015 Persian {100 - VA2015_LOCAL[u]:>3}  "
              + ", ".join(f"{lab.split(';')[0]} {n / tot * 100:.0f}"
                          for lab, n in sorted(v.items(), key=lambda kv: -kv[1])
                          if lab != PERSIAN and n / tot >= 0.05))

    # check against Values and Attitudes 2015 (Spearman over the 31 provinces)
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
    us = sorted(pop)
    a = [pooled[u].get(PERSIAN, 0) / sum(pooled[u].values()) for u in us]
    b = [100 - VA2015_LOCAL[u] for u in us]
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    rho = (sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
           / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5)
    print(f"  Spearman, WVS pooled Persian share vs Values and Attitudes 2015: {rho:+.3f}")
    if rho < 0.7:
        raise SystemExit("the pooled shares disagree with the 2015 survey's province ranking")
    gaps = sorted(us, key=lambda u: -abs(a[us.index(u)] * 100 - b[us.index(u)]))[:6]
    print("  largest Persian-share gaps (WVS - V&A): "
          + ", ".join(f"{u} {a[us.index(u)] * 100 - b[us.index(u)]:+.0f}" for u in gaps))
    natl_pers = natl.get(PERSIAN, 0) / CENSUS_TOTAL * 100
    print(f"  drawn Persian nationally {natl_pers:.1f}% (V&A 2015 printed 49.1% of 14,686)")

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
        print(f"      {lab:<34} {n:>11,}  {n / CENSUS_TOTAL * 100:5.2f}%")


if __name__ == "__main__":
    main()
