"""Thailand: World Values Survey "language at home", which names the regional Thai varieties
(Central, Northeastern/Isan, Northern/Lanna, Southern) that the 2010 census lumps as Thai
-> data/normalized/th_wvs.csv, read by countries/th.py to split the census's Thai.

    python sources/th_wvs.py [--fetch]

THE SOURCE. Three WVS waves for Thailand ask language at home with the four Thai varieties on the
card (IHSN catalogue DDIs, read 2026-10-06; frequencies there match the online tool's):

  wave 7 (2018, 1,500 adults)  Q272 x N_REGION_ISO, 49 changwat sampled; 482 Esan, 134 Northern
                               Thai, 132 Southern Thai, 632 Central Thai, 65 Malay, 1 Mandarin,
                               48 other, 6 other local
  wave 6 (2013, 1,200)         V247 x V256 (5 regions: Bangkok, Central, North, Northeast, South)
  wave 5 (2007, 1,534)         V222 x V257 (North, Central incl. Bangkok, South, Northeast, and an
                               "East" of 8 interviews)

THE ROUTE is sources/ir_wvs.py's: the WVS online analysis tool's JSP form posts, no login, one
crosstab per page (column percentages and each column's N). Thailand's ids: country 764; sample
wave 7 `3223`, wave 6 `2214`, wave 5 `368`; question wave 7 `C_Q272`, waves 5 and 6 `007_003`;
cross variable wave 7 region ISO `2437884`, wave 6 region `43783`, wave 5 region `1512`.

THE TOOL WEIGHTS Thailand's tables (percent x N is not a whole number, unlike Iran's), so counts
here are weighted, percent x N as floats.

THE SPLIT, per changwat, among respondents who named one of the four Thai varieties (Malay,
Chinese, "other" and "other local" answers are left out: the census counts those languages
itself):
  share_v = (wave 7 weighted count_v + K x zone prior_v) / (wave 7 Thai-variety n + K),  K = 30
  * zone = Thailand's official six regions (North, Northeast, West, East, South, Central with
    Bangkok and the lower north); its prior is its sampled changwat pooled. 28 changwat had no
    wave 7 interviews and take the prior as it is.
  * Bangkok also gets wave 6's Bangkok column (the only wave 5/6 column that is one changwat).
  * Wave 7's Bueng Kan (TH-38) is pooled into Nong Khai (443): the census's Nong Khai is the
    pre-2011 changwat that still held Bueng Kan.
  * RECODED and PRIOR_OUT below: Suphan Buri's "Northern Thai" and Prachuap Khiri Khan's
    one-cluster Esan; reasons there.
OUTPUT data/normalized/th_wvs.csv: unit, zone, n_thai_answers, central, isan, northern, southern
(shares summing to 1).

CHECKS: each column's percentages sum to 100 of its N; per-label national weighted totals within
20% of the IHSN catalogue's unweighted frequencies (they agree within 2% except wave 7's Malay,
weighted 76 against 65); the 76 changwat each in one zone. Printed, not asserted: waves 5 and 6
by NSO region against wave 7 pooled the same way.
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "sources"))
import ir_wvs  # noqa: E402  (reuses its crosstab reader and form chain)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW = HERE / "data" / "raw" / "th"
OUT = HERE / "data" / "normalized" / "th_wvs.csv"
BASE = ir_wvs.BASE
UA = ir_wvs.UA

# key: (file, wave id, sample id, question MAIDX, cross variable)
PAGES = {
    "w7": ("wvs_w7_q272_province.html", "1562", "3223", "C_Q272", "2437884"),
    "w6": ("wvs_w6_v247_region.html", "1", "2214", "007_003", "43783"),
    "w5": ("wvs_w5_v222_region.html", "2", "368", "007_003", "1512"),
}

# IHSN catalogue frequencies (catalog.ihsn.org/catalog/12307 Q272, /9027 V247, /8955 V222),
# unweighted, keyed by the online tool's labels (the DDIs' own: wave 6 "Thai: Esan", wave 5
# "Thai: Northeastern" and so on). Read 2026-10-06.
CEN, ISAN, NTH, STH = "central", "isan", "northern", "southern"
IHSN = {
    "w7": {"Esan": 482, "Malay; Malaysian": 65, "Northern Thai; Lanna": 134,
           "Southern Thai; Dambro; Pak Thai": 132, "Thai; Central Thai": 632, "Other": 48},
    "w6": {"Esan": 403, "Northern Thai; Lanna": 81, "Southern Thai; Dambro; Pak Thai": 106,
           "Thai; Central Thai": 560, "Other": 12},
    "w5": {"Isan; North Eastern Thai": 494, "Malay; Malaysian": 10, "Northern Thai; Lanna": 106,
           "Southern Thai; Dambro; Pak Thai": 175, "Thai; Central Thai": 641, "Other": 101},
}
LABEL_ALIAS = {}
VARIETIES = (CEN, ISAN, NTH, STH)
VARIETY_OF = {
    "Thai; Central Thai": CEN,
    "Esan": ISAN, "Isan; North Eastern Thai": ISAN,
    "Northern Thai; Lanna": NTH,
    "Southern Thai; Dambro; Pak Thai": STH,
}
# Answers not counted as the variety they name. Suphan Buri's wave 7 sample answered "Northern
# Thai; Lanna" 74% (52 interviews, a sampling point or two). Suphan Buri has no Lanna community;
# its non-Central Tai speakers are Lao Khrang, Lao Song (Tai Dam) and Lao Wiang (Mahidol's
# Ethnolinguistic Maps of Thailand), the first of which the census counts in its own row. The
# card had no Lao answer, so those households took the nearest box. Left out of the Thai split.
RECODED = {("272", NTH)}
# Kept for their own changwat but left out of their zone's prior: Prachuap Khiri Khan's 12
# interviews answered 71% Esan and 0% Central Thai, one sampling point that would otherwise put
# 16% Isan on unsampled Tak and Kanchanaburi.
PRIOR_OUT = {"277"}
K = 30          # prior weight, in interviews

# Thailand's official six-region grouping (National Geographical Committee, 1977); a zone's
# sampled changwat, pooled, are the prior for its unsampled and thinly sampled ones.
ZONES = {
    "north": {"350", "351", "352", "353", "354", "355", "356", "357", "358"},
    "northeast": {str(u) for u in range(430, 450) if u != 438},
    "west": {"363", "271", "270", "276", "277"},
    "east": {"220", "221", "222", "223", "224", "225", "227"},
    "south": {"580", "581", "582", "583", "584", "585", "586", "590", "591", "592", "593",
              "594", "595", "596"},
    "central": {"110", "211", "212", "213", "214", "215", "216", "217", "218", "219", "226",
                "272", "273", "274", "275", "360", "361", "362", "364", "365", "366", "367"},
}
# NSO's four regions plus Bangkok, as waves 5 and 6 code them (wave 5 has no Bangkok: its
# Central holds it)
NSO_REGION = {}
for _u in [u for zs in ZONES.values() for u in zs]:
    _i = int(_u)
    NSO_REGION[_u] = ("Bangkok" if _u == "110" else "North" if 350 <= _i < 370
                      else "Northeast" if 430 <= _i < 450 else "South" if _i >= 580
                      else "Central")
REGION_COLS = {
    "North": ("TH: The North", "TH: The North"),
    "Central": ("TH: The Central", "TH: The Central"),
    "Northeast": ("TH: The Northeast", "TH: The Northeast"),
    "South": ("TH: The South", "TH: The South"),
    "Bangkok": (None, "TH: Bangkok"),
}

# wave 7 column label -> religiondots changwat code (th_lookup.csv `unit`)
W7_UNITS = {
    "TH-10 Bangkok": "110", "TH-11 Samut Prakan": "211", "TH-12 Nonthaburi": "212",
    "TH-13 Pathum Thani": "213", "TH-14 Phra Nakhon Si Ayutthaya": "214", "TH-16 Lop Buri": "216",
    "TH-18 Chai Nat": "218", "TH-20 Chon Buri": "220", "TH-22 Chanthaburi": "222",
    "TH-23 Trat": "223", "TH-27 Sa Kaeo": "227", "TH-30 Nakhon Ratchasima": "430",
    "TH-32 Surin": "432", "TH-33 Si Sa Ket": "433", "TH-34 Ubon Ratchathani": "434",
    "TH-35 Yasothon": "435", "TH-37 Amnat Charoen": "437", "TH-38 Bueng Kan": "443",
    "TH-40 Khon Kaen": "440", "TH-41 Udon Thani": "441", "TH-42 Loei": "442",
    "TH-43 Nong Khai": "443", "TH-44 Maha Sarakham": "444", "TH-45 Roi Et": "445",
    "TH-46 Kalasin": "446", "TH-47 Sakon Nakhon": "447", "TH-48 Nakhon Phanom": "448",
    "TH-50 Chiang Mai": "350", "TH-55 Nan": "355", "TH-57 Chiang Rai": "357",
    "TH-58 Mae Hong Son": "358", "TH-60 Nakhon Sawan": "360", "TH-64 Sukhothai": "364",
    "TH-65 Phitsanulok": "365", "TH-66 Phichit": "366", "TH-67 Phetchabun": "367",
    "TH-70 Ratchaburi": "270", "TH-72 Suphan Buri": "272", "TH-73 Nakhon Pathom": "273",
    "TH-76 Phetchaburi": "276", "TH-77 Prachuap Khiri Khan": "277",
    "TH-80 Nakhon Si Thammarat": "580", "TH-82 Phangnga": "582", "TH-83 Phuket": "583",
    "TH-84 Surat Thani": "584", "TH-86 Chumphon": "586", "TH-90 Songkhla": "590",
    "TH-94 Pattani": "594", "TH-96 Narathiwat": "596",
}


def fetch():
    import requests
    import urllib3
    urllib3.disable_warnings()
    RAW.mkdir(parents=True, exist_ok=True)
    for key, (name, wave, said, maidx, x1) in PAGES.items():
        s = requests.Session()
        s.verify = False                     # the site omits an intermediate certificate
        s.headers["User-Agent"] = UA
        s.get(BASE + "WVSOnline.jsp", timeout=60)
        s.get(BASE + "AJOnline.jsp?WAVE=&COUNTRY=", timeout=60)
        form = {"ulthost": "WVS", "CMSID": "", "WAVE": wave, "MAIDX": "", "SAIDS": said,
                "AMIDS": "764", "SATITULOS": "Thailand", "COUNTRY": "", "CRUCEX": ""}
        s.post(BASE + "AJOnlineCountries.jsp", data=form, timeout=60)
        s.post(BASE + "AJOnlineIndex.jsp", data=form, timeout=60)
        form["MAIDX"] = maidx
        s.post(BASE + "AJOnlineQtn.jsp", data=form, timeout=120)
        form2 = {"ulthost": "WVS", "CMSID": "", "WAVE": wave, "SAIDS": said,
                 "SATITULOS": "Thailand", "AMIDS": "764", "MAIDX": maidx, "MACRUCE1": x1,
                 "MACRUCE2": "", "CRUCES_ROTARXY": "", "CRUCE_TYPE": "TAB",
                 "AJArchive": "WVS Data Archive"}
        r = s.post(BASE + "AJOnlineQtn.jsp", data=form2, timeout=120)
        r.raise_for_status()
        if "JDSTableCellHeader" not in r.text:
            raise SystemExit(f"{name}: no crosstab in the response")
        tmp = (RAW / name).with_suffix(".part")
        tmp.write_text(r.text, encoding="utf-8")
        tmp.replace(RAW / name)
        print(f"  {name}: {len(r.text):,} chars")


def read(key):
    """{column: {label: weighted count}}. The tool prints WEIGHTED column percentages for Thailand
    (unlike Iran's, percent x N is not a whole number), so counts are percent x N as floats."""
    name = PAGES[key][0]
    out = {}
    for filt, cols, data, ns in ir_wvs.tables(RAW / name):
        if filt is not None or ns is None or len(ns) != len(cols):
            raise SystemExit(f"{name}: unexpected table {filt} {cols} {ns}")
        for j, col in enumerate(cols):
            if col in ("TOTAL", "No answer"):
                continue
            got = {lab: float(c[j].rstrip("%")) * ns[j] / 100.0
                   for lab, c in data.items() if c[j] not in ("-", "")}
            if abs(sum(got.values()) - ns[j]) > 0.002 * ns[j] + 0.1:
                raise SystemExit(f"{name} {col}: percentages sum to {sum(got.values()):.2f} "
                                 f"of N {ns[j]}")
            out[col] = {k: v for k, v in got.items() if k not in ir_wvs.NOT_DRAWN}
    return out


def zone_of(unit):
    for z, units in ZONES.items():
        if unit in units:
            return z
    raise SystemExit(f"unit {unit} in no zone")


def main():
    if "--fetch" in sys.argv:
        fetch()
    waves = {k: read(k) for k in PAGES}

    # 1. national totals against the IHSN catalogue's unweighted frequencies (weights move them
    #    a little; a wrong page or a relabelled answer moves them a lot)
    for key, cols in waves.items():
        tot = {}
        for got in cols.values():
            for lab, n in got.items():
                tot[lab] = tot.get(lab, 0) + n
        for lab, want in IHSN[key].items():
            have = tot.get(LABEL_ALIAS.get(lab, lab), 0)
            flag = "" if abs(have - want) <= max(12, 0.2 * want) else "  <-- off"
            print(f"  {key} {lab:34s} weighted {have:7.1f}  IHSN {want:4d}{flag}")
            if flag:
                raise SystemExit(f"{key} {lab}: weighted total far from the IHSN frequency")

    # 2. Thai-variety answers per changwat: wave 7, plus wave 6's Bangkok column for Bangkok
    w7 = {}
    for col, got in waves["w7"].items():
        if col not in W7_UNITS:
            raise SystemExit(f"w7 column {col!r} has no unit")
        u = W7_UNITS[col]
        d = w7.setdefault(u, dict.fromkeys(VARIETIES, 0.0))
        for lab, n in got.items():
            v = VARIETY_OF.get(lab)
            if v and (u, v) not in RECODED:
                d[v] += n
    for lab, n in waves["w6"]["TH: Bangkok"].items():
        v = VARIETY_OF.get(lab)
        if v:
            w7["110"][v] += n
    missing = set(W7_UNITS.values()) - set(w7)
    if missing:
        raise SystemExit(f"wave 7 provinces missing: {missing}")

    # 3. zone priors: the zone's sampled changwat pooled, PRIOR_OUT left out
    prior = {}
    for z, units in ZONES.items():
        p = dict.fromkeys(VARIETIES, 0.0)
        for u in units:
            if u in w7 and u not in PRIOR_OUT:
                for v in VARIETIES:
                    p[v] += w7[u][v]
        tot = sum(p.values())
        prior[z] = {v: p[v] / tot for v in VARIETIES}
        print(f"  zone {z:9s} n {tot:6.1f}  " + "  ".join(
            f"{v} {prior[z][v]:.3f}" for v in VARIETIES))

    # 4. per changwat: (wave 7 counts + K x zone prior) / (n + K)
    units = [u for zs in ZONES.values() for u in zs]
    if len(units) != 76 or len(set(units)) != 76:
        raise SystemExit(f"zones hold {len(units)} units, {len(set(units))} distinct")
    rows = []
    for u in sorted(units):
        z = zone_of(u)
        d = w7.get(u, dict.fromkeys(VARIETIES, 0.0))
        n = sum(d.values())
        sh = {v: (d[v] + K * prior[z][v]) / (n + K) for v in VARIETIES}
        rows.append({"unit": u, "zone": z, "n_thai_answers": round(n, 2),
                     **{v: round(sh[v], 5) for v in VARIETIES}})
    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(OUT)
    print(f"wrote {OUT.relative_to(HERE)}: {len(rows)} changwat")

    # 5. corroboration: waves 5 and 6 by region against wave 7 pooled the same way
    print("  region check, share of Thai-variety answers (w5 / w6 / w7):")
    w7reg = {}
    for col, got in waves["w7"].items():
        r = NSO_REGION[W7_UNITS[col]]
        d = w7reg.setdefault(r, dict.fromkeys(VARIETIES, 0.0))
        for lab, n in got.items():
            v = VARIETY_OF.get(lab)
            if v:
                d[v] += n
    for r, (c5, c6) in REGION_COLS.items():
        line = []
        for d in (waves["w5"].get(c5) if c5 else None, waves["w6"].get(c6), None):
            if d is None:
                continue
            dd = dict.fromkeys(VARIETIES, 0.0)
            for lab, n in d.items():
                v = VARIETY_OF.get(lab)
                if v:
                    dd[v] += n
            t = sum(dd.values())
            line.append("/".join(f"{dd[v] / t:.2f}" for v in VARIETIES))
        t = sum(w7reg[r].values())
        line.append("/".join(f"{w7reg[r][v] / t:.2f}" for v in VARIETIES))
        print(f"    {r:10s} " + "   ".join(line) + f"   ({'/'.join(VARIETIES)})")


if __name__ == "__main__":
    main()
