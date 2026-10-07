"""Papua New Guinea: a modelled church mix for each of the 22 provinces, from what the census printed.

Writes data/normalized/pg.csv. `sources/pg.md` is the record; sources.md §11ab and its 2026-09-09
re-check are why there is no census table to read instead.

PNG HAS NEVER PUBLISHED RELIGION BY PROVINCE AS A TABLE (§11ab swept the office's 287 files). What
it did publish, in the Summary Indicators of the 2011 and 2000 National Reports, is ONE line per
province: the largest church there and its share ("Main religion (% of population)"). The build
uses every number the census printed and invents the rest by the least-informative rule that
reproduces them:

  1. National, 2011 (Table 2.4 and Figure 2.1, p33): Christian 95.6%, Non-Christian 1.4%, No
     religion 0.0%, Not stated 3.1%; and the eleven churches as shares of Christians (they sum to
     100.2, so they are shares of Christians and not of the population, though the text says
     "of the population"). Applied to every province alike: nothing printed varies them.
  2. Per province, 2011 (Summary Indicators, pp28-29): the largest church at its printed share,
     read as a share of Christians like the national row it matches (national R/Cath 26.0 =
     Figure 2.1's 26.0).
  3. The rest of each province's Christians is split over the other ten churches by iterative
     proportional fitting, so that every church's national total matches Figure 2.1 on the 2011
     census populations, and no church other than the largest exceeds the largest's share.
  4. Two outside patterns seed the fit (they move people between provinces; the census margins
     still fix every total):
       * Catholics: each diocese's Catholic share in the Annuario Pontificio's 2004 figures as
         published by catholic-hierarchy.org (`DIOCESES`), over the national rate of the same
         figures. Most dioceses are one province.
       * The 2000 census's largest church, where it is not 2011's: New Ireland's United Church
         (40.3% in 2000) and Southern Highlands' Other Christian (22.3% in 2000; Hela was part
         of it then), each over its 2000 national share.
  5. The shares go onto the 2024 census count of each province (Final Figures, Table 2).

Every row is `modelled` (spec §7b): only the largest church in each province, and the national
shares, were ever counted, and none at the 2024 population.

Usage:
    python sources/pg.py --fetch    the three NSO PDFs and the catholic-hierarchy page into data/raw/pg/
    python sources/pg.py            rebuild from data/raw/pg/
"""

import csv
import os
import re
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "pg")
OUT = os.path.join(ROOT, "data", "normalized", "pg.csv")

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
FILES = {
    "png_national_report_2011.pdf":
        "https://www.nso.gov.pg/download/51/population-housing/2152/png-national-report-2011-census.pdf",
    "png_national_report_2000.pdf":
        "https://www.nso.gov.pg/download/51/population-housing/2151/png-national-report-2000-census.pdf",
    "png_2024_census_final_figures.pdf":
        "https://www.nso.gov.pg/download/51/population-housing/4310/2024-national-population-census-final-figures.pdf",
    "catholic_hierarchy_scpg1.html":
        "https://www.catholic-hierarchy.org/country/scpg1.html",
}

# ---- the 22 provinces: COD p-code, name in the 2024 Final Figures' Table 2, short label ----
PROVINCES = [
    ("PG01", "Western", "Western"),
    ("PG02", "Gulf", "Gulf"),
    ("PG03", "Central", "Central"),
    ("PG04", "National Capital District", "NCD"),
    ("PG05", "Milne Bay", "Milne Bay"),
    ("PG06", "Northern", "Northern (Oro)"),
    ("PG07", "Southern Highlands", "Southern Highlands"),
    ("PG08", "Enga", "Enga"),
    ("PG09", "Western Highlands", "Western Highlands"),
    ("PG10", "Chimbu", "Chimbu (Simbu)"),
    ("PG11", "Eastern Highlands", "Eastern Highlands"),
    ("PG12", "Morobe", "Morobe"),
    ("PG13", "Madang", "Madang"),
    ("PG14", "East Sepik", "East Sepik"),
    ("PG15", "West Sepik", "West Sepik (Sandaun)"),
    ("PG16", "Manus", "Manus"),
    ("PG17", "New Ireland", "New Ireland"),
    ("PG18", "East New Britain", "East New Britain"),
    ("PG19", "West New Britain", "West New Britain"),
    ("PG20", "Autonomous Region of Bougainville", "Bougainville"),
    ("PG21", "Hela", "Hela"),
    ("PG22", "Jiwaka", "Jiwaka"),
]
TOTAL_2024 = 10_185_363
TOTAL_2011 = 7_275_324          # Table 2's 2011 column, summed; equals COD-PS 2011 (asserted in pg_geo.py)

# ---- 2011 National Report, Table 2.4 (p33): citizens in private dwellings, % ----
TOP_2011 = {"Christian": 95.6, "Non-Christian": 1.4, "No Religion": 0.0, "Not Stated": 3.1}

# ---- 2011 National Report, Figure 2.1 (p33): % of Christians (sums to 100.2) ----
FIG_2011 = {
    "Roman Catholic": 26.0, "Evangelical Lutheran": 18.4, "Seventh Day Adventist": 12.9,
    "Pentecostals": 10.4, "United Church": 10.3, "Other Christian": 9.7,
    "Evangelical Alliance": 5.9, "Anglican": 3.2, "Baptist": 2.8, "Salvation Army": 0.4,
    "Kwato Church": 0.2,
}
BODIES = list(FIG_2011)

# the Summary Indicators' abbreviations
ABBR = {"R/Cath.": "Roman Catholic", "Evan.Luth": "Evangelical Lutheran", "SDA": "Seventh Day Adventist",
        "U/Church": "United Church", "Evan.All.": "Evangelical Alliance", "Anglican": "Anglican",
        "O.Christ": "Other Christian", "United": "United Church"}

# ---- 2011 National Report, Summary Indicators, Provinces (pp28-29): Main religion, Total ----
MAIN_2011 = {
    "PG01": ("Evan.All.", 37.1), "PG02": ("U/Church", 30.1), "PG03": ("U/Church", 40.0),
    "PG04": ("U/Church", 23.0), "PG05": ("U/Church", 54.9), "PG06": ("Anglican", 60.6),
    "PG07": ("R/Cath.", 19.7), "PG08": ("Evan.Luth", 26.4), "PG09": ("Evan.Luth", 26.0),
    "PG10": ("R/Cath.", 34.4), "PG11": ("SDA", 39.6), "PG21": ("Evan.All.", 19.7),
    "PG22": ("R/Cath.", 29.6), "PG12": ("Evan.Luth", 67.0), "PG13": ("Evan.Luth", 38.4),
    "PG14": ("R/Cath.", 43.0), "PG15": ("R/Cath.", 40.4), "PG16": ("R/Cath.", 38.5),
    "PG17": ("R/Cath.", 31.3), "PG18": ("R/Cath.", 42.8), "PG19": ("R/Cath.", 55.3),
    "PG20": ("R/Cath.", 68.4),
}
# page and the order the values appear in its text layer, for the check against the PDF
MAIN_2011_PAGES = {28: ["PG01", "PG02", "PG03", "PG04", "PG05", "PG06",
                        "PG07", "PG08", "PG09", "PG10", "PG11", "PG21", "PG22"],
                   29: ["PG12", "PG13", "PG14", "PG15", "PG16", "PG17", "PG18", "PG19", "PG20"]}

# ---- 2000 National Report, Summary Indicators, Provinces (pp25-26, a scan; read at 200 dpi) ----
# Hela was part of Southern Highlands and Jiwaka of Western Highlands in 2000.
MAIN_2000 = {
    "PG01": ("Evan.All.", 37.5), "PG02": ("U/Church", 37.8), "PG03": ("U/Church", 42.7),
    "PG04": ("U/Church", 30.4), "PG05": ("U/Church", 60.9), "PG06": ("Anglican", 61.5),
    "PG07": ("O.Christ", 22.3), "PG21": ("O.Christ", 22.3), "PG08": ("Evan.Luth", 30.1),
    "PG09": ("R/Cath.", 31.6), "PG22": ("R/Cath.", 31.6), "PG10": ("R/Cath.", 35.9),
    "PG11": ("SDA", 36.6), "PG12": ("Evan.Luth", 71.6), "PG13": ("Evan.Luth", 38.2),
    "PG14": ("R/Cath.", 44.8), "PG15": ("R/Cath.", 47.4), "PG16": ("R/Cath.", 45.7),
    "PG17": ("United", 40.3), "PG18": ("R/Cath.", 51.1), "PG19": ("R/Cath.", 56.5),
    "PG20": ("R/Cath.", 69.5),
}
# 2000 national shares (of the population) for the two seeded bodies: United Church 12% from the
# text on p31; Other Christian 8.2% measured off Figure 2.1's bar in pixels (18 px a point at 170
# dpi; the bars carry no labels in 2000).
NATIONAL_2000 = {"United Church": 12.0, "Other Christian": 8.2}

# ---- Catholic dioceses (catholic-hierarchy.org scpg1, Annuario Pontificio 2005, data for 2004) ----
# diocese -> provinces it covers (CBC PNG/SI's diocese pages; Port Moresby's 2004 population only
# fits with Oro inside it: NCD 248,948 + Central outside Bereina ~100,000 + Oro 132,952 in 2000,
# against 512,386). Central is split between Bereina (Kairuku and Goilala) and Port Moresby.
DIOCESES = {
    "Aitape": ["PG15"], "Vanimo": ["PG15"], "Wewak": ["PG14"], "Madang": ["PG13"], "Lae": ["PG12"],
    "Rabaul": ["PG18"], "Kimbe": ["PG19"], "Kavieng": ["PG17", "PG16"], "Bougainville": ["PG20"],
    "Goroka": ["PG11"], "Kundiawa": ["PG10"], "Mount Hagen": ["PG09", "PG22"], "Wabag": ["PG08"],
    "Mendi": ["PG07", "PG21"], "Daru-Kiunga": ["PG01"], "Kerema": ["PG02"],
    "Port Moresby": ["PG04", "PG06"], "Alotau-Sideia": ["PG05"],
}
CENTRAL_2000 = 183_805          # 2000 National Report p4 (summary), Central's citizens
CAP_MARGIN = 0.1                # a non-largest church stays this many points under the largest


def fetch():
    import requests
    os.makedirs(RAW, exist_ok=True)
    for name, url in FILES.items():
        dst = os.path.join(RAW, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 5_000:
            print(f"  have {name} ({os.path.getsize(dst):,} bytes)")
            continue
        r = requests.get(url, headers={"User-Agent": UA}, timeout=600)
        r.raise_for_status()
        if name.endswith(".pdf") and r.content[:4] != b"%PDF":
            raise SystemExit(f"{url} did not return a PDF")
        with open(dst + ".part", "wb") as fh:
            fh.write(r.content)
        os.replace(dst + ".part", dst)
        print(f"  got  {name} ({len(r.content):,} bytes)")


def pdf_page(name, page):
    import fitz
    with fitz.open(os.path.join(RAW, name)) as doc:
        return doc[page - 1].get_text()


def check_2011_text():
    """Every transcribed 2011 figure must be in the page's own text layer, in order."""
    t = pdf_page("png_national_report_2011.pdf", 33)
    for k, v in TOP_2011.items():
        if not re.search(rf"{re.escape(k)}\s+{v:.1f}\b", t):
            raise SystemExit(f"Table 2.4: {k} {v} not on p33")
    nums = re.findall(r"^\s*(\d+\.\d)\s*$", t[t.index("(See Summary Indicators)"):], re.M)
    labels = re.findall(r"^\s{4,}(\S.*?)\s*$", t, re.M)
    labels = [x for x in labels if x in FIG_2011]
    got = dict(zip(labels, nums[:len(labels)]))
    if {k: float(v) for k, v in got.items()} != FIG_2011:
        raise SystemExit(f"Figure 2.1 text layer {got} != transcription")
    print(f"  Table 2.4 and Figure 2.1 match p33's text layer (11 churches, sum {sum(FIG_2011.values()):.1f})")
    for page, order in MAIN_2011_PAGES.items():
        s = " ".join(pdf_page("png_national_report_2011.pdf", page).split())
        for blk in ([order[:6], order[6:]] if page == 28 else [order[:4], order[4:]]):
            want = [MAIN_2011[u][1] for u in blk]
            labs = [MAIN_2011[u][0] for u in blk]
            # the labels print with or without a final dot (Hela's `Evan.All`, ESP's `R/Cath`)
            pat = r"\s+".join(re.escape(x.rstrip(".")) + r"\.?" for x in labs)
            m = re.search(pat, s)
            if not m:
                raise SystemExit(f"p{page}: labels {labs} not found in order")
            after = s[m.end():]                # the Islands block has its heading before the labels
            nums = re.findall(r"\d+\.\d", after.split("Main religion (% of population)", 1)[-1])[:len(blk)]
            if [float(x) for x in nums] != want:
                raise SystemExit(f"p{page}: {labs} read {nums}, transcribed {want}")
            s = s[m.end():]
    print("  the 22 provinces' Main religion labels and Totals match pp28-29's text layer")


def read_2024():
    t = pdf_page("png_2024_census_final_figures.pdf", 8)
    s = " ".join(t.split())
    pop11, pop24 = {}, {}
    for pc, name, _ in PROVINCES:
        key = "Autonomous Region of Bougainville" if pc == "PG20" else name
        m = re.search(rf"{re.escape(key)}\s+([\d,]+)\s+([\d,]+)\s+\d+\.\d", s)
        if not m:
            raise SystemExit(f"2024 Table 2: no row for {key}")
        pop11[pc], pop24[pc] = (int(m.group(i).replace(",", "")) for i in (1, 2))
    if sum(pop24.values()) != TOTAL_2024 or sum(pop11.values()) != TOTAL_2011:
        raise SystemExit(f"Table 2 sums {sum(pop11.values()):,} / {sum(pop24.values()):,}")
    print(f"  2024 Final Figures Table 2: 22 provinces, 2011 {TOTAL_2011:,}, 2024 {TOTAL_2024:,}")
    return pop11, pop24


def read_dioceses():
    t = open(os.path.join(RAW, "catholic_hierarchy_scpg1.html"), encoding="utf-8", errors="replace").read()
    t = re.sub(r"<[^>]+>", "|", t)
    rows = {}
    for m in re.finditer(r"\|([\d,]+)\|+([\d,]+)\|+([\d.]+)%\|+([A-Za-z\- ]+?)( \((?:Arch)?[Dd]iocese\))?\|", t):
        rows[m.group(4).strip()] = (int(m.group(1).replace(",", "")), int(m.group(2).replace(",", "")))
    if set(rows) != set(DIOCESES) | {"Bereina"}:
        raise SystemExit(f"catholic-hierarchy dioceses {sorted(rows)} != expected")
    return rows


def catholic_seed(dio):
    cath = sum(c for c, _ in dio.values())
    pop = sum(p for _, p in dio.values())
    nat = cath / pop
    share = {}
    for d, provs in DIOCESES.items():
        for pc in provs:
            share.setdefault(pc, []).append(d)
    out = {}
    for pc, ds in share.items():
        c = sum(dio[d][0] for d in ds)
        p = sum(dio[d][1] for d in ds)
        out[pc] = c / p
    # Central: Bereina on its own 2004 population, the archdiocese's rate on the rest of Central
    b_c, b_p = dio["Bereina"]
    pm_c, pm_p = dio["Port Moresby"]
    out["PG03"] = (b_c + pm_c / pm_p * max(CENTRAL_2000 - b_p, 0)) / CENTRAL_2000
    print(f"\n  dioceses: {len(dio)}, {cath:,} Catholics of {pop:,} ({nat:.1%}), 2004")
    return {pc: v / nat for pc, v in out.items()}, out


def fit(pop11, cseed):
    P = [pc for pc, _, _ in PROVINCES]
    B = BODIES
    C = np.array([pop11[p] * TOP_2011["Christian"] / sum(TOP_2011.values()) for p in P])
    fig = np.array([FIG_2011[b] for b in B]) / sum(FIG_2011.values())
    T = fig * C.sum()
    main = np.array([B.index(ABBR[MAIN_2011[p][0]]) for p in P])
    m = np.array([MAIN_2011[p][1] / 100.0 for p in P])
    fixed = np.zeros((len(P), len(B)))
    fixed[np.arange(len(P)), main] = m * C
    R = C - fixed.sum(1)
    K = T - fixed.sum(0)
    for j, b in enumerate(B):
        if K[j] <= 0:
            raise SystemExit(f"{b}: the provinces where it is largest already hold more than its "
                             f"national total ({K[j]:,.0f})")
    seed = np.ones((len(P), len(B)))
    seed[:, B.index("Roman Catholic")] = [cseed[p] for p in P]
    for i, p in enumerate(P):
        b0 = ABBR[MAIN_2000[p][0]]
        if b0 != ABBR[MAIN_2011[p][0]] and b0 != "Roman Catholic":
            seed[i, B.index(b0)] = MAIN_2000[p][1] / NATIONAL_2000[b0]
            print(f"  seed: {p} {b0} x{seed[i, B.index(b0)]:.2f} (the 2000 census's largest church)")
    seed[np.arange(len(P)), main] = 0.0
    cap = np.maximum(m - CAP_MARGIN / 100.0, 0) * C
    X = seed.copy()
    frozen = np.zeros_like(X, dtype=bool)
    for it in range(5000):
        for axis in (1, 0):
            tgt = R if axis == 1 else K
            fz = np.where(frozen, X, 0).sum(axis)
            fr = np.where(frozen, 0, X).sum(axis)
            f = np.divide(tgt - fz, fr, out=np.ones_like(fr), where=fr > 0)
            X = np.where(frozen, X, X * (f[:, None] if axis == 1 else f[None, :]))
            over = (~frozen) & (X > cap[:, None])
            if over.any():
                X[over] = np.broadcast_to(cap[:, None], X.shape)[over]
                frozen |= over
        err = max(np.abs(X.sum(1) - R).max() / R.min(), np.abs(X.sum(0) - K).max() / K.min())
        if err < 1e-10:
            break
    else:
        raise SystemExit(f"the fit did not converge (margin error {err:.2e})")
    print(f"  fit converged in {it + 1} sweeps; {int(frozen.sum())} cells held at the largest church's share")
    full = X + fixed
    return P, B, full, C, frozen


def main():
    if "--fetch" in sys.argv or not all(os.path.exists(os.path.join(RAW, n)) for n in FILES):
        fetch()
    check_2011_text()
    pop11, pop24 = read_2024()
    dio = read_dioceses()
    cseed, dshare = catholic_seed(dio)
    P, B, full, C, frozen = fit(pop11, cseed)
    name = {pc: lab for pc, _, lab in PROVINCES}

    # shares of each province's whole population
    tot = sum(TOP_2011.values())
    rows = []
    print("\n  modelled share of each province's people (2011 fit), largest church starred:")
    hdr = "".join(f"{b[:9]:>10}" for b in B)
    print(f"      {'province':<20}{hdr}")
    for i, p in enumerate(P):
        sh = {b: full[i, j] / pop11[p] for j, b in enumerate(B)}
        sh["Non-Christian"] = TOP_2011["Non-Christian"] / tot
        sh["No Religion"] = TOP_2011["No Religion"] / tot
        sh["Not Stated"] = TOP_2011["Not Stated"] / tot
        if abs(sum(sh.values()) - 1) > 1e-9:
            raise SystemExit(f"{p}: shares sum to {sum(sh.values())}")
        mainb = ABBR[MAIN_2011[p][0]]
        cells = "".join(f"{100 * sh[b]:>9.1f}{'*' if b == mainb else ' '}" for b in B)
        print(f"      {name[p]:<20}{cells}")
        # largest-remainder rounding onto the 2024 count
        cats = list(sh)
        raw = np.array([sh[c] * pop24[p] for c in cats])
        n = np.floor(raw).astype(int)
        n[np.argsort(-(raw - n))[: pop24[p] - n.sum()]] += 1
        for c, k in zip(cats, n):
            rows.append((p, name[p], c, int(k)))

    print("\n  witness, Catholics: modelled share of Christians against the diocese's 2004 share")
    j = B.index("Roman Catholic")
    for i, p in enumerate(P):
        print(f"      {name[p]:<20} model {100 * full[i, j] / C[i]:5.1f}   diocese {100 * dshare[p]:5.1f}"
              f"{'   (census, largest church)' if ABBR[MAIN_2011[p][0]] == 'Roman Catholic' else ''}")

    print("\n  national shares on the 2024 counts against Figure 2.1 (2011), % of Christians:")
    chr24 = sum(k for p, _, c, k in rows if c in FIG_2011)
    for b in B:
        n = sum(k for p, _, c, k in rows if c == b)
        print(f"      {b:<22} {100 * n / chr24:5.1f}   2011 {FIG_2011[b] * 100 / sum(FIG_2011.values()):5.1f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT + ".part", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["geo_id", "geo_level", "geo_name", "source_category", "count", "basis", "year",
                    "source_id", "note"])
        for p, nm, c, k in rows:
            w.writerow([p, "province", nm, c, k, "self_id", "2011", "pg_census2011_main_religion_fit",
                        "2011 census: national shares and the province's largest church printed; the "
                        "rest fitted (sources/pg.py), on the 2024 census count"])
    os.replace(OUT + ".part", OUT)
    print(f"\nwrote {OUT}: {len(rows)} rows, {sum(k for *_, k in rows):,} people")


if __name__ == "__main__":
    main()
