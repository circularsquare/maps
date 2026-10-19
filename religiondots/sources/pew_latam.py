"""Pew Research Center, *Religion in Latin America* (2014): Protestant composition per country.

Reads the .sav in data/raw/pew_latam/ and writes data/normalized/pew_latam.csv, one row per
country: the number of Protestants interviewed and the weighted share of them in each family.
The build reads only the CSV, through `latam_split.py` at the religiondots root, so nothing
downstream needs pyreadstat. Why this exists and what it is used for:
unspecified_splits/latam.md, and Anita's ruling of 2026-10-16 recorded in `latam_split.py`.

THE FILE. Fieldwork October 2013 to February 2014, face to face, 18 countries plus Puerto Rico,
adults, 1,500 to 2,000 interviews each and 116 to 638 Protestants a country. Pew's own download
needs an account; `--fetch` takes the open mirror on opendata.com.pk instead (the dataset and
its codebook in one zip, 6,235,278 bytes). `check()` asserts that the file reproduces Pew's
published chart "% of Protestants who belong to a Pentecostal denomination" (report p.63) to
within 0.6 points for all nineteen, which is what shows the mirror is Pew's file.

THE QUESTION. `QPROT`, put to every Protestant: "Do you belong to (1) a historical Protestant
church, for example Baptist, Seventh Day Adventist, Methodist, Lutheran or Presbyterian, (2) a
Pentecostal church, for example Assemblies of God, or (3) another Protestant church?" Then
`QDENOM<cc>` for answers 1 and 3, read from a country card. Pentecostal is QPROT=2, the same
question in every country. Baptist is QDENOM=2, which is on most cards and absent from
Colombia's, Ecuador's, Paraguay's and Uruguay's (a family off the card is written as empty, not
as zero). Pew's "Protestant" leaves Jehovah's Witnesses and Latter-day Saints out, as every
source this is applied to does.

THE PUBLIC FILE HAS NO GEOGRAPHY (its only non-question columns are Country, CASEID and
WEIGHT), so these are national shares.

Usage:
    python sources/pew_latam.py --fetch   # download and unzip the mirror, then build
    python sources/pew_latam.py           # rebuild the CSV from the .sav and run the checks
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "pew_latam"
ZIP = RAW / "religion-in-latin-america-dataset-and-codebook.zip"
SAV = RAW / "Religion in Latin America Dataset.sav"
OUT = HERE / "data" / "normalized" / "pew_latam.csv"
MIRROR = ("https://opendata.com.pk/dataset/0e281ef2-e927-4058-b029-4606fcbf85d6/resource/"
          "e13b64f4-5787-4741-a04e-2b331c734500/download/"
          "religion-in-latin-america-dataset-and-codebook.zip")
ZIP_BYTES = 6_235_278

# Pew's `Country` value label -> this map's country code, and the suffix of its card variables.
COUNTRY = {
    "Argentina": ("ar", "ARG"), "Bolivia": ("bo", "BOL"), "Brazil": ("br", "BRA"),
    "Chile": ("cl", "CHI"), "Colombia": ("co", "COL"), "Costa Rica": ("cr", "CRI"),
    "Dominican Republic": ("do", "DOM"), "Ecuador": ("ec", "ECU"), "El Salvador": ("sv", "ESA"),
    "Guatemala": ("gt", "GUA"), "Honduras": ("hn", "HON"), "Mexico": ("mx", "MEX"),
    "Nicaragua": ("ni", "NIC"), "Panama": ("pa", "PAN"), "Paraguay": ("py", "PAR"),
    "Peru": ("pe", "PER"), "Puerto Rico": ("pr", "PUE"), "Uruguay": ("uy", "URU"),
    "Venezuela": ("ve", "VEN"),
}
# Pew's published chart, report p.63, "% of Protestants who belong to a Pentecostal denomination".
PUBLISHED_PENT = {"do": 72, "br": 60, "pa": 58, "ni": 61, "gt": 57, "ar": 62, "hn": 48, "ec": 56,
                  "cl": 45, "pr": 50, "cr": 46, "py": 44, "sv": 46, "mx": 46, "ve": 39, "co": 47,
                  "uy": 42, "pe": 41, "bo": 43}
PUBLISHED_TOL = 0.6
QPROT_PENT = 2
QDENOM = {"baptist": 2, "methodist": 3, "presbyterian": 4, "adventist": 6}
QDENOM_LABEL = {"baptist": "Baptist", "methodist": "Methodist",
                "presbyterian": "Presbyterian/Reformed", "adventist": "Seventh Day Adventist"}


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    if not (ZIP.exists() and ZIP.stat().st_size == ZIP_BYTES):
        req = urllib.request.Request(MIRROR, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=300).read()
        if len(data) != ZIP_BYTES:
            raise SystemExit(f"mirror returned {len(data):,} bytes, expected {ZIP_BYTES:,}")
        tmp = ZIP.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(ZIP)
        print(f"  {len(data):,} bytes -> {ZIP}")
    with zipfile.ZipFile(ZIP) as z:
        for n in z.namelist():
            if Path(n).name != n or n.startswith("."):
                raise SystemExit(f"unexpected path in zip: {n!r}")
        z.extractall(RAW)


def build():
    import pyreadstat
    if not SAV.exists():
        raise SystemExit(f"{SAV} missing -- run python sources/pew_latam.py --fetch")
    d, meta = pyreadstat.read_sav(str(SAV))
    clab = meta.value_labels[meta.variable_to_label["Country"]]
    plab = meta.value_labels[meta.variable_to_label["QPROT"]]
    if plab.get(float(QPROT_PENT)) != "a Pentecostal Church":
        raise SystemExit(f"QPROT {QPROT_PENT} is {plab.get(float(QPROT_PENT))!r}")
    rlab = meta.value_labels[meta.variable_to_label["QCURRELrec"]]
    if rlab.get(2.0) != "Protestant":
        raise SystemExit(f"QCURRELrec 2 is {rlab.get(2.0)!r}")
    rows = []
    for code, name in clab.items():
        cc, suf = COUNTRY[name]
        x = d[d["Country"] == code]
        p = x[x["QCURRELrec"] == 2]
        w = p["WEIGHT"]
        W = float(w.sum())
        r = {"cc": cc, "country": name, "n_all": len(x), "n_prot": len(p),
             "prot_share_pop": float(W / x["WEIGHT"].sum()),
             "pentecostal": float(w[p["QPROT"] == QPROT_PENT].sum() / W)}
        var = f"QDENOM{suf}rec"
        labs = meta.value_labels.get(meta.variable_to_label.get(var), {})
        for fam, k in QDENOM.items():
            lab = labs.get(float(k))
            if lab not in (None, QDENOM_LABEL[fam]):
                raise SystemExit(f"{cc} {var} code {k} is {lab!r}, expected {QDENOM_LABEL[fam]!r}")
            r[fam] = float(w[p[var] == k].sum() / W) if lab else None
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("cc")
    check(t)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False, float_format="%.6f")
    print(f"wrote {OUT} ({len(t)} countries)")
    return t


def check(t):
    bad = []
    for _, r in t.iterrows():
        pub = PUBLISHED_PENT[r["cc"]]
        got = 100 * r["pentecostal"]
        off = abs(got - pub) > PUBLISHED_TOL
        print(f"  {r['cc']}  Pentecostal {got:5.1f}% of {r['n_prot']:>3} Protestants; "
              f"Pew published {pub}%" + ("  <-- OFF" if off else ""))
        if off:
            bad.append(r["cc"])
    if bad:
        raise SystemExit(f"the file does not reproduce Pew's published chart for {bad}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if "--fetch" in sys.argv:
        fetch()
    build()
