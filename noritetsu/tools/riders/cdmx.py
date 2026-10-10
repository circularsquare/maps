"""Mexico City: Metro CDMX daily entries per station ("Afluencia diaria del Metro CDMX").

NOT YET RUN. datos.cdmx.gob.mx (and every other cdmx.gob.mx host) refused or timed out every
connection on 2026-10-09, from this machine and from a second network, so the file has not been
seen. Until a file is in data/raw/riders/cdmx/ this module is skipped (READY). Download the
"afluenciastc_simple" CSV(s) from
    https://datos.cdmx.gob.mx/dataset/afluencia-diaria-del-metro-cdmx
into data/raw/riders/cdmx/ and run `python tools/station_riders.py cdmx`; then read
data/raw/riders/_reports/cdmx.txt, since the column names and station spellings below are
from memory of the dataset, not from the file.

Expected columns (any case; "año" or "anio"): fecha, anio, linea, estacion, afluencia; one row
per day, line and station. afluencia is entries through the turnstiles (the Metro counts
people coming in, not going out), so n = 2 x the mean daily entries, for on + off. The latest
year with all twelve months is used; the mean is over the days that station has rows for.
A transfer station is one row per line; the lines' rows land on one station id where the map
has one id for the complex and are added (COMBINE sum), and stay apart where the map splits it
("Candelaria L1" / "Candelaria L4"). match_hook matches each row only to stations of its own
line (by the line's ref: "Línea 1" -> "1", "Línea A" -> "A") inside Mexico City.
"""
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from station_riders import name_score, name_variants, squash

KEY = "cdmx"
CC = "mx"
FOLDER = "cdmx"
COMBINE = "sum"
MODES = {"metro"}
META = {
    "label": "Metro CDMX entries",
    "name": "Afluencia diaria del Metro CDMX, Sistema de Transporte Colectivo Metro "
            "(Portal de Datos Abiertos de la Ciudad de México)",
    "url": "https://datos.cdmx.gob.mx/dataset/afluencia-diaria-del-metro-cdmx",
    "licence": "Datos abiertos de la Ciudad de México (CC BY 4.0)",
    "counts": "2 x turnstile entries per day (the Metro counts entries only; doubled for "
              "on + off), mean over the days of the year",
    "note": "Mexico City Metro only",
}
ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw" / "riders" / FOLDER
READY = RAW.is_dir() and any(RAW.glob("*.csv"))
BOX = (-99.4, 19.1, -98.8, 19.7)
FORCE = {}


def _key(h):
    h = unicodedata.normalize("NFKD", h or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", h)


def records(raw):
    tot = defaultdict(float)
    days = defaultdict(set)
    months = defaultdict(set)
    for p in sorted(raw.glob("*.csv")):
        with open(p, encoding="utf-8-sig", newline="") as f:
            rd = csv.DictReader(f)
            cols = {_key(h): h for h in rd.fieldnames or ()}
            need = {"fecha", "linea", "estacion", "afluencia"}
            if not need <= set(cols):
                raise ValueError(f"{p.name}: expected columns {sorted(need)}, got "
                                 f"{rd.fieldnames}")
            c_year = cols.get("anio") or cols.get("ano")
            for r in rd:
                try:
                    n = float(r[cols["afluencia"]])
                except ValueError:
                    continue
                fecha = r[cols["fecha"]].strip()
                year = int(r[c_year]) if c_year else int(re.search(r"(20\d\d)", fecha).group(1))
                k = (year, r[cols["linea"]].strip(), r[cols["estacion"]].strip())
                tot[k] += n
                days[k].add(fecha)
                months[year].add(r[cols["mes"]].strip().lower() if "mes" in cols
                                 else fecha[:7])
    full = [y for y, ms in months.items() if len(ms) >= 12]
    if not full:
        raise ValueError("no year with twelve months in the files")
    year = max(full)
    out = []
    for (y, line, name), t in tot.items():
        if y != year or t <= 0:
            continue
        ref = line.split()[-1].upper() if line.split() else ""
        out.append({"name": name, "alt": [], "line": line, "refs": [ref], "x": None,
                    "y": None, "box": BOX, "n": 2 * t / len(days[(y, line, name)]),
                    "year": year})
    return out


_lines = {}


def line_refs():
    if not _lines:
        for l in json.loads((ROOT / "dist" / "data" / CC / "lines.json")
                            .read_text(encoding="utf-8"))["lines"]:
            _lines[l["id"]] = (l.get("ref") or "").upper()
    return _lines


def match_hook(rec, S):
    """Only stations of the record's own Metro line (by ref) inside Mexico City."""
    refs = set(rec["refs"])
    lr = line_refs()
    w, s, e, n = BOX
    rn = name_variants(rec["name"], *rec["alt"])
    whole = {squash(x) for x in name_variants(rec["name"], *rec["alt"], whole=True)}
    best = []
    for k, v in S.st.items():
        if not (w <= v["x"] <= e and s <= v["y"] <= n):
            continue
        if not any(lr.get(l) in refs for l in v.get("l", ())):
            continue
        # the whole name first: line B has both "Garibaldi/Lagunilla" and "Lagunilla"
        if whole & {squash(x) for x in name_variants(v.get("n"), v.get("e"), whole=True)}:
            best.append((-4, k))
            continue
        sc = name_score(rn, S.names[k])
        if sc == 0 and any(len(squash(a)) >= 5 and squash(a) in squash(b)
                           for a in rn for b in S.names[k]):
            sc = 2
        if sc:
            best.append((-sc, k))
    if not best:
        return None, None
    best.sort()
    if len(best) > 1 and best[1][0] == best[0][0]:
        return None, None
    return best[0][1], "line+name"
