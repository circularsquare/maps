"""Bolivia — two tables off INE's REDATAM webserver: the 2024 census count, and 1992's religion item.

Writes, under data/raw/bo/:
    cpv2024_pop_depto.csv, cpv2024_pop_provin.csv          people counted in 2024, by sex
    cpv1992_religion_depto.csv, cpv1992_religion_provin.csv  1992 item 16, persons per group
and keeps every raw result page in data/raw/bo/redatam/ so a rebuild does not touch the server.

## WHERE THESE ARE

`https://redatam.ine.gob.bo/`, INE's own Redatam webserver. Its home page lists the 2001, 2012 and
2024 bases; **the 1992 base (`PHCCEN92ESP`) is live but its link is commented out**, found by a
helper session on 2026-09-14 (`sources/bo.md` §6). The server sends an incomplete certificate
chain, so this skips verification: a server that omits its intermediate is a server fault and not
a wall (spec §12, Mozambique).

## 2024: THE POPULATION EVERY BOLIVIAN ROW IS DRAWN ON

INE published 11,365,333 as the final count (results bulletin, p. 1, with 5,682,835 women and
5,682,498 men). Nothing on INE's pages prints the nine departments in a table; the Wikipedia
table that does cites press. So the count is taken from the base itself, `PERSONA.SEXO` by
`DEPTO` and `PROVIN`, and the bulletin's three national figures are asserted.

## 1992: THE LAST CENSUS THAT ASKED RELIGION

Household item 16 asked how many members **belong to no religion, are Catholic, evangelical, or
of another religion** (`VIVIENDA.RNINGUNA`, `RCATOLICA`, `RVANGELICA`, `ROTRAS`; `RELIGNOR` is
unknown). The webserver only offers a frequency of HOUSEHOLDS by how many members answered each
way, so persons are the sum of k times the households with k. That is exact, not an estimate,
and each area's household total is asserted against its own `Total` row. The universe is people
in private households, 6,292,819 of the 6,420,792 counted.

The national figures are checked against INE's own 2008 web page (Wayback capture of
`ine.gov.bo/censo/censo1992.aspx`, *Población en hogares, por religión que profesan*): the four
named groups agree to the person, and unknown is 416,514 there against 416,424 here.

Usage:
    python sources/bo_census.py --fetch    query the server (~14 requests, a minute or two)
    python sources/bo_census.py            re-parse the saved pages
"""

import html
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "bo")
RED = os.path.join(RAW, "redatam")

HOST = "https://redatam.ine.gob.bo"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots/1.0 (census tabulation)"
BASE24, BASE92 = "PHCCEN24ESPV1", "PHCCEN92ESP"
LEVELS = {"DEPTO": 9, "PROVIN": 112}

# results bulletin, p. 1
CENSUS_2024 = 11_365_333
WOMEN_2024, MEN_2024 = 5_682_835, 5_682_498

# The 2024 base's province break has one area more than COD-AB's 112 provinces: an indigenous
# territory (TIOC) counted as a unit of its own, code 0809, 3,973 people. Kept as its own row
# here and asserted by code and name; sources/bo_geo.py folds it into the province around it.
TIOC_2024 = {"PROVIN": {"0809": "Territorio Indígena Multiétnico"}}

VARS92 = {"catolica": "VIVIENDA.RCATOLICA", "evangelica": "VIVIENDA.RVANGELICA",
          "otras": "VIVIENDA.ROTRAS", "ninguna": "VIVIENDA.RNINGUNA",
          "ignorada": "VIVIENDA.RELIGNOR", "totper": "VIVIENDA.TOTPER"}
# INE's 2008 page, national, persons in households
INE_2008 = {"catolica": 4_992_897, "evangelica": 636_551, "otras": 77_918, "ninguna": 169_029,
            "ignorada": 416_514}
SUM5_1992, TOTPER_1992 = 6_292_819, 6_420_792

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def _get(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers=headers or {"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, context=CTX, timeout=600) as f:
            return f.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} from {url}")


def frequency(base, item, row, areabreak, name):
    """One FREQUENCY run, cached. Returns the result page that carries the area blocks."""
    path = os.path.join(RED, f"{name}.htm")
    if os.path.exists(path) and "--fetch" not in sys.argv:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    fields = {"MAIN": "WebServerMain.inl", "BASE": base, "LANG": "ESP", "CODIGO": "XXUSUARIOXX",
              "ITEM": item, "MODE": "RUN", "inputTitle": "", "ROW": row, "AREABREAK": areabreak,
              "SELECTION": "ALL", "INLINESELECTION": "", "UNIVERSE": "", "FILTER": "",
              "TEXT_FILTER": "", "PERCENT": "OFF", "FORMAT": "HTML", "Submit": "Ejecutar"}
    hdr = {"User-Agent": UA, "Content-Type": "application/x-www-form-urlencoded",
           "Referer": f"{HOST}/binbol/RpWebStats.exe/Frequency?BASE={base}&ITEM={item}&lang=ESP"}
    body = _get(f"{HOST}/binbol/RpWebStats.exe/Frequency?",
                urllib.parse.urlencode(fields).encode("ascii"), hdr)
    pages = [body]
    for link in sorted(set(re.findall(
            r'(?:src|href)="((?:https://redatam\.ine\.gob\.bo)?/redbol/+tempo/[^"]+)"', body))):
        pages.append(_get(link if link.startswith("http") else HOST + link))
    hits = [p for p in pages if "AREA #" in p]
    if len(hits) != 1:
        raise SystemExit(f"{name}: {len(hits)} result pages carry area blocks; the server's "
                         "output has changed")
    os.makedirs(RED, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(hits[0])
    print(f"  fetched {name}")
    time.sleep(1.0)
    return hits[0]


def _cells(row):
    return [(c, html.unescape(re.sub(r"<[^>]+>", "", t)).strip().replace("\xa0", ""))
            for c, t in re.findall(r'<td class="(c\d)"[^>]*>(.*?)</td>', row, re.S)]


def _num(s):
    return int(re.sub(r"[\s.,]", "", s))


def parse(text):
    """Area blocks of a FREQUENCY table -> (order, {code: {name, rows: {label: cases}, total}})."""
    areas, order, cur = {}, [], None

    def open_area(code, name):
        areas[code] = {"name": name, "rows": {}, "total": None}
        order.append(code)

    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", text, re.S):
        cs = _cells(row)
        vals = [v for _, v in cs]
        m = next((re.match(r"AREA # (\S+)", v) for v in vals if v.startswith("AREA #")), None)
        if m:
            i = vals.index(m.group(0))
            cur = m.group(1)
            open_area(cur, vals[i + 1] if i + 1 < len(vals) else "")
            continue
        if any(c == "c5" and v == "Total" for c, v in cs):
            nums = [v for c, v in cs if c == "c6" and re.fullmatch(r"[\d\s.,]+", v)]
            if cur is None:
                cur = "NATIONAL"
                if cur not in areas:
                    open_area(cur, "national")
            areas[cur]["total"] = _num(nums[0]) if nums else None
            cur = None
            continue
        data = [(c, v) for c, v in cs if v]
        if len(data) >= 2 and data[0][0] == "c1" and data[1][0] == "c4":
            if cur is None:
                cur = "NATIONAL"
                if cur not in areas:
                    open_area(cur, "national")
            areas[cur]["rows"][data[0][1]] = areas[cur]["rows"].get(data[0][1], 0) + _num(data[1][1])
    return order, areas


def census_2024(level):
    order, areas = parse(frequency(BASE24, "RESPER", "PERSONA.SEXO", level,
                                   f"cpv2024_sexo_{level.lower()}"))
    rows = []
    for code in order:
        a = areas[code]
        women = sum(v for k, v in a["rows"].items() if "mujer" in k.lower())
        men = sum(v for k, v in a["rows"].items() if "hombre" in k.lower())
        if len(a["rows"]) != 2 or women + men != a["total"]:
            raise SystemExit(f"2024 {level} {code}: rows {a['rows']} against total {a['total']}")
        rows.append((code, a["name"], women + men, women, men))
    df = pd.DataFrame(rows, columns=["code", "name", "pop", "women", "men"])
    nat = df[df["code"] == "NATIONAL"]
    df = df[df["code"] != "NATIONAL"].reset_index(drop=True)
    extra = TIOC_2024.get(level, {})
    got = dict(zip(df.loc[df["code"].isin(extra), "code"], df.loc[df["code"].isin(extra), "name"]))
    if got != extra or len(df) != LEVELS[level] + len(extra):
        raise SystemExit(f"2024 {level}: {len(df)} areas with extra {got}, expected "
                         f"{LEVELS[level]} plus {extra}")
    if (int(df["pop"].sum()), int(df["women"].sum()), int(df["men"].sum())) != \
            (CENSUS_2024, WOMEN_2024, MEN_2024):
        raise SystemExit(f"2024 {level}: {int(df['pop'].sum()):,} people "
                         f"({int(df['women'].sum()):,} women, {int(df['men'].sum()):,} men), "
                         f"not the bulletin's {CENSUS_2024:,} ({WOMEN_2024:,}, {MEN_2024:,})")
    if len(nat) and int(nat["pop"].iloc[0]) != CENSUS_2024:
        raise SystemExit(f"2024 {level}: the national block says {int(nat['pop'].iloc[0]):,}")
    print(f"  2024 {level}: {len(df)} areas, {CENSUS_2024:,} people, women and men equal to the "
          "bulletin's")
    return df


def census_1992(level):
    per = {}
    names = {}
    order = None
    for key, var in VARS92.items():
        o, areas = parse(frequency(BASE92, "FREQVIV", var, level,
                                   f"cpv1992_{key}_{level.lower()}"))
        order = order or o
        if o != order:
            raise SystemExit(f"1992 {level} {key}: area order differs from the first variable")
        for code in o:
            a = areas[code]
            # `TOTPER` carries one labelled row, POBLACION_TOTAL, counted among the households in
            # the Total row and not a household size; it is left out of the persons, which then
            # reproduce the census's 6,420,792 exactly. Any other label stops the build.
            odd = {k: v for k, v in a["rows"].items() if not re.fullmatch(r"\d+", k)}
            if set(odd) - {"POBLACION_TOTAL"}:
                raise SystemExit(f"1992 {level} {code} {key}: non-numeric rows {odd}")
            if sum(a["rows"].values()) != a["total"]:
                raise SystemExit(f"1992 {level} {code} {key}: households {sum(a['rows'].values())}"
                                 f" against the Total row {a['total']}")
            per.setdefault(code, {})[key] = sum(int(k) * v for k, v in a["rows"].items()
                                                if k not in odd)
            per[code]["households"] = a["total"]
            names[code] = a["name"]
    rows = [dict(code=c, name=names[c], **per[c]) for c in order]
    df = pd.DataFrame(rows)
    df["sum5"] = df[["catolica", "evangelica", "otras", "ninguna", "ignorada"]].sum(axis=1)
    nat = df[df["code"] == "NATIONAL"]
    df = df[df["code"] != "NATIONAL"].reset_index(drop=True)
    if len(df) != LEVELS[level]:
        raise SystemExit(f"1992 {level}: {len(df)} areas, expected {LEVELS[level]}")
    tot = df.drop(columns=["code", "name"]).sum()
    if (int(tot["sum5"]), int(tot["totper"])) != (SUM5_1992, TOTPER_1992):
        raise SystemExit(f"1992 {level}: {int(tot['sum5']):,} in the five groups and "
                         f"{int(tot['totper']):,} in households, not {SUM5_1992:,} / "
                         f"{TOTPER_1992:,}")
    for k in ("catolica", "evangelica", "otras", "ninguna"):
        if int(tot[k]) != INE_2008[k]:
            raise SystemExit(f"1992 {level}: {k} {int(tot[k]):,}, INE's 2008 page {INE_2008[k]:,}")
    print(f"  1992 {level}: {len(df)} areas; Catholic, evangelical, other and none equal INE's "
          f"2008 page to the person; unknown {int(tot['ignorada']):,} against its "
          f"{INE_2008['ignorada']:,}")
    if len(nat):
        if int(nat["sum5"].iloc[0]) != SUM5_1992:
            raise SystemExit(f"1992 {level}: the national block says {int(nat['sum5'].iloc[0]):,}")
    return df


def main():
    os.makedirs(RAW, exist_ok=True)
    for level in LEVELS:
        census_2024(level).to_csv(os.path.join(RAW, f"cpv2024_pop_{level.lower()}.csv"),
                                  index=False, encoding="utf-8")
        census_1992(level).to_csv(os.path.join(RAW, f"cpv1992_religion_{level.lower()}.csv"),
                                  index=False, encoding="utf-8")
    print(f"wrote four CSVs to {RAW}")


if __name__ == "__main__":
    main()
