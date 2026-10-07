"""Costa Rica, X Censo Nacional de Poblacion y VI de Vivienda 2011 (INEC): indigenous people who
speak an indigenous language, per distrito.

    python sources/cr_censo.py [--fetch]

Writes data/normalized/cr.csv: one row per (distrito, code). Every person counted in 2011
(4,301,712) is in exactly one row. `code` is the derived variable LNG defined in the program
below:

    2          does not consider themselves indigenous (P07 = 2); not asked P09
    10*p + a   indigenous, pueblo p (P08 1-10) and answer a to P09 (1 speaks an indigenous
               language, 2 does not)
    1, 3       P07 yes with no P09 answer / P07 missing: both must be zero (checked)

THE QUESTIONS (boleta censal 2011, printed in the Territorios Indigenas volume, p54): P07 "Se
considera (nombre) indigena?"; if yes, P08 which pueblo (Bribri, Brunca o Boruca, Cabecar,
Chorotega, Huetar, Maleku o Guatuso, Ngobe o Guaymi, Teribe o Terraba, de otro pais, ningun
pueblo); and P09 "Habla (nombre) alguna lengua indigena?", yes/no. P09 is asked only of the
104,143 who said yes to P07 (the national crosstab shows the other 4,197,569 as `No Aplica`). It
asks about ANY indigenous language and never names it; taxonomy/cr2011.py says what each pueblo's
yes is drawn as.

THE SOURCE is INEC's own REDATAM webserver (sistemas.inec.cr:8443/bininec, base CP2011, open,
no login). Its certificate chain is incomplete: Windows completes it from the certificate's AIA
link and caches the intermediate, and this script verifies TLS against the Windows store
(ssl.create_default_context). If it fails with CERTIFICATE_VERIFY_FAILED, open the base URL once
in a browser or with PowerShell's Invoke-WebRequest and run again.

CHECKS (the script stops unless all hold):
  1. 472 distritos (the census's own count; REDATAM's DISTRITO frequency); codes sum to
     4,301,712, the 2011 census population; codes 1 and 3 are empty.
  2. the distritos rebuild the 7 provincias on every code (a separate query).
  3. per distrito, LNG agrees with separate tabulations of the raw P07, P08 and P09 (each pueblo
     column of P08 = that pueblo's two LNG codes; P09 yes / no = all x1 / all x2; P07 no = code 2).
  4. per pueblo, distrito sums equal the national P08 x P09 crosstab.
  5. the printed volume, INEC "Territorios Indigenas: principales indicadores demograficos y
     socioeconomicos" (2013, data/raw/cr/territorios_indigenas_2011.pdf), CUADRO 3, last column:
     the share of indigenous people who speak an indigenous language in each of the 24
     territories, one decimal, reproduced from a TERRINDI x P09 crosstab of the same base.
"""
import html
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "cr"
NORM = HERE / "data" / "normalized"

HOST = "https://sistemas.inec.cr:8443/bininec"
BASE = "CP2011"

LNG = """DEFINE POBLACIO.LNG
 AS SWITCH
  INCASE POBLACIO.LENGIND = 1 OR POBLACIO.LENGIND = 2
   ASSIGN POBLACIO.PUEBLOIN * 10 + POBLACIO.LENGIND
  INCASE POBLACIO.CONSIND = 1
   ASSIGN 1
  INCASE POBLACIO.CONSIND = 2
   ASSIGN 2
  DEFAULT 3
 TYPE INTEGER
 RANGE 0-110
"""

_HEAD = "RUNDEF Job\n SELECTION ALL\n"
PROGRAMS = {
    "dist_lng": _HEAD + LNG + "TABLE T\n AS AREALIST\n OF DISTRITO, POBLACIO.LNG\n",
    "prov_lng": _HEAD + LNG + "TABLE T\n AS AREALIST\n OF PROVINCI, POBLACIO.LNG\n",
    "dist_p07": _HEAD + "TABLE T\n AS AREALIST\n OF DISTRITO, POBLACIO.CONSIND\n",
    "dist_p08": _HEAD + "TABLE T\n AS AREALIST\n OF DISTRITO, POBLACIO.PUEBLOIN\n",
    "dist_p09": _HEAD + "TABLE T\n AS AREALIST\n OF DISTRITO, POBLACIO.LENGIND\n",
    "nat_p08_p09": _HEAD + "TABLE T\n AS FREQUENCY\n OF POBLACIO.PUEBLOIN BY POBLACIO.LENGIND\n",
    "terr_p09": _HEAD + "TABLE T\n AS FREQUENCY\n OF VIVIENDA.TERRINDI BY POBLACIO.LENGIND\n",
    "dist_names": _HEAD + "TABLE T\n AS FREQUENCY\n OF DISTRITO.DISTRITO\n",
    "dist_area": _HEAD + "TABLE T\n AS AREALIST\n OF DISTRITO, DISTRITO.EXTTER\n",
}

# P08's labels, codes 1-10, in the order REDATAM prints them. The order is not taken on trust:
# check 3 matches each label's AREALIST column against the LNG codes 10p+1 and 10p+2 per distrito.
PUEBLOS = {1: "Bribrí", 2: "Brunca o Boruca", 3: "Cabécar", 4: "Chorotega", 5: "Huetar",
           6: "Maleku o Guatuso", 7: "Ngöbe o Guaymí", 8: "Teribe o Térraba",
           9: "De otro país", 10: "Ningún pueblo"}
ANSWERS = {1: "speaks an indigenous language", 2: "does not speak an indigenous language"}
OTHER_CODES = {2: "not indigenous (not asked)"}
CENSUS_POPULATION = 4_301_712
N_DIST, N_PROV = 472, 7

# THE PRINTED WITNESS, typed from the PDF (printed p35, CUADRO 3, last column "% de indig. habla
# idioma indigena"), sharing no code path with the query: territory -> percent, one decimal.
TERRITORY_PCT = {
    "Salitre": 53.4, "Cabagra": 43.6, "Talamanca Bribrí": 60.8, "Këköldi": 36.3,
    "Boruca": 5.9, "Curré": 4.4, "Chirripó": 96.7, "Ujarrás": 71.4, "Tayni": 86.7,
    "Talamanca Cabécar": 64.9, "Telire": 86.5, "Bajo Chirripó": 86.6, "Nairi Awari": 94.6,
    "China Kicha": 39.1, "Matambú": 0.4, "Zapatón": 0.8, "Quitirrisí": 0.7, "Guatuso": 67.5,
    "Abrojo Montezuma": 86.4, "Osa": 87.0, "Conteburica": 67.3, "Coto Brus": 88.2,
    "Altos de San Antonio": 13.9, "Térraba": 9.9,
}
# Also printed in the same volume (p34, the table before CUADRO 3, row "Costa Rica"): 104,143
# people self-identified as indigenous, 26,070 of them with no pueblo. Asserted against the
# P08 x P09 crosstab.
INDIGENOUS_2011 = 104_143
NO_PUEBLO_2011 = 26_070

_CTX = ssl.create_default_context()


def _get(url, data=None, timeout=900):
    if data is not None:
        data = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=_CTX, timeout=timeout) as r:
        raw = r.read()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _post(program):
    body = _get(HOST + "/RpWebStats.exe/CmdSet?",
                data={"MAIN": "WebServerMain.inl", "BASE": BASE, "LANG": "esp",
                      "CODIGO": "XXUSUARIOXX", "ITEM": "PROGRED", "MODE": "RUN",
                      "CMDSET": program, "Submit": "Ejecutar"})
    tmps = re.findall(r"LFN=([^\"'&<>]+?\.htm)", html.unescape(body))
    if not tmps:
        raise SystemExit("REDATAM returned no output file. Body starts:\n" + body[:600])
    return _get(HOST + "/RpWebUtilities.exe/Text?LFN=" + urllib.parse.quote(tmps[0])
                + "&TYPE=TMP")


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, program in PROGRAMS.items():
        dest = RAW / f"cr_{name}.htm"
        if dest.exists() and dest.stat().st_size > 1_000:
            print("already have", dest.name)
            continue
        print("RUN", name)
        body = _post(program)
        if "Tabla vac" in body or "<table" not in body.lower():
            raise SystemExit(f"{name}: REDATAM returned no table\n{body[:600]}")
        dest.write_text(body, encoding="utf-8")
        (RAW / f"cr_{name}.txt").write_text(program, encoding="utf-8")
        print(f"  {dest.stat().st_size:,} bytes")
        time.sleep(1)


def _rows(name):
    path = RAW / f"cr_{name}.htm"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run with --fetch first")
    body = path.read_text(encoding="utf-8")
    out = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S | re.I):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).replace("\xa0", " ").strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)]
        cells = [c for c in cells if c]
        if cells:
            out.append(cells)
    return out


def _int(tok):
    t = tok.replace(" ", "").replace(",", "")
    if not re.fullmatch(r"\d+", t):
        raise SystemExit(f"{tok!r} is not a figure")
    return int(t)


def _areal(name, expected):
    """AREALIST -> {area code: {column label: count}}, `Total` asserted against the row."""
    rows = _rows(name)
    header = next((r for r in rows if r[0] == "Código"), None)
    if header is None:
        raise SystemExit(f"{name}: no header row starting `Código`")
    cols = header[1:]
    if cols[-1] != "Total":
        raise SystemExit(f"{name}: last column is {cols[-1]!r}, not Total")
    out = {}
    for r in rows:
        if len(r) != len(cols) + 1 or not re.fullmatch(r"\d+", r[0].replace(" ", "")):
            continue
        vals = dict(zip(cols, (_int(x) for x in r[1:])))
        tot = vals.pop("Total")
        if sum(vals.values()) != tot:
            raise SystemExit(f"{name}: area {r[0]} categories sum to {sum(vals.values())}, "
                             f"Total {tot}")
        if r[0] in out:
            raise SystemExit(f"{name}: area {r[0]} twice")
        out[r[0].strip()] = vals
    if len(out) != expected:
        raise SystemExit(f"{name}: {len(out)} areas, expected {expected}")
    return out, cols[:-1]


def _crosstab(name):
    """FREQUENCY ... BY P09 -> {row label: (si, no)}, each row's Total asserted."""
    out = {}
    for r in _rows(name):
        if len(r) == 4 and r[0] not in ("Sí", "Total") and re.fullmatch(r"[\d ]+", r[1]):
            si, no, tot = (_int(x) for x in r[1:])
            if si + no != tot:
                raise SystemExit(f"{name}: {r[0]} {si} + {no} != {tot}")
            out[r[0]] = (si, no)
        elif r[0] == "Total" and len(r) == 4:
            out["Total"] = tuple(_int(x) for x in r[1:3])
    return out


def district_names():
    """{code: name} from REDATAM's DISTRITO frequency. The labels for 60111-60116 are shifted by
    one against their codes (60111 is labelled Chacarita, but its area, 316.6 km2, is Cobano's);
    sources/cr_geo.py joins on CODE and says so."""
    out = {}
    for r in _rows("dist_names"):
        m = re.fullmatch(r"(\d{5})(?:\s+(.+))?", r[0].strip())
        if m:
            out[m.group(1)] = (m.group(2) or "").strip()
    if len(out) != N_DIST:
        raise SystemExit(f"dist_names: {len(out)} names, expected {N_DIST}")
    return out


def district_areas():
    """{code: km2}, the census base's own EXTTER per distrito."""
    out = {}
    for r in _rows("dist_area"):
        if re.fullmatch(r"\d{5}", r[0]) and len(r) >= 2:
            out[r[0]] = float(r[1].replace(" ", "").replace(",", "."))
    if len(out) != N_DIST:
        raise SystemExit(f"dist_area: {len(out)} areas, expected {N_DIST}")
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()
    ok = True

    def say(good, msg):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK ' if good else 'BAD'} {msg}")

    dist, cols = _areal("dist_lng", N_DIST)
    allowed = {1, 2, 3} | {10 * p + a for p in PUEBLOS for a in ANSWERS}
    bad = sorted({int(c) for c in cols} - allowed)
    if bad:
        raise SystemExit(f"LNG codes outside the plan: {bad}")
    prov, _ = _areal("prov_lng", N_PROV)
    names = district_names()
    say(set(dist) == set(names), f"{N_DIST} distritos, the same codes as REDATAM's own list")

    # 1. every person once
    nat = {}
    for v in dist.values():
        for k, n in v.items():
            nat[int(k)] = nat.get(int(k), 0) + n
    tot = sum(nat.values())
    say(tot == CENSUS_POPULATION, f"codes sum to {tot:,} (census population "
        f"{CENSUS_POPULATION:,})")
    say(nat.get(1, 0) == 0 and nat.get(3, 0) == 0,
        f"nobody indigenous without a P09 answer ({nat.get(1, 0)}) or without P07 "
        f"({nat.get(3, 0)})")

    # 2. distritos rebuild provincias (first digit of the five-digit code)
    rolled = {}
    for code, v in dist.items():
        acc = rolled.setdefault(code[0], {})
        for k, n in v.items():
            acc[k] = acc.get(k, 0) + n
    bad = [(p, k) for p in prov for k in set(prov[p]) | set(rolled.get(p, {}))
           if prov[p].get(k, 0) != rolled.get(p, {}).get(k, 0)]
    say(set(rolled) == set(prov) and not bad,
        f"the {N_DIST} distritos rebuild the {N_PROV} provincias on every code "
        f"({len(bad)} failures)")

    # 3. against the raw variables, per distrito
    p07, p07c = _areal("dist_p07", N_DIST)
    p08, p08c = _areal("dist_p08", N_DIST)
    p09, p09c = _areal("dist_p09", N_DIST)
    print(f"      P07 columns {p07c}; P09 columns {p09c}")
    if set(p08c) != set(PUEBLOS.values()):
        raise SystemExit(f"P08 columns changed: {p08c}")
    if set(p09c) != {"Sí", "No"} or set(p07c) - {"Sí", "No"}:
        raise SystemExit(f"P07/P09 columns changed: {p07c} {p09c}")
    bad = []
    for d, v in dist.items():
        g = lambda k: v.get(str(k), 0)  # noqa: E731
        for p, lab in PUEBLOS.items():
            if p08[d].get(lab, 0) != g(10 * p + 1) + g(10 * p + 2):
                bad.append((d, "P08", lab))
        for a, lab in ((1, "Sí"), (2, "No")):
            if p09[d].get(lab, 0) != sum(g(10 * p + a) for p in PUEBLOS):
                bad.append((d, "P09", lab))
        si = sum(g(10 * p + a) for p in PUEBLOS for a in ANSWERS)
        if (p07[d].get("Sí", 0), p07[d].get("No", 0)) != (si, g(2)):
            bad.append((d, "P07"))
    say(not bad, f"LNG agrees with separate P07, P08 and P09 tabulations on all {N_DIST} "
        f"distritos ({len(bad)} failures) {bad[:4]}")

    # 4. national crosstab
    xt = _crosstab("nat_p08_p09")
    bad = [p for p, lab in PUEBLOS.items()
           if xt.get(lab) != (nat.get(10 * p + 1, 0), nat.get(10 * p + 2, 0))]
    say(not bad, f"per pueblo, distrito sums equal the national P08 x P09 crosstab ({bad})")
    say(sum(xt["Total"]) == INDIGENOUS_2011,
        f"{sum(xt['Total']):,} self-identified indigenous (published {INDIGENOUS_2011:,})")
    say(sum(xt["Ningún pueblo"]) == NO_PUEBLO_2011,
        f"{sum(xt['Ningún pueblo']):,} of no pueblo (published {NO_PUEBLO_2011:,})")

    # 5. printed territory shares
    terr = _crosstab("terr_p09")
    bad = []
    for t, want in TERRITORY_PCT.items():
        si, no = terr.get(t, (0, 0))
        got = round(100 * si / (si + no), 1) if si + no else None
        if got is None or abs(got - want) > 0.051:
            bad.append((t, got, want))
    say(not bad, f"the {len(TERRITORY_PCT)} territories' speaker shares equal CUADRO 3 as "
        f"printed in 2013 ({bad})")
    out_si, out_no = terr.get("Fuera de territorio indígena", (0, 0))
    print(f"      outside the territories: {out_si:,} speakers of {out_si + out_no:,} "
          f"indigenous ({100 * out_si / (out_si + out_no):.1f}%)")

    if not ok:
        raise SystemExit("reconciliation FAILED")

    rows = []
    for d in sorted(dist):
        for k, n in sorted(dist[d].items(), key=lambda kv: int(kv[0])):
            k = int(k)
            cat = OTHER_CODES.get(k) or f"{PUEBLOS[k // 10]}: {ANSWERS[k % 10]}"
            if n:
                rows.append(dict(geo_level="distrito", geo_id=d, geo_name=names[d],
                                 code=k, source_category=cat, count=n))
    df = pd.DataFrame(rows)
    NORM.mkdir(parents=True, exist_ok=True)
    df.to_csv(NORM / "cr.csv", index=False)
    print(f"\nwrote {NORM / 'cr.csv'} ({len(df):,} rows)\n\nnational:")
    for k in sorted(nat):
        if nat[k]:
            cat = OTHER_CODES.get(k) or f"{PUEBLOS[k // 10]}: {ANSWERS[k % 10]}"
            print(f"  {k:>3} {nat[k]:>10,}  {cat}")


if __name__ == "__main__":
    main()
