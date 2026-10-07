"""Honduras, Censo 2013: country of birth per municipio, for the immigrant languages
countries/hn.py had drawn as Spanish. Session edd42a8c-latn, 2026-10-05; the record is
sources/hn.md, "Immigrant languages"; the shared rule is sources/latam_immig.py.

    python sources/hn_imm.py [--fetch]

Writes data/normalized/hn_imm.csv: geo_id (municipio, 4 digits), origin (ISO alpha-2, or
"US_U18" for the US-born under 18), count, for countries where Spanish is not the main language.

THE VARIABLE: P08C_PAIS "Pais de nacimiento" (INE's own three-digit codes, printed with their
names) on the base sources/hn_censo.py reads (181.115.7.199/binhnd, CPVHND2013NAC). A derived
PAISX numbers the non-Spanish-speaking countries 1..k (AREALIST makes a column per value of the
range); the US-born are split by P03 age, under 18 and 18+. Regional remainders ("Otros paises
asiaticos", 130) and "Ignorado" (1,176) stay Spanish.
CHECKS: 298 municipios; per country, municipio sums equal the national frequency; the zero
column plus the countries equals 7,657,684.
"""
import re
import sys
import time
import urllib.parse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hn_censo as hc  # noqa: E402
import latam_immig  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_H = "RUNDEF Job\n SELECTION ALL\n"
N_MUN = 298
POP = 7_657_684
# INE code -> ISO, non-Spanish-speaking countries only (the rest stay Spanish)
CODE_ISO = {
    107: "BZ", 201: "CA", 202: "US", 303: "BR", 307: "GY", 402: "BB", 404: "HT", 405: "JM",
    407: "KY", 409: "VC", 410: "TT", 501: "DE", 502: "AT", 503: "BE", 504: "DK", 506: "FI",
    507: "FR", 508: "GR", 509: "NL", 510: "IT", 511: "GB", 512: "NO", 513: "PL", 514: "PT",
    515: "RU", 516: "SE", 517: "CH", 601: "KP", 602: "KR", 603: "IL", 604: "CN", 605: "TW",
    606: "IN", 607: "IR", 608: "JP", 609: "JO", 610: "LB", 611: "PS", 612: "SY", 613: "TR",
    701: "DZ", 702: "EG", 704: "MA", 705: "ZA", 706: "TN", 801: "AU", 802: "NZ",
}
COLS = [(c, None) for c in sorted(CODE_ISO) if c != 202] + [(202, "adult"), (202, "u18")]


def program():
    lines = ["DEFINE PERSONA.PAISX\n AS SWITCH\n"]
    for i, (c, age) in enumerate(COLS, 1):
        cond = f"PERSONA.P08C_PAIS = {c}"
        if age == "adult":
            cond += " AND PERSONA.P03 >= 18"
        elif age == "u18":
            cond += " AND PERSONA.P03 < 18"
        lines.append(f"  INCASE {cond}\n   ASSIGN {i}\n")
    lines.append(f"  DEFAULT 0\n TYPE INTEGER\n RANGE 0-{len(COLS)}\n")
    return _H + "".join(lines) + "TABLE T\n AS AREALIST\n OF MUNIC, PERSONA.PAISX\n"


PROGRAMS = {
    "hn_nat_pais": _H + "TABLE T\n AS FREQUENCY\n OF PERSONA.P08C_PAIS\n",
    "hn_mun_paisx": program(),
}


def fetch():
    import requests
    hc.RAW.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0"
    for name, prog in PROGRAMS.items():
        dest = hc.RAW / f"{name}.htm"
        if dest.exists() and dest.stat().st_size > 2_000:
            continue
        print("RUN", name)
        r = s.post(f"{hc.HOST}/RpWebStats.exe/CmdSet?", data={
            "MAIN": "WebServerMain.inl", "BASE": hc.BASE, "LANG": "esp", "CODIGO": "XXUSUARIOXX",
            "ITEM": "PROGRED", "MODE": "RUN", "CMDSET": prog, "Submit": "Ejecutar"}, timeout=1800)
        r.raise_for_status()
        m = re.search(r"(RpBases[^\"'&<>]*?\.htm)", r.text)
        if not m:
            raise SystemExit(f"{name}: no output file\n{r.text[:800]}")
        t = s.get(f"{hc.HOST}/RpWebUtilities.exe/Text?LFN=" + urllib.parse.quote(m.group(1))
                  + "&TYPE=TMP", timeout=1800)
        t.raise_for_status()
        body = t.content.decode("utf-8", errors="replace")
        if "<table" not in body.lower():
            raise SystemExit(f"{name}: no table\n{body[:600]}")
        (hc.RAW / f"{name}.program.txt").write_text(prog, encoding="utf-8")
        dest.write_text(body, encoding="utf-8")
        print(f"  {dest.stat().st_size:,} bytes")
        time.sleep(1)


def national():
    out = {}
    for r in hc._rows("hn_nat_pais"):
        m = re.fullmatch(r"(\d{3}) .+", r[0])
        if m and len(r) == 4:
            out[int(m.group(1))] = int(r[1].replace(",", ""))
    return out


def main():
    if "--fetch" in sys.argv:
        fetch()
    nat = national()
    for c in CODE_ISO:
        assert c in nat, c
    rows = hc._rows("hn_mun_paisx")
    head = next(r for r in rows if r[0] == "Código")
    cols = head[1:]
    assert cols[-1] == "Total", cols[-1]
    data = {}
    for r in rows:
        if re.fullmatch(r"\d{3,4}", r[0].replace(" ", "")) and len(r) == len(cols) + 1:
            v = [int(x.replace(",", "")) if x != "-" else 0 for x in r[1:]]
            row = dict(zip((int(c) for c in cols[:-1]), v[:-1]))
            assert sum(row.values()) == v[-1], r[0]
            data[r[0].strip().zfill(4)] = row
    assert len(data) == N_MUN, len(data)
    tot = {k: sum(d.get(k, 0) for d in data.values()) for k in range(len(COLS) + 1)}
    assert sum(tot.values()) == POP, sum(tot.values())
    for c in CODE_ISO:
        got = sum(tot[i] for i, (cc, _) in enumerate(COLS, 1) if cc == c)
        assert got == nat[c], (c, got, nat[c])
    out = []
    for geo, row in data.items():
        for i, (c, age) in enumerate(COLS, 1):
            n = row.get(i, 0)
            if n:
                out.append((geo, "US_U18" if age == "u18" else CODE_ISO[c], n))
    df = pd.DataFrame(out, columns=["geo_id", "origin", "count"])
    df.to_csv(hc.NORM / "hn_imm.csv", index=False, encoding="utf-8")
    s = df.groupby("origin")["count"].sum().sort_values(ascending=False)
    print(f"  checks pass: {N_MUN} municipios, {len(CODE_ISO)} countries equal the national "
          f"frequency, all {POP:,} people once")
    print("  " + ", ".join(f"{k} {v:,}" for k, v in s.head(15).items()))


if __name__ == "__main__":
    main()
