"""Iceland: Icelandic plus immigrant languages proxied by citizenship, per municipality, from
Statistics Iceland (Hagstofa) MAN04203 -> data/normalized/is.csv.

    python sources/is_pop.py --fetch     download MAN04203 (1 January 2026) into data/raw/is/
    python sources/is_pop.py             build data/normalized/is.csv, printing the checks

NO LANGUAGE QUESTION (Iceland's register-based census asks none). Under Anita's 2026-10-05
ruling for rich countries (AGENT_BRIEF §2): the national language plus immigrant languages
proxied by citizenship, every row `derived`. Iceland has no regional language.

Per municipality (62), from MAN04203 "Population by sex, municipality and citizenship",
1 January 2026, both sexes:
  1. Icelandic citizens on Icelandic (naturalised immigrants included; nothing counts them).
  2. each foreign citizenship on a language mix:
     - HOME_MIX origins (multilingual, drawn on this map): the origin's own drawn mix, languages
       of 1%+ kept and scaled back to 100% (sources/sa.md's method, sa_census.mix_from);
     - the rest: fr_build.COUNTRY_LANG's main language, with Portugal's splits (pt_censos.py)
       for Ukraine-less multilingual states (Switzerland, Belgium, Canada) and France added.
  3. retention: the share of each origin region's immigrants who speak only the host language
     with their children (France's TeO2, fr_build.TEO_FRENCH, as Portugal and Belgium use) is
     moved onto Icelandic.
  4. stateless and unspecified on `other`.
The record is sources/is.md.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "sources"))
sys.path.insert(0, str(HERE / "taxonomy"))
import fr_build  # noqa: E402

RAW = HERE / "data" / "raw" / "is" / "man04203_2026.json"
OUT = HERE / "data" / "normalized" / "is.csv"
URL = ("https://px.hagstofa.is/pxen/api/v1/en/Ibuar/mannfjoldi/3_bakgrunnur/Rikisfang/"
       "MAN04203.px")
QUERY = {"query": [{"code": "Kyn", "selection": {"filter": "item", "values": ["0"]}},
                   {"code": "Ár", "selection": {"filter": "item", "values": ["2026"]}}],
         "response": {"format": "json-stat2"}}
TOTAL_2026 = 394_324
MUNIS = 62
ICELANDIC = "indoeuropean.germanic.north.icelandic"

# multilingual origins with 200+ citizens in Iceland, at their own drawn mix on this map. Not
# here: origins whose drawn mix is mostly their own immigrants (Germany, France, the US,
# Sweden, the Netherlands), which would give a German citizen in Iceland a Turkish share.
HOME_MIX = ["UA", "LV", "LT", "EE", "ES", "PH", "IN", "NG", "AF", "IQ", "GH", "PK", "IR", "CN",
            "RO", "SK", "BG", "RU", "RS", "IT", "PS", "SO"]
MIN_SHARE = 0.01
COUNTRY_LANG = dict(fr_build.COUNTRY_LANG)
COUNTRY_LANG.update({
    "FR": "French",                                   # France's own list has no France
    "CA": {"English": 0.75, "French": 0.25},          # as Portugal (pt_censos.py)
    "CH": {"German": 0.65, "French": 0.25, "Italian": 0.1},
    "BE": {"Dutch": 0.6, "French": 0.4},
})
HAG_TO_EU = {"GB": "UK", "GR": "EL"}    # Hagstofa writes ISO; COUNTRY_LANG writes Eurostat
# Hagstofa's non-country rows: stateless, unspecified foreign, ex-Yugoslavia unspecified
# (Serbo-Croatian, Slovene, Macedonian or Albanian; too mixed for one node), former units
OTHER = {"XZ", "XX", "XY", "XR", "CS", "SU", "YU", "ZR"}
# TeO2 region for codes fr_build's continent fallback needs
BLOCK = {"EUR": {"IS", "LI", "NO", "CH", "UK", "BA", "ME", "MD", "MK", "GE", "AL", "RS", "TR",
                 "UA", "XK", "AD", "BY", "VA", "MC", "RU", "SM", "AM", "AZ"},
         "AME": {"CA", "US", "AG", "AR", "BB", "BO", "BR", "BS", "BZ", "CL", "CO", "CR", "CU",
                 "DM", "DO", "EC", "GD", "GT", "GY", "HN", "HT", "JM", "KN", "MX", "NI", "PA",
                 "PE", "PR", "PY", "SR", "SV", "TT", "UY", "VC", "VE", "LC"},
         "OCE": {"AU", "FJ", "NR", "NZ", "PG", "SB", "TO", "VU"}}
AFRICA = {"AO", "BF", "BJ", "BW", "CD", "CG", "CI", "CM", "CV", "DJ", "DZ", "EG", "EH", "ER",
          "ET", "GA", "GH", "GM", "GN", "GQ", "GW", "KE", "LR", "LS", "LY", "MA", "MG", "ML",
          "MR", "MU", "MW", "MZ", "NA", "NE", "NG", "RW", "SC", "SD", "SL", "SN", "SO", "SS",
          "TD", "TG", "TN", "TZ", "UG", "ZA", "ZM", "ZW"}


def block_of(iso):
    for b, s in BLOCK.items():
        if iso in s:
            return b
    return "AFR" if iso in AFRICA else "ASI"


def fetch():
    RAW.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(QUERY).encode()
    for _ in range(40):     # the API answers 429 for whole minutes when busy
        req = urllib.request.Request(URL, data=body, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Content-Type": "application/json"})
        try:
            RAW.write_bytes(urllib.request.urlopen(req, timeout=120).read())
            print("wrote", RAW)
            return
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(15)
    raise SystemExit("Hagstofa kept answering 429")


def table():
    d = json.load(open(RAW, encoding="utf-8"))
    assert d["id"][:2] == ["Sveitarfélag", "Ríkisfang"] and d["size"][2:] == [1, 1], d["id"]
    mun = list(d["dimension"]["Sveitarfélag"]["category"]["index"])
    mlab = d["dimension"]["Sveitarfélag"]["category"]["label"]
    cit = list(d["dimension"]["Ríkisfang"]["category"]["index"])
    v = d["value"]
    nc = len(cit)
    cells = {(m, c): int(v[i * nc + j] or 0) for i, m in enumerate(mun) for j, c in enumerate(cit)}
    return mun, mlab, cit, cells


def mix_of(iso):
    import fr2023
    if iso in HOME_MIX:
        from countries import load_one
        df = load_one(iso.lower())["counts"]()
        s = df.groupby("node")["count"].sum()
        s = s[s > 0] / s.sum()
        s = s[s >= MIN_SHARE]
        return (s / s.sum()).to_dict()
    v = COUNTRY_LANG[HAG_TO_EU.get(iso, iso)]
    items = [(v, 1.0)] if isinstance(v, str) else list(v.items())
    return {fr2023.NAMES[lab]: f for lab, f in items}


def main():
    if "--fetch" in sys.argv or not RAW.exists():
        fetch()
    mun, mlab, cit, cells = table()
    munis = [m for m in mun if m != "IS"]
    assert len(munis) == MUNIS, len(munis)
    assert cells[("IS", "01")] == TOTAL_2026, cells[("IS", "01")]
    leaves = [c for c in cit if c != "01"]
    for m in mun:   # citizenships partition every municipality
        assert sum(cells[(m, c)] for c in leaves) == cells[(m, "01")], m
    for c in cit:   # municipalities sum to Iceland
        assert sum(cells[(m, c)] for m in munis) == cells[("IS", c)], c
    print(f"Iceland 1 Jan 2026: {TOTAL_2026:,}; {cells[('IS', 'IS')]:,} Icelandic citizens, "
          f"{TOTAL_2026 - cells[('IS', 'IS')]:,} foreign or stateless; partitions checked")

    mixes, keep = {}, {}
    for c in leaves:
        if c == "IS" or c in OTHER or not cells[("IS", c)]:
            continue
        mixes[c] = mix_of(c)
        assert abs(sum(mixes[c].values()) - 1) < 1e-9, c
        eu = HAG_TO_EU.get(c, c)
        keep[c] = 1 - fr_build.TEO_FRENCH[fr_build.teo_region(eu, block_of(eu))]

    rows, before, moved = [], defaultdict(float), 0.0
    for m in munis:
        langs = defaultdict(float)
        langs[ICELANDIC] += cells[(m, "IS")]
        for c in leaves:
            n = cells[(m, c)]
            if not n or c == "IS":
                continue
            if c in OTHER:
                langs["other"] += n
                continue
            for node, f in mixes[c].items():
                before[node] += n * f
                if node == ICELANDIC:
                    langs[node] += n * f
                    continue
                langs[node] += n * f * keep[c]
                langs[ICELANDIC] += n * f * (1 - keep[c])
                moved += n * f * (1 - keep[c])
        tot = cells[(m, "01")]
        assert abs(sum(langs.values()) - tot) < 1e-6, m
        # whole people, largest remainder within the municipality
        s = pd.Series(langs)
        n = s.astype(int)
        n[(s - n).sort_values(ascending=False).index[:tot - int(n.sum())]] += 1
        for node, k in n.items():
            if k > 0:
                rows.append(dict(geo_id=m, geo_level="municipality", geo_name=mlab[m],
                                 source_category=node, count=int(k), tier="derived",
                                 source_id="hagstofa_man04203_2026_x_teo2", year=2026))
    df = pd.DataFrame(rows)
    assert int(df["count"].sum()) == TOTAL_2026 and df["geo_id"].nunique() == MUNIS
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    nat = df.groupby("source_category")["count"].sum().sort_values(ascending=False)
    print(f"wrote {OUT}: {MUNIS} municipalities, {len(nat)} languages, {nat.sum():,} people")
    print(f"TeO2 retention moved {moved:,.0f} onto Icelandic")
    for k, v in nat.head(20).items():
        print(f"  {k:<48} {v:>9,}  ({v / nat.sum():.2%}; before retention {before.get(k, 0):,.0f})")


if __name__ == "__main__":
    main()
