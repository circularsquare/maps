"""Caribbean Netherlands (Bonaire, Sint Eustatius, Saba): CBS Omnibus survey 2021, main language
(voertaal) of persons aged 15 and over, per island, on CBS's island populations
-> data/normalized/bq.csv.

    python sources/bq_survey.py [--fetch]

THE TABLE. CBS StatLine 82867NED, *Caribisch Nederland; gesproken talen en voertaal,
persoonskenmerken* (rounds 2013, 2017/2018, 2021; the 2021 figures are marked provisional). The
Omnibus survey asked "welke taal of welke talen spreekt u?" (several answers: Papiaments, Engels,
Nederlands, Spaans, Anders) and, of anyone naming more than one, "welke taal spreekt u het meest?".
The *Voertaal* columns (Papiaments_8 .. Anders_12) are the language each respondent speaks most,
one per person, as a percentage of persons 15+ per island with a 95% margin (Marges B000150).
Fieldwork October to December 2021.

The shares are applied to each island's population on 1 January 2022 (StatLine 83774NED), the
date nearest the fieldwork, as religiondots does with the same survey's religion table: the
islands have grown by about a fifth since, through immigration, and a later base would describe
people the survey did not. Every count is a survey share times a population, so `tier` is
`modelled`. Children under 15 were not asked and are drawn on their island's adult shares.

WITHHELD CELLS. CBS prints nothing where an estimate is too unreliable (Saba's Papiamentu in 2021).
The printed shares sum to 100.1 (Bonaire), 100.1 (Sint Eustatius) and 99.8 (Saba); where a cell is
withheld, the remainder becomes a `Niet gepubliceerd` row, not drawn.

THE CHECKS: the three populations pinned (22,573 / 3,242 / 1,911); a few 2021 shares pinned so a
revision of the provisional figures fails loudly; each island's printed shares sum to 99-101; every
main-language share is at most the share who speak that language at all (Gesproken talen,
Papiaments_3 .. Anders_7), since the language spoken most is one of those spoken. The CN01
(all islands) row is NOT population-weighted (English 43.1% there, against 19.8% from the islands
weighted by population) and is not used; see sources/bq.md.
"""
import csv
import json
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "raw" / "bq"
OUT = HERE / "data" / "normalized" / "bq.csv"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
API = "https://opendata.cbs.nl/ODataApi/odata/"
FILES = {
    "lang": (API + "82867NED/TypedDataSet?$filter=Persoonskenmerken%20eq%20%27T009002%27"
                   "&$format=json", RAW / "82867NED_T009002.json"),
    "meta": (API + "82867NED/TableInfos?$format=json", RAW / "82867NED_TableInfos.json"),
    "pop": (API + "83774NED/TypedDataSet?$format=json", RAW / "83774NED_TypedDataSet.json"),
}
SOURCE_ID = "cbs_82867NED_2021"
PERIOD, POP_PERIOD = "2021JJ00", "2022JJ00"
SHARE, MARGIN = "MW00000", "B000150"
ISLANDS = {"GM9001": "Bonaire", "GM9002": "Sint Eustatius", "GM9003": "Saba"}
EXPECTED_POP = {"GM9001": 22_573, "GM9002": 3_242, "GM9003": 1_911}
# (main-language column, spoken-at-all column, label)
TOPICS = [("Papiaments_8", "Papiaments_3", "Papiaments"), ("Engels_9", "Engels_4", "Engels"),
          ("Nederlands_10", "Nederlands_5", "Nederlands"), ("Spaans_11", "Spaans_6", "Spaans"),
          ("Anders_12", "Anders_7", "Anders")]
PINNED = {("GM9001", "Papiaments"): 62.4, ("GM9002", "Engels"): 81.2,
          ("GM9003", "Engels"): 83.3, ("GM9003", "Spaans"): 9.9}
WITHHELD = "Niet gepubliceerd"


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for url, path in FILES.values():
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        body = urllib.request.urlopen(req, timeout=120).read()
        json.loads(body)                      # a 200 is not a download
        tmp = path.with_suffix(".part")
        tmp.write_bytes(body)
        tmp.replace(path)
        print(f"  {path.name}: {len(body):,} bytes")


def rows(key):
    path = FILES[key][1]
    if not path.exists():
        raise SystemExit(f"missing {path}; run with --fetch")
    return json.loads(path.read_text(encoding="utf-8"))["value"]


def main():
    if "--fetch" in sys.argv:
        fetch()
    lang = rows("lang")
    pops = {r["CaribischNederland"].strip(): r["BevolkingOp1Januari_1"]
            for r in rows("pop") if r["Perioden"] == POP_PERIOD}
    out, drawn = [], 0
    weighted = {lab: 0.0 for _, _, lab in TOPICS}
    for code, island in ISLANDS.items():
        pop = pops.get(code)
        if pop != EXPECTED_POP[code]:
            raise SystemExit(f"{island}: population {pop} on {POP_PERIOD}, expected "
                             f"{EXPECTED_POP[code]:,}")
        pick = {r["Marges"]: r for r in lang
                if r["CaribischNederland"].strip() == code and r["Perioden"] == PERIOD}
        if set(pick) != {SHARE, MARGIN}:
            raise SystemExit(f"{island}: no {PERIOD} share and margin rows")
        s, m = pick[SHARE], pick[MARGIN]
        printed = {lab: s[k] for k, _, lab in TOPICS if s[k] is not None}
        for (c, lab), want in PINNED.items():
            if c == code and printed.get(lab) != want:
                raise SystemExit(f"{island}: {lab} is {printed.get(lab)}, pinned {want}")
        tot = sum(printed.values())
        if not 99.0 <= tot <= 101.0:
            raise SystemExit(f"{island}: printed main-language shares sum to {tot:.1f}")
        for k, k_any, lab in TOPICS:
            if s[k] is not None and s[k_any] is not None and s[k] > s[k_any] + 0.05:
                raise SystemExit(f"{island}: {lab} main {s[k]} > spoken at all {s[k_any]}")
        used = 0
        for k, k_any, lab in TOPICS:
            if s[k] is None:
                continue
            n = round(s[k] / 100.0 * pop)
            used += n
            weighted[lab] += s[k] * pop
            out.append(dict(geo_id=code, geo_level="island", geo_name=island,
                            source_category=lab, count=n, tier="modelled", source_id=SOURCE_ID,
                            year=2021, note=f"main language {s[k]}% (margin {m[k]}), speaks it "
                                            f"{s[k_any]}%; persons 15+; population 1 Jan 2022 "
                                            f"{pop}"))
        withheld = [lab for k, _, lab in TOPICS if s[k] is None]
        rest = pop - used
        if withheld:
            out.append(dict(geo_id=code, geo_level="island", geo_name=island,
                            source_category=WITHHELD, count=rest, tier="modelled",
                            source_id=SOURCE_ID, year=2021,
                            note=f"100 minus the printed shares ({tot:.1f}); withheld: "
                                 f"{', '.join(withheld)}"))
        drawn += used
        print(f"  {island:<15} pop {pop:>6,}  printed {tot:5.1f}%  drawn {used:>6,}  "
              f"rest {rest:>4}  withheld: {', '.join(withheld) or 'none'}")
        # the earlier rounds, for the record
        for per in ("2013JJ00", "2017JJ00"):
            r = next(r for r in lang if r["CaribischNederland"].strip() == code
                     and r["Perioden"] == per and r["Marges"] == SHARE)
            print(f"      {per[:4]}: " + ", ".join(f"{lab} {r[k]}" for k, _, lab in TOPICS))

    total_pop = sum(EXPECTED_POP.values())
    cn = next(r for r in lang if r["CaribischNederland"].strip() == "CN01"
              and r["Perioden"] == PERIOD and r["Marges"] == SHARE)
    print("  islands weighted by population vs CBS's CN01 row:")
    for k, _, lab in TOPICS:
        print(f"      {lab:<11} {weighted[lab] / total_pop:5.1f}  vs  {cn[k]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["geo_id", "geo_level", "geo_name", "source_category",
                                          "count", "tier", "source_id", "year", "note"])
        w.writeheader()
        w.writerows(out)
    tmp.replace(OUT)
    print(f"  wrote {OUT.relative_to(HERE)}: {len(out)} rows, {drawn:,} drawn of {total_pop:,}")


if __name__ == "__main__":
    main()
