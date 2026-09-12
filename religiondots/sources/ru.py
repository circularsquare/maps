"""Russia — Sreda "Arena" 2012, the only subnational religion source the country has.

Writes data/normalized/ru.csv: 18 religion categories x 79 federal subjects, as PEOPLE.

Usage:
    python sources/ru.py --fetch     # download the Arena workbook and the population table
    python sources/ru.py             # rebuild ru.csv from data/raw/ru/

WHAT MAKES THIS SOURCE UNLIKE EVERY OTHER ONE HERE.

Russia's census has not asked about religion since 1937, and the 2021 census does not ask
either, so there is no census route at all — this is the first country on the map drawn
from a SURVEY as its primary source rather than as a rescale or a supplement (the US uses
Pew, but on top of ASARB's county census). Sreda's Arena polled 56,900 people across 79 of
the then 83 federal subjects in summer 2012 and published one clean cross-tabulation.

Three consequences, all of which the reader has to be told about:

  * IT IS PERCENTAGES, NOT PEOPLE. Arena publishes shares of respondents. The magnitude has
    to come from somewhere else, and spec §3.4 says where: structure from the detailed
    source, totals from the recent one. The 2021 census supplies the population per subject
    and Arena's 2012 shares are applied to it. That is deliberate rather than convenient —
    the Muslim republics grew and the Russian oblasts shrank over the nine years, and
    applying 2012 shares to 2021 populations carries that shift instead of freezing it.

  * ~720 RESPONDENTS PER SUBJECT. The large categories are solid and the tail is noise: a
    0.2% answer in one subject is one or two people. Nothing here is a count of anybody.
    `basis` is self_id and the tier is `modelled`, which is what §7 exists to draw.

  * FOUR SUBJECTS ARE MISSING and they are not missing at random. Arena covers 79 of 83:
    it has no Chechnya, no Ingushetia, no Nenets Autonomous Okrug and no Chukotka. The
    first two are the most Muslim republics in the country, 2.02 million people between
    them, so the hole in this map sits exactly where Sunni Islam is densest. Dagestan, the
    third, IS present. sources/ru.md records what would fill them.

THE CATEGORY LIST IS THE REASON TO BOTHER. Arena asked one question with seventeen offered
answers and they separate things almost nothing else does: the Russian Orthodox Church from
Orthodoxy outside it and from the Old Believers; Sunni from Shia from Muslims who decline
both; and "I believe in God but profess no particular religion" from "I do not believe in
God", which is 25.2% and 13.0% of the country respectively. Those two alone are why this is
worth drawing at 79 blobs.

THE POPULATION TABLE IS SECOND-HAND, AND SAYS SO. Rosstat is the primary source and cannot
be fetched: rosstat.gov.ru presents a certificate chain no Western trust store can verify,
and curl, urllib and the WebFetch service all fail on it identically — a server-side
omission by §9h's test, not local interception. Rather than disable verification for a
whole country's magnitudes, the 2021 census figures are taken from the Wikipedia table that
carries and cites them, parsed from WIKITEXT so the parse is deterministic, and then checked
against the Kontur population surface, which was built independently of both. See ru.md §3
for the Rosstat URLs if the primary is ever wanted.
"""

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "ru")
OUT = os.path.join(ROOT, "data", "normalized", "ru.csv")

ARENA_URL = "https://sreda.org/maps/arena_russia_main/arena_statistic_en.xls"
ARENA_XLS = os.path.join(RAW, "arena_statistic_en.xls")

WP_PAGE = "List of federal subjects of Russia by population"
WP_API = ("https://en.wikipedia.org/w/api.php?action=parse&prop=wikitext&format=json&page="
          + urllib.parse.quote(WP_PAGE))
WP_RAW = os.path.join(RAW, "wp_federal_subjects.wikitext")

SOURCE_ID = "ru_arena_2012"
YEAR = 2012
POP_YEAR = 2021

# Row indices in the Arena sheet's single 'Data' worksheet. The religion question is one
# block of 18 offered answers; everything below row 31 is a different question (agreement
# statements, "I have read the gospel", demographics) and is NOT a partition of anybody.
RELIGION_ROWS = list(range(14, 32))
HEADER_GROUP, HEADER_REGION = 10, 11
NATIONAL_COL = 1

# ---------------------------------------------------------------------------------------
# Arena's own English region names -> ISO 3166-2:RU. Written out in full and asserted
# exhaustive rather than matched by similarity: spec §9j is the case where a name/code join
# matched 96% of a country and put 762,824 people in the wrong district in silence. 79 keys,
# one per Arena column, and the four subjects Arena does not cover are absent by design.
# ---------------------------------------------------------------------------------------
ARENA_TO_ISO = {
    # Central Federal District
    "Belgorod Oblast": "RU-BEL",
    "Bryansk Oblast": "RU-BRY",
    "Vladimir Oblast": "RU-VLA",
    "Voronezh Oblast": "RU-VOR",
    "Ivanovo Oblast": "RU-IVA",
    "Kaluga Oblast": "RU-KLU",
    "Kostroma Oblast": "RU-KOS",
    "Kursk Oblast": "RU-KRS",
    "Lipetsk Oblast": "RU-LIP",
    "Moscow Oblast": "RU-MOS",
    "Oryol Oblast": "RU-ORL",
    "Ryazan Oblast": "RU-RYA",
    "Smolensk Oblast": "RU-SMO",
    "Tambov Oblast": "RU-TAM",
    "Tver Oblast": "RU-TVE",
    "Tula Oblast": "RU-TUL",
    "Yaroslavl Oblast": "RU-YAR",
    "Moscow": "RU-MOW",
    # North-West
    "Republic of Karelia": "RU-KR",
    "Komi Republic": "RU-KO",
    "Arkhangelsk Oblast": "RU-ARK",
    "Vologda Oblast": "RU-VLG",
    "Kaliningrad Oblast": "RU-KGD",
    "Leningrad Oblast": "RU-LEN",
    "Murmansk Oblast": "RU-MUR",
    "Novgorod Oblast": "RU-NGR",
    "Pskov Oblast": "RU-PSK",
    "St.Petersburg": "RU-SPE",
    # Southern
    "Republic of Adygea": "RU-AD",
    "Republic of Kalmykia": "RU-KL",
    "Krasnodar Krai": "RU-KDA",
    "Astrakhan Oblast": "RU-AST",
    "Volgograd Oblast": "RU-VGG",
    "Rostov Oblast": "RU-ROS",
    # North Caucasus  (no Chechnya, no Ingushetia -- see the module docstring)
    "Republic of Dagestan": "RU-DA",
    "Kabardino-Balkar Republic": "RU-KB",
    "Karachay–Cherkess Republic": "RU-KC",
    "Republic of North Ossetia–Alania": "RU-SE",
    "Stavropol Krai": "RU-STA",
    # Volga
    "Republic of Bashkortostan": "RU-BA",
    "Mari El Republic": "RU-ME",
    "Republic of Mordovia": "RU-MO",
    "Republic of Tatarstan": "RU-TA",
    "Udmurt Republic": "RU-UD",
    "Chuvash Republic": "RU-CU",
    "Kirov Oblast": "RU-KIR",
    "Nizhny Novgorod Oblast": "RU-NIZ",
    "Orenburg Oblast": "RU-ORE",
    "Penza Oblast": "RU-PNZ",
    "Samara Oblast": "RU-SAM",
    "Saratov Oblast": "RU-SAR",
    "Ulyanovsk Oblast": "RU-ULY",
    "Perm Krai": "RU-PER",
    # Ural
    "Kurgan Oblast": "RU-KGN",
    "Sverdlovsk Oblast": "RU-SVE",
    "Tyumen Oblast": "RU-TYU",
    "Chelyabinsk Oblast": "RU-CHE",
    "Khanty–Mansi Autonomous Okrug": "RU-KHM",
    "Yamalo-Nenets Autonomous Okrug": "RU-YAN",
    # Siberian
    "Altai Republic": "RU-AL",
    "Republic of Buryatia": "RU-BU",
    "Tyva Republic": "RU-TY",
    "Republic of Khakassia": "RU-KK",
    "Altai Krai": "RU-ALT",
    "Krasnoyarsk Krai": "RU-KYA",
    "Irkutsk Oblast": "RU-IRK",
    "Kemerovo Oblast": "RU-KEM",
    "Novosibirsk Oblast": "RU-NVS",
    "Omsk Oblast": "RU-OMS",
    "Tomsk Oblast": "RU-TOM",
    "Zabaykalsky Krai": "RU-ZAB",
    # Far East  (no Chukotka)
    "Sakha (Yakutia) Republic": "RU-SA",
    "Primorsky Krai": "RU-PRI",
    "Khabarovsk Krai": "RU-KHA",
    "Amur Oblast": "RU-AMU",
    "Magadan Oblast": "RU-MAG",
    "Sakhalin Oblast": "RU-SAK",
    "Jewish Autonomous Oblast": "RU-YEV",
    "Kamchatka Krai": "RU-KAM",
}

# The Wikipedia table's own names for the same subjects. A THIRD spelling of every region,
# which is why both dictionaries are explicit: Arena says "Tyva Republic", geoBoundaries
# says "Tuva", Wikipedia says "Tuva", and none of the three is derivable from another.
WP_TO_ISO = {
    "Moscow": "RU-MOW", "Moscow Oblast": "RU-MOS", "Krasnodar Krai": "RU-KDA",
    "Saint Petersburg": "RU-SPE", "Sverdlovsk Oblast": "RU-SVE",
    "Rostov Oblast": "RU-ROS", "Bashkortostan": "RU-BA", "Tatarstan": "RU-TA",
    "Chelyabinsk Oblast": "RU-CHE", "Dagestan": "RU-DA", "Samara Oblast": "RU-SAM",
    "Nizhny Novgorod Oblast": "RU-NIZ", "Stavropol Krai": "RU-STA",
    "Krasnoyarsk Krai": "RU-KYA", "Novosibirsk Oblast": "RU-NVS",
    "Kemerovo Oblast": "RU-KEM", "Perm Krai": "RU-PER", "Volgograd Oblast": "RU-VGG",
    "Saratov Oblast": "RU-SAR", "Irkutsk Oblast": "RU-IRK", "Voronezh Oblast": "RU-VOR",
    "Altai Krai": "RU-ALT", "Leningrad Oblast": "RU-LEN", "Orenburg Oblast": "RU-ORE",
    "Omsk Oblast": "RU-OMS", "Primorsky Krai": "RU-PRI", "Tyumen Oblast": "RU-TYU",
    "Chechnya": "RU-CE", "Belgorod Oblast": "RU-BEL", "Tula Oblast": "RU-TUL",
    "Udmurtia": "RU-UD", "Vladimir Oblast": "RU-VLA", "Penza Oblast": "RU-PNZ",
    "Tver Oblast": "RU-TVE", "Yaroslavl Oblast": "RU-YAR", "Ulyanovsk Oblast": "RU-ULY",
    "Chuvashia": "RU-CU", "Bryansk Oblast": "RU-BRY", "Kirov Oblast": "RU-KIR",
    "Vologda Oblast": "RU-VLG", "Lipetsk Oblast": "RU-LIP", "Ryazan Oblast": "RU-RYA",
    "Kaluga Oblast": "RU-KLU", "Kursk Oblast": "RU-KRS", "Tomsk Oblast": "RU-TOM",
    "Kaliningrad Oblast": "RU-KGD", "Zabaykalsky Krai": "RU-ZAB", "Buryatia": "RU-BU",
    "Tambov Oblast": "RU-TAM", "Astrakhan Oblast": "RU-AST",
    "Kabardino-Balkaria": "RU-KB", "Ivanovo Oblast": "RU-IVA",
    "Smolensk Oblast": "RU-SMO", "Mordovia": "RU-MO", "Amur Oblast": "RU-AMU",
    "Kurgan Oblast": "RU-KGN", "Komi Republic": "RU-KO", "Oryol Oblast": "RU-ORL",
    "North Ossetia–Alania": "RU-SE", "Mari El": "RU-ME", "Murmansk Oblast": "RU-MUR",
    "Pskov Oblast": "RU-PSK", "Novgorod Oblast": "RU-NGR", "Kostroma Oblast": "RU-KOS",
    "Ingushetia": "RU-IN", "Khakassia": "RU-KK", "Karelia": "RU-KR", "Adygea": "RU-AD",
    "Karachay-Cherkessia": "RU-KC", "Sakhalin Oblast": "RU-SAK", "Tuva": "RU-TY",
    "Kamchatka Krai": "RU-KAM", "Kalmykia": "RU-KL", "Altai Republic": "RU-AL",
    "Jewish Autonomous Oblast": "RU-YEV", "Magadan Oblast": "RU-MAG",
    "Chukotka": "RU-CHU", "Nenets Autonomous Okrug": "RU-NEN",
    "Khanty–Mansi Autonomous Okrug": "RU-KHM", "Khabarovsk Krai": "RU-KHA",
    "Sakha": "RU-SA", "Arkhangelsk Oblast": "RU-ARK",
    "Yamalo-Nenets Autonomous Okrug": "RU-YAN",
}


def fetch():
    """Both files. Neither host needs a key, a session or a browser."""
    os.makedirs(RAW, exist_ok=True)
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) religiondots"}

    req = urllib.request.Request(ARENA_URL, headers=ua)
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
    # §5a: assert size AND type. An HTTP 200 carrying an error page is the usual failure.
    if body[:4].hex() != "d0cf11e0" or len(body) < 100_000:
        raise SystemExit(f"Arena download is not an OLE2 workbook: {len(body)} bytes, "
                         f"magic {body[:4].hex()}")
    open(ARENA_XLS, "wb").write(body)
    print(f"  arena_statistic_en.xls  {len(body):,} bytes")

    req = urllib.request.Request(WP_API, headers=ua)
    with urllib.request.urlopen(req, timeout=90) as r:
        doc = json.load(r)
    text = doc["parse"]["wikitext"]["*"]
    open(WP_RAW, "w", encoding="utf-8").write(text)
    print(f"  wp_federal_subjects.wikitext  {len(text):,} chars")


def read_population():
    """2021 census population per federal subject, keyed by ISO code.

    The table writes a subject four different ways -- `{{flag|X}}`, `{{flag|X|name=Y}}`,
    with a trailing `{{efn|...}}` footnote, and with a third historical figure inside the
    change template -- so the pattern allows all four and the count is asserted afterwards.
    """
    if not os.path.exists(WP_RAW):
        raise SystemExit(f"missing {WP_RAW} -- run sources/ru.py --fetch first")
    text = open(WP_RAW, encoding="utf-8").read()

    row = re.compile(
        r"\|\s*(?:''')?\{\{flag\|([^}|]+?)(?:\|name=[^}]*)?\}\}(?:''')?"
        r"(?:\{\{efn\|[^}]*\}\})?\s*\n"
        r"\|\s*\{\{change\|invert=on\|([\d,]+)\|([\d,]+)"
    )
    pop, seen = {}, {}
    for m in row.finditer(text):
        name = m.group(1).strip()
        if name == "Russian Federation":
            continue
        iso = WP_TO_ISO.get(name)
        if iso is None:
            raise SystemExit(f"Wikipedia names a subject this file does not know: {name!r}. "
                             "The table changed -- update WP_TO_ISO rather than guessing.")
        # group(2) is the 2025 estimate, group(3) the 2021 census; the header order was
        # verified against Moscow (13,010,112 in 2021, 13,258,262 in 2025).
        pop[iso] = int(m.group(3).replace(",", ""))
        seen[iso] = name

    missing = set(WP_TO_ISO.values()) - set(pop)
    if missing:
        raise SystemExit(f"no population parsed for {sorted(missing)} -- the row markup "
                         "changed again; fix the pattern, do not drop the subjects")
    print(f"  population: {len(pop)} subjects, {sum(pop.values()):,} people ({POP_YEAR} census)")
    return pop


def read_arena():
    """The religion block: {iso: {category: share}}, plus the national column."""
    import pandas as pd

    if not os.path.exists(ARENA_XLS):
        raise SystemExit(f"missing {ARENA_XLS} -- run sources/ru.py --fetch first")
    df = pd.read_excel(ARENA_XLS, sheet_name="Data", header=None)

    cats = []
    for i in RELIGION_ROWS:
        v = df.iat[i, 0]
        if not isinstance(v, str) or not v.strip():
            raise SystemExit(f"Arena row {i} has no label -- the sheet layout changed")
        cats.append((i, v.strip()))
    if len(cats) != 18:
        raise SystemExit(f"expected 18 religion categories, found {len(cats)}")

    cols = {}
    for j in range(1, df.shape[1]):
        name = df.iat[HEADER_REGION, j]
        if isinstance(name, str) and name.strip():
            cols[j] = name.strip()

    unknown = sorted(set(cols.values()) - set(ARENA_TO_ISO))
    if unknown:
        raise SystemExit(f"Arena columns with no ISO code: {unknown}")
    if len(cols) != 79:
        raise SystemExit(f"expected 79 region columns, found {len(cols)}")

    out = {}
    for j, name in cols.items():
        iso = ARENA_TO_ISO[name]
        shares = {}
        for i, cat in cats:
            v = df.iat[i, j]
            shares[cat] = 0.0 if v is None or (isinstance(v, float) and v != v) else float(v)
        total = sum(shares.values())
        # Arena's own columns sum to 100 in all 79 subjects. Anything else is a layout
        # change, not a rounding artefact, so it stops the run.
        if abs(total - 100.0) > 1.5:
            raise SystemExit(f"{name} ({iso}) sums to {total:.2f}, not 100 -- wrong rows?")
        out[iso] = shares

    national = {cat: float(df.iat[i, NATIONAL_COL]) for i, cat in cats}
    return out, national, [c for _, c in cats]


def to_people(shares, population):
    """shares (%) x population -> whole people that sum EXACTLY to the population.

    Largest remainder, and it is safe here in a way spec §4.1a says it is not in general:
    that warning is about spreading one category across many UNITS, where rank on absolute
    count hands every dot to the densest places. This apportions the categories WITHIN one
    subject, so no geography is being decided and the only thing at stake is which category
    absorbs the last few people.

    The shares are renormalised to sum to 1 first. Arena publishes each share rounded
    independently, so a column can total 99.6 or 100.4 -- Kostroma is 100.37 -- and taking
    the percentages at face value makes the floors overshoot the population and the
    remainder negative. Renormalising is the difference between an apportionment and an
    off-by-a-few-people bug that only shows up in some subjects.
    """
    scale = sum(shares.values())
    if scale <= 0:
        raise SystemExit("a subject has no religion shares at all")
    exact = {k: v / scale * population for k, v in shares.items()}
    floors = {k: int(v) for k, v in exact.items()}
    short = population - sum(floors.values())
    order = sorted(exact, key=lambda k: exact[k] - floors[k], reverse=True)
    for k in order[:short]:
        floors[k] += 1
    return floors


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true", help="download the two raw files first")
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    if args.fetch:
        fetch()

    import pandas as pd

    pop = read_population()
    arena, national, cats = read_arena()

    have = set(arena)
    absent = sorted(set(WP_TO_ISO.values()) - have)
    lost = sum(pop[i] for i in absent)
    print(f"  Arena covers {len(have)} of {len(WP_TO_ISO)} subjects; absent: {absent} "
          f"({lost:,} people, {lost / sum(pop.values()):.2%} of Russia)")

    rows = []
    drawn_pop = 0
    for iso in sorted(arena):
        p = pop[iso]
        drawn_pop += p
        counts = to_people(arena[iso], p)
        if sum(counts.values()) != p:
            raise SystemExit(f"{iso}: apportionment lost people")
        for cat in cats:
            rows.append({
                "geo_id": iso, "geo_level": "subject", "geo_name": iso,
                "source_category": cat, "count": counts[cat],
                "basis": "self_id", "year": YEAR, "source_id": SOURCE_ID,
                "note": (f"share={arena[iso][cat]:.4f}%; population={p} "
                         f"({POP_YEAR} census); survey n~720 in this subject"),
            })

    # The national column, for cross-checking only. countries.py reads geo_level=='subject'.
    for cat in cats:
        rows.append({
            "geo_id": "RU", "geo_level": "country", "geo_name": "Russian Federation",
            "source_category": cat, "count": round(national[cat] / 100.0 * drawn_pop),
            "basis": "self_id", "year": YEAR, "source_id": SOURCE_ID,
            "note": f"share={national[cat]:.4f}%; Arena's own national column, n=56,900",
        })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"  wrote {OUT}  {len(out):,} rows")
    print(f"  population on the 79 drawn subjects: {drawn_pop:,}")

    # ---- what the national column says, biggest first, as the readable check ----
    print("\n  Arena's national shares:")
    for cat in sorted(cats, key=lambda c: -national[c]):
        print(f"    {national[cat]:6.2f}%  {cat[:88]}")

    # ---- and the check that the SUBJECT columns agree with the national one. They are
    # separately weighted, so this is a relationship, not an identity (§9i).
    print("\n  subject-weighted vs Arena's own national column:")
    for cat in sorted(cats, key=lambda c: -national[c])[:8]:
        w = sum(arena[i][cat] * pop[i] for i in arena) / drawn_pop
        print(f"    {w:6.2f}% vs {national[cat]:6.2f}%   {cat[:70]}")


if __name__ == "__main__":
    main()
