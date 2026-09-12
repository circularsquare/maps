"""Russia — the four federal subjects Arena did not survey, filled from census ethnicity.

Reads  data/normalized/ru.csv          (79 subjects, Arena 2012)
       data/raw/ru/Tom5_tab1_VPN-2020.xlsx   (2021 census national composition)
Writes data/normalized/ru_filled.csv   (83 subjects; countries.py reads this one)

Usage:
    python ru_fill.py --fetch    # download the census ethnicity workbook first
    python ru_fill.py

WHY THIS EXISTS. Sreda's Arena covers 79 of Russia's 83 federal subjects. The four it
misses are Chechnya, Ingushetia, Nenets AO and Chukotka AO — and the first two are the most
Muslim republics in the country, 2.02 million people. Leaving them blank does not read as
"no data" on a dot map, it reads as "nobody lives there", and it makes Russia look less
Muslim than it is. Anita's call, 2026-09-05.

WHAT IS AND IS NOT CLAIMED. spec §14.4 says never estimate a magnitude a source does not
publish. This estimates one, so it is worth being exact about the footing:

  * The *geography* is measured — these are real subjects with real census populations.
  * The *religion* is inferred from the 2021 census ethnic composition, which IS measured,
    through a relationship FITTED ON ARENA'S OWN 79 MEASURED SUBJECTS. It is not a guess
    about Chechnya imported from outside; it is what Arena's instrument records for
    populations of this ethnic composition everywhere else in Russia.
  * Every row is `tier = modelled` — but so is every other Russian row, because a
    720-per-region survey is already modelled. The fill is marked by `source_id` instead,
    so it can always be separated out.

THE FIT, and it is better than expected. Across the 79 subjects Arena measured:

    islam_answer = 0.724 x muslim_ethnic_share        R² = 0.964

That slope is the point of the whole exercise. **Only about seven in ten ethnically Muslim
Russians give an Islam answer**; the rest say "I believe in God but profess no particular
religion", "I do not believe in God", or "difficult to answer". A naive fill would put 98%
of Chechnya on Islam and make it the most religiously uniform unit on the entire map —
more uniform than anything anyone has actually measured anywhere.

WHICH RATIO IS USED, AND WHY NOT THE NATIONAL ONE FOR ALL FOUR. The linear fit
underpredicts at the top: Dagestan is 95.9% Muslim by ethnicity and answered 82.6% Islam,
against 69.4% predicted. Religiosity rises with how homogeneous the community is, so
extrapolating a straight line to Chechnya's 98.5% is exactly where it is weakest. Chechnya
and Ingushetia therefore take **Dagestan's own measured ratio (0.861)** — the nearest
republic, the same Shafi'i Sufi tradition, and the only comparable one Arena reached. The
two Arctic okrugs take the national slope, where the Muslim share is ~2% and the choice
changes nothing.

THE REST OF THE PROFILE comes from a donor subject Arena did measure, renormalised to fill
whatever Islam does not take:

    Chechnya, Ingushetia  <- Dagestan            (the Muslim North Caucasus republic)
    Nenets AO             <- Arkhangelsk Oblast  (Nenets AO is administratively inside it)
    Chukotka AO           <- Magadan Oblast      (its neighbour, same settlement history)

ONE LIMITATION WORTH NAMING. Chukotka is 28% Chukchi and Nenets AO 18% Nenets, and their
donors have much smaller indigenous populations — so traditional religion is understated in
both. The same ethnic method would fix it, but it is 89,000 people between them, about 89
dots, and a second fitted model doubles the surface for error to move them. Not done, and
recorded here rather than left to be rediscovered.
"""

import argparse
import os
import re
import sys
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "4")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "sources"))

RAW = os.path.join(HERE, "data", "raw", "ru")
XLSX = os.path.join(RAW, "Tom5_tab1_VPN-2020.xlsx")
NORM = os.path.join(HERE, "data", "normalized", "ru.csv")
OUT = os.path.join(HERE, "data", "normalized", "ru_filled.csv")

# rosstat.gov.ru cannot be verified by any Western trust store (sources/ru.md §3), but the
# Wayback Machine can, and it holds the file the census cites. This is the general answer to
# a TLS-walled statistical office and it is worth remembering.
XLSX_URL = ("https://web.archive.org/web/20221230204643if_/"
            "https://rosstat.gov.ru/storage/mediabank/Tom5_tab1_VPN-2020.xlsx")

FILL_SOURCE_ID = "ru_ethnic_fill_2021"

# Sheet name -> the subject being filled, its ISO, and the subject whose measured profile
# supplies everything Islam does not take.
FILLS = {
    "RU-CE": dict(sheet="Чеченская Республика", donor="RU-DA", ratio="dagestan"),
    "RU-IN": dict(sheet="Республика Ингушетия", donor="RU-DA", ratio="dagestan"),
    "RU-NEN": dict(sheet="Ненецкий автономный округ", donor="RU-ARK", ratio="national"),
    "RU-CHU": dict(sheet="Чукотский автономный округ", donor="RU-MAG", ratio="national"),
}

# Peoples whose religious tradition is Islam, at Rosstat's top level (Code02). Two absences
# are deliberate and both would be common errors: OSSETIANS are majority Orthodox with a
# large traditional-religion minority, and YAZIDIS are not Muslim however often they are
# filed as such.
MUSLIM_PEOPLES = {
    "абазины", "аварцы", "агулы", "адыгейцы", "азербайджанцы", "балкарцы", "башкиры",
    "даргинцы", "ингуши", "кабардинцы", "казахи", "карачаевцы", "киргизы", "кумыки",
    "курды", "лакцы", "лезгины", "ногайцы", "рутульцы", "табасараны", "таджики",
    "татары", "турки", "туркмены", "узбеки", "уйгуры", "цахуры", "черкесы", "чеченцы",
    "афганцы", "арабы", "крымские татары", "шапсуги", "андийцы",
}

ISLAM_CATEGORY = "I profess Islam, but am neither Sunni nor Shia"
ALL_ISLAM = re.compile(r"\bIslam\b")


def fetch():
    os.makedirs(RAW, exist_ok=True)
    req = urllib.request.Request(XLSX_URL, headers={"User-Agent": "religiondots/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        body = r.read()
    if body[:2] != b"PK" or len(body) < 500_000:
        raise SystemExit(f"census ethnicity download is not an xlsx: {len(body)} bytes")
    open(XLSX, "wb").write(body)
    print(f"  Tom5_tab1_VPN-2020.xlsx  {len(body):,} bytes")


def _name(cell):
    """Rosstat writes 'Аварцы (аварал, …)'; the head word before the bracket is the name."""
    s = re.sub(r"\(.*?\)", "", str(cell)).strip().lower()
    return re.sub(r"\s+", " ", s)


def muslim_shares():
    """{sheet name: (population, stated-ethnicity base, Muslim share of the stated base)}.

    Only Code02 rows are summed. Rosstat nests sub-peoples under their parent as Code01 —
    Andi and Didoi inside Avars, Kryashen inside Tatars — and adding both levels counts the
    same people twice. Brazil's classification-133 trap (sources.md §9b), in Cyrillic.
    """
    import pandas as pd

    if not os.path.exists(XLSX):
        raise SystemExit(f"missing {XLSX} -- run `python ru_fill.py --fetch` first")
    xl = pd.ExcelFile(XLSX)
    out = {}
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, header=None)
        tag = df.iloc[:, 0].astype(str)
        total = stated = None
        muslim = 0.0
        for i in range(len(df)):
            t = tag.iat[i]
            if not isinstance(t, str):
                continue
            try:
                v = float(df.iat[i, 2])
            except (TypeError, ValueError):
                continue
            if "[Hierarchy].[All]" in t:
                total = v
            elif "[Code03]" in t:
                stated = v
            elif "[Code02]" in t and _name(df.iat[i, 1]) in MUSLIM_PEOPLES:
                muslim += v
        if total and stated:
            out[sheet] = (total, stated, muslim / stated)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        fetch()

    import pandas as pd
    from ru import read_population

    pop = read_population()
    shares = muslim_shares()

    # ---- join sheets to ISO codes BY POPULATION. Each sheet carries its own "Все
    # население" and the census population per subject is already keyed by ISO, so a unique
    # numeric match identifies the sheet and checks itself — no fourth spelling of 83
    # region names to maintain.
    by_pop = {}
    for iso, p in pop.items():
        by_pop.setdefault(p, []).append(iso)
    iso_of = {}
    for sheet, (total, _stated, _m) in shares.items():
        hit = by_pop.get(int(total), [])
        if len(hit) == 1:
            iso_of[sheet] = hit[0]
    print(f"  {len(shares)} sheets, {len(iso_of)} matched an ISO by population")

    m_by_iso = {iso_of[s]: shares[s][2] for s in iso_of}

    # ---- Arena's measured profiles
    df = pd.read_csv(NORM, dtype={"geo_id": str}, low_memory=False)
    sub = df[df["geo_level"] == "subject"]
    tot = sub.groupby("geo_id")["count"].sum()
    isl = sub[sub["source_category"].str.contains(ALL_ISLAM, na=False)] \
        .groupby("geo_id")["count"].sum()
    arena_islam = (isl / tot).reindex(tot.index).fillna(0.0)

    # ---- the fit, on the 79 measured, reported every run so it cannot rot silently
    both = [(m_by_iso[i], arena_islam[i]) for i in tot.index if i in m_by_iso]
    xs = pd.Series([a for a, _ in both]); ys = pd.Series([b for _, b in both])
    slope = (xs * ys).sum() / (xs ** 2).sum()
    r2 = 1 - ((ys - slope * xs) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    dag = arena_islam["RU-DA"] / m_by_iso["RU-DA"]
    print(f"  fit on {len(both)} measured subjects: islam = {slope:.4f} x ethnic  "
          f"R2 = {r2:.4f}")
    print(f"  Dagestan's own ratio: {dag:.4f} "
          f"(ethnic {m_by_iso['RU-DA']:.3f} -> measured {arena_islam['RU-DA']:.3f})")

    cats = list(sub[sub["geo_id"] == "RU-DA"]["source_category"])
    rows = []
    for iso, spec in FILLS.items():
        m = m_by_iso.get(iso)
        if m is None:
            raise SystemExit(f"no ethnic share for {iso} -- sheet {spec['sheet']!r} "
                             "did not match a population")
        ratio = dag if spec["ratio"] == "dagestan" else slope
        islam = min(m * ratio, m)          # cannot exceed the ethnically Muslim population
        population = pop[iso]

        donor = sub[sub["geo_id"] == spec["donor"]].set_index("source_category")["count"]
        donor = donor / donor.sum()
        non_islam = donor[[c for c in donor.index if not ALL_ISLAM.search(c)]]
        non_islam = non_islam / non_islam.sum() * (1.0 - islam)

        profile = {ISLAM_CATEGORY: islam}
        for c, v in non_islam.items():
            profile[c] = profile.get(c, 0.0) + float(v)

        # whole people, summing exactly to the census population (sources/ru.py's rule)
        exact = {c: v * population for c, v in profile.items()}
        floors = {c: int(v) for c, v in exact.items()}
        short = population - sum(floors.values())
        for c in sorted(exact, key=lambda k: exact[k] - floors[k], reverse=True)[:short]:
            floors[c] += 1
        if sum(floors.values()) != population:
            raise SystemExit(f"{iso}: apportionment lost people")

        print(f"  {iso}: ethnic {m:.4f} -> islam {islam:.4f}  "
              f"({floors[ISLAM_CATEGORY]:,} of {population:,}), donor {spec['donor']}")
        for c in cats:
            rows.append({
                "geo_id": iso, "geo_level": "subject", "geo_name": iso,
                "source_category": c, "count": floors.get(c, 0),
                "basis": "self_id", "year": 2012, "source_id": FILL_SOURCE_ID,
                "note": (f"filled from 2021 census ethnicity: muslim_ethnic_share={m:.4f}, "
                         f"ratio={ratio:.4f}, donor={spec['donor']}; population={population}"),
            })

    out = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    drawn = out[out["geo_level"] == "subject"].groupby("geo_id")["count"].sum()
    print(f"\n  wrote {OUT}  {len(out):,} rows, {drawn.index.nunique()} subjects, "
          f"{drawn.sum():,} people")

    filled_islam = out[(out["source_id"] == FILL_SOURCE_ID)
                       & (out["source_category"] == ISLAM_CATEGORY)]["count"].sum()
    print(f"  the fill adds {filled_islam:,} Muslims and "
          f"{drawn.sum() - sub['count'].sum():,} people in total")


if __name__ == "__main__":
    main()
