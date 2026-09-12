"""Where England's Orthodox Christians are, from country of birth.

`sources/uk_orthodox.py` -> `data/normalized/uk_orthodox.csv`

WHY A PROXY AT ALL, WHEN EVERY OTHER LEG HAS A REGISTER. Orthodoxy is the one denomination
in England's split that **neither placement source can find**:

  * The English Church Census 2005 got 49 Orthodox churches at a **7% response rate** -- its
    worst by a factor of four, because 8 May 2005 fell the Sunday after Orthodox Easter and
    many were shut. Zero of its 94 county-by-settlement cells reach three churches.
  * OpenStreetMap has **129 Orthodox churches in all of England**, a large undercount:
    Orthodox parishes here are mostly post-2004, many rent Anglican buildings on a Sunday
    afternoon and are mapped as the Anglican church they meet in, and a congregation without
    its own building is usually not mapped at all. Placing by that count put **6.0% of
    Norfolk's Christians on Orthodoxy**, which is nonsense.

Both blindnesses have the same cause and it is not carelessness: **a 2005 survey and a map of
buildings are both describing a country that Orthodoxy had not yet arrived in.** The British
Election Study puts Orthodoxy at 2.17% of England's Christians, about 568,000 people, larger
than the Baptists. Almost all of that is Romanian, Bulgarian, Greek, Ukrainian and Moldovan
migration, and most of it postdates the church census entirely.

So this file places Orthodoxy by the thing the 2021 census does record at fine geography and
which does track that population: **where people born in Orthodox-majority countries live.**
Anita, 2026-09-07: "okay we can place orthodox by origin."

WHAT IT IS AND IS NOT. This is `modelled` (§7), the only leg in England's split that is. It
is not a count of Orthodox people and nothing here claims to be one: the census asked country
of birth, not religion, and the magnitude comes entirely from the anchor. What the proxy
supplies is the SHAPE -- Haringey and Enfield for the Greek Cypriots, Harrow and Wembley for
the Romanians, Peterborough, Boston and Wisbech for the Romanians and Bulgarians in the
fenland agriculture, Newham and Barking for the more recent arrivals.

**THE GEOGRAPHY IS MSOA AND THAT IS THE FLOOR, NOT A CHOICE.** ONS publishes country of birth
at 60 categories down to local authority and at 190 categories down to MSOA; asking for
either below MSOA returns **HTTP 400 with an empty error body**, which is disclosure control
refusing rather than a row cap. The distinction matters and cost half an hour: at MSOA the
same request returns **403 "Too many rows"**, which IS a row cap and is solved by asking for
200 areas at a time. A 400 is a wall and a 403 is a queue.

**60 CATEGORIES IS NOT ENOUGH AND NOMIS ONLY HAS 60.** In the classification the bulk
downloads use, Greece is inside `Other member countries in March 2001` with the Netherlands
and Sweden, Bulgaria and Cyprus are inside `Other EU countries` with Czechia and Hungary, and
Ukraine, Russia, Serbia and Moldova are inside `Rest of Europe: Other Europe` with Norway and
Switzerland. **Romania is the only Orthodox country it separates.** Only the 190-category
classification, which is API-only, names the rest.

THE WEIGHTS, AND WHY THEY ARE NOT OPTIONAL. Born-in is not believes-in, and the error is not
even across countries. Unweighted, **Albania would carry 5.9% of England's Orthodox geography
while being about 7% Orthodox** -- Albania is a Muslim-majority country, and much of its
diaspora here is Kosovar. Cyprus is a second case: the Republic is about 89% Orthodox but
Britain's Cypriot population includes a large Turkish Cypriot minority the census cannot
separate from a birthplace.

So each origin is weighted by the Orthodox share of its own population, from that country's
own census where it has one. **These are the only invented-looking numbers in England's
split**, so each carries its source, and `check()` prints the weighted national mix for
inspection. They move the shape and not the size: the anchor fixes the total either way.

Run: python sources/uk_orthodox.py            # -> data/normalized/uk_orthodox.csv
     python sources/uk_orthodox.py --report   # print the mix and the checks, write nothing
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "uk")
COB = os.path.join(RAW, "msoa_country_of_birth.csv")
OUT = os.path.join(ROOT, "data", "normalized", "uk_orthodox.csv")

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

SOURCE_ID = "uk_en_orthodox_proxy_2021"
BASIS = "estimate"        # a compiler's judgement over a census variable, spec §3.1
YEAR = 2021

# Country-of-birth category id (country_of_birth_190a) -> (name, Orthodox share).
#
# The share is of that country's OWN population, Eastern and Oriental Orthodox together,
# because the anchor's option is a single "Orthodox Christian" and does not separate them.
# Sources are each country's own census unless noted.
ORIGINS = {
    35: ("Romania", 0.855),      # 2021 census, 85.3% Orthodox of those stating
    26: ("Bulgaria", 0.715),     # 2021 census, 71.5% Eastern Orthodox of those answering
    19: ("Greece", 0.90),        # no religion question since 1951; the conventional figure
    27: ("Cyprus", 0.70),        # Republic census ~89%, reduced for the Turkish Cypriot
                                 # share of Britain's Cypriot population, which a birthplace
                                 # cannot separate. The least certain number in this table.
    49: ("Moldova", 0.90),       # 2014 census, 90.1%
    52: ("Russia", 0.70),        # no census question; Sreda Arena 2012, the same source
                                 # sources/ru.md uses for Russia itself
    56: ("Ukraine", 0.75),       # no census since 2001; survey consensus
    53: ("Serbia", 0.81),        # 2022 census, 81.1%
    42: ("Belarus", 0.75),       # no census question; survey consensus
    45: ("Georgia", 0.834),      # 2014 census, 83.4%
    48: ("North Macedonia", 0.46),   # 2021 census, 46.1% -- a plurality, not a majority
    50: ("Montenegro", 0.71),    # 2011 census, 72.1%
    43: ("Bosnia and Herzegovina", 0.31),   # 2013 census, 30.7% Orthodox
    39: ("Albania", 0.07),       # 2011 census, 6.8% Orthodox; Muslim-majority country
    40: ("Armenia", 0.92),       # 2011 census, 92.5% Armenian Apostolic (Oriental Orthodox)
    88: ("Ethiopia", 0.43),      # 2007 census, 43.5% Ethiopian Orthodox (Oriental)
    87: ("Eritrea", 0.40),       # no census; Orthodox Tewahedo about 40%, Muslim about half
}


def shape():
    """(rows keyed by MSOA, diagnostics) -- the Orthodox placement weight per MSOA."""
    import pandas as pd
    if not os.path.exists(COB):
        sys.exit(f"missing {COB}\n  see sources/uk_orthodox.md for the fetch")
    df = pd.read_csv(COB, dtype={"msoa": str})
    df = df[df["msoa"].str.startswith("E")]          # England; Wales has no split
    keep = df[df["cob_id"].isin(ORIGINS)].copy()
    keep["country"] = keep["cob_id"].map(lambda i: ORIGINS[i][0])
    keep["weight"] = keep["count"] * keep["cob_id"].map(lambda i: ORIGINS[i][1])

    by_msoa = keep.groupby("msoa", as_index=False)["weight"].sum()
    by_msoa = by_msoa[by_msoa["weight"] > 0]

    nat = keep.groupby("country").agg(born=("count", "sum"),
                                      weighted=("weight", "sum"))
    diag = {
        "msoas_in_file": df["msoa"].nunique(),
        "msoas_with_weight": len(by_msoa),
        "born_abroad_total": int(keep["count"].sum()),
        "weighted_total": float(keep["weight"].sum()),
        "national": nat.sort_values("weighted", ascending=False),
    }
    return by_msoa, diag


def check(by_msoa, diag):
    fails = 0
    print("England's Orthodox placement, from country of birth at MSOA")
    print(f"  MSOAs in the file              {diag['msoas_in_file']:,}")
    print(f"  MSOAs with any weight          {diag['msoas_with_weight']:,}")
    print(f"  people born in these countries {diag['born_abroad_total']:,}")
    print(f"  after Orthodox weighting       {diag['weighted_total']:,.0f}")
    if diag["msoas_in_file"] < 6000:
        print("  ! far fewer MSOAs than England's 6,856; the fetch was probably truncated")
        fails += 1

    nat = diag["national"].copy()
    nat["share of shape %"] = 100 * nat["weighted"] / nat["weighted"].sum()
    nat["share unweighted %"] = 100 * nat["born"] / nat["born"].sum()
    print("\n  what each origin contributes to the shape")
    print(nat.round(1).to_string())

    # The weighting exists to stop Albania carrying Orthodox geography. Say so numerically.
    if "Albania" in nat.index:
        a = nat.loc["Albania"]
        print(f"\n  Albania: {a['share unweighted %']:.1f}% of the shape unweighted, "
              f"{a['share of shape %']:.1f}% weighted. That gap is why the table exists.")

    top = by_msoa.nlargest(10, "weight")
    print("\n  the ten MSOAs carrying the most Orthodox weight")
    for r in top.itertuples(index=False):
        print(f"    {r.msoa}  {r.weight:8,.0f}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    by_msoa, diag = shape()
    fails = check(by_msoa, diag)
    if args.report:
        return 1 if fails else 0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in by_msoa.itertuples(index=False):
            w.writerow({
                "geo_id": r.msoa, "geo_level": "msoa", "geo_name": r.msoa,
                "source_category": "orthodox", "count": round(r.weight, 2),
                "basis": BASIS, "year": YEAR, "source_id": SOURCE_ID,
                "note": "placement weight, not people; residents born in Orthodox-majority "
                        "countries weighted by each country's Orthodox share",
            })
    print(f"\nwrote {len(by_msoa):,} rows -> {OUT}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
