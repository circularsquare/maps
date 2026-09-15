# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _me_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "me_grid_400m.gpkg", "sources/me_geo.py")


def _me_counts():
    """MONSTAT Popis 2023 at municipality: 10 nodes on 23 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.
    me.csv also carries the national row, which is the same people again, and the two
    residual categories this drops.

    **93.4% OF THE COUNTED COUNTRY IS DRAWN, AND THE TWO THINGS MISSING ARE DIFFERENT.**
    581,388 of 622,537. `Ne želi da se izjasni` is 11,924 people who declined the question
    and are not drawn per §3.5. `zaštićen podatak` is 29,225 people MONSTAT withheld as `z` —
    disclosure control, not an answer — and §3.8 removes those rather than placing them;
    §6.3a's `unknown` names suppression as the one thing it will not accept.

    **THE SUPPRESSION IS THE WHOLE CAVEAT AND IT IS NOT EVENLY SPREAD.** It protects small
    counts, a small count is a LOCAL minority, and so it lands hardest where the national
    majority is scarce: Petnjica loses 29.4% of its people to it, Šavnik 25.4%, Rožaje 21.3%,
    against Tivat 0.8% and Podgorica 0.9%. Per category — which is the only honest way to
    report it (§3.8, Lithuania) — **Islam loses 10.6% of itself against Orthodoxy's 2.7%**,
    measured against MONSTAT's own published national figures. So every municipality on this
    map under-reports whichever of the two is its own minority, and the Bosniak and Albanian
    municipalities under-report most.

    **AND A FURTHER 219 SETTLEMENTS ARE NOT DRAWN AT ALL**, because MONSTAT withheld their
    populations too — between 219 and 1,971 people at the disclosure threshold of ten. They
    are in `gap=` rather than estimated.
    """
    from me2023 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "me.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 23:
        raise SystemExit(f"{df['geo_id'].nunique()} municipalities, expected 23 -- re-run "
                         "sources/me.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "me": dict(
        name="Montenegro",
        source="Popis 2023 (MONSTAT)",
        basis="self-identification",
        view=[18.3, 41.75, 20.5, 43.65],
        gap=("6.6%: 4.7% withheld by MONSTAT's disclosure control and 1.9% who declined to "
             "declare; a further 219 small settlements have their populations withheld too"),
        gap_share=0.0661,
        note_public=(
            "**The sharpest religious boundary in Europe over the shortest distance.** "
            "Montenegro is 69% Orthodox and 18% Muslim in a country of 620,000 people and "
            "13,800 km², and the two do not mix so much as sit either side of a line. In "
            "the northeast, in the Sandžak, **Gusinje is 84% Muslim, Rožaje 77%, Plav 72% "
            "and Petnjica 70%** — Bosniak municipalities against **Mojkovac at 94% "
            "Orthodox and Nikšić at 92%**, an hour's drive away. "
            "**The Catholics are two separate communities in two corners.** 3.1% "
            "nationally, but **Tivat is 16.3% and Kotor 10.0%** — the Croats of the Bay of "
            "Kotor, whose Catholicism is Venetian and six hundred years old — while "
            "**Ulcinj at 9.2%** on the Albanian border is a different community "
            "altogether, Albanian rather than Croat. The Archdiocese of Bar between them "
            "is one of the oldest sees in the region. "
            "**The Orthodox category is one box and two churches are claiming it.** The "
            "census asks 'Orthodox' and names no jurisdiction. The Serbian Orthodox Church "
            "holds the great majority and the monasteries; the Montenegrin Orthodox "
            "Church, self-declared autocephalous in 1993, is recognised by nobody and "
            "claims a substantial minority. That dispute brought down a government in 2020 "
            "and the census was taken with it still live, so these dots say 'Orthodox' and "
            "deliberately do not take a side. "
            "**There is no 'no religion' box at all.** The irreligious answers are "
            "*atheist* and *agnostic*, which are positions rather than an absence, and "
            "together they are 2.5% — concentrated on the coast and in the old capital, "
            "**Budva 4.8%, Herceg Novi 4.1%, Cetinje 4.0%**. Read that as a floor: "
            "Montenegro was never offered the question Czechia was. "
            "**Nearly 5% of the country is missing from this map and it is missing "
            "unevenly.** MONSTAT withholds any cell it judges too small to publish, which "
            "means it withholds *local minorities* — and so the places that lose most are "
            "the ones where the national majority is scarce. **Petnjica loses 29% of its "
            "people to this, Šavnik 25%, Rožaje 21%, Ulcinj 15%**, against under 1% in "
            "Tivat and Podgorica. Islam loses 10.6% of itself nationally and Orthodoxy "
            "2.7%. So every municipality here under-shows whichever religion is its own "
            "minority, and the Bosniak and Albanian ones under-show most. "
            "**Tuzi and Ulcinj are Montenegro's Albanian municipalities and only one of "
            "them is on this map.** Tuzi was split from Podgorica in 2018 and no boundary "
            "file yet published has it, so its people are drawn inside Podgorica and its "
            "Catholic and Muslim majority is averaged into the capital's."),
        how="census, 2023",
        grain="municipalities, 27,000 people on average",
        counts=_me_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "me" / "me_grid_400m.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_me_place_weight,
        note="**§11c AND §11k BOTH RECORDED THAT MONSTAT PUBLISHES NO XLSX AND BOTH WERE "
             "WRONG, FOR A REASON THAT GENERALISES.** Montenegro's 2023 census lives under "
             "`uploads/files/popis 2021/` — the year it was first scheduled for, before it "
             "was postponed twice — so every path search for `2023` misses it, and the two "
             "earlier sweeps walked the census landing pages without descending into their "
             "fourteen subpages. Enumerating `page.php?id=` across the range found it on "
             "id=2342 in one pass. **When an office's folder names disagree with its "
             "publication titles, enumerate the pages rather than guessing the paths.** "
             "**AND WHAT IT PUBLISHES IS FINER THAN THE MAP CAN USE.** `naselja vjera popis "
             "2023..xlsx` is religion at the **settlement** — 1,462 of them, 425 people "
             "each, which would be the finest counting geography on this map after "
             "Estonia. It is aggregated to 23 municipalities here because **no settlement "
             "geometry exists**: geoBoundaries has only ADM0 and ADM1 for Montenegro, and "
             "the GISCO LAU file that covers Albania's 61 bashki, Serbia's 169 and "
             "Liechtenstein's 11 has **zero Montenegrin features** — checked directly, not "
             "inferred. The counts for a much better map are on disk and waiting for a "
             "boundary file. "
             "**THE `z` SENTINEL IS SUPPRESSION AND THE WORKBOOK SAYS SO IN ITS OWN "
             "LEGEND** — `\"z\" zaštićen podatak`, beside `\"-\" nema pojave` for a true "
             "zero. So this is Lithuania's case (§9q) and not Kosovo's (§9w), the two look "
             "identical from the cell alone, and reading it as zero would delete 29,225 "
             "people. `sources/me.py` asserts the legend is still present, because it is "
             "the only evidence for the reading. "
             "**IT CANNOT BE DIFFERENCED OUT, AND THAT IS WORTH KNOWING BEFORE TRYING.** "
             "The obvious attack is that a settlement with one suppressed cell gives it "
             "away by subtraction from its own total. **Of the 745 settlements that publish "
             "a total and carry a `z`, exactly zero have only one.** MONSTAT's "
             "complementary suppression is properly done and the check asserts it, so a "
             "sloppier future vintage is noticed rather than silently exploited. "
             "**THE DISCLOSURE THRESHOLD IS TEN AND IS READABLE OFF THE DATA** — no value "
             "anywhere in either 2023 settlement workbook is below it — which bounds a "
             "primary suppression at 1-9. It does **not** bound the total: 223 settlements "
             "have a gap larger than 9 x their `z` count, so those cells are complementary "
             "rather than small, and they hold 23,932 of the 29,225. "
             "**THE TOTAL COLUMN IS SUPPRESSED TOO, IN 219 SETTLEMENTS**, which this build "
             "assumed away first and which showed up as the categories out-summing the "
             "country by 31 people. Where no denominator is published there is no residual "
             "to compute, so those settlements are dropped whole rather than half-drawn. "
             "**AND THE BOUNDARY VINTAGE COSTS A REAL UNIT.** Montenegro has been splitting "
             "municipalities for a decade — Petnjica off Berane 2013, Gusinje off Plav "
             "2014, Tuzi off Podgorica 2018, Zeta 2022 — and the geoBoundaries cut has the "
             "first two and not the last two. Tuzi and Zeta are folded back into Podgorica, "
             "23 units against the census's 25. Podgorica's Kontur ratio of 0.95x against a "
             "median of 1.08x is what says the merge is right, and it is the only place a "
             "wrong one would show.",
    ),
}
