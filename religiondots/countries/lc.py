# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _lc_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "lc_hexes.gpkg", "sources/lc_grid.py")


def _lc_counts():
    """CSO 2022 census at district: 21 nodes on 10 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.

    **95.89% of the household population is drawn** — 164,764 of 171,834 — the missing part
    being `Not reported`, 7,064 people, excluded by taxonomy/lc2022.py per §3.5.

    **WHAT IS DRAWN IS THE CENSUS'S OWN ESTIMATE AND NOT ITS RAW COUNT.** CSO measured a
    **23.3% undercount** and weighted every district back up before publishing — factors
    1.107 in Anse La Raye to 1.507 in Laborie (`sources/lc.py`). That is the exact inverse
    of Barbados (`_bb_counts`), where BSS publishes the uncorrected count, warns that its
    parish tables are understated, and this project declines to scale them (§14.4). Nothing
    here scales anything either; the difference is entirely on the publisher's side.

    **`Mennonite` IS DRAWN AS `christianity.evangelical`, AND THAT IS THE ONE READING
    DECISION ON THIS COUNTRY.** 3,760 people, 2.19%. The census's own questionnaire calls
    that option `Evangelical`, the 2010 census has `Evangelical` at the same 2.2% and no
    Mennonite row, and there is no Mennonite community of that size in Saint Lucia.
    `sources/lc.py` machine-checks the questionnaire against the table on every run and
    refuses to build if the discrepancy changes shape; `taxonomy/lc2022.py` has the full
    chain. **`lc.csv` still carries the source's own label** (§12).

    **THE TABLE MISSES ITS OWN MARGINS BY UP TO 3 PEOPLE** — the cells are independently
    rounded weighted estimates — so the drawn total is 171,829 against a published 171,834.
    Five people in 0.003%; `sources/lc.py` prints the whole spread.
    """
    from lc2022 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "lc.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    df = df[df["geo_level"] == "district"].copy()
    if df["geo_id"].nunique() != 10:
        raise SystemExit(f"{df['geo_id'].nunique()} districts, expected 10 -- re-run "
                         "sources/lc.py")
    if "Mennonite" not in set(df["source_category"]):
        raise SystemExit("lc.csv has no `Mennonite` category -- that is the row the "
                         "census questionnaire calls `Evangelical`, 2.2% of Saint Lucia "
                         "(taxonomy/lc2022.py). Re-run sources/lc.py.")

    df["count"] = pd.to_numeric(df["count"])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"])
                      - {"Not reported", "Total"})
    if unmapped:
        raise SystemExit(f"lc.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "lc": dict(
        name="Saint Lucia",
        source="2022 Population and Housing Census, provisional release 2 "
               "(Central Statistics Office)",
        basis="self-identification",
        view=[-61.09, 13.69, -60.86, 14.12],
        gap="4.1% who did not report a religion, and 6.9% of Castries",
        gap_share=0.04115,
        note_public=(
            "**The most Catholic country on this map, and the one losing it fastest.** "
            "50.6% Roman Catholic — against 31.5% in Grenada, 21.6% in Trinidad and 3.8% "
            "in Barbados — in an island Britain and France traded fourteen times and where "
            "the French church stayed after the British navy left. "
            "**The census's own back-series is the finding.** Saint Lucia was **92.4% "
            "Catholic in 1960**, 85.6% in 1980, 67.5% in 2001, 61.1% in 2010 and 50.6% in "
            "2022: forty-two points in sixty-two years, and still nine points a decade at "
            "the end. About half of what left went to two churches — Seventh Day "
            "Adventists rose 1.8% to 10.8% and Pentecostals 0.0% to 9.0% over the same "
            "span — and about half went to no church at all. "
            "**And the church held where the roads did not.** Catholicism is 71.7% of "
            "Choiseul, 70.1% of Soufriere and 66.1% of Canaries, the south-west coast, "
            "against **44.1% in Anse La Raye, 44.8% in Castries and 45.8% in Gros Islet** "
            "— the north-west, where nearly two thirds of Saint Lucians now live. The "
            "Adventists run the other way: 18.6% of Anse La Raye and 16.3% of Canaries "
            "against 5.3% of Soufriere. "
            "**This census asks whether you believe in God and whether you belong to "
            "anything, separately, which almost nothing else here does.** 14.1% answer "
            "*no religion but believe in God* and **0.30% answer *do not believe in "
            "God*** — a 47-fold gap, and the strongest evidence on this map that "
            "Caribbean irreligion is lapsed affiliation rather than atheism. The "
            "non-affiliated are the mirror image of the Catholics: 16.2% of Castries "
            "against **4.8% of Choiseul**, the most Catholic district in the country. "
            "**Anglicans are 1.3% here and 23.9% in Barbados**, which is the whole "
            "difference between an island the British kept and an island they only "
            "captured. Where they are is odd and worth a look: Choiseul (3.7%) and Laborie "
            "(2.8%), in the Catholic south rather than the anglophone north. "
            "**One row on this map does not say what the census says.** The report prints "
            "2.2% of the country as `Mennonite`; the census's own questionnaire calls that "
            "option `Evangelical`, and it is drawn as Evangelical here. The note below sets "
            "out why."),
        how="census, 2022",
        grain="districts, 17,200 people on average",
        counts=_lc_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "lc" / "lc_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_lc_place_weight,
        note="**THE `Mennonite` ROW IS THE FORM'S `Evangelical` OPTION, AND IT IS 2.2% OF "
             "THE COUNTRY — 3,760 PEOPLE.** This is the largest reading decision on any "
             "Caribbean source here and it is machine-checked rather than argued. **CSO "
             "publishes the 2022 census instrument** (*St Lucia Census 2022, Version 4*) on "
             "its own site, and question 1.5 offers 22 options. Table D.2's 23 rows are "
             "those 22 options **in the same order**, plus `Not reported`; twenty-two of "
             "the twenty-three match one for one, and the one that does not is **option 6, "
             "`Evangelical` on the form and `Mennonite` in the report**. **The 2010 census "
             "agrees**: its Table 40 has `Evangelical` at the same 2.2%, in the same place "
             "in a similar list, and no Mennonite row at all — while the 2022 report has no "
             "Evangelical row at all. And there is no Mennonite community of 3,760 people "
             "in Saint Lucia; Grenada's 2021 census, whose form has BOTH cells, counts 280 "
             "Mennonites (0.26%) beside 2.36% Evangelical, which is what the real pair "
             "looks like in this region. `sources/lc.py` asserts the whole shape of the "
             "discrepancy on every run and refuses to build if it changes. **The normalised "
             "file still carries the source's own label**; the reading happens in the "
             "mapping module, which is why the two layers are separate. "
             "**THE PUBLISHED FIGURES ARE ALREADY CORRECTED FOR UNDERCOUNT, AND THAT IS "
             "CSO'S DOING RATHER THAN THIS MAP'S.** The enumeration was **23.3% short** and "
             "CSO applied per-district weight factors — 1.107 in Anse La Raye up to 1.507 "
             "in Laborie — to reach *estimated full values*, which are what every table in "
             "the report holds. **That is the exact inverse of Barbados**, where BSS "
             "publishes the raw count, warns that its parish tables are understated, and "
             "this project declines to scale them (§14.4). Nothing here scales anything "
             "either. It is also why the cells miss their own margins by one to three "
             "people — independently rounded estimates — so the drawn total is 171,829 "
             "against a published 171,834. "
             "**THE FORM ASKS ABOUT HINDUISM TWICE.** Options 7 and 19 are `Hindu` and "
             "`Hinduism`, 253 and 66 people, and they are drawn at one node. A duplicated "
             "option, not two religions. "
             "**THE JOIN IS 10/10 BOTH WAYS**, on COD-AB's ADM1, which is the government's "
             "own boundary file and is CSO's district tier exactly. One name differs, by a "
             "hyphen. **COD's district AREAS do not match the census's own** — Dennery is "
             "1.71x CSO's published figure and Soufriere 0.73x — which in Cayman was a "
             "boundary error that moved dots. **Here it is not, and that was measured:** "
             "geoBoundaries' independent set matches CSO's areas within 8.3%, the two sets "
             "agree on 528 of COD's own 547 settlements, and summing Kontur's population "
             "grid inside each gives the same answer district by district — Dennery holds "
             "10,581 people under COD and 10,571 under geoBoundaries **despite COD's "
             "Dennery being 51 km² larger**. The extra land is the Central Forest Reserve "
             "and nobody lives in it. "
             "**PLACEMENT IS KONTUR'S 400 m GRID**, 772 hexes, and it is here because Saint "
             "Lucia is a volcanic ridge with every settlement on the coast road — Castries "
             "alone holds 60,614 people on 87.4 km², nearly all of them between Bois "
             "d'Orange and Cul de Sac. The grid and the census agree nationally (1.07x) and "
             "not per district (0.58x in Laborie to 1.54x in Anse La Raye); only the shape "
             "*within* a district is used, so no district gets the wrong number of dots. "
             "**Both methods find fewer people in the rural south than CSO's correction "
             "asserts**, which is worth recording and is not something this map can settle. "
             "**A FINER GEOGRAPHY EXISTS AND CARRIES NO RELIGION.** COD's ADM2 has 547 "
             "settlements, ~310 people each, which would be among the finest tiers on this "
             "map. Nothing in the report cuts religion below the district, so it is not "
             "used: it would be placement with no counts to place.",
    ),
}
