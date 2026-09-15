# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _gh_counts():
    """GSS 2021 PHC at district: 7 drawn categories on 272 units.

    ONE level in effect, no allocation, nothing modelled — GSS publishes these categories
    at this geography and the map draws exactly that, so every row is `measured` and may
    ring.

    THE DRAWN TIER IS TWO geo_levels AND THAT IS THE ONLY FIDDLY THING HERE. The cube holds
    Ghana, 16 regions, 261 MMDAs, and — for six metropolitan districts — a further split
    into 17 sub-metros. The sub-metros REPLACE their parents (§12's Czechia/Estonia rule)
    because GSS ships boundaries for both tiers, so the drawn set is
    `district` + `submetro` = 255 + 17 = 272. Taking `district` alone is the silent failure
    to avoid: it looks right, reconciles against nothing, and loses 1.7M people in Accra,
    Kumasi, Tema, Tamale, Sekondi-Takoradi and Cape Coast.

    TWO CATEGORIES RESOLVE TO NOTHING and neither is a person lost. `Total` is the universe
    row, and `Christian` is a parent published beside all four of its children, which sum
    to it exactly — see taxonomy/gh2021.py. So the drawn population is the whole universe,
    30,753,327, which is 99.74% of the census count; the other 0.26% never answered the
    question and GSS publishes no cell for them.
    """
    from gh2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "gh.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"].isin(["district", "submetro"])].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "gh" / "gh_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"{len(missing)} gh.csv units have no polygon ({missing[:3]}) -- "
                         "re-run sources/gh_geo.py, the lookup is stale")
    if df["unit"].nunique() != 272:
        raise SystemExit(f"{df['unit'].nunique()} drawn units, expected 272")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations"]]


ENTRY = {
    "gh": dict(
        name="Ghana",
        source="2021 Population and Housing Census (Ghana Statistical Service)",
        basis="self-identification",
        view=[-3.35, 4.5, 1.3, 11.25],
        note_public=(
            "The first African country on this map, and the shape of its question is the "
            "first thing to know about it. Ghana's census offers **four Christian boxes** "
            "— Catholic, Protestant, Pentecostal/Charismatic, Other Christian — and one "
            "box each for Islam, Traditionalist, Other and No Religion. So 71% of the "
            "country is resolved four ways and everyone else is resolved once, which is a "
            "fact about what Ghana counts rather than about who lives there. Nothing here "
            "separates Sunni from the large and old Ahmadi community, and nothing names a "
            "single one of the African Independent Churches, which are most of 'Other "
            "Christian'. "
            "**Pentecostal and Charismatic Christianity is the largest answer in the "
            "country at 31.6%**, bigger than any single Christian category anywhere else "
            "on this map outside the United States, and it is a 20th-century arrival "
            "rather than a mission inheritance: the Church of Pentecost and the Accra "
            "megachurches together outnumber Catholics three to one. It peaks in Greater "
            "Accra at 47.3% and in the Ada districts at nearly 60%. "
            "**The north–south divide is the strongest thing on the map and it is not a "
            "gradient.** Islam is 66.5% of the Northern region and 4.7% of Volta; four "
            "districts around Tamale — Nanton, Kumbungu, Tolon, Savelugu — are between 95% "
            "and 99% Muslim, which is as near-total as any unit on this map outside Sulu. "
            "**And there is a Catholic island inside the Muslim north.** Upper West is "
            "33.8% Catholic against a national 10%, and Nandom is **88.6%** — the highest "
            "single-body share of any district in Ghana. That is one mission field, the "
            "White Fathers at Navrongo and Jirapa from 1906, still legible as a hard edge "
            "a century later; its neighbours Jirapa, Nadowli Kaleo and Lawra all run "
            "48-64%. "
            "**Traditional religion survives in a belt, not a scatter.** 3.25% nationally, "
            "but 43.6% in Tatale Sanguli, 41.0% in Nabdam and 40.1% in Nanumba South, and "
            "close to zero across the whole Akan south — the median district is 0.5%. Read "
            "the number as a floor: the form makes Traditionalist exclusive of the "
            "Christian and Muslim boxes, and in Ghana traditional practice very often "
            "accompanies one of those rather than replacing it, so anyone who would answer "
            "both is counted in the other column."),
        how="census, 2021",
        grain="districts, 113,000 people on average",
        counts=_gh_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "gh" / "gh_districts.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        note="272 units for 30.8M people — the 261 MMDAs with the six metropolitan "
             "districts replaced by their 17 sub-metros, which is GSS's own finest "
             "publication of religion and its own boundary file for it (sources/gh_geo.py; "
             "geoBoundaries has 260 units on a 2019 vintage and no sub-metros). About "
             "113,000 people per unit, so read a cluster as a district and never as a "
             "neighbourhood. Placement is uniform inside each district: nothing on the "
             "Ghanaian side of this map weights dots by where people live, so the empty "
             "half of a large northern district gets as many dots as the town in it. "
             "Ghana is the first country to need INLAND water subtracted — GSS runs its "
             "districts straight across Lake Volta, and 397 of the first build's 30,750 "
             "dots, 1.29%, were on open water. sources/gh_geo.py cuts HydroLAKES out of "
             "the placement polygons and that is now zero (spec §8.2; water.py does the "
             "sea only and names this as its known gap). The universe is the 99.74% who answered the "
             "question; GSS publishes no 'not stated' cell and the missing 78,692 are "
             "spread evenly across every age and education band, so they are missingness "
             "rather than a group, and are not scaled up (sources/gh.md §4).",
    ),
}
