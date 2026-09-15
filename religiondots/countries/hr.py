# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _hr_counts():
    """DZS Popis 2021 at town/municipality: 12 categories, mapped at branch level.

    ZAGREB'S 17 DISTRICTS ARE SUMMED BACK INTO ONE UNIT, which is the reverse of what
    Czechia and Estonia do and is not a choice. DZS publishes religion for the 17 gradske
    četvrti and not for Grad Zagreb as a whole — the census has 555 municipalities where
    Croatia has 556 — but no boundary source for the districts was found (GISCO stops at
    the municipality, and OSM has nothing at admin_level 9 or 10 inside Zagreb). So the
    data supports the split and the geometry does not, and 18.4% of Croatia is one polygon.
    sources/hr_geo.py writes the lookup that routes all 17 districts to LAU 01333.

    The census carries NO geographic codes — rows are (županija, name) — so the resolution
    to LAU codes is done once in sources/hr_geo.py and read from disk here, as Romania does.
    """
    from hr2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "hr.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df = df[df["geo_level"].isin(("municipality", "city_district"))].copy()

    lut = pd.read_csv(HERE / "data" / "geo" / "hr" / "hr_lookup.csv",
                      dtype={"geo_id": str, "kod": str})
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["kod"])))
    missing = df["unit"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} hr.csv rows have no LAU code -- re-run "
                         "sources/hr_geo.py, the lookup is stale")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["congregations"] = 0
    # the 17 Zagreb districts collapse onto one unit, so re-aggregate
    return (df.groupby(["unit", "node"], as_index=False)
              .agg(count=("count", "sum"), congregations=("congregations", "max")))


ENTRY = {
    "hr": dict(
        name="Croatia",
        source="Popis stanovništva 2021 (Croatian Bureau of Statistics)",
        basis="self-identification",
        view=[13.3, 42.3, 19.5, 46.6],
        note_public=(
            "Croatia is 79% Catholic, so what this map shows is the other fifth: the "
            "Serbian Orthodox belt along the Bosnian and Serbian borders, the Muslim "
            "populations of the cities, and Istria — which is by a distance the least "
            "religious part of the country. The categories are shallow here by choice "
            "rather than by necessity: the census also names 54 individual churches at "
            "this same geography, including four Orthodox jurisdictions counted "
            "separately and eleven Jewish communities, and that table is not yet drawn."),
        how="census, 2021",
        grain="municipalities, 6,700 people on average",
        counts=_hr_counts,
        # Zagreb is one polygon holding 18.4% of the country. The census would allow 17,
        # but the district boundaries were not found — see _hr_counts().
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "hr" / "hr_opcine.gpkg",
        place_unit=lambda g: g["kod"].astype(str),
        note="DZS neither rounds nor suppresses; the categories partition every unit "
             "exactly. Zagreb's 17 districts are summed into one (sources/hr_geo.md).",
    ),
}
