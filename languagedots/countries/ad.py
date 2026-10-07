# Andorra. No census: language at home from the World Values Survey wave 7 (2018, 1,004
# interviews), national shares on the 2018 population (sources/ad_wvs.py). Every row modelled.
# Placed on religiondots' Kontur hexes for the country as one unit (read-only). Record:
# sources/ad.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import ad2018
    df = pd.read_csv(NORM / "ad.csv")
    if len(df) != 6:
        raise SystemExit("ad.csv: expected six answers -- run sources/ad_wvs.py")
    df["node"] = df["source_category"].map(ad2018.resolve)
    df["unit"] = "AD"   # religiondots' ad_hexes.gpkg: one unit
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Andorra",
    source=("World Values Survey wave 7, Andorra 2018 (Institut d'Estudis Andorrans), Q272; "
            "population 31 December 2018 (Observatori Social d'Andorra)"),
    how=("no census: survey, one round of 1,004 interviews in 2018, language at home; national "
         "shares on the 2018 population"),
    parts=[dict(covers="Everyone",
                source="World Values Survey 2018, language at home, 1,004 adults", rest=True)],
    grain="the country as one unit, 76,200 people",
    view=[1.40, 42.42, 1.80, 42.66],
    counts=_counts,
    mappings=["ad2018"],
    place=RD_GEO / "ad" / "ad_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Andorra has no census. In the 2018 World Values Survey, 1,004 residents aged 18 and "
        "over were asked which language they speak at home: 37% said Spanish, 35% Catalan, the "
        "only official language, 11% Portuguese and 5% French, and 11% named another language "
        "that the survey does not identify. The shares are national, with a sampling error of "
        "a few points, so the dots of each language follow population alone."),
)
