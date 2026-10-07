# Vatican City. No language statistics: the 882 inhabitants of 2024, the Swiss Guard on
# Switzerland's languages and everyone else on Italian (sources/va_pop.py). Placed on Kontur
# hexes for the city (data/geo/va/va_hexes.gpkg). Record: sources/va.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import va2024
    df = pd.read_csv(NORM / "va.csv")
    if int(df["count"].sum()) != 882:
        raise SystemExit("va.csv: expected 882 people -- run sources/va_pop.py")
    df["node"] = df["source_category"].map(va2024.resolve)
    df["unit"] = "VA"
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="Vatican City",
    source="Vatican City State, Population, figures at 31 December 2024 (vaticanstate.va)",
    how=("no language statistics: the 120 Swiss Guards drawn on Switzerland's languages, "
         "everyone else as Italian"),
    parts=[
        dict(covers="Swiss Guards", source="2024 figure, drawn on Switzerland's languages",
             people=120),
        dict(covers="Everyone else", source="2024 population, drawn as Italian", rest=True),
    ],
    grain="the city as one unit, 882 people",
    view=[12.445, 41.900, 12.460, 41.908],
    counts=_counts,
    mappings=["va2024"],
    place=ROOT / "data" / "geo" / "va" / "va_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Vatican City had 882 inhabitants at the end of 2024 and publishes no figures on "
        "language or nationality. The 120 members of the Swiss Guard are drawn on "
        "Switzerland's languages, mostly German. Everyone else is drawn as Italian, the "
        "language the state works in; the clergy living there come from many countries, and "
        "their own first languages are not counted anywhere."),
)
