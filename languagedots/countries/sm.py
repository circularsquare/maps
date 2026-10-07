# San Marino. No language statistics: residents by citizenship (register, end of 2024), Italian
# and Romagnol (at Rimini's ISTAT rate) for Sammarinese and Italian citizens (sources/sm_census.py).
# Placed on Kontur hexes for the whole republic (data/geo/sm/sm_hexes.gpkg). Record: sources/sm.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import sm2024
    df = pd.read_csv(NORM / "sm.csv")
    if int(df["count"].sum()) != 34_045:
        raise SystemExit("sm.csv: expected 34,045 people -- run sources/sm_census.py")
    df["node"] = df["source_category"].map(sm2024.resolve)
    df["unit"] = "SM"
    out = by_unit(df)
    out["tier"] = "derived"
    return out


ENTRY = dict(
    name="San Marino",
    source=("Ufficio Informatica, Tecnologia, Dati e Statistica, Bollettino di Statistica I "
            "trimestre 2025, Tavola 1.8 (residents by citizenship, December 2024); ISTAT 2024 "
            "dialect-in-the-family rate for Rimini, as drawn on Italy's map"),
    how=("no language question: Sammarinese and Italian citizens drawn as Italian, with a share "
         "on Romagnol borrowed from neighbouring Rimini; other citizens as language not named"),
    parts=[
        dict(covers="Romagnol",
             source="ISTAT 2024, share speaking dialect in the family in neighbouring Rimini",
             nodes=["indoeuropean.romance.romagnol"]),
        dict(covers="Other citizenships",
             source="2024 residents register, drawn as language not named", people=811),
        dict(covers="Everyone else",
             source="2024 residents register, Sammarinese and Italian citizens, drawn as "
                    "Italian",
             rest=True),
    ],
    grain="the republic as one unit, 34,045 people",
    view=[12.38, 43.88, 12.53, 44.00],
    counts=_counts,
    mappings=["sm2024"],
    place=ROOT / "data" / "geo" / "sm" / "sm_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "San Marino does not count languages. Its residents are Sammarinese (83%) or Italian "
        "(15%) citizens, and nearly all speak Italian. The local dialect is a form of "
        "Romagnol, spoken mostly by older people; no Sammarinese count exists, so the share "
        "drawn, 17.6%, is the rate at which people in neighbouring Rimini speak dialect in "
        "the family in Italy's 2024 survey. The register does not publish the other "
        "residents' nationalities, so they are drawn as language not named."),
)
