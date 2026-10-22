# Tokelau. 2016 census Table 5.5, language spoken most often at home, national (sources/tk_census.py),
# people aged 5 and over present on census night. One unit, on religiondots' Kontur hexes for the
# three atolls re-keyed to TK (sources/tk_geo.py). Record: sources/tk.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import tk2016
    df = pd.read_csv(NORM / "tk.csv")
    if df["geo_id"].nunique() != 1 or int(df["count"].sum()) != 1_062:
        raise SystemExit("tk.csv: expected 1,062 people in one unit -- run sources/tk_census.py")
    df["node"] = df["source_category"].map(tk2016.resolve)
    df["unit"] = "TK"   # data/geo/tk/tk_hexes.gpkg: the three atolls as one unit
    out = by_unit(df)
    out["tier"] = "measured"
    return out


ENTRY = dict(
    name="Tokelau",
    source=("2016 Tokelau Census of Population and Dwellings, social profile Table 5.5 (Tokelau "
            "National Statistics Office and Statistics New Zealand)"),
    how="census, 2016, language spoken most often at home, people aged 5 and over present on census night",
    parts=[dict(covers="People aged 5 and over present on census night",
                source="2016 census, language spoken most often at home", rest=True)],
    grain="the three atolls as one unit, 1,060 people",
    gap="302 usual residents overseas on census night, 128 children under 5, and 7 who gave no answer",
    view=[-173.0, -9.9, -170.9, -8.2],
    counts=_counts,
    mappings=["tk2016"],
    place=GEO / "tk" / "tk_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2016 census asked residents aged 5 and over which language they speak most often at "
        "home. Of the 1,062 who answered, 82.6% said Tokelauan, 8.5% English, 5.1% Tuvaluan and 2.0% Samoan. The "
        "table is for the three atolls together, so the dots follow where Kontur puts people on "
        "each atoll. It covers the 1,069 residents aged 5 and over who were in Tokelau on census "
        "night; another 302 usual residents were overseas and were not asked. Of the 90 who "
        "said English, 69 had lived overseas for six months or more."),
)
