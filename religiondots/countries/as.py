# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403

_AS_UNITS = 10
_AS_CENSUS_2020 = 49_710


def _as_place_weight(place):
    """countries.py hook. `place` is the Kontur 400m hex layer scatter.py has read.

    Tutuila is steep and its people live along the coast and on the flat south-west of the
    island; Manu'a's villages are on the shores of Ta'u, Ofu and Olosega (sources/as_geo.py).
    """
    return _kontur_place_weight(place, "as_hexes.gpkg", "sources/as_geo.py")


def _as_counts():
    """2015 HIES Table 1.6 shares on the 2020 census count, at 10 units: EVERY ROW `modelled`.

    The survey's county totals are its sample scaled by one flat weight, not populations
    (sources/as.py), so each unit's people are the 2020 census count (sources/as_geo.py) and the
    survey supplies the shares. The categories in as2015.OWN_GEOGRAPHY take their own county
    share. The rest failed the county test and share each unit's remainder in their
    territory-wide proportions, as Puerto Rico's and Taiwan's failing categories do. Household
    surveys here are all `modelled` (pr, uy, do, ht, tw).
    """
    import as2015

    df = pd.read_csv(HERE / "data" / "normalized" / "as.csv",
                     low_memory=False, keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "as" / "as_lookup.csv",
                      keep_default_na=False, na_values=[""])
    census = dict(zip(lut["unit"], lut["census_2020"].astype(int)))
    units = sorted(df["geo_id"].unique())
    if len(units) != _AS_UNITS or set(units) != set(census):
        raise SystemExit(f"as.csv units {units} and as_lookup.csv units {sorted(census)} differ -- "
                         "re-run sources/as.py and sources/as_geo.py")
    if sum(census.values()) != _AS_CENSUS_2020:
        raise SystemExit(f"as_lookup.csv sums to {sum(census.values())}, expected {_AS_CENSUS_2020}")
    unmapped = sorted(set(df["source_category"]) - set(as2015.MAP))
    if unmapped:
        raise SystemExit(f"as.csv categories with no node: {unmapped}")

    cells = df.pivot_table(index="geo_id", columns="source_category", values="count",
                           aggfunc="sum", fill_value=0)
    own = [c for c in cells.columns if c in as2015.OWN_GEOGRAPHY]
    rest = [c for c in cells.columns if c not in as2015.OWN_GEOGRAPHY]
    national_rest = cells[rest].sum()
    rest_mix = national_rest / national_rest.sum()

    rows = []
    for u in units:
        share = cells.loc[u] / cells.loc[u].sum()
        people = census[u]
        for c in own:
            rows.append((u, c, people * share[c]))
        remainder = people * (1.0 - share[own].sum())
        for c in rest:
            rows.append((u, c, remainder * rest_mix[c]))
    out = pd.DataFrame(rows, columns=["unit", "source_category", "count"])
    if abs(out["count"].sum() - _AS_CENSUS_2020) > 1e-6:
        raise SystemExit(f"modelled counts sum to {out['count'].sum()}, expected {_AS_CENSUS_2020}")
    out["node"] = out["source_category"].map(as2015.resolve)
    out = out[out["count"] > 0].groupby(["unit", "node"], as_index=False)["count"].sum()
    out["tier"] = "modelled"
    out["congregations"] = 0
    return out[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "as": dict(
        name="American Samoa",
        source="2015 Household Income and Expenditure Survey, Table 1.6 (American Samoa "
               "Department of Commerce), on the 2020 census count",
        basis="self-identification, household members",
        note_public=(
            "**No census in American Samoa asks about religion, so this map is drawn from a "
            "survey.** The Department of Commerce's 2015 Household Income and Expenditure "
            "Survey wrote down the religion of everyone in 1,838 households, about one in six, "
            "and published it for ten areas: the nine counties of Tutuila and Aunu'u, and the "
            "Manu'a islands together. The shares are the survey's. The number of people in "
            "each area is the 2020 census count, 49,710 in all, because the survey's own "
            "totals are its sample multiplied by one weight and differ from the 2010 census by "
            "as much as 37% in a county. "
            "**The Congregational Christian Church in American Samoa is a third of the "
            "territory.** It is **33.3%** of the survey and **88.3%** of Manu'a, where the "
            "survey recorded no Catholics and no Latter-day Saints. Catholics are 30.5% of "
            "Lealataua and 25.9% of Maoputasi. Tualauta, the largest county, is **25.0%** "
            "Latter-day Saint and 11.0% Methodist, against 4 to 8% Methodist elsewhere; it "
            "also holds 1,397 of the survey's 1,607 Tongans. "
            "**Eight other answers share what is left in each county.** Assembly of God "
            "(9.5%), Baptist, Full Gospel, Pentecostal, Bahá'í, Jewish, Orthodox and no religion "
            "differ between counties by no more than a sample of this many households would by "
            "chance, so each county's remaining people are shared among them in territory-wide "
            "proportions."),
        how="household survey, one round in 2015",
        grain="counties, 5,000 people on average",
        counts=_as_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "as" / "as_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_as_place_weight,
        note="2015 HIES REPORT TABLE 1.6 (printed p.57, doi.gov, pinned) is religion x 10 county "
             "columns in weighted persons: one flat weight of 5.99668 on 9,578 sampled persons "
             "in 1,838 households, every cell a whole multiple of it to half a person. Checked "
             "against Tables 1.1, A, 2.3, 3.3 and 4.3 (NR row 0). The census cannot ask religion "
             "and UNSD has no row. The survey's county totals are 0.83x to 1.37x the 2010 census, "
             "so the shares are laid on the 2020 census count per county (phc table 1). No "
             "microdata, so no split-half: a household-level simulation test "
             "(sources/as.py::geography_test) gives CCCAS, Catholic, Methodist, SDA, LDS, "
             "Jehovah's Witness, Nazarene and Other religion their own county shares; the other "
             "eight share each county's remainder. Units are TIGER/Line 2020 county "
             "subdivisions, Manu'a's five dissolved; Rose and Swains are not units (0 people in "
             "2020). CCCAS is a new node beside Samoa's cccs (independent since 1980). "
             "sources/as.md has the record.",
    ),
}
