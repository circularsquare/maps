# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ky_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read."""
    return _kontur_place_weight(place, "ky_hexes.gpkg", "sources/ky_grid.py")


def _ky_counts():
    """ESO 2021 census at district: 15 nodes on 6 units.

    ONE level, no allocation, nothing modelled — every drawn row is `measured` and may ring.

    **`keep_default_na=False`**, as for Belize, Trinidad and the Bahamas. `None` is a
    category name here too and it is the SECOND largest answer in the country — 11,502
    people, 16.72% — so a bare `pd.read_csv` would delete a sixth of the Cayman Islands
    while every check in `sources/ky.py` still passed. Guarded below.

    **THE UNIVERSE IS SMALLER THAN THE CENSUS AND IT IS NOT ONLY REFUSALS.** ESO's tables
    all run on the *census survey tabular population count*, 68,811. The census counted
    71,432; the gap is 327 people in institutions plus a **2,294-person weighted
    non-response estimate** published only nationally. Neither is scaled in (§14.4).

    **98.58% of that universe is drawn** — 67,836 of 68,811. What is not: `DK/NS`, 967
    people, excluded by taxonomy/ky2021.py per §3.5; and eight people in cells ESO printed
    as a dash, dropped by the `count > 0` filter along with every other empty cell.
    """
    from ky2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ky.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[])
    df = df[df["geo_level"] == "district"].copy()
    if df["geo_id"].nunique() != 6:
        raise SystemExit(f"{df['geo_id'].nunique()} districts, expected 6 -- re-run "
                         "sources/ky.py")
    if "None" not in set(df["source_category"]):
        raise SystemExit("ky.csv has no `None` category -- it has been read as NaN. "
                         "pd.read_csv needs keep_default_na=False here; without it a sixth "
                         "of the Cayman Islands disappears silently.")

    df["count"] = pd.to_numeric(df["count"])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"])
                      - {"DK/NS", "Total"})
    if unmapped:
        raise SystemExit(f"ky.csv has unmapped source categories: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "ky": dict(
        name="Cayman Islands",
        name_in="the Cayman Islands",
        source="2021 Census of Population and Housing (Economics and Statistics Office)",
        basis="self-identification",
        view=[-81.50, 19.20, -79.65, 19.82],
        gap_share=0.037,
        gap=("the institutional population and a national non-response estimate, 3.7% together, "
             "which are outside every table in the report"),
        note_public=(
            "**The most evenly religious country on this map.** The largest answer reaches "
            "only 19.5%, and the top five are five different things — a Holiness church, no "
            "religion at all, Roman Catholicism, Adventism and non-denominational "
            "Christianity. Nowhere else drawn here is this flat. "
            "**The reason is that over half the residents were born abroad**, and almost "
            "every category's geography is really a map of that. Roman Catholicism is 18.4% "
            "in George Town against 3.6% in North Side — the Filipino and Latin American "
            "workforce in the capital. The Hindu share peaks at 6.6% in East End. "
            "**The Church of God is the exception and is effectively the national church**: "
            "27.2% in North Side, 25.3% in Bodden Town, 23.5% on the Sister Islands, 20.9% "
            "in East End, 18.3% in West Bay, 16.8% in George Town. Nothing else is that "
            "even. It is the **Anderson, Indiana** church — Holiness rather than "
            "Pentecostal, and the same body Jamaica's census counts separately — which "
            "arrived through the Cayman Islands Regional Mission Council. "
            "**And Cayman Brac is a different country.** The Sister Islands are **30.5% "
            "Baptist** against 2.8% to 9.5% in every Grand Cayman district, the sharpest "
            "single contrast in the territory: the old Brac Baptist settlement still "
            "visible through a population that immigration has otherwise remade. They are "
            "also the least Presbyterian place here, at 0.66% against 13.8% in North Side. "
            "**16.7% report no religion**, the second largest answer, and it is highest in "
            "West Bay and East End rather than in the capital. "
            "**3.7% of the territory is outside this map before any of that.** Every table "
            "in the census report runs on what the office calls the *census survey tabular "
            "population count* — 68,811 of the 71,432 people counted. The difference is 327 "
            "people living in institutions and a **2,294-person non-response estimate**, "
            "weighted from household refusals and verified no-contacts, that exists only as "
            "a national figure and so cannot be put anywhere on a map."),
        how="census, 2021",
        grain="districts, 11,500 people on average",
        counts=_ky_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ky" / "ky_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ky_place_weight,
        note="**THE ROW LIST IS NOT FIXED AND NEITHER IS THE COLUMN COUNT.** Two "
             "irregularities, neither announced. **North Side's table has no `Muslim` row "
             "at all** — not a zero, not a dash, the row is absent — and **Table 4.10F "
             "(Sister Islands) has eleven figures per row where the other five have "
             "twelve**, dropping the Non-Caymanian DK/NS column. So the parse can assume "
             "neither a fixed row sequence (the Bahamas' problem, §9ar) nor a fixed column "
             "count: it reads a row as `label, then a run of figures`, takes the first, and "
             "asserts the run length is constant WITHIN a table. "
             "**AN OMITTED ROW IS NOT A ZERO, AND THE NATIONAL TABLE PRICES IT.** The six "
             "districts are short by exactly 3 on `Muslim` and North Side is the only "
             "district omitting it, which pins the missing cell at 3 people. **It is not "
             "added back** (§14.4): the value is implied by a residual rather than "
             "published, and ESO prints a dash for a genuine zero elsewhere in the same "
             "table, so the omission means something it does not say. The map draws North "
             "Side with no Muslims. "
             "**ESO'S TABLES DO NOT INTERNALLY RECONCILE, BY ONE TO FIVE PEOPLE.** A "
             "district's rows fall 1 short of its own printed Total in George Town, Bodden "
             "Town and the Sister Islands and 1 over in East End; North Side is 5 short, of "
             "which 3 is the omitted Muslim row. The NATIONAL table is internally exact, so "
             "the discrepancy is the district tables' and not the category list's. "
             "Two-sided and bounded at five people on a 68,811-person table; the whole "
             "spread prints on every run rather than being absorbed by a tolerance. "
             "**THE BOUNDARIES ARE WRONG AND WERE CHOSEN ANYWAY, ON A MEASUREMENT.** "
             "COD-AB's ADM1 is ESO's six districts exactly — same names, `Sister Islands` "
             "included, 6/6 both ways with the pcodes cross-checked — but its **Bodden Town "
             "is an 8.3 km² coastal strip** and its North Side reaches south across the "
             "island. OSM carries the same six districts at `admin_level=8`, so this was a "
             "choice. Both were tested. **Settlements**: all 69 OSM place nodes located in "
             "both sets, 60 agree and 9 do not — COD wrongly puts six eastern Bodden Town "
             "villages (Breakers, Northward, Frank Sound, Midland Acres, Pease Bay, Belford "
             "Estates) in North Side, and OSM wrongly puts Savannah and Pedro Castle in "
             "George Town. **Population**: Kontur summed per polygon against ESO's own "
             "district counts gives a total absolute error of **11,785 for COD against "
             "25,748 for OSM** — COD's mistakes are on villages, OSM's are on a town. COD "
             "is used, and `sources/ky_geo.py` re-runs the settlement test every build and "
             "asserts the three known failures are exactly those three, so a reissue that "
             "fixes or breaks one stops the run. "
             "**THE VISIBLE SYMPTOM IS THE RATIO TABLE**: North Side reads 2.03x its census "
             "population and Bodden Town 0.86x, West Bay 0.78x. That is the boundary error, "
             "not Kontur — nationally the grid is 1.007x. Only the within-district shape is "
             "used, so no district gets the wrong number of dots (§9t). "
             "**PLACEMENT IS THE THINNEST KONTUR EXTRACT ON THE MAP**, 392 hexes for the "
             "whole country — but the units are 8-89 km² and a hex is 0.67 km², so the grid "
             "is still finer than the tier it weights, which is the test Saint Vincent "
             "failed (§9ac). 6.29% of it lands outside every district and **none of that is "
             "genuinely offshore** — the measured maximum distance is 500 m — so it is "
             "snapped to the nearest district within 1 km, as in the Bahamas.",
    ),
}
