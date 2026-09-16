# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _so_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Bari, Sanaag, Sool, Nugaal and Mudug are 245,000 km2 of arid plateau. Drawn flat, their dots
    would sit on empty ground (sources/so_grid.py).
    """
    return _kontur_place_weight(place, "so_hexes.gpkg", "sources/so_grid.py")


def _so_counts():
    """The 2026 humanitarian planning estimate, everyone on Islam: 18 regions.

    EVERY ROW IS `modelled` (§7b). No census or survey asks Somalis their religion, so each
    region's people are drawn on Islam, as Mauritania's nationals are. sources/so.py and
    sources/so.md.
    """
    from so2026 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "so.csv",
                     dtype={"geo_id": str}, keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "so" / "so_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"so.csv regions with no polygon: {missing}; re-run sources/so_geo.py")
    if df["unit"].nunique() != 18:
        raise SystemExit(f"{df['unit'].nunique()} regions, expected 18")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"so.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "so": dict(
        name="Somalia",
        source="No census or survey asks; each region's people in the 2026 humanitarian planning "
               "estimate (OCHA Somalia; national figure from the Federal Government of Somalia "
               "through UNFPA), drawn on Pew Research Center's 2020 estimate and the US State "
               "Department's 2023 International Religious Freedom Report",
        basis="everyone drawn as Muslim, which nobody asked",
        note_public=(
            "**Nobody in Somalia is asked their religion.** No census or survey there has the "
            "question: not the UN Population Fund's Population Estimation Survey of 2014, not the "
            "Somali Health and Demographic Survey of 2020, and not the World Bank's High Frequency "
            "Surveys of 2016 and 2017. So all **19,442,160** people in the 2026 population "
            "estimate are drawn as Muslim. The published estimates agree: Pew Research Center puts "
            "Somalia at 99.833% Muslim in 2020, and the US State Department's 2023 report cites "
            "Somalia's Ministry of Endowments and Religious Affairs for more than 99% Sunni "
            "Muslim. Nobody counted these dots, and they are drawn desaturated to say so. "
            "**Somalis who are not Muslim are not on this map.** The same State Department report "
            "gives a Christian community of about 1,000, a figure it takes from World Atlas, and "
            "reports that al-Shabaab threatened to execute anyone suspected of converting to "
            "Christianity. Pew's 2020 estimate, which counts foreign residents too, is 4,367 "
            "Christians. No source says where they live, and the map does not place them anywhere. "
            "**The people are placed by region.** Each of the 18 regions takes the sum of the "
            "district figures the UN's humanitarian office published for planning in 2026, which "
            "it describes as planning estimates and not official statistics; the national total "
            "is the government's. Inside a region, dots follow Kontur's population grid. "
            "Somaliland, which has declared itself independent and governs itself, is drawn as "
            "part of Somalia, because every population figure used here counts its five regions "
            "as Somalia's. "
            "**Foreign residents and refugees are not on this map.** The UN Population Division "
            "estimates **77,972** people born abroad or registered as refugees living in Somalia "
            "in mid-2024, 28,964 of them from Ethiopia and 17,680 from Yemen, and UNHCR counted "
            "41,763 refugees and asylum seekers at the end of 2024. No source says how many live "
            "in each region, so they are the not drawn part of the bar."),
        how="no source asks; everyone drawn as Muslim, on published national estimates",
        grain="regions, 1.08 million people on average",
        gap="foreign residents and refugees, whom no source counts by region: 77,972 in mid-2024 "
            "by the UN Population Division's estimate, 0.4% of residents; and Somalis who are not "
            "Muslim, whom no source has counted",
        gap_share=0.00399,
        counts=_so_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "so" / "so_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_so_place_weight,
        note="BUILT ON ANITA'S MAGHREB AND MAURITANIA RULINGS (ask/RULINGS.md 2026-09-15 and "
             "2026-09-16: draw a near-uniformly Muslim country on a compiler's figure) AND ASK 033. "
             "sources/so.md is the record. NOTHING ASKS: PESS 2014, SHDS 2020, SHFS 2016-17, "
             "Somaliland HS 2012 have no item (sources.md 11aq); not in any Afrobarometer round. "
             "All 19,442,160 on islam (Pew 2020 99.833%; State Dept IRF 2023 'more than 99 "
             "percent' Sunni); Pew's residual not drawn (Mauritania's construction). SECTION 14: "
             "Somali Christians (IRF 2023: about 1,000, per World Atlas; al-Shabaab threatens "
             "converts) placed nowhere, named in note and gap only. POPULATION: COD-PS 2026 HRP "
             "planning estimate (2026.V1, 90 districts summed to 18 COD-AB regions; national "
             "figure the government's via UNFPA); PESS 2014 printed as witness (Kontur half-L1 "
             "0.161 vs PESS, 0.154 vs 2026). SOMALILAND: SO11-SO15 inside so, NE's SOL added to "
             "the outline (country_shapes.py ALSO). PLACEMENT: Kontur SO; border hexes snapped "
             "within 2 km, or 10 km inside NE Somalia/Somaliland (Cabudwaaq, 124,326 within 5 "
             "km), hexes already in et/ke place layers left to them (112,427 people); 28 cap "
             "blocks reviewed by a rule set before the numbers (12 real, 16 capped; "
             "kontur_cap.csv). NOT DRAWN: foreigners and refugees (UN DESA IMS 2024 mid-2024, "
             "type I R, 77,972; gap_share on the estimate plus it; UNHCR end-2024 41,763 witness).",
    ),
}
