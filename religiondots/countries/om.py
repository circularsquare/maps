# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _om_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Oman is 310,000 km2, mostly desert; Kontur decides where people are inside each wilaya,
    uncalibrated, since nothing finer than the wilaya is published (sources/om_grid.py).
    """
    return _kontur_place_weight(place, "om_hexes.gpkg", "sources/om_grid.py")


def _om_counts():
    """End-2024 register: Omanis on Islam, expatriates by nationality and sex, 61 wilayat.

    EVERY ROW IS `modelled` (§7b). No source asks religion in Oman, so every Omani is drawn on
    Islam; each governorate's male workers, female workers and dependants take their own national
    nationality mix through Pew 2020, with India's Hindu share set by Pew's Oman estimate, and each
    wilaya's expatriates take their governorate's mix. Both halves are the register's counts per
    wilaya, so they partition each unit. Two register wilayat newer than the boundaries (Sinaw, Al
    Jabal al Akhdar) are drawn inside the polygons that hold their seats. sources/om.py, om.md.
    """
    from om2024 import resolve

    nat = pd.read_csv(HERE / "data" / "normalized" / "om.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    nat["node"] = nat["source_category"].map(resolve)
    unmapped = sorted(nat.loc[nat["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"om.csv categories with no node: {unmapped}")
    ext = pd.read_csv(HERE / "data" / "normalized" / "om_foreign.csv", dtype={"geo_id": str})
    df = pd.concat([nat[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    lut = pd.read_csv(HERE / "data" / "geo" / "om" / "om_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"om rows with no unit: {missing}; re-run sources/om_geo.py")
    if df["unit"].nunique() != 61:
        raise SystemExit(f"{df['unit'].nunique()} wilayat, expected 61")
    df = df[df["count"] > 0]
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "om": dict(
        name="Oman",
        source="No census or survey asks; the National Centre for Statistics and Information's "
               "register at the end of 2024 (Statistical Year Book 2025): Omanis and expatriates in "
               "each wilaya, expatriate workers by sex in each governorate and by nationality and sex "
               "for the country, with expatriates by nationality in 2018 as mirrored by the Gulf "
               "Labour Markets, Migration and Population programme; through Pew Research Center's "
               "2020 estimates",
        basis="Omanis drawn as Muslim, which nobody asked; foreign residents by nationality and sex",
        note_public=(
            "**Nobody in Oman is asked their religion.** The 2003 and 2010 censuses had no question "
            "on it, and since 2020 the population has been counted from registers that record none. "
            "No source counts an Omani who is not Muslim, so the **2,984,793** Omanis on the "
            "register at the end of 2024 are all drawn as Muslim. Most are Ibadi or Sunni, with a "
            "small Shia minority in Muscat and on the Batinah coast. Published estimates of the "
            "Ibadi share among Omanis run from under half to three quarters, and none gives a "
            "figure for any governorate or wilaya, so the map does not split them. Nobody counted "
            "these dots, and they are drawn desaturated to say so. "
            "**Every non-Muslim drawn is a foreign resident.** The register counts **2,283,279** "
            "expatriates, 43.3% of the people living in Oman, in every wilaya. It gives the "
            "nationality of expatriate workers only for the whole country, by sex, and the number "
            "of male and female expatriate workers in each governorate. Men and women come from "
            "different places: 609,996 of the Bangladeshi workers are men and 25,791 women, while "
            "32,696 of the workers from Myanmar are women and 363 men. So each governorate's male "
            "workers are drawn at the national mix of male workers, its female workers at the mix "
            "of female workers, and its 474,353 expatriates who are not workers at the nationality "
            "mix of expatriates who were not workers in 2018, the latest year found with both "
            "counts. Each nationality is drawn at Pew Research Center's 2020 estimate for its home "
            "country, which cannot see anyone who converted or stopped practising. "
            "**Indians are not drawn at India's own figure.** Pew says of its own migration "
            "estimates that Indians in Muslim-majority Middle Eastern countries are mostly Muslim, "
            "so Indians are drawn at the Hindu share that gives Pew's figure for Hindus in Oman "
            "(9.5% of everyone): 55.4% of Indians, not India's 79%. "
            "That puts **720,302** people on religions other than Islam: 502,123 Hindus, 133,118 "
            "Christians, 60,559 Buddhists and 16,477 Sikhs. Pew's estimate for everyone living in "
            "Oman is 81.8% Muslim and 8.1% Christian, and the US State Department's is 95% "
            "Muslim; this map draws 86.3% and 2.5%. Its Buddhists are about ten times Pew's "
            "figure, most of them workers from Myanmar and Sri Lanka."),
        how="no source asks; Omanis drawn as Muslim, foreign residents by nationality and sex",
        grain="wilayat, 86,000 people on average",
        gap="Omanis who are not Muslim, whom no source counts; and foreign residents outside the "
            "register, whom nobody has counted",
        counts=_om_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "om" / "om_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_om_place_weight,
        note="BUILT ON ANITA'S PRIORITY LINE AND THE MAGHREB AND MAURITANIA RULINGS (ask/RULINGS.md "
             "2026-09-15 and 2026-09-16), as sa. sources/om.md is the record. OMANIS: nothing asks "
             "(2003 form no item; 2003, 2010, 2020 results none; register; not in the Arab "
             "Barometer); all 2,984,793 on islam; no Ibadi/Sunni/Shia split (national figures only, "
             "21-75% Ibadi; spec §14, the July 2024 attack on a Shia mosque in Muscat; ask 043). "
             "EXPATRIATES: 2,283,279 per wilaya (NCSI yearbook 2025 Table 7-2, end-2024; 2023 column "
             "equals GLMM's copy); per governorate male workers, female workers (Table 8-4) and "
             "dependants (expatriates less workers), each at its own national mix: workers by "
             "nationality and sex from Tables 17-4 and 18-4, other women on 2018's Uganda, "
             "Indonesia, Ethiopia and Nepal women, other men on the named men's mix, other Arabs on "
             "Pew's MENA row; dependants at mid-2018 population less 2018 workers by nationality "
             "(GLMM). Pew 2020 per nationality, Muslim branches folded; India's Hindu share 55.38% "
             "so the layer's Hindus equal Pew's Oman 9.531%. WITNESS: Christians 0.31 of Pew's Oman "
             "share (band 0.2-2.0, written after a scouting sum); Buddhists 10x, not corrected. "
             "GEOGRAPHY: geoBoundaries OMN ADM2, NCSI's 2020 wilayat (61), Sinaw and Al Jabal al "
             "Akhdar merged by seat; COD-AB not used (governorate lines pre-2011). PLACEMENT: "
             "Kontur OM uncalibrated (0.889 of the register; rank witness +0.837); no block at the "
             "cap.",
    ),
}
