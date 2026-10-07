# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _lb_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Kontur decides where people are inside each caza, uncalibrated (sources/lb_grid.py): the
    Lebanese dots per caza follow the register, not residence, so there is no caza total to
    calibrate Kontur to.
    """
    return _kontur_place_weight(place, "lb_hexes.gpkg", "sources/lb_grid.py")


def _lb_counts():
    """2022 electoral register by sect, carried to the 26 cazas and scaled to OCHA's resident
    Lebanese; Syrians and Palestinians by caza from OCHA at Pew 2020's origin mixes
    (sources/lb.py, sources/lb.md §10).

    Tiers come from the normalized file: `measured` for the five cazas that are whole electoral
    districts, `derived` where a district's sects were split between its cazas on the 2014
    register, `modelled` for the refugees. No row rolls (taxonomy/lb2022.py: a derived cell's
    sect was counted for the district, not the caza).
    """
    from lb2022 import resolve
    from rollup import NOWHERE

    df = pd.read_csv(HERE / "data" / "normalized" / "lb.csv", keep_default_na=False,
                     na_values=[""])
    df["node"] = df["source_category"].map(resolve)
    df = df[df["source_category"] != "Migrants (not drawn)"]
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"lb.csv categories with no node: {unmapped}")
    lut = pd.read_csv(HERE / "data" / "geo" / "lb" / "lb_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"lb rows with no unit: {missing}; re-run sources/lb_geo.py")
    if df["unit"].nunique() != 26:
        raise SystemExit(f"{df['unit'].nunique()} cazas, expected 26")
    bad = sorted(set(df["tier"]) - {"measured", "derived", "modelled"})
    if bad:
        raise SystemExit(f"lb.csv tiers {bad}")
    df = df[df["count"] > 0]
    rank = {"measured": 0, "derived": 1, "modelled": 2}
    df = df.groupby(["unit", "node"], as_index=False).agg(
        count=("count", "sum"), tier=("tier", lambda t: max(t, key=rank.get)))
    df["congregations"] = 0
    # spec §3.10: a carried or modelled count cannot establish that anyone is present
    df["may_ring"] = df["tier"] == "measured"
    df["roll"] = NOWHERE
    return df[["unit", "node", "count", "congregations", "tier", "roll", "may_ring"]]


ENTRY = {
    "lb": dict(
        name="Lebanon",
        source="2022 electoral register by sect (Ministry of Interior and Municipalities), as "
               "published by electoral district in Information International's The Monthly no. 187; "
               "split between cazas on the 2014 register by caza (lub-anan.com) and L'Orient "
               "Today's 2022 voter counts; residents, Syrians and Palestinians from OCHA's 2026 "
               "population package; refugees' religion from Pew Research Center's 2020 estimates",
        basis="Sect of record in the electoral register, by the caza where a family is registered",
        note_public=(
            "**Lebanon has not held a census since 1932, so this map draws the electoral "
            "register.** Every Lebanese citizen's civil record names a sect, and the Ministry of "
            "Interior counts registered voters by sect before each election: **3,967,507** in "
            "2022. The register files people under the caza where their family's record is kept, "
            "not where they live, and the map follows it: each caza is drawn with as many people, "
            "and the same mix, as are registered there. Kontur's population grid decides where "
            "the dots fall inside a caza. "
            "**Beirut and its suburbs are where this matters most.** Many people living in "
            "Greater Beirut are registered in the villages their families came from. Baabda, El "
            "Meten and Kesrwane hold **27%** of the Lebanese living in Lebanon, by OCHA's "
            "figures, but **11%** of the register, so they are drawn with fewer than half the "
            "Lebanese who live there, while Marjayoun, Bint Jbeil and El Hermel are drawn with "
            "three to five times as many. The southern suburbs, where most residents are Shia "
            "families registered in the South and the Bekaa, are drawn with Baabda's registered "
            "mix: 36% Maronite, 26% Shia and 18% Druze. Beirut itself has about twice as many "
            "people registered as Lebanese living there, and is drawn at its registered mix of "
            "49% Sunni and 16% Shia. "
            "**The register counts voters aged 21 and over, including emigrants.** The counts are "
            "scaled down evenly to OCHA's **3,864,296** Lebanese living in the country, a figure "
            "from the Central Administration of Statistics' 2018-19 labour force survey. Nothing "
            "adjusts for which communities have more children or more members abroad. The sect "
            "is the family record's: a married woman's record moves to her husband's family, so "
            "she is usually counted under his sect. The register still lists 4,309 Jewish voters, "
            "almost all in Beirut; most of the families left decades ago, and they are drawn "
            "where they are registered, as everyone is. "
            "**In 2022 the register is 29.5% Sunni, 29.3% Shia, 19.3% Maronite, 6.6% Greek "
            "Orthodox, 5.6% Druze and 4.3% Greek Catholic**, with 2.1% Armenian Orthodox, 1.0% "
            "Alawite and smaller churches. The Alawites are drawn as Muslims with no sect, since "
            "the map has no Alawite category. "
            "**The 2022 figures are published by electoral district, not by caza.** Information "
            "International printed the register for the fifteen districts of the 2017 electoral "
            "law. Five cazas are whole districts (Beirut is two) and are drawn as published, "
            "about a third of the Lebanese dots. Elsewhere each district's sects are split "
            "between its cazas using the 2014 register by caza, published by lub-anan.com, and "
            "matched to each caza's 2022 total from L'Orient Today. Those dots disappear when "
            "inferred dots are turned off. "
            "**Syrians and Palestinians are drawn where OCHA counted them for 2026**: "
            "**1,120,000** Syrians and **224,791** Palestinian refugees, by caza. Their religion "
            "is Pew Research Center's 2020 estimate for Syria and for the Palestinian "
            "territories, not a count of the people in Lebanon, so these dots also disappear "
            "when inferred dots are turned off. The Syrians' Muslims are not split into Sunni and "
            "other sects, and Christians are not split into churches. About 164,000 migrant "
            "workers are not drawn, since nothing gives their nationalities by caza. Shebaa "
            "Farms, held by Israel, are left out of Hasbaya."),
        how="electoral register, 2022, every voter's sect of record, by electoral district",
        fill="from the 2014 register by caza",
        grain="cazas of family registration, not of residence; 200,000 people on average",
        gap="164,097 migrant workers, 3.1% of the people living in Lebanon",
        gap_share=0.0305,
        counts=_lb_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "lb" / "lb_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_lb_place_weight,
        note="BUILT ON ASK 049 (Anita, 2026-10-03): the register by caza, labelled as where "
             "families are registered. sources/lb.md §10 is the record. LEVEL: The Monthly 187 "
             "Tables 1-16 (MoIM figures), 2022 column, 15 districts; nine tables' rows off their "
             "totals by 1-44 voters, scaled (ROW_SUM_OFF); Table 1 agrees on ten sects to the "
             "voter, Greek Orthodox +2,554 and the Syriac, Chaldean and Others rows grouped "
             "differently (NATIONAL_OFF). CAZAS: IPF of lub-anan's 2014 caza x sect (personal "
             "sect, MoIM 2014 lists) onto each district's 2022 sect rows and L'Orient's 25 "
             "minor-district totals (Tableau CSV export only; the packaged workbook holds the "
             "named voter roll and is not used); three caza pairs split on 2014. Tiers: Beirut, "
             "El Meten, Baabda, Zahle, Akkar measured (1,305,483 of 3,864,296), rest derived, "
             "roll NOWHERE. SCALE: x 3,864,296 / 3,967,507 (OCHA 2026 LRP, CAS LFHLCS). "
             "REFUGEES: OCHA 2026 by caza, Syrians 1,120,000 on Pew 2020 Syria (Muslims bare "
             "islam, Christians bare), Palestinians 224,791 on Pew PS (Sunni); modelled. GAP: "
             "migrants 164,097. NODES: Maronite and Melkite added under eastern Catholic; Greek "
             "Orthodox on .antiochian; Alawite on bare islam. GEOGRAPHY: COD-AB 2026 adm2, 26 "
             "cazas by pcode, Shebaa Farms clipped (17.9 km2). PLACEMENT: Kontur LB uncalibrated, "
             "ratio 0.997 against OCHA's residents, caza rank witness +0.847 (shuffle 99th "
             "+0.448); per-caza band failed in six cazas (Baabda 2.12, Akkar 0.23), printed. Cap: "
             "Beirut-Dahiyeh and Tripoli real, one Baabda block capped.",
    ),
}
