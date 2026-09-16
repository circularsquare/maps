# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _cd_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer, calibrated to COD-PS 2024 by territoire.

    26 provinces over 2.34 million km2 is 90,000 km2 a unit, and the people are not spread over
    them: Kinshasa is 14.7 million on 10,615 km2, while Tshopo holds 3.0 million on 200,478 km2 of
    forest. Raw Kontur has lost most of Sankuru (0.09 of COD-PS), so sources/cd_geo.py scales every
    hex to its territoire's COD-PS 2024 total, and in Sankuru's five worst territoires spreads the
    total evenly over the hexes Kontur has; see its docstring.
    """
    return _kontur_place_weight(place, "cd_hexes.gpkg", "sources/cd_geo.py")


def _cd_counts():
    """The Enquête 1-2-3's household heads at province: 8 answers, 26 units, EVERY ROW `modelled`.

    Each province's shares are the religion of household heads in the 2005 and 2012 rounds, as the
    U.S. Census Bureau counted them by district, weighted by district population inside the
    province, and laid on OCHA's COD-PS 2024 projection. Nobody counted any cell (§7b). Animiste
    fails the split-half and is at its national share in every province. sources/cd.md is the
    record.
    """
    from cd2012 import EXCLUDED, resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "cd.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "cd" / "cd_lookup.csv", dtype=str)
    missing = sorted(set(df["geo_id"]) - set(lut["unit"]))
    if missing:
        raise SystemExit(f"cd.csv provinces with no polygon: {missing} -- re-run sources/cd_geo.py")
    if df["geo_id"].nunique() != 26:
        raise SystemExit(f"{df['geo_id'].nunique()} provinces in cd.csv, expected 26")

    df["unit"] = df["geo_id"]
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]) - set(EXCLUDED))
    if unmapped:
        raise SystemExit(f"cd.csv answers with no node: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)].copy()
    # EVERY row, without exception: a survey share on a projection (§7).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "cd": dict(
        name="DR Congo",
        name_in="the Democratic Republic of the Congo",
        source="Enquête 1-2-3, 2005 and 2012 (Institut National de la Statistique), household heads "
               "as tabulated by the U.S. Census Bureau, on OCHA's COD-PS 2024 province populations",
        basis="self-identification, religion of the household head",
        view=[12.2, -13.5, 31.3, 5.4],
        note_public=(
            "**No census has been taken in the DR Congo since 1984, so this map is drawn from a "
            "household survey.** The national statistics institute's Enquête 1-2-3, a survey of "
            "work and household spending, asked about religion in its 2005 and 2012 rounds, and "
            "the U.S. Census Bureau counted the answers of **31,755 household heads** by province "
            "and district. Those counts are what is drawn: each province's shares, laid on the UN "
            "humanitarian office's 2024 population projection. The dots are desaturated because "
            "they are survey shares and not a count, and each province rests on between 570 heads "
            "(Tanganyika) and 2,758 (Kinshasa). "
            "**Everyone in a household is drawn in the head's column.** 35.3% Catholic means that "
            "share of Congolese live in a household whose head gave Catholic as their religion, "
            "not that share of Congolese asked one by one. "
            "**The survey is ten to twenty years old, and the country has moved since.** The "
            "2023-24 Demographic and Health Survey asked women and men aged 15 to 49 for their own "
            "religion and found **23.7%** of women Catholic, 27.4% Protestant and **39.1%** in "
            "non-denominational churches, the revival churches this survey could only record as "
            "other Christian (22.3% here, drawn as Christianity unspecified). It prints those "
            "figures for the whole country only, so they cannot be drawn, but they say the "
            "Catholic share on this map is high for today and the unspecified Christian share "
            "low. "
            "**The sharpest line is between the Catholic north-east and the revival churches of "
            "Kasaï.** Ituri is **73.7%** Catholic and Haut-Uele 67.0%; in Kasaï-Oriental 55.0% "
            "named another Christian church and 13.3% were Catholic. Kinshasa is 36.9% other "
            "Christian and 31.9% Catholic. Protestants, members of the churches federated in the "
            "Église du Christ au Congo, are half of Haut-Lomami (50.6%). "
            "**Islam is 1.7% of the country and a sixth of Maniema** (16.4%), whose river towns, "
            "Kindu and Kasongo among them, were settled by traders from Zanzibar in the nineteenth "
            "century. Kimbanguists, of the church founded by Simon Kimbangu in 1921, are **10.5%** "
            "of Kongo-Central, where it began, and 12.7% of Sankuru. "
            "**Animist answers are drawn at 0.6% in every province.** Only 167 heads gave that "
            "answer, too few for the survey to say where they are, and as everywhere on this map "
            "a box offered as an alternative to the churches is a floor. "
            "**Inside each province the dots follow a population grid corrected to the 2024 "
            "territory figures.** The grid has lost most of Sankuru, where it holds under a tenth "
            "of the projected population, so in five of Sankuru's six territories the dots are "
            "spread evenly over the settled places the grid does show."),
        how="household survey, two rounds in 2005 and 2012",
        grain="provinces, 4.5 million people on average",
        # `gap` (§6.12): `Manquant`, 68,543 of COD-PS 2024's 117,808,872 as laid on the provinces
        # (tools/gap_share.py, rows method; it does not write a NEW rows-only residual itself).
        gap_share=0.000582,
        gap="0.06% of household heads, whose religion was not recorded",
        counts=_cd_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "cd" / "cd_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_cd_place_weight,
        note="§11h REFUSED THIS TABLE ON 2026-09-05 AND THIS BUILD REVERSES IT, on the rulings and "
             "builds since. §11h's reason was that the cells are 31,755 sampled household heads, "
             "not people; `do` and `hn` have since been drawn from exactly that shape (MICS `HC1A`, "
             "the head's religion applied to the household), and every survey country here draws "
             "sample shares on a population base. The open alternatives were checked first and "
             "print religion nationally only: EDS-RDC III 2023-24 (FR393 Tableau 3.1), MICS-Palu "
             "2017-18 (asks HC1A, tabulates nothing), the Enquête 1-2-3 2012 results report (no "
             "religion table). The UNSD oracle has no DRC row; DR Congo is not an Afrobarometer, "
             "WVS or Global Flourishing Study country; Pew's 2008-09 Africa survey is behind a Pew "
             "account. The person-level 1-2-3 files (roster item M27 asks every member) sit behind "
             "the University of Antwerp's registration form; the DHS recode files CDIR81 and CDMR81 "
             "(v130, 26 provinces, 2023-24) behind a DHS registration. "
             "DISTRICT SHARES ARE WEIGHTED BY DISTRICT POPULATION INSIDE EACH PROVINCE (COD-PS "
             "2019, USCB's own sheet), because the 2012 sample was fixed per district: pooling heads "
             "straight moves Ituri's Catholic share by 9.5 points. "
             "THE SPLIT-HALF IS ON DISTRICTS, because no cluster id survives USCB's summing: 400 "
             "random halvings of each province's sampled districts, against 2,000 regroupings of "
             "the 147 halvable districts into provinces. Seven answers pass (p 0.0005 to 0.0035, "
             "chi-square decisive, no district over 17% of an answer); Animiste fails at p 0.059 "
             "and is flat. "
             "THE POPULATION IS COD-PS 2024 (HPC projection by health zone, 117,808,872). The name "
             "join to USCB's provinces is witnessed by COD-PS 2019 in the same workbook: Spearman "
             "+0.902 against a best shuffled pairing of +0.709. "
             "THE GRID IS KONTUR CALIBRATED TO THE 164 TERRITOIRES, because raw Kontur reads 0.09 of "
             "COD-PS in Sankuru and 0.35 in Kongo-Central; five Sankuru territoires more than 10x "
             "the national factor are spread evenly over Kontur's hexes instead of scaled "
             "(a scaled Lubefu drew a 214,456/km2 hex). sources/cd_geo.py.",
    ),
}
