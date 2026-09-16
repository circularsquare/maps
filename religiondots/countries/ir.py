# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ir_place_weight(place):
    """countries.py hook. `place` is the Kontur 400m hex layer scatter.py has read.

    31 provinces over 1.62 million km2 is 52,000 km2 a unit, and most of Iran is empty: Semnan is
    97,000 km2 holding 702,360 people, while Tehran province's 13.3 million live in 13,600 km2.

    The layer's `pop` is Kontur calibrated to the 1395 census's 429 county totals (COD-PS ADM2),
    because raw Kontur places 18% of Iran in the wrong county, with false cities at its density
    limit in Sarvestan, Kavar and Kherameh holding more weight than Shiraz (sources/ir_geo.py).
    """
    return _kontur_place_weight(place, "ir_hexes.gpkg", "sources/ir_geo.py")


def _ir_counts():
    """Iran census 1395 (November 2016) at province: 6 nodes on 31 units.

    SCI Statistical Yearbook 1395, Table 3-18, in counts (sources/ir.py). Not stated (124,572) is
    in the table and off the tree (taxonomy/ir2016.py EXCLUDED); it is `gap`.

    THE `مسلمان` (MUSLIM) ROWS ARE REPLACED, NOT SUPPLEMENTED (ir_split.py, 2026-09-15), the way
    in_split.csv replaces India's `Muslim`. ir_split.csv holds every province's Muslims again:
    Masaili's Sunnis in 16 provinces (`derived`, rolling back to `islam` through ir2016.COLUMNS)
    plus a smaller census `مسلمان` for the rest, summing to the census figure. Adding them without
    dropping these would count 7.6M Muslims twice, and the check below stops that.
    """
    from ir2016 import resolve
    import ir2016

    df = pd.read_csv(HERE / "data" / "normalized" / "ir.csv",
                     low_memory=False, keep_default_na=False, na_values=[""])
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 31:
        raise SystemExit(f"{df['geo_id'].nunique()} provinces in ir.csv, expected 31 -- re-run "
                         "sources/ir.py")
    replaced = int(df.loc[df["source_category"] == "مسلمان", "count"].sum())
    df = df[df["source_category"] != "مسلمان"].assign(tier="measured")
    split = pd.read_csv(HERE / "data" / "normalized" / "ir_split.csv",
                        low_memory=False, keep_default_na=False, na_values=[""])
    split["count"] = split["count"].astype(int)
    if int(split["count"].sum()) != replaced:
        raise SystemExit(f"ir_split.csv draws {int(split['count'].sum()):,} Muslims against the "
                         f"{replaced:,} it replaces; re-run ir_split.py")
    df = pd.concat([df, split], ignore_index=True)
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(c for c in set(df.loc[df["node"].isna(), "source_category"])
                      if c not in ir2016.EXCLUDED)
    if unmapped:
        raise SystemExit(f"ir.csv / ir_split.csv categories with no node: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)].copy()
    df = _add_roll(df, ir2016.COLUMNS)
    # A Sunni share laid over a province's Muslims says nothing about where a small group is, so
    # only census rows may ring (as India's split).
    df["may_ring"] = df["tier"] == "measured"
    df["congregations"] = 0
    df = df.rename(columns={"geo_id": "unit"})
    return df[["unit", "node", "count", "congregations", "may_ring", "tier", "roll"]]


ENTRY = {
    "ir": dict(
        name="Iran",
        source="General Census of Population and Housing 1395 (Statistical Centre of Iran), "
               "Statistical Yearbook 1395, Table 3-18; Sunni Muslims from Masaili, Atlas-e "
               "Towsifi-ye Ahl-e Sonnat-e Iran (2023)",
        basis="self-identification",
        note_public=(
            "**Iran's 2016 census counts four named religions.** The answers are Muslim, "
            "Zoroastrian, Christian and Jewish, which are Islam and the three minorities Iran's "
            "constitution recognises, plus other and not stated. The Statistical Centre of Iran "
            "prints the counts for all 31 provinces in its statistical yearbook. Muslims are "
            "**99.6%**. "
            "**Sunni Muslims are drawn from one researcher's estimate.** No Iranian census "
            "publishes Muslims by sect. Mehdi Masaili, a Shia cleric who teaches at the Isfahan "
            "seminary, put Iran's Sunnis at **7.6 million**, 9.5% of the country, in his "
            "*Atlas-e Towsifi-ye Ahl-e Sonnat-e Iran* (2023), from library and field research "
            "he did alone. He gives each province a share of its 2016 census population: "
            "**82%** of Kurdistan, **64%** of Sistan and Baluchestan, 38% of Golestan, 35% of "
            "West Azerbaijan and of Hormozgan, 26% of Kermanshah, and 2% to 15% in eight more. "
            "The map takes those Sunnis out of each province's census Muslims. His other "
            "500,000, in Tehran and the central provinces, are not assigned to a province and "
            "are drawn in Tehran and Alborz by population. The other 15 provinces, Khuzestan and "
            "Isfahan among them, draw no Sunnis, though his table does not say they have none. "
            "His paper argues against claims that Sunnis are a fifth or more of Iran, and a "
            "Sunni author, Soltani (2015), puts Iran's Sunni Kurds about two million higher. "
            "The rest of Iran's Muslims are drawn without a branch; most are Twelver Shia, but "
            "no source counts them. "
            "**Zoroastrians and Jews live mostly in a few provinces.** Zoroastrians are 23,109 "
            "people and **0.32%** of Yazd, against 0.03% of Iran, though Tehran province has "
            "more of them (8,579 against Yazd's 3,600). Half of Iran's 9,826 Jews are in Tehran "
            "province; Fars has 2,816, the highest share of any province at 0.06%, and Isfahan "
            "has 1,007. "
            "**Christians are 130,158, and a third of them are in Tehran province.** Tehran "
            "(**0.33%**), West Azerbaijan (0.23%) and Isfahan (0.17%) are the only provinces "
            "above 0.16%; in the other 28 the census counts Christians at 0.08% to 0.16%. There "
            "is one Christian answer, so the Armenian and Assyrian churches are one colour. "
            "**There is no Baháʼí answer on the census.** People who answered other, 40,551 in "
            "all, are drawn as other religion. The 124,572 people (0.16%) who did not state a "
            "religion are not drawn."),
        how="census, 2016; Sunni Muslims from one researcher's 2023 estimate",
        fill="from Masaili's 2023 estimate of Sunni Muslims by province",
        grain="provinces, 2.6 million people on average",
        gap="the 0.16% who did not state a religion",
        gap_share=0.00156,
        counts=_ir_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ir" / "ir_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ir_place_weight,
        note="SCI's 1395 STATISTICAL YEARBOOK PRINTS RELIGION BY PROVINCE IN COUNTS. Table 3-18, "
             "chapter 3, read from the Iran Data Portal's copy of SCI's PDF because amar.org.ir "
             "resets every connection from here. sources.md §11n said Iran published no "
             "geography and the 2026-09-14 scout found 1390 shares in SCI's Amar magazine; this "
             "is the next census, in counts, with other and not stated apart and one Christian "
             "column. CHECKED FOUR WAYS: 31 rows parsed off the page equal the transcription and "
             "close; they sum to the national row, which equals UNSD table 28 Iran 2016 in all "
             "six categories; Table 3-17 on the page before reproduces UNSD 2011 and 2006; the "
             "1395 detailed-results workbook 3-jamiat-k.xls gives every province total to the "
             "person. The 1390 Amar table is a witness: same leading province for each minority. "
             "Ask 023 (Anita, 2026-09-14) ruled Iran drawn at its 31 provinces with other on its "
             "node. THE MUSLIM COLUMN IS SPLIT BY ir_split.py (2026-09-15, Anita's ruling of "
             "2026-09-16): Masaili's 2023 Sunni estimates by province (Haft Aseman 26(88), "
             "Table 1, re-read off the PDF), 7,608,500 Sunnis `derived` and rolling back to "
             "islam; the rest stays on islam, not islam.shia, because nobody published a Shia "
             "figure. sources/ir.md §9 has the record.",
    ),
}
