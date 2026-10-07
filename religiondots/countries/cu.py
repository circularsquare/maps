# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _cu_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Kontur 2023 holds 0.17 of Granma's share of ONEI's count and about 0.4 of Santiago de Cuba's
    and Guantánamo's, near uniformly inside each, so the hexes are scaled to each province's 2024
    count, and two false blocks at the cap in Havana are lowered first (sources/cu_grid.py).
    """
    return _kontur_place_weight(place, "cu_hexes.gpkg", "sources/cu_grid.py")


def _cu_counts():
    """NORC 2016's national shares on ONEI's 2024 count: 16 provinces, one mix.

    EVERY ROW IS `modelled` (§7b). No Cuban census asks religion, and NORC's public file has no
    region below the nation, so each province's people take the national mix of 835 adults'
    answers. sources/cu.py and sources/cu.md.
    """
    from cu2016 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "cu.csv",
                     dtype={"geo_id": str}, keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "cu" / "cu_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"cu.csv provinces with no polygon: {missing}; re-run sources/cu_geo.py")
    if df["unit"].nunique() != 16:
        raise SystemExit(f"{df['unit'].nunique()} provinces, expected 16")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"cu.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "cu": dict(
        name="Cuba",
        source="NORC at the University of Chicago, A Rare Look Inside Cuban Society, survey of "
               "2016 (public use file), against each province's people at the end of 2024 "
               "(Oficina Nacional de Estadística e Información, Anuario Demográfico de Cuba 2024)",
        basis="self-identification, Cuban adults 18 and over",
        note_public=(
            "**No Cuban census asks about religion.** None of the census forms from 1899 to 2012 "
            "has the question. This map uses the one survey fielded inside Cuba whose answers "
            "are public: NORC at the University of Chicago interviewed 840 adults face to face "
            "in 2016, and **835** of them named a religion or none. Their answers, weighted as "
            "NORC weighted them, are applied to each province's people in the National Office "
            "of Statistics and Information's count for the end of 2024, **9,748,007**. Nobody "
            "counted these dots, so they disappear when inferred dots are turned off. "
            "**Every province is drawn at the same mix.** NORC's public file records whether "
            "someone lived in a town or in the countryside, but not where, so nothing places any "
            "religion below the whole country. Santería looked more common in towns, 19% "
            "against 10% of the 82 people interviewed in the countryside, but with so few rural "
            "interviews that could be chance, so the map does not draw it. Children are drawn "
            "at the adults' shares. "
            "**What people said.** Catholic **28.3%**; Santería, also called the Regla de Ocha, "
            "**16.9%**; another Christian church 6.5%, most of them answering only \"Christian\"; "
            "believing in God without belonging to a religion **21.8%**; none of the religions "
            "offered 24.4%, drawn as no religion; atheist 1.1%; something else 1.0%. Santería is "
            "often practised alongside Catholicism, and a question with one answer records "
            "whichever a person names, so its share counts those who named it first. "
            "**Other figures put the Christian share much higher.** Pew Research Center's "
            "estimate for 2020 has Cuba at 60.7% Christian, 21.6% religiously unaffiliated and "
            "17.4% in other religions. A 2015 poll of 1,200 Cubans by Bendixen & Amandi for "
            "Univision and The Washington Post found 27% Catholic and 44% saying they were not "
            "religious, which is closer to NORC. "
            "**The survey missed part of the east.** Hurricane Matthew struck during the "
            "fieldwork in October 2016, and NORC left out several areas of eastern Cuba holding "
            "about 15% of the population; they are drawn at the national mix like everywhere "
            "else. The survey also comes before the large emigration of the 2020s, and nothing "
            "says whether the people who left differ in religion from those who stayed. The "
            "Guantánamo Bay Naval Base, which the United States administers, is not drawn."),
        how="survey, 2016, one national share and mix",
        grain="provinces, 609,000 people on average",
        counts=_cu_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "cu" / "cu_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_cu_place_weight,
        note="REOPENED AND BUILT ON ANITA'S RULINGS (ask/RULINGS.md 2026-09-15 priority holes, "
             "2026-09-16 Mauritania: a country no census asks is drawn on the best survey or "
             "compiler figure, with the method said). sources/cu.md is the record. NOTHING "
             "COUNTS: census forms 1899-2012 read (sources.md 11ap, scout-2026-09-15-negatives; "
             "1970 and 1981 unread). SURVEY: NORC 2016 public use file (norc.org, no login), Z10, "
             "840 adults, 5 DK/refused out, finalwt; no region variable, only urban/rural "
             "(chi-square p 0.138, Santeria perm p 0.039, 0.27 Bonferroni over 7: not drawn). "
             "Topline reproduced to a point. Evangelical, Protestant and Christian (other) merged "
             "in the file -> christianity. NEW NODES afrodiasporic.santeria (ask 044) and other.cu. "
             "WITNESSES: Pew 2020 Christian 60.7 / unaffiliated 21.6 / other 17.4 (not drawn; "
             "compiler); Bendixen & Amandi 2015 (no microdata). POPULATION: ONEI Anuario "
             "Demografico 2024 Tabla 1.5, poblacion efectiva 31 Dec 2024, 9,748,007; COD-PS 2024 "
             "(projection from 2012, 11.3M, 1.16x) rejected. GEOGRAPHY: COD-AB v01 (GADM) "
             "16 provinces; La Habana 819 km2 vs ONEI 728 pinned. PLACEMENT: Kontur CU, "
             "Guantanamo base (NE USG) dropped, 2 Havana cap blocks capped, scaled to ONEI per "
             "province (Granma raw 0.17, Santiago 0.39, Guantanamo 0.41).",
    ),
}
