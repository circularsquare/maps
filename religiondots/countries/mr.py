# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _mr_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Tiris Zemmour, Adrar and Tagant are 59% of Mauritania and hold 5.4% of its people. Kontur's hexes
    are calibrated to the census's 63 moughataas, because Kontur puts Nouakchott's people in Tevragh
    Zeina (4.09x its count) and not in El Mina, Dar Naim or Riyad (sources/mr_grid.py).
    """
    return _kontur_place_weight(place, "mr_hexes.gpkg", "sources/mr_grid.py")


def _mr_counts():
    """RGPH 2023 nationals on Islam, foreign residents by nationality: 15 wilayas.

    EVERY ROW IS `modelled` (§7b). No source asks Mauritanians their religion, so every national is
    drawn on Islam; foreign residents take the census's national nationality groups through Pew
    2020, with the 46,800 refugees placed in Hodh Chargui. Both halves are the same census's counts
    per wilaya, so they partition each unit. sources/mr.py and sources/mr.md.
    """
    from mr2023 import resolve

    nat = pd.read_csv(HERE / "data" / "normalized" / "mr.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    nat["node"] = nat["source_category"].map(resolve)
    unmapped = sorted(nat.loc[nat["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"mr.csv categories with no node: {unmapped}")
    ext = pd.read_csv(HERE / "data" / "normalized" / "mr_foreign.csv", dtype={"geo_id": str})
    df = pd.concat([nat[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    lut = pd.read_csv(HERE / "data" / "geo" / "mr" / "mr_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"mr rows with no unit: {missing}; re-run sources/mr_geo.py")
    if df["unit"].nunique() != 15:
        raise SystemExit(f"{df['unit'].nunique()} wilayas, expected 15")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "mr": dict(
        name="Mauritania",
        source="No census or survey asks; the 2023 census count of Mauritanians and of foreign "
               "residents in each wilaya (ANSADE, RGPH 2023), with foreign residents' nationality "
               "through Pew Research Center's 2020 estimates and the United Nations' International "
               "Migrant Stock 2024",
        basis="nationals drawn as Muslim, which nobody asked; foreign residents by nationality",
        note_public=(
            "**Nobody in Mauritania is asked their religion.** The 2013 census form has no question "
            "on it, none of the sixteen reports on the 2023 census has a table on it, and the Arab "
            "Barometer, the Afrobarometer and the Demographic and Health Survey all leave it out "
            "there. So the **4,801,598** Mauritanians the 2023 census counted are all drawn as "
            "Muslim, which is also what the CIA World Factbook gives: \"Muslim (official) 100%\". "
            "Any Mauritanians who are not Muslim are not on this map, because no source counts "
            "them. Nobody counted these dots, and they are drawn desaturated to say so. "
            "**Every non-Muslim drawn is a foreign resident.** The census counted **125,933** "
            "foreign residents, 2.6% of the people living in the country, in every wilaya, but "
            "their nationality only for the whole country: 83,681 from Mali, 22,906 from Senegal, "
            "11,660 from other African countries, 2,280 from other Arab countries, 1,602 from "
            "Morocco, 284 from Algeria, 600 from Europe and 2,920 from everywhere else. Each "
            "nationality is drawn at Pew Research Center's 2020 estimate for its home country, "
            "which cannot see anyone who converted or stopped practising, and the other African "
            "and other Arab groups are split by the United Nations' 2024 count of Mauritania's "
            "migrants by country of origin. That puts **10,345** people on religions other than "
            "Islam, 6,136 of them Christian and 2,497 with no religion. Pew's own estimate for "
            "everyone living in Mauritania is 99.185% Muslim with 10,754 Christians; this map "
            "draws 99.790% Muslim. "
            "**The 46,800 refugees are drawn in Hodh Chargui.** Hodh Chargui holds 59,555 of the "
            "foreign residents, and the census reports put that down to the Malian refugees of the "
            "Mbera camps, so the refugees are drawn there and the other foreign residents at the "
            "national mix in every wilaya. The refugees came from northern Mali, so they are not "
            "drawn at Pew's estimate for Mali. UNHCR's 2018 map of where the Mbera camp's refugees "
            "came from puts 89.7% of them from Tombouctou, 6.5% from Mopti and 3.6% from Ségou, "
            "and each is drawn at the shares Mali's 2022 census gives the same area. That makes "
            "379 of the 46,800 not Muslim, where Pew's figure for Mali would make 2,745. "
            "**The census says it missed some foreigners.** Migrants without a home were missed by "
            "the door-to-door count, gold miners move around, and some foreigners said they were "
            "Mauritanian because the count coincided with a campaign to issue residence cards. "
            "It also found far fewer refugees than UNHCR: at the end of 2023, while the count was "
            "under way, UNHCR had **118,476** refugees and asylum seekers registered in Mauritania, "
            "against the census's 46,800. The census counted about 41,000 people in the Mbera "
            "camps, which UNHCR put at almost 100,000 after the arrivals of 2023, so most of the "
            "difference was never counted, rather than counted as Mauritanian. Those 71,676 people "
            "are not on this map; they are the not-drawn part of the bar."),
        how="no source asks; nationals drawn as Muslim, foreign residents by nationality",
        grain="wilayas, 328,000 people on average",
        gap="refugees and asylum seekers the census did not count, 1.4% of residents: UNHCR's "
            "118,476 registered at the end of 2023 less the census's 46,800; and Mauritanians "
            "who are not Muslim, whom no source has counted",
        gap_share=0.01434,
        counts=_mr_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "mr" / "mr_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_mr_place_weight,
        note="BUILT ON ANITA'S MAGHREB RULINGS (ask/RULINGS.md 2026-09-15 and 2026-09-16: draw it on a "
             "compiler's figure, foreigners by wilaya, no rings). sources/mr.md is the record. "
             "NATIONALS: nothing asks (RGPH 2013 form; no RGPH 2023 table; AB VII-VIII Q1012 empty; "
             "Afrobarometer and DHS skip it); all 4,801,598 on islam, Factbook \"Muslim (official) "
             "100%\". Pew 2020's non-Muslims are for all residents, so Pew's residual on nationals "
             "was not drawn (double count; sources/mr.md §4); the foreigner layer alone gives 0.53 "
             "of Pew's Christian share. FOREIGNERS: 125,933 per wilaya (Thème 16, Tableau 16.6), "
             "eight national groups (16.2), 46,800 refugees placed in Hodh Chargui as Malians at "
             "northern Mali's RGPH 2022 shares weighted by UNHCR's 2018 Mbera origins (sources/mr.md "
             "§7), the rest at the national mix without them; other African and other Arab split by UN DESA "
             "2024's named origins, Europe and the rest by Pew's regional rows (50% cover bar); Pew "
             "2020 per nationality, Muslim branches folded to islam. GEOGRAPHY: COD-AB ADM1 15 "
             "wilayas, RGPH 2023 Thème 1 p.3 populations. PLACEMENT: Kontur MR calibrated to the "
             "census's 63 moughataas (Tableau A.1.4); Nouakchott's Tevragh Zeina 4.09x before; "
             "Bassiknou's Vessale commune still carries about 30,000 too many. NOT DRAWN: "
             "Mauritanian non-Muslims (no source), rings (no rings), refugees the census missed "
             "(gap_share: UNHCR end-2023 118,476 less 46,800; ask 033; sources/mr.md §8).",
    ),
}
