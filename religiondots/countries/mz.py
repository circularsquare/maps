# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403

# The two provinces whose 2007 district volume is not archived, drawn whole from 2017.
_MZ_WHOLE = ("MZ02", "MZ06")   # Cabo Delgado, Manica


def _mz_counts():
    """INE Moçambique: 2007 districts sized to the 2017 census in nine provinces; two provinces whole.

    TWO FILES, ONE MAP. `mz_districts.csv` (sources/mz_2007.py) is 121 districts as they were
    in 2007, in nine provinces: each district's religion is the 2007 census's Quadro 8.1 shape,
    fitted by IPF to the 2017 census's province totals per answer (Quadro 11) and district
    populations (Quadro 3). Every one of those rows is `derived`, because the 2017 numbers were
    measured at province and spread over districts, which is Switzerland's case (countries/ch.py)
    and spec §7a-i-1's rule: the column was not measured at the unit it is drawn on. So their
    roll is NOWHERE and `inferred dots: not shown` leaves those nine provinces empty.
    `mz.csv` (sources/mz.py) supplies Cabo Delgado and Manica, whose 2007 district volumes the
    Wayback Machine never captured, as 2017's own provincial counts, `measured`.

    The nine provinces' districts are asserted to sum to their 2017 Quadro 11 rows answer by
    answer, to the person, so the two files partition the country. `Total` and `Desconhecida`
    (674,680, 2.5%) resolve to nothing.
    """
    from mz2017 import resolve
    from rollup import NOWHERE

    norm = HERE / "data" / "normalized"
    kw = dict(dtype={"geo_id": str}, low_memory=False, keep_default_na=False, na_values=[""])
    prov = pd.read_csv(norm / "mz.csv", **kw)
    prov = prov[prov["geo_level"] == "province"].copy()
    dist = pd.read_csv(norm / "mz_districts.csv", **kw)
    if dist["geo_level"].ne("district2007").any() or dist["geo_id"].nunique() != 121:
        raise SystemExit(f"mz_districts.csv: {dist['geo_id'].nunique()} units, expected 121 "
                         "2007 districts -- re-run sources/mz_2007.py")

    # The partition: the nine provinces' districts are their Quadro 11 rows, to the person.
    d = dist.assign(p=dist["geo_id"].str[:4]).groupby(["p", "source_category"])["count"].sum()
    p = prov.set_index(["geo_id", "source_category"])["count"]
    p = p[~p.index.get_level_values(0).isin(_MZ_WHOLE)].rename_axis(["p", "source_category"])
    if not d.sort_index().equals(p.sort_index()):
        raise SystemExit("mz_districts.csv does not sum to mz.csv's nine provinces -- one of the "
                         "two files is stale; re-run sources/mz.py and sources/mz_2007.py")

    whole = prov[prov["geo_id"].isin(_MZ_WHOLE)].assign(tier="measured", roll=None)
    dist = dist.assign(tier="derived", roll=NOWHERE)
    df = pd.concat([dist, whole], ignore_index=True)
    df["unit"] = df["geo_id"]

    lut = pd.read_csv(HERE / "data" / "geo" / "mz" / "mz_units.csv", dtype=str)
    missing = sorted(set(df["unit"]) - set(lut["unit"]))
    if missing or df["unit"].nunique() != 123:
        raise SystemExit(f"units with no polygon: {missing}; {df['unit'].nunique()} units, "
                         "expected 123 -- re-run sources/mz_geo.py")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)].copy()
    df["may_ring"] = df["tier"] == "measured"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier", "roll", "may_ring"]]


def _mz_place_weight(place):
    """countries.py hook. `place` is the hex layer scatter.py has read.

    Kenya's weighter unchanged. With 2007 districts the grid matters less than it did at
    province, but Niassa's Mecula and Marrupa are still mostly reserve with their people on a
    few roads, and Cabo Delgado and Manica are still whole provinces.
    """
    if "pop" not in place.columns:
        print("  !! mz_hexes.gpkg has no `pop` column — run sources/mz_grid.py; "
              "placing on equal shares (§8.2)")
        return None
    return _KeHexWeighter(place)


ENTRY = {
    "mz": dict(
        name="Mozambique",
        source="Recenseamento Geral da População e Habitação 2017 (Quadros 3 and 11) and 2007 "
               "(Quadro 8.1 of the district volumes), INE Moçambique",
        basis="self-identification, whole census population",
        note_public=(
            "**Mozambique's census counts the Zionist churches as an answer of their own.** "
            "The *Zione/Sião* box holds **4.2 million** people, 15.6% of the country, and "
            "nearly three quarters of them live in the six provinces from Manica and Sofala "
            "south to Maputo. In Inhambane's Funhalouro and Mabote more than half the people "
            "give it; in Cabo Delgado 0.4% do. "
            "**The districts are drawn as they were in 2007, resized to 2017.** The 2017 census "
            "published religion by province only. The 2007 census printed it for every "
            "district, one volume per province, and nine of the eleven volumes survive in the "
            "Wayback Machine. So each district's mix of answers here is 2007's, scaled so that "
            "every province's total for each answer, and every district's population, are the "
            "2017 census's own. Where a whole province changed in those ten years its districts "
            "move together: Maputo Província's evangelical and Pentecostal share went from "
            "**16.9%** to **34.4%**, and every district in it rises with it. Cabo Delgado and "
            "Manica are drawn as whole provinces, because their 2007 district volumes were not "
            "archived. "
            "**Islam is the north.** Niassa is **59.0%** Muslim and Cabo Delgado 52.6%, and "
            "Mavago on the Tanzanian border, Muembe and N'gauma are each over 95%. Lago on the "
            "lake shore is **38.7%** Anglican, "
            "opposite the Likoma mission, and Nipepe on the Nampula border is 72% Catholic. "
            "**The census's no-religion answer is drawn as no religion.** Its questionnaire "
            "words that answer *Sem religião (ateu, animista, agnóstico,...)*, so it could hold "
            "followers of traditional religion too. Surveys that offer both answers find few: "
            "in Afrobarometer's rounds from 2008 to 2023, 0.17% of adults name traditional "
            "religion and 7.6% name none. Those surveys record what people call themselves, not "
            "what they practise. The answer is **32.8%** of Tete, and **62.1%** of Tete's "
            "Changara, against under 1% of Niassa."),
        how="census, 2007, resized to 2017 census totals",
        grain="districts as of 2007, and two whole provinces; 219,000 people on average",
        fill="from the 2007 census",
        gap="2.5% whose religion was recorded as unknown",
        gap_share=0.0251,
        counts=_mz_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "mz" / "mz_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_mz_place_weight,
        note="THE 2017 CENSUS PUBLISHED RELIGION BY PROVINCE; THE 2007 CENSUS PRINTED IT BY "
             "DISTRICT. 2017's Quadro 11 is one xlsx per province on INE's retired Plone site, "
             "read from the Wayback Machine's 2019-11-14 captures (sources/mz.py); the eleven "
             "provinces sum to the national table's total to the person. 2007's *Indicadores "
             "Sócio-Demográficos Distritais* print Quadro 8.1, religion by district in the same "
             "eight answers, as one-decimal shares with each district's N; nine of eleven "
             "volumes are captured (sources/mz_2007.py), Cabo Delgado and Manica at no "
             "timestamp. In those nine provinces each 2007 district's shares are fitted by IPF "
             "to 2017's Quadro 11 answer totals and to 2017's Quadro 3 district populations, "
             "grouped where administrative posts changed district after 2007 (six posts, "
             "POST_MOVES), with Zambézia on one province-wide group because its 2017 district "
             "tables are not captured. Every fitted row is `derived` with no roll, as "
             "Switzerland's. The 2007 volumes' own checks: every row closes to 100, every N "
             "column to its Total row, every N-weighted share to the Total row within 0.1 "
             "except Gaza's, whose printed Total row is the right eight numbers in the wrong "
             "cells (pinned). 2007 districts are dissolved from COD-AB's posts and tile each "
             "province exactly (sources/mz_geo.py); dots are spread over Kontur 400m hexagons "
             "(sources/mz_grid.py). `Sem religião` includes animists by the questionnaire's "
             "own wording and is drawn as `unaffiliated`, Anita's ruling of the night of "
             "2026-09-14; taxonomy/mz2017.py REVIEW has why, and sources/mz.md §6 the national "
             "sources on the split. sources/mz.md §7 is the district upgrade's record.",
    ),
}
