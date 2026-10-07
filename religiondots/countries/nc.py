# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _nc_place_weight(place):
    """countries.py hook. `place` is the Kontur 400 m hex layer scatter.py has read.

    Kontur NC reads the Loyalty Islands at about half their census share (Maré 0.47, Ouvéa 0.55,
    Lifou 0.58 of the national ratio; sources/nc_geo.py). Placement is inside each commune only,
    so that moves nobody between communes.
    """
    return _kontur_place_weight(place, "nc_hexes.gpkg", "sources/nc_geo.py")


def _nc_counts():
    """Pew 2020's national levels, placed by Kohler's 1978 church count on the 2019 census's
    communities by commune: 33 communes.

    EVERY ROW IS `modelled` (§7b). No census or survey asks; sources/nc.py and sources/nc.md.
    """
    from nc2020 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "nc.csv", keep_default_na=False,
                     na_values=[""], dtype={"geo_id": str})
    lut = pd.read_csv(HERE / "data" / "geo" / "nc" / "nc_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"nc.csv communes with no polygon: {missing}; re-run sources/nc_geo.py")
    if df["unit"].nunique() != 33:
        raise SystemExit(f"{df['unit'].nunique()} communes, expected 33")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"nc.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    # spec §3.10: a modelled share cannot put a ring on a commune.
    df["may_ring"] = False
    return df[["unit", "node", "count", "congregations", "tier", "may_ring"]]


ENTRY = {
    "nc": dict(
        name="New Caledonia",
        source="Pew Research Center's 2020 estimate (World Religion Database), placed by "
               "J.-M. Kohler's 1978 church count (ORSTOM) on the 2019 census's communities by "
               "commune (ISEE)",
        basis="a compiler's national estimate, placed by a 1978 count of church members",
        note_public=(
            "**No census or survey in New Caledonia asks about religion.** The 1996, 2009 and "
            "2019 census forms leave it out, and no survey that asks has been found. The levels "
            "here are Pew Research Center's estimate for 2020, which comes from the World "
            "Religion Database: 85.1% Christian, 10.5% with no religion, 2.8% Muslim, and small "
            "shares of Buddhists, Bahá'ís and others. "
            "**Where each church is drawn comes from a count made in 1978.** Jean-Marie Kohler "
            "of ORSTOM counted the members of every church at the start of 1978, community by "
            "community, and for Kanak by home commune. The map applies those 1978 mixes to each "
            "commune's communities in the 2019 census, then scales the whole to Pew's levels. "
            "That draws **59.1%** of New Caledonia as Catholic and **24.6%** as Protestant, and "
            "the Loyalty Islands as **71.1%** Protestant; Kohler found the Kanak of Lifou and "
            "Maré more than 80% Protestant and those of Ouvéa 61% Catholic. Kanak living away "
            "from their home commune, most of them in Greater Nouméa, are given the mix of the "
            "communes that have lost people since 1978. "
            "**This is a model, and there was nothing to check it against.** It assumes each "
            "community's religion has not changed since 1978 and is the same wherever its "
            "members live. The weakest part is no religion. Kohler's count saw very few people "
            "outside the churches, so Pew's 10.5% is placed where his small count of them was "
            "largest, among Europeans, which puts **14.1%** of Nouméa at no religion; treat the "
            "location of no religion as the least reliable thing on this map. Muslims are "
            "placed where Kohler's were, most of them of Indonesian descent. Buddhists and "
            "other religions are drawn at an even share, since nothing places them. Nobody "
            "counted these dots, so they disappear when inferred dots are turned off. "
            "Inside each commune the dots follow Kontur's population map, scaled to the "
            "commune's count in the 2019 census."),
        how="no source asks; a compiler's national estimate, placed by a 1978 church count",
        grain="communes, 8,200 people on average",
        counts=_nc_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "nc" / "nc_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_nc_place_weight,
        note="BUILT ON ANITA'S RULINGS (ask/RULINGS.md 2026-09-15 priority holes, 2026-09-16 "
             "Mauritania: a country no source asks is drawn on the best compiler figure, method "
             "said). sources/nc.md is the record. NO CENSUS RELIGION: forms 1996, 2009, 2019 "
             "(sources.md §scout-2026-09-14-asia-oceania); Kohler 1979 p.8, earlier censuses "
             "did not record it either. LEVEL: Pew 2020 New Caledonia (WRD; Pew 2012 Appendix "
             "B): Christian 85.10, unaffiliated 10.52, Muslim 2.76, other 0.95 (split by WRD "
             "2025 parts, ARDA u=162c: Baha'i 0.37 to bahai, the rest plus Jews 0.04 to "
             "other.nc), Buddhist 0.62. GEOGRAPHY AND CHURCHES: J.-M. Kohler, Religions et "
             "dynamique sociale en Nouvelle-Calédonie, fasc. II (ORSTOM 1979), Tableau 1 (church "
             "x community, start of 1978) and Tableau 2 (Melanesian Catholics/Protestants by "
             "commune of origin), a roll; laid on ISEE RP2019 community x commune "
             "(rp-structure-communautes.xls) with Tahitian/Indonesian/Vietnamese/Ni-Vanuatu "
             "split at province proportions; Kanak migrants at the mix of the communes that "
             "lost people; then IPF to Pew's levels and the commune counts. spec §14.12: no "
             "second cut to check against, said in the note; weakest cell no religion. "
             "GEOGRAPHY: data.gouv.nc communes-nc-limites-terrestres-simplifiees (Licence "
             "Ouverte 2.0), joined on code_com. PLACEMENT: Kontur NC 2023 (ratio 1.080; Loyalty "
             "Islands 0.47-0.58 of it).",
    ),
}
