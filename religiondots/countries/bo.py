# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _bo_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Bolivia needs it for both of §8.2's reasons. Beni, Pando and the Santa Cruz lowlands are
    most of the land and a small part of the people, the Altiplano's salt flats are empty, and
    La Paz-El Alto, Santa Cruz de la Sierra and Cochabamba hold close to half the country
    (sources/bo_grid.py).
    """
    return _kontur_place_weight(place, "bo_hexes.gpkg", "sources/bo_grid.py")


def _bo_counts():
    """LAPOP AmericasBarometer single-country files, waves 2010-2023 pooled, at departamento:
    11 categories, all 9 units, and EVERY ROW IS `modelled` IN §7.

    sources/bo.py builds it and sources/bo.md is the record. LAPOP's `prov` is its own
    department order in 2010-2018 and a province code in 2023; `municipio` names, and 2023's
    INE municipality codes, place every sampled municipality in the department its label
    names. Each wave is post-stratified to the 2024 census department shares. The 1992
    census, the last that asked, is a witness for the department ordering and is not drawn
    (sources/bo_checks.py).
    """
    from bo2023 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "bo.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "bo" / "bo_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"bo.csv departments with no polygon: {missing} -- re-run "
                         "sources/bo_geo.py, the lookup is stale")
    if df["unit"].nunique() != 9:
        raise SystemExit(f"{df['unit'].nunique()} departments, expected 9")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"bo.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    # three LAPOP answers share other.bo
    return (df.groupby(["unit", "node"], as_index=False)
              .agg(count=("count", "sum"), congregations=("congregations", "sum"),
                   tier=("tier", "first")))


ENTRY = {
    "bo": dict(
        name="Bolivia",
        source="AmericasBarometer, six rounds 2010 to 2023 (LAPOP Lab, Vanderbilt "
               "University), against the 2024 census department populations (INE Bolivia)",
        basis="self-identification, adults 18 and over",
        view=[-69.8, -23.0, -57.4, -9.6],
        note_public=(
            "**No Bolivian census since 1992 has asked about religion, so this is a survey "
            "standing where a census would be.** The map is drawn from the LAPOP "
            "AmericasBarometer, **13,882 people** across six rounds between 2010 and 2023, "
            "pooled and applied to each department's 2024 census count. The dots are drawn "
            "desaturated to say so. "
            "**La Paz is the least Catholic department.** It is 60.3% Catholic, against 83.2% "
            "in Chuquisaca. Evangelicals are **24.0% of Pando** and 17.9% of Beni, against 5.5% "
            "of Chuquisaca. The 1992 census orders the departments in much the same way for "
            "Catholics and for other Christians. "
            "**Five answers are drawn where the survey found them.** Catholic, evangelical, "
            "traditional Protestant, believing in God without belonging to a religion, and "
            "agnostic or atheist. The other six are spread at the national rate. "
            "**The level is an average over 2010 to 2023.** Catholic identification runs from "
            "80.3% in 2010 to **64.8% in 2023**, and evangelicals from 8.5% to 17.9%, so this "
            "map is more Catholic than any round since 2016 found. From 2018 "
            "the survey stopped offering Jehovah's Witnesses and Mormons as answers of their "
            "own, so both are undercounts."),
        how="survey, six rounds 2010 to 2023 pooled",
        grain="departments, 1.3 million people on average",
        counts=_bo_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bo" / "bo_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_bo_place_weight,
        note="A SURVEY ON A CENSUS COUNT AND EVERY ROW IS `modelled` (§7b): LAPOP's single-country "
             "files give a department share and the 2024 census, read off INE's REDATAM base, "
             "the people. sources/bo.py builds it, sources/bo.md is the record, sources.md "
             "§11ap scouted it and §9dm writes it up. THE LABELS WERE CHECKED BEFORE THEY WERE "
             "TRUSTED: prov is LAPOP's own department order in 2010-2018 and a province code in "
             "2023, and municipio names place every sampled municipality in the labelled "
             "department in every wave. Each wave is post-stratified to the census department "
             "shares. Placed on department shares: Catholic, evangelical, traditional "
             "Protestant, believer without a church, agnostic or atheist. The 1992 census "
             "(REDATAM, persons in households) is a witness for the department ordering and is "
             "not drawn: its province pattern inside departments did not replicate in LAPOP "
             "(sources/bo_checks.py)."),
}
