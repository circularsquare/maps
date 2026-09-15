# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ve_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Venezuela needs it for both of §8.2's reasons. Bolívar and the llanos of Apure and Guárico
    are most of the drawn land and a small part of the people, and three quarters of Venezuelans
    live along the coastal mountains, in and around Caracas, Maracaibo, Valencia, Barquisimeto
    and Maracay (sources/ve_grid.py).
    """
    return _kontur_place_weight(place, "ve_hexes.gpkg", "sources/ve_grid.py")


def _ve_counts():
    """LAPOP AmericasBarometer single-country files, waves 2010-2016/17 pooled, at estado:
    10 categories, 21 of 25 federal entities drawn, and EVERY ROW IS `modelled` IN §7.

    sources/ve.py builds it and sources/ve.md is the record. LAPOP's `prov` is alphabetical in
    2010 and in its own order in 2012-2016; `municipio` names place every sampled municipality in
    the labelled state in every wave. Each wave is post-stratified to the 2011 census state
    shares. Amazonas, Delta Amacuro, Nueva Esparta and the Dependencias Federales were never
    sampled and are NOT in ve.csv; they are in `gap=`.
    """
    from ve2016 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ve.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "ve" / "ve_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"ve.csv states with no polygon: {missing} -- re-run "
                         "sources/ve_geo.py, the lookup is stale")
    if df["unit"].nunique() != 21:
        raise SystemExit(f"{df['unit'].nunique()} states, expected 21")
    blank = {"VE02", "VE10", "VE17", "VE25"}
    if blank & set(df["unit"]):
        raise SystemExit(f"{sorted(blank & set(df['unit']))} is in ve.csv and must not be")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"ve.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    # three LAPOP answers share other.ve
    return (df.groupby(["unit", "node"], as_index=False)
              .agg(count=("count", "sum"), congregations=("congregations", "sum"),
                   tier=("tier", "first")))


ENTRY = {
    "ve": dict(
        name="Venezuela",
        source="AmericasBarometer, four rounds 2010 to 2016/17 (LAPOP Lab, Vanderbilt "
               "University), against the 2011 census state populations (INE Venezuela)",
        basis="self-identification, adults 18 and over",
        view=[-73.4, 0.6, -59.7, 12.3],
        note_public=(
            "**No Venezuelan census asks about religion, so this is a survey standing where a "
            "census would be.** The map is drawn from the LAPOP AmericasBarometer, **5,894 "
            "people** across four rounds between 2010 and 2016/17, pooled and applied to each "
            "state's count in the 2011 census, the last one. The dots are drawn desaturated to "
            "say so. "
            "**Evangelicals are most common in Bolívar and the llanos.** They are **20.4% of "
            "Bolívar** and 16.2% of Portuguesa, against 5.0% of Miranda and 2.9% of the Distrito "
            "Capital. "
            "**Catholics are drawn by the survey's six regions rather than by state.** The state "
            "figures did not hold up when the survey's rounds were split in half, and the "
            "regional ones did. They run from 62.9% in the llanos region (Portuguesa, Guárico, "
            "Apure and Barinas) to 81.1% in the Andes (Mérida, Táchira and Trujillo). Jehovah's "
            "Witnesses are drawn by region too, and are highest in Zulia and Falcón at 3.2%. "
            "Most of the people who chose the answer for traditional religions live in Caracas, "
            "and it is drawn by state. The other answers are spread across what is left of each "
            "state at their national proportions. "
            "**Four states are blank because the survey never went there.** Amazonas, Delta "
            "Amacuro, Nueva Esparta and the Dependencias Federales hold **2.96%** of Venezuelans. "
            "Apure, Barinas, Monagas and La Guaira were visited in 2010 only. Apure and Barinas "
            "take the rates of the other llanos states and La Guaira those of Caracas; Monagas "
            "takes its region's Catholic share and the national rate for the rest. "
            "**The level is an average over 2010 to 2016/17, and there is no later round.** "
            "Catholic identification runs from 78.1% in 2010 to **67.5% in 2016/17**, and "
            "evangelicals from 6.1% to 13.0%, so this map is more Catholic than the last round "
            "found. The population is the 2011 count, taken before the emigration of the late "
            "2010s."),
        how="survey, four rounds 2010 to 2016/17 pooled",
        grain="states, 1.3 million people on average",
        gap=("Amazonas, Delta Amacuro, Nueva Esparta and the Dependencias Federales, 805,770 "
             "people, 2.96% of the country in the 2011 census, which the survey never sampled"),
        gap_share=0.0296,
        counts=_ve_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ve" / "ve_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ve_place_weight,
        note="A SURVEY ON A CENSUS COUNT AND EVERY ROW IS `modelled` (§7b): LAPOP's single-country "
             "files give a state share and the 2011 census (INE's Cuadro 2.2, equal to COD-PS) "
             "the people. sources/ve.py builds it, sources/ve.md is the record, sources.md "
             "§11ap scouted it and §ve-2026-09-14 writes it up. THE LABELS WERE CHECKED BEFORE "
             "THEY WERE TRUSTED: prov is alphabetical in 2010 and LAPOP's own order in 2012-2016, "
             "and municipio names place every sampled municipality in the labelled state in every "
             "wave. Each wave is post-stratified to the census state shares. Placed on state "
             "shares: evangelical, traditional religions; on 2010's six design regions: Catholic "
             "(fails at the states at p=0.052 on 20,000 draws, passes at the regions), Jehovah's "
             "Witnesses. The residual reverses believers without a church in Falcón and "
             "traditional Protestants in Bolívar against the survey's own state shares "
             "(sources/ve.md §8). Apure, Barinas (llanos) and La Guaira (capital) take their "
             "region's every-round states; Monagas the national rate; four entities blank."),
}
