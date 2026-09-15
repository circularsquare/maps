# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _hn_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    18 departments over 112,000 km2 is about 6,200 km2 a unit, and Honduras needs the grid for
    both of §8.2's reasons: Gracias a Dios and Olancho are over a third of the land and 7% of
    the people, and Francisco Morazán's polygon runs from Tegucigalpa out into empty mountains
    (sources/hn_grid.py).
    """
    return _kontur_place_weight(place, "hn_hexes.gpkg", "sources/hn_grid.py")


def _hn_counts():
    """INE ENDESA-MICS 2019, at departamento: 7 categories, 18 units, EVERY ROW `modelled`.

    The household questionnaire's `HC1`, religion of the household head, on 20,669 households
    in all 18 departments, laid on INE's 2024 projection (no count since 2013). INE's files
    carry no weights, so sources/hn.py rebuilds them from the report's Tablas SR.3.1 and SD.1
    and checks them against three rows of SR.3.1 the fit never saw. Catholic, evangelical, no
    religion and the Witnesses carry department shares; Adventists keep their measured share
    in Islas de la Bahía only; Latter-day Saints and `OTRO` are flat. sources/hn.md is the
    record.
    """
    from hn2019 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "hn.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "hn" / "hn_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"hn.csv departments with no polygon: {missing} -- re-run "
                         "sources/hn_geo.py")
    if df["unit"].nunique() != 18:
        raise SystemExit(f"{df['unit'].nunique()} departments, expected 18")
    df["node"] = df["source_category"].map(resolve)
    import hn2019
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]) - set(hn2019.EXCLUDED))
    if unmapped:
        raise SystemExit(f"hn.csv categories with no node: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)].copy()
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "hn": dict(
        name="Honduras",
        source="Encuesta Nacional de Demografía y Salud ENDESA-MICS 2019 (Instituto Nacional "
               "de Estadística), against INE's 2024 department population projections",
        basis="self-identification, religion of the household head",
        view=[-89.5, 12.9, -83.1, 16.6],
        note_public=(
            "No Honduran census has asked about religion. This map is drawn from INE's 2019 "
            "demographic and health survey, fielded as MICS6 with UNICEF, which asked the "
            "religion of the head of each of **20,669 households** in all 18 departments. "
            "Everyone living in a household is drawn in its head's column, so 41% Catholic "
            "means 41% of Hondurans living in a Catholic-headed household. The survey asks "
            "nobody else, so how often the rest of a household differs is not known. The "
            "shares are laid on INE's 2024 population projection. "
            "**Catholics and evangelicals are almost exactly level**, about 41% each, and "
            "14.9% report no religion. Catholic identification runs from **68.7%** of Intibucá "
            "and 61.4% of Lempira, in the western highlands, down to 28.3% of Cortés and 16.7% "
            "of the Bay Islands. The evangelical share is highest in Gracias a Dios, the "
            "Mosquitia, at **60.3%**, and is above half in Cortés and Atlántida on the north "
            "coast. The Bay Islands have the highest no-religion share, 24.5%, and Gracias a "
            "Dios the lowest, 2.6%. "
            "**Adventists are drawn at 8.3% of the Bay Islands**, the one department the "
            "survey clearly sets apart for them, and at 0.6% everywhere else. Latter-day "
            "Saints and other religions are too few in the survey to place, so they are drawn "
            "at their national shares in every department."),
        how="household survey, one round in 2019",
        grain="departments, 550,000 people on average",
        gap="the 0.5% of people in households whose head's religion was not given",
        gap_share=0.0047,
        counts=_hn_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "hn" / "hn_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_hn_place_weight,
        note="§11ad CALLED HONDURAS'S LAPOP GEOGRAPHY BROKEN AND §11ap FOUND IT WAS NOT, and found "
             "the office source beside it. Drawn from INE's ENDESA-MICS 2019 household file "
             "with weights rebuilt from the report; LAPOP 2012-2023, decoded from municipio "
             "names, is the cross-check (Catholic r=+0.84). sources/hn.md.",
    ),
}
