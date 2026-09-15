# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _bn_place_weight(place):
    """countries.py hook. `place` is the Kontur 400m hex layer scatter.py has read.

    Four districts over 5,800 km2, and inside each the people are in a few places: Belait's are
    in three coastal mukims out of eight, Temburong's along one river (sources/bn_geo.py).
    """
    return _kontur_place_weight(place, "bn_hexes.gpkg", "sources/bn_geo.py")


def _bn_counts():
    """Brunei BPP 2021 at district: 4 nodes on 4 units, every row `measured`.

    Table A4 in persons, the whole enumerated population of 440,715, temporary residents
    included (sources/bn.py).
    """
    from bn2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "bn.csv",
                     low_memory=False, keep_default_na=False, na_values=[""])
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 4:
        raise SystemExit(f"{df['geo_id'].nunique()} districts in bn.csv, expected 4 -- re-run "
                         "sources/bn.py")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"bn.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["may_ring"] = True
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


ENTRY = {
    "bn": dict(
        name="Brunei",
        source="Population and Housing Census 2021 (Department of Economic Planning and "
               "Statistics), Annex A, Table A4",
        basis="self-identification, total population",
        note_public=(
            "**Brunei's 2021 census publishes religion for its four districts.** Islam is "
            "**82.1%** of the population and every district has a Muslim majority, from 84.5% "
            "in Brunei-Muara to 70.3% in Belait, where 11.0% are Buddhist and 10.7% Christian. "
            "Temburong is **12.8%** Christian. "
            "**The census counted temporary residents too.** They are 18% of the population and "
            "most of its Christians, 18,653 of 29,462. "
            "**Hindus are drawn inside Others.** The form offered Hindu as an answer, but every "
            "published table adds Hindus to Others, together with everyone who wrote in "
            "another answer; the form had no box for no religion. Others is **11.0%** of "
            "Tutong, where 4,718 of the 5,192 are Brunei citizens, and 3.2% of Brunei-Muara, "
            "where 7,615 of the 10,074 are temporary residents."),
        how="census, 2021",
        grain="districts, 110,000 people on average",
        counts=_bn_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bn" / "bn_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_bn_place_weight,
        note="TABLE A4 OF THE BPP 2021 REPORT'S ANNEX A is district x religion in persons, from "
             "the tables workbook `EXCEL TABLE A-C.xls` (Wayback 20231023184321) and checked "
             "against the PDF page (Wayback 20240624173053; the 2023-09-30 capture is a 1 MiB "
             "fragment), A1, A10, A11, A12, C1 and UNSD table 28 to the person. The form's E10 "
             "has five codes (Islam, Christianity, Buddhism, Hindu, Others please specify) and "
             "no no-religion or not-stated code; the tables fold Hindu into Others, drawn on "
             "other.bn. No non-response row, so no gap. Districts are geoBoundaries gbOpen BRN "
             "ADM1 (2011), witnessed by its 38 ADM2 mukims against census Table C1. "
             "sources/bn.md has the record.",
    ),
}
