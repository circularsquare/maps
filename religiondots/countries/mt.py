# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _mt_place_weight(place):
    """countries.py hook. `place` is sources/mt_geo.py's layer: Kontur hexes cut to the 68
    localities, each piece carrying its share of the hex's people by area.

    The localities are small (median 2.94 km2, four hexes), which is below spec §8.2e's floor for a
    centroid join, so the hexes are cut rather than joined, and the grid was kept only after it
    agreed with the census per locality (p10 0.68, p90 1.38; sources/mt.md §5). It is a
    POPULATION weight inside a locality, never a religion one.
    """
    return _kontur_place_weight(place, "mt_hexes.gpkg", "sources/mt_geo.py")


def _mt_counts():
    """NSO Census 2021, Volume 1, Table 5.3: ten answers by 68 localities, every row `measured`.

    Counted at the locality, which is the unit drawn, so nothing is spread and nothing rolls up.
    """
    from mt2021 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "mt.csv", dtype={"geo_id": str},
                     keep_default_na=False, na_values=[""], low_memory=False)
    if df["geo_id"].nunique() != 68:
        raise SystemExit(f"{df['geo_id'].nunique()} localities, expected 68; re-run "
                         "`python sources/mt.py`")
    df["node"] = df["source_category"].map(resolve)
    if df["node"].isna().any():
        raise SystemExit(f"unmapped: {sorted(df.loc[df['node'].isna(), 'source_category'].unique())}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["tier"] = "measured"
    return (df.groupby(["unit", "node"], as_index=False)
              .agg(count=("count", "sum"), congregations=("congregations", "max"),
                   tier=("tier", "first")))


ENTRY = {
    # ---- MALTA (sources/mt.py, sources/mt_geo.py, sources/mt.md) ---------------------------
    "mt": dict(
        name="Malta",
        source="Census of Population and Housing 2021, Final Report Volume 1, Table 5.3 (NSO Malta)",
        basis="self-identification, persons aged 15 and over",
        how="a census question, 2021, asked of people aged 15 and over",
        grain="localities, 7,600 people on average",
        gap="children under 15, 13.1% of residents, who were not asked the religion question",
        gap_share=0.1305,
        note_public=(
            "**The 2021 census was the first in Malta to ask about religion.** It asked the "
            "451,746 residents aged 15 and over, and printed the answers in counts for all 68 "
            "localities, which is the grain drawn here. Roman Catholics are **82.6%** of them, "
            "people with no religion 5.1%, Muslims 3.9% and Orthodox Christians 3.6%, followed "
            "by Hindus, the Church of England, other Protestants, Buddhists, Jews and other "
            "religions. The census prints no figure for people who did not answer, and each "
            "locality's total is its whole population aged 15 and over. The report says details "
            "for people who did not take part were estimated from government registers, and it "
            "does not say how religion was handled for them. "
            "**Most residents who are not Roman Catholic are not Maltese citizens.** Maltese "
            "citizens are **96.4%** Roman Catholic, and 84% of the residents who are not Roman "
            "Catholic are among the 104,169 without Maltese citizenship, who are 36.7% Roman "
            "Catholic, 15.3% of no religion, 15.1% Muslim and 14.5% Orthodox. "
            "**The localities differ.** San Pawl il-Baħar (St Paul's Bay), whose population "
            "almost doubled between 2011 and 2021, has 4,427 Orthodox Christians, **15.8%** of "
            "its residents aged 15 and over and more than a quarter of Malta's Orthodox. Muslims "
            "are **14.0%** of Il-Marsa and 10.7% of Birżebbuġa, Hindus 8.3% of L-Imsida, and "
            "people with no religion 12.4% of Tas-Sliema and 12.3% of San Ġiljan (St Julian's). "
            "Santa Luċija is 96.8% Roman Catholic."),
        counts=_mt_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "mt" / "mt_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_mt_place_weight,
        note="NSO CENSUS 2021, VOLUME 1, TABLE 5.3, IN COUNTS BY THE 68 LOCALITIES, persons aged "
             "15 and over; UNSD has no Maltese row. CHECKED (sources/mt.py): Tables 5.1 to 5.4 "
             "reconcile with each other and with a transcription of 5.2; every locality's "
             "religion total equals its population aged 15 and over in Table 1.5, so the "
             "under-15s (67,816, 13.05%) are the whole gap and there is no not-stated column. "
             "p.171 says non-participants were estimated from registers; how religion was "
             "handled is not stated. UNITS: GISCO LAU 2021 joined by name, witnessed by the "
             "code's district and Table 1.10's printed areas (0.961 to 1.030). PLACEMENT: Kontur "
             "hexes cut to the localities, kept after measuring against uniform placement "
             "(p10 0.68, p90 1.38; sources/mt.md §5).",
    ),
}
