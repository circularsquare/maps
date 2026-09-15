# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _sn_place_weight(place):
    """countries.py hook. `place` is the Kontur 400m hex layer scatter.py has read.

    Twelve units over 197,000 km2, from Dakar (542 km2, 1.49 million people in 1988) to the
    Tambacounda région with Kédougou (59,000 km2). Kontur is 2023 and the counts 1988, so this
    places 1988's dots where people live now; Mbacké (Touba) has grown 5.2 times since
    (sources/sn_geo.py).
    """
    return _kontur_place_weight(place, "sn_hexes.gpkg", "sources/sn_geo.py")


def _sn_counts():
    """Senegal RGPH 1988: 7 nodes on 12 units (9 régions, Diourbel's 3 départements), all `measured`.

    Tableau 1.15's one-decimal shares on Tableau 1.2's région populations, and the Diourbel
    report's Tableau 1.12 counts raked to its two printed margins (sources/sn.py).
    """
    from sn1988 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "sn.csv",
                     low_memory=False, keep_default_na=False, na_values=[""])
    df["count"] = df["count"].astype(int)
    if df["geo_id"].nunique() != 12:
        raise SystemExit(f"{df['geo_id'].nunique()} units in sn.csv, expected 12 -- re-run "
                         "sources/sn.py")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]))
    if unmapped:
        raise SystemExit(f"sn.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0].rename(columns={"geo_id": "unit"})
    df["congregations"] = 0
    df["may_ring"] = True
    df["tier"] = "measured"
    return df[["unit", "node", "count", "congregations", "may_ring", "tier"]]


ENTRY = {
    "sn": dict(
        name="Senegal",
        source="RGPH 1988, résultats définitifs (Direction de la Prévision et de la Statistique), "
               "Tableau 1.15, and the Diourbel regional report's Tableau 1.12",
        basis="self-identification, residents of ordinary households",
        note_public=(
            "**Senegal's 1988 census asked Muslims which Sufi brotherhood they belonged to.** It "
            "is the latest Senegalese census found with religion published below the national "
            "level; the 2002, 2013 and 2023 censuses asked the same question, but no regional "
            "table from them has been found. The map is drawn at the ten regions of 1988, with "
            "Diourbel divided into its three departments. The four regions created since then "
            "are drawn inside the regions they were taken from, so Saint-Louis includes Matam, "
            "Kaolack includes Kaffrine, Tambacounda includes Kédougou and Kolda includes Sédhiou. "
            "**The Tijaniyya was the largest brotherhood.** It counted **47.3%** of residents, "
            "and 80.2% in Saint-Louis. The Mourides were 29.7%, and **91.5%** of Mbacké "
            "department, where Touba is; they were also 45.9% of Louga and 44.7% of Thiès. The "
            "Qadiriyya was 11.7%, and strongest in the south and east, at 32.0% of Ziguinchor "
            "and 26.0% of Kolda. Three quarters of the Layène, 0.6% of the country, lived in the "
            "Dakar region. Another 5.1% were Muslims who named none of the four. "
            "**Christians and other religions were most common in Ziguinchor.** It was **17.1%** "
            "Christian and 7.7% other religions, which the census report describes as mainly "
            "animism. The census printed Catholics and other Christians together, except in "
            "Diourbel. "
            "**The dots are placed using today's population.** The counts are from 1988, when "
            "Senegal had 6.9 million people, but inside each unit they are spread by a 2023 "
            "population grid, so a fast-growing town such as Touba holds more of its "
            "department's dots than its 1988 population would. Some borders between regions "
            "have also moved since 1988."),
        how="census, 1988",
        grain="regions of 1988, and departments in Diourbel; 575,000 people on average",
        # Tableau 1.1's 6,896,808 are residents of ordinary households; p8 adds "35000
        # personnes vivant dans la population comptée à part" (barracks, prisons), in no table.
        # 35,000 / 6,931,808 = 0.50%. Hand-written: tools/gap_share.py cannot see them.
        gap="0.5%: about 35,000 people counted apart, in barracks, prisons and the like, who "
            "are in no religion table",
        gap_share=0.005,
        counts=_sn_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "sn" / "sn_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_sn_place_weight,
        note="TABLEAU 1.15 OF THE 1988 NATIONAL REPORT is région x religion with the Sufi "
             "brotherhoods as the Muslim answers, one decimal, % of all residents (the five order "
             "columns sum to Musulmans). Each row's seven leaves normalised and applied to "
             "Tableau 1.2's populations. THE NATIONAL ROW PRINTS DIOURBEL'S KHADRIYA UNDER "
             "LAYÈNE; the Diourbel regional report's Tableau 1.12 counts settle it, and Diourbel "
             "is drawn from that table at its 3 départements, raked to Tableau 1.2 and its région "
             "column. The Ensemble row does not come back from the région rows (Khadriya 11.7 "
             "against 10.9) and is not used. 1988 régions rebuilt from COD-AB v02 by pcode; five "
             "units' areas differ from 1988's by 5-22% (sources/sn_geo.py). sources/sn.md has "
             "the record.",
    ),
}
