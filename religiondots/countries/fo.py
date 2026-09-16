# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _fo_place_weight(place):
    """countries.py hook. `place` is sources/fo_geo.py's layer: Kontur hexes labelled with the
    census district and the nearest village, each carrying that village's November 2011 register
    population shared over its hexes in Kontur's proportions.

    Kontur alone put Tórshavn at about half its register population and spread the rest into
    the villages around it (sources/fo.md §5), so the weight is the office's village count and
    Kontur only shapes it inside a village. It is a POPULATION weight inside a district, never
    a religion one.
    """
    return _kontur_place_weight(place, "fo_hexes.gpkg", "sources/fo_geo.py")


def _fo_counts():
    """Hagstova Census 2011, MT325: six answers by 7 districts, every row `measured`.

    Counted at the district, which is the unit drawn. sources/fo.py has already scaled each
    district's body columns from ticks to people (the question allowed several ticks), inside
    the district's own cells, so nothing is spread between units and nothing rolls up.
    """
    from fo2011 import EXCLUDED, resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "fo.csv", dtype={"geo_id": str},
                     keep_default_na=False, na_values=[""], low_memory=False)
    if df["geo_id"].nunique() != 7:
        raise SystemExit(f"{df['geo_id'].nunique()} districts, expected 7; re-run "
                         "`python sources/fo.py`")
    df = df[~df["source_category"].isin(EXCLUDED)].copy()
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
    # ---- FAROE ISLANDS (sources/fo.py, sources/fo_geo.py, sources/fo.md) --------------------
    "fo": dict(
        name="Faroe Islands",
        name_in="the Faroe Islands",
        source="Census 2011, table MT325, congregation association by district (Hagstova Føroya)",
        basis="self-reported association with a Christian congregation, persons aged 15 and over",
        how="a voluntary census question, 2011, asked of people aged 15 and over",
        grain="districts, 6,900 people on average",
        gap=("28.8% of residents: children under 15 (21.5%), who were not asked, and people aged "
             "15 or over who left the voluntary question blank or did not return the census "
             "form (7.3%)"),
        gap_share=0.2877,
        note_public=(
            "**The 2011 census, the first in the Faroe Islands since 1977, asked people aged 15 "
            "and over which Christian church, congregation or community they were associated "
            "with.** The question was voluntary and 34,436 people answered it. The census "
            "printed the answers for seven districts, which is the grain drawn here, and inside "
            "each district the dots are placed by the villages' registered populations in "
            "November 2011. "
            "**People could tick more than one box.** 4,596 did, so the census counts 39,558 "
            "associations for 34,436 people. The map scales each district's associations down "
            "so that each person is one dot, which assumes the extra ticks are spread across the "
            "churches in proportion to their size. "
            "**The National Church is the largest answer.** It is **68.3%** of people after that "
            "scaling, and another 10.1% named a mission house, a congregation house, KFUM or KFUK "
            "or the Salvation Army, which the census groups as movements close to the National "
            "Church. Both are drawn as Lutheran here, although the Salvation Army is not "
            "Lutheran. The missionary movements are 16.6% of the answers in Eysturoy and 6.0% in "
            "southern Streymoy. "
            "**The Brethren are strongest in the northern islands.** They are 13.4% of people "
            "nationally and **30.7%** in Norðoyar, the district around Klaksvík, against 4.5% in "
            "northern Streymoy and 3.5% in Sandoy. Pentecostal congregations are 3.2%, and 7.5% "
            "in Sandoy. People who named no congregation are 3.6%, and 5.6% in southern "
            "Streymoy, the district of Tórshavn. "
            "**The smallest churches are hidden by district.** The census does not publish a cell "
            "under three people, so the Adventist, Catholic, Orthodox and Jehovah's Witness "
            "answers, 479 people nationally, are inside each district's other congregations and "
            "are drawn together as other Christians. The religion question asked beside this one "
            "counted 124 people naming Islam, Hinduism, Buddhism, Judaism, the Bahá'í faith or "
            "Sikhism, who are not separated here."),
        counts=_fo_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "fo" / "fo_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_fo_place_weight,
        note="HAGSTOVA CENSUS 2011, MT325, IN COUNTS BY THE 7 DISTRICTS, persons aged 15 and over "
             "who answered voluntary question E23 (select all that apply); UNSD has no Faroese "
             "row. CHECKED (sources/fo.py): MT325, MT321 and MT1 pinned and reconciled; MT1's "
             "15+ equals MT325's persons in every district. TICKS TO PEOPLE: each district's "
             "body columns scaled by answers-minus-None over ticks (0.80 to 0.91); the "
             "National-Church-plus-missionary reading of the overlaps fails in 6 of 7 districts. "
             "SUPPRESSED: Adventist, Catholic, Orthodox, Jehovah's Witness live in district Other "
             "(585 = those four + national Other). UNITS: districts are village lists that sum "
             "to Hagstova's register regions to the person; hexes assigned by GADM municipality, "
             "Sunda by OSM island, Tórshavn by nearest village; 115 villages witness the rule. "
             "PLACEMENT: Kontur calibrated to the November 2011 village register (sources/fo.md "
             "§5), because Kontur alone halves Tórshavn.",
    ),
}
