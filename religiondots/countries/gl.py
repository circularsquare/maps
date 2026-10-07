# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _gl_place_weight(place):
    """countries.py hook. `place` is one disc per town or settlement (sources/gl_geo.py), with
    `pop` its Greenland-born residents on 1 January 2026, so a municipality's dots go to its
    localities in proportion to who lives there. Kontur holds about 20,000 of Greenland's
    56,740 people and is not used."""
    return _kontur_place_weight(place, "gl_hexes.gpkg", "sources/gl_geo.py")


def _gl_counts():
    """SLiCA 2003-2006's national Christian share on the Greenland-born, by municipality.

    EVERY ROW IS `modelled` (§7b). No census in Greenland asks religion; the Survey of Living
    Conditions in the Arctic asked "do you consider yourself to be a Christian?" of 1,197 people
    born in Greenland, aged 15 and over, and 98% said yes. That one mix is applied in every
    municipality, as Cuba's, Eritrea's and Comoros's are (Anita's rulings, 2026-10-03). The
    Christians are split by the 2026 church roll (spec 3.1 allows a roll to split, never to
    add): taxonomy/gl2006.py. The universe is the Greenland-born, as the survey's was; the
    7,019 born outside Greenland are `gap`. sources/gl.py and sources/gl.md.
    """
    from gl2006 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "gl.csv",
                     dtype={"geo_id": str}, keep_default_na=False, na_values=[""])
    df["unit"] = df["geo_id"]
    want = {"GL-KU", "GL-SM", "GL-QE", "GL-QT", "GL-AV", "GL-UO"}
    if set(df["unit"]) != want:
        raise SystemExit(f"gl.csv units {sorted(set(df['unit']))}, expected {sorted(want)}")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"gl.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "gl": dict(
        name="Greenland",
        source="Survey of Living Conditions in the Arctic (SLiCA), Greenland, 2003 to 2006, "
               "Results Tables (Institute of Social and Economic Research, University of "
               "Alaska Anchorage, 2007), Table 162; Christians split by Statistics Greenland's "
               "church membership table BEXKIRK, 1 January 2026; Greenland-born residents by "
               "locality, Statistics Greenland BEXSTD, 1 January 2026",
        basis="self-identification, people born in Greenland aged 15 and over",
        note_public=(
            "**No census in Greenland asks about religion.** Statistics Greenland keeps the "
            "church roll, and this map draws what people say instead, as it does for Denmark. "
            "The one survey that asked is the Survey of Living Conditions in the Arctic: "
            "Statistics Greenland interviewed **1,197** people born in Greenland, aged 15 and "
            "over, between December 2003 and August 2006, and **98%** said they consider "
            "themselves Christian. That share is applied to the Greenland-born people of each "
            "municipality on 1 January 2026, **49,721** in all, and placed in the towns and "
            "settlements where they live. Nobody counted these dots, so they disappear when "
            "inferred dots are turned off. "
            "**The church roll decides which church.** The survey asked only whether people "
            "are Christian. On the roll, **96.4%** of the Greenland-born belong to the Church "
            "of Greenland, the Lutheran folk church, so that much of the 98% is drawn as "
            "Lutheran and the other 1.6% as Christian with no church named. The 2% who said "
            "they are not Christian are drawn as religion unknown, because the question cannot "
            "tell no religion from Inuit belief or another faith. "
            "**Every municipality is drawn at the same mix.** The survey's five regions ran "
            "from 97% to 99% Christian, closer together than its sample can separate, and they "
            "do not match today's municipalities. "
            "**The survey is twenty years old.** Church membership among the Greenland-born "
            "fell from 97.5% in 2012 to 96.4% in 2026, so the Christian share is probably a "
            "little lower now than drawn. "
            "**The 7,019 people born outside Greenland are not drawn.** The survey did not "
            "interview them; 3,904 were born in Denmark and 2,001 in Asia, and about half of "
            "them are on the church roll."),
        how="survey, 2003 to 2006, one national share; Christians split by the 2026 church roll",
        grain="municipalities, 9,900 people on average; dots placed by town and settlement",
        gap="the 7,019 people born outside Greenland, 12.4% of residents, whom the survey "
            "did not interview",
        gap_share=0.1237,
        counts=_gl_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "gl" / "gl_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_gl_place_weight,
        note="CLOSED 2026-09-15 ON THE CHURCH ROLL, REOPENED AND BUILT 2026-10-03 (scout "
             "fafd1067-gaps, sources/gl.md): SLiCA's Greenland tables carry a self-identified "
             "Christian share, so Greenland is drawn the way Denmark is, from what people say, "
             "with the roll only splitting the Christians. One national mix as for Cuba, "
             "Eritrea and Comoros. Universe the Greenland-born (SLiCA sampled only them). "
             "Placement by locality discs from BEXSTD and GeoNames, not Kontur. geoBoundaries "
             "ADM1 leaves out Disko Island; its two localities are placed off the polygon.",
    ),
}
