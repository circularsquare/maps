# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _tw_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Taiwan's counties put their people on the western plain and a few east-coast towns, with the
    Central Mountain Range empty between them: Nantou, Hualien and Taitung are mostly mountain,
    and Kaohsiung and Taichung reach far into it (sources/tw_geo.py).
    """
    return _kontur_place_weight(place, "tw_hexes.gpkg", "sources/tw_geo.py")


def _tw_counts():
    """Taiwan Social Change Survey, seven rounds 1994-2018 pooled, at county and city: 11
    categories, 19 of 22 units drawn, EVERY ROW `modelled` (§7).

    sources/tw.py builds it and sources/tw.md is the record. Each county's pattern is its
    observed answers over what its respondents' rounds predict nationally, the level is the 2014
    and 2018 religion modules, and the two are fitted to the MOI household register at the end of
    2025. Penghu, Kinmen and Lienchiang were never sampled; they are not in tw.csv and are in
    `gap=`.
    """
    from tw2018 import resolve
    import tw2018

    df = pd.read_csv(HERE / "data" / "normalized" / "tw.csv", dtype={"geo_id": str},
                     low_memory=False, keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "tw" / "tw_lookup.csv", dtype=str)
    missing = sorted(set(df["geo_id"]) - set(lut["geo_id"]))
    if missing:
        raise SystemExit(f"tw.csv counties with no polygon: {missing} -- re-run sources/tw_geo.py")
    if df["geo_id"].nunique() != 19 or {"TW-PEN", "TW-KIN", "TW-LIE"} & set(df["geo_id"]):
        raise SystemExit(f"tw.csv has {df['geo_id'].nunique()} counties, expected the 19 sampled")
    df["unit"] = df["geo_id"]
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(set(df.loc[df["node"].isna(), "source_category"]) - set(tw2018.EXCLUDED))
    if unmapped:
        raise SystemExit(f"tw.csv categories with no node: {unmapped}")
    df = df[df["node"].notna() & (df["count"] > 0)].copy()
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "tw": dict(
        name="Taiwan",
        source="Taiwan Social Change Survey, seven rounds 1994 to 2018 (Institute of Sociology, "
               "Academia Sinica; open copies at the Association of Religion Data Archives), against "
               "the Ministry of the Interior's household register at the end of 2025",
        basis="self-identification, adults",
        view=[119.3, 21.8, 122.2, 25.4],
        note_public=(
            "**Taiwan's census does not ask about religion, so this map is drawn from a "
            "survey.** The Taiwan Social Change Survey asked **13,395 adults** about their "
            "religion in seven rounds between 1994 and 2018, and their answers are laid on the "
            "household register at the end of 2025. The dots are drawn desaturated to say so. "
            "**How many people count as Buddhist depends on how the question was put.** On the "
            "1994 card Buddhism was **38.5%** of answers; on the 2018 card, where the "
            "interviewer records *worships the gods* as folk religion and *worships the Buddha* "
            "as Buddhism, it was **13.5%**. The national figures here come from the 2014 and "
            "2018 rounds, which used the same card, and the older rounds only show where each "
            "answer is more or less common. "
            "**Folk religion is strongest in the rural southwest and centre.** It is about "
            "**74% of Chiayi County** and 68% of Yunlin, against 38% of Taipei. Taoism is a "
            "fifth or more of Tainan, Kaohsiung, Pingtung and Yilan. Protestants are **9.5% of "
            "Taipei** and about a tenth of Hualien and Taitung, the two counties with the "
            "largest indigenous share of residents. "
            "**Catholics, Yiguan Dao and the smaller answers are spread at the national rate.** "
            "They did not differ reliably between counties when the survey's townships were "
            "split in half, so each county's remainder is shared among them in national "
            "proportions. The same goes for people who named Buddhism and Taoism together, "
            "whose strongest county rested on one township. "
            "**Penghu, Kinmen and Lienchiang (Matsu) are blank because no round of the survey "
            "went there.** They hold **1.11%** of registered residents."),
        how="survey, seven rounds 1994 to 2018 pooled",
        grain="counties and cities, 1.2 million people on average",
        gap=("Penghu, Kinmen and Lienchiang, 259,650 registered residents, 1.11% of the "
             "country, which the survey never sampled"),
        gap_share=0.0111,
        counts=_tw_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "tw" / "tw_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_tw_place_weight,
        note="A SURVEY ON THE HOUSEHOLD REGISTER AND EVERY ROW IS `modelled` (§7b). TSCS 1994, "
             "1999, 2004, 2009, 2014, 2015, 2018 from ARDA; residence from each round's postcode "
             "(Chunghwa Post's 368 codes) or county label. Pattern is a within-round "
             "observed/expected index (the cards differ: 1994 and 2015 short, the rest long), "
             "level the 2014 and 2018 religion modules, fitted to MOI ODRP048 114. Split-half on "
             "townships with a regrouping null: none, folk, Buddhism, Taoism, Protestant carry; "
             "`Buddhism and Taoism` refused, Hsinchu County's reading is one township. "
             "sources/tw.py, sources/tw.md, sources.md §tw-2026-09-14.",
    ),
}
