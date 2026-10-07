# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _bi_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    17 provinces over 27,000 km2. Kontur 2023 hexes, each scaled so its commune sums to the 2008
    census count (sources/bi_geo.py), so the dots follow 2008's spread between communes.
    """
    return _kontur_place_weight(place, "bi_hexes.gpkg", "sources/bi_geo.py")


def _bi_counts():
    """The 2008 census's national religion rows, by urban and rural, on an Afrobarometer pattern:
    8 nodes, 17 provinces, and EVERY ROW IS `modelled` IN §7.

    THREE EXACT CENSUS TABLES AND A SURVEY BETWEEN THEM, Togo's construction with the census's
    urban and rural split added. Tableau 1.5 gives each province's urban and rural residents,
    Tableau 1.13 the religion rows by urban and rural, Tableau 1.4 the collective households; the
    survey supplies only the seed. Catholic, Protestant and Muslim carry their own pattern; the
    rest follow the census's urban and rural shares. sources/bi.py has the construction.
    """
    from bi2008 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "bi.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "bi" / "bi_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"bi.csv units with no polygon: {missing} -- re-run "
                         "sources/bi_geo.py, the lookup is stale")
    if df["unit"].nunique() != 17:
        raise SystemExit(f"{df['unit'].nunique()} units, expected 17")

    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)].copy()
    # EVERY row, without exception -- nobody counted any cell (§7b).
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "bi": dict(
        name="Burundi",
        source="2008 census (ISTEEBU), the national religion table by urban and rural, with the "
               "provincial pattern from two rounds of the Afrobarometer, 2012 and 2014",
        basis="self-identification, whole census population",
        note_public=(
            "**Burundi's 2008 census asked everyone's religion and published the answer only for "
            "the whole country, split into towns and countryside.** Of the people whose religion "
            "was recorded, it counted **63.1%** Catholic, **22.0%** Protestant, 6.3% with no "
            "religion, 3.3% in another religion, **2.6%** Muslim and 2.4% Adventist. Muslims were "
            "14.7% of the towns and 1.3% of the countryside. "
            "**Where those people are comes from a survey.** Two rounds of the Afrobarometer are "
            "pooled, **2,395** adults interviewed in late 2012 and late 2014. The census fixes each "
            "province's town and country population and each religion's total in town and country, "
            "and the survey decides only how a province's people divide between religions. Nobody "
            "counted these dots, so they disappear when inferred dots are turned off. "
            "**Three religions are placed by the survey, because their order across the provinces "
            "repeats from one round to the other.** Protestants are **48.6%** of Bururi and 46.8% "
            "of Makamba in the south, and 6.7% of Gitega, which is **81.9%** Catholic. Islam is "
            "**14.8%** of Bujumbura Mairie and under 5% everywhere else; the survey found no Muslims "
            "in Muramvya, Mwaro or Rutana, so none are drawn there. "
            "**The other religions are not placed.** Adventists, Jehovah's Witnesses, people with "
            "no religion and the census's other religions are spread in every province at the "
            "census's own town and country shares, so they vary only with how urban a province is. "
            "The provinces are the 17 of 2008: Rumonge, made a province in 2015, is drawn inside "
            "Bururi and Bujumbura Rural, and the five provinces of the 2025 reform are not used. "
            "The dots are the people counted in August 2008, not today's population; "
            "the 2024 census has published no religion table yet."),
        how="census totals, 2008, given a provincial pattern by a pooled survey",
        grain="provinces as of 2008; 474,000 people on average",
        gap="2.8%: 1.7% whose religion the census gives as not declared, and 1.1% in collective "
            "households, whom the religion table leaves out",
        gap_share=0.02775,                      # 223,452/8,053,574, exact; tools/gap_share.py "rows only"
        counts=_bi_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bi" / "bi_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_bi_place_weight,
        note="ISTEEBU PUBLISHED RELIGION FROM THE 2008 CENSUS ONLY FOR THE NATION, URBAN AND RURAL "
             "(Tableau 1.13; sources.md §11ah, sources/bi.md). Built on Togo's construction with a "
             "third dimension: an IPF of province x urban/rural x religion to Tableaux 1.5, 1.13 "
             "and 1.4, seeded by Afrobarometer R5-R6, every row modelled. 17 UNITS: the 2008 "
             "provinces, rebuilt from geoBoundaries' 119 communes (Rumonge's five returned to "
             "Bururi and Bujumbura Rural; sources/bi_geo.py). CARRIED on the split-half: Catholic, "
             "Protestant, Muslim; the rest seeded flat and given the census's urban/rural shares.",
    ),
}
