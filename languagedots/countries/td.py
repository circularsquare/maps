# Chad. No census table gives language below the whole country (RGPH2 2009 printed only national
# figures). MICS6 2019 (INSEED and UNICEF), mother tongue of the household head, read as every
# member's; heads who answered "other" are sent to their ethnic group's languages as the 2009
# census record gives them; weighted shares per province applied to the 2009 census région
# populations (sources/td_mics.py). Placed on religiondots' Kontur 400 m hexes for the 22 régions
# of 2009, read-only. Record: sources/td.md (§0 for MICS; the national census build it replaced
# below that, sources/td_rgph.py).
from _shared import *  # noqa: F401,F403

POP_2009 = 10_941_682      # RGPH2 2009, Tableau 5.07, 22 régions (religiondots' td.csv)
HC1B_NAMED = 6_892_198     # people under a head who named a language on MICS's list
ARAB_OTHER = 336_322       # Arab heads who answered "other", drawn as Chadian Arabic


def _counts():
    import td2019
    df = pd.read_csv(NORM / "td.csv")
    if df["geo_id"].nunique() != 22:
        raise SystemExit(f"td.csv: {df['geo_id'].nunique()} régions, expected 22")
    if int(df["count"].sum()) != POP_2009:
        raise SystemExit(f"td.csv sums to {df['count'].sum():,}, expected {POP_2009:,}")
    if set(df["source_id"]) != {"mics6_2019_hc1b"}:
        raise SystemExit(f"td.csv sources {sorted(set(df['source_id']))}: rerun sources/td_mics.py")
    named = int(df[df["source_category"].str.isupper()]["count"].sum())
    arab = int(df[df["source_category"] == "Arabe local"]["count"].sum())
    if (named, arab) != (HC1B_NAMED, ARAB_OTHER):
        raise SystemExit(f"td.csv parts moved ({named:,}, {arab:,}): update HC1B_NAMED, ARAB_OTHER")
    df["node"] = df["source_category"].map(td2019.resolve)
    df["unit"] = df["geo_id"]
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Chad",
    source=("Chad Multiple Indicator Cluster Survey 2019 (MICS6; INSEED, UNICEF), microdata; "
            "RGPH2 2009, État et structures de la population (INSEED), Tableaux 5.02, 5.07 and "
            "5.10 and Annexes 2-3"),
    how=("a household survey, 2019, mother tongue of the household head, read as every member's; "
         "heads who answered \"other\" split among their ethnic group's languages by the 2009 "
         "census; weighted shares per province applied to each région's 2009 population"),
    parts=[
        dict(covers="Households whose head named a language on the survey's list",
             source="UNICEF MICS 2019, about 19,000 households, mother tongue of the household "
                    "head",
             people=HC1B_NAMED),
        dict(covers="Arab heads who answered \"other\"",
             source="drawn as Chadian Arabic: interviewers in the same village coded the same "
                    "answer differently",
             people=ARAB_OTHER),
        dict(covers="Other heads who answered \"other\"",
             source="shared among the languages of the head's ethnic group, in the proportions of "
                    "the 2009 census",
             rest=True),
    ],
    grain="22 régions, 500,000 people on average",
    gap=("households that were not interviewed, left out before the shares; Tibesti, which the "
         "survey did not reach, is drawn on Borkou's shares"),
    view=[13.4, 7.4, 24.0, 23.5],
    counts=_counts,
    mappings=["td2019"],
    place=RD_GEO / "td" / "td_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Chad's 2009 census published language for the whole country only, so these are shares "
        "from UNICEF's 2019 household survey of about 19,000 households, applied to each "
        "province's 2009 population. Each household is drawn on the mother tongue of its head. "
        "The survey named 13 languages; for the 37% of people whose head answered \"other\", "
        "the head's ethnic group decides the language, shared among that group's languages in "
        "the proportions of the 2009 census, and where the census did not name them they are "
        "drawn as unnamed. Arab heads who answered \"other\" are drawn as Chadian Arabic, since "
        "interviewers in the same village coded the same answer both ways. Ngambay and Sar are "
        "drawn apart from the other Sara languages. Tibesti was not surveyed and is drawn on "
        "Borkou's shares. Inside a province dots follow population."),
)
