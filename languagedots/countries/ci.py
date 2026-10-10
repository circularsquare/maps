# Côte d'Ivoire. RGPH 2021, the Ivorian language each Ivorian speaks most, by région
# (sources/ci_rgph.py: Tableau 4.20's shares x annex 25's Ivorians per région, rescaled to the
# national counts of annex 20), on religiondots' Kontur hexes for the same 33 units. Residents of
# other nationalities from sources/ci_foreign.py (Tableaux 4.24 and 4.21 + origin_mix), derived.
from _shared import *  # noqa: F401,F403


OTHERS = "Ensemble des autres langues nationales parlées"


def _counts():
    import ci2021
    df = pd.read_csv(NORM / "ci.csv")
    df = df[(df["geo_level"] == "region") & (df["count"] > 0)].copy()
    df = df.rename(columns={"geo_id": "unit"})
    if df["unit"].nunique() != 33:
        raise SystemExit(f"ci: {df['unit'].nunique()} régions, expected 33")
    # the remainder is replaced by its share-out into 82 labels (ask 011, sources/ci_model.py,
    # every row `modelled`); the script asserts it meets each région's remainder exactly
    rem = df[df["source_category"] == OTHERS].set_index("unit")["count"]
    df = df[df["source_category"] != OTHERS].assign(tier="measured")
    mod = pd.read_csv(NORM / "ci_model.csv")
    if set(mod["tier"]) != {"modelled"}:
        raise SystemExit(f"ci_model.csv: tiers {sorted(set(mod['tier']))}")
    off = (mod.groupby("unit")["count"].sum().reindex(rem.index) - rem).abs()
    if off.isna().any() or (off > 0.01 * rem + 50).any():
        raise SystemExit(f"ci_model.csv does not match the régions' remainders:\n{off}")
    df = pd.concat([df, mod], ignore_index=True)
    df["node"] = df["source_category"].map(ci2021.resolve)
    # residents of other nationalities (sources/ci_foreign.py): Tableau 4.24's count per région
    # on the national nationality mix (Tableau 4.21), each nationality on its home languages
    # (origin_mix); rows carry node ids already, every one `derived`
    fo = pd.read_csv(NORM / "ci_foreign.csv")
    if set(fo["tier"]) != {"derived"} or abs(fo["count"].sum() - 6_460_062) > 5:
        raise SystemExit("ci_foreign.csv: run sources/ci_foreign.py")
    fo["node"] = fo["source_category"]
    df = pd.concat([df, fo], ignore_index=True)
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Côte d'Ivoire",
    source=("General Census of Population and Housing (RGPH) 2021, thematic report vol. 1, "
            "Tableaux 4.20, 4.21 and 4.24 and annexes 20 and 25 (Agence Nationale de la "
            "Statistique)"),
    how=("census, 2021, Ivorian language spoken most, Ivorian citizens; the 18% on languages "
         "the census names nationally only placed into regions by a model; residents of other "
         "nationalities counted per region by the census, drawn on their nationality's "
         "languages, nationality mix national"),
    parts=[
        dict(covers="Residents of other nationalities",
             source="2021 census count per region, the national mix of nationalities, each "
                    "drawn on its home country's languages",
             people=6_460_061),
        dict(covers="Languages named nationally only",
             source="2021 census national totals, placed into regions by where each language "
                    "is spoken (Glottolog) and where its ethnic group lives",
             people=3_850_763),
        dict(covers="Everyone else", source="2021 census, Ivorian language spoken most, by "
             "region", rest=True),
    ],
    grain="33 regions and autonomous districts, 645,000 Ivorians on average",
    gap="Ivorian children under three, about 1.5 million, who were not asked",
    view=[-8.7, 4.3, -2.4, 10.8],
    counts=_counts,
    mappings=["ci2021"],
    place=RD_GEO / "ci" / "ci_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked which of the country's own languages each person speaks most. "
        "French, the official language, was not an answer, and the 3.4% of Ivorians who speak "
        "no Ivorian language (7.5% in Abidjan) are drawn as unnamed; French is the likeliest "
        "for most of them. Because the question is about use, Dioula, the trade language of "
        "the north and of the towns, is drawn wherever people gave it rather than their own "
        "language. The regional table names only 13 languages; the rest together are 18% of "
        "Ivorians. For them the census gives each region's total and each language's national "
        "total, but not the two together, so their regional figures here are an estimate "
        "that meets both totals. Tried on 12 languages the census does give by region, the "
        "method put about a quarter of each language's speakers in the wrong region. "
        "The census published answers for Ivorian citizens only. The 6.5 million residents of "
        "other nationalities (22% of the population, about 44% in Gboklè and Cavally) are counted "
        "in each region by the census, but their languages are an estimate: each region is "
        "given the national mix of nationalities (63% Burkinabè, 17% Malian, 5% Guinean), "
        "and each nationality the languages of its home country, so about half are drawn as "
        "Mòoré, Bambara or Fula speakers. Two thirds of them were born in Côte d'Ivoire, and many "
        "will speak Dioula more than their parents' language; the census does not say how "
        "many."),
)
