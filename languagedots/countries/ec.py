# Ecuador. Censo 2022, languages spoken (several allowed, age 1+), per canton, from INEC's
# tabulado (sources/ec_censo.py), each person shared across the classes they named using INEC's
# own combination table, so the split is exact; the foreign-language shares are drawn as Spanish
# (Anita, 2026-10-05, consistent with the neighbours). Placed on Kontur hexes re-keyed from
# religiondots' province layer to COD-AB's 221 cantons (sources/ec_geo.py). Record: sources/ec.md.
from _shared import *  # noqa: F401,F403


SPANISH = "indoeuropean.romance.spanish"
FOREIGN = "Idioma extranjero"


def _counts():
    import ec2022
    df = pd.read_csv(NORM / "ec.csv")
    df["node"] = df["source_category"].map(ec2022.resolve)
    gap = df[df["node"].isna()].groupby(["province", "canton"])["count"].sum()
    df = df[df["node"].notna() & (df["count"] > 0)].copy()  # no-habla and under-1s: in `gap`
    # Foreign languages are second languages here, and the neighbours (co, pe, and the
    # indigenous-only rule of spec §3.5) draw only indigenous languages as measured and everyone
    # else as Spanish (Anita, 2026-10-05). So each person's foreign-language share goes to
    # Spanish; indigenous and sign shares stay exactly as counted. The 25,949 who named only a
    # foreign language go to Spanish too, as a neighbour's remainder would. sources/ec.md.
    df.loc[df["source_category"] == FOREIGN, "node"] = SPANISH
    lut = pd.read_csv(GEO / "ec" / "ec_lookup.csv", dtype=str)
    key = dict(zip(zip(lut["province"], lut["canton"]), lut["unit"]))
    df["unit"] = [key.get(k) for k in zip(df["province"], df["canton"])]
    if df["unit"].isna().any():
        raise SystemExit("ec: cantons missing from data/geo/ec/ec_lookup.csv; "
                         "re-run sources/ec_geo.py")
    # per canton, the drawn shares are everyone the census asked who speaks (5.1's 1+ total
    # less no-habla): the whole population less the gap rows
    pop = pd.read_csv(NORM / "ec.csv").groupby(["province", "canton"])["count"].sum()
    drawn = df.groupby(["province", "canton"])["count"].sum()
    bad = ((drawn - (pop - gap.reindex(pop.index, fill_value=0))).abs() > 1e-6).sum()
    if bad or len(drawn) != 221:
        raise SystemExit(f"ec: drawn shares do not add up in {bad} cantons ({len(drawn)} drawn)")
    # every row is a person shared across the languages they named (spec §3.6)
    df = df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    return _immigrants(df)


# Immigrants (2026-10-06, session 5d7dac7e-br): the census's foreign-born per canton (table 1.1)
# split by their province's country-of-birth mix (table 7; sources/ec_immig.py), on their
# origin's languages (origin_mix), retained at France's TeO2 rate (sources/latam_immig.py), taken
# out of the canton's Spanish. The census asked everyone about indigenous languages, so
# indigenous-American languages are taken out of the origin mixes (DROP_ROOTS), as in mx.
def _immigrants(df):
    sys.path.insert(0, str(ROOT / "sources"))
    import latam_immig
    if latam_immig.active():          # an origin's home mix for another country's build
        return df
    imm = pd.read_csv(NORM / "ec_immig.csv", keep_default_na=False)     # "NA" is Namibia
    # Spanish-speaking origins stay on Spanish whole, as the northern pass's rule (spain's
    # Catalan, Galician, Basque shares are not drawn for the Spain-born, mostly children of
    # Ecuadorians who came back)
    imm = imm[(imm["count"] > 0) & ~imm["iso"].isin(latam_immig.HISPANIC)]
    rows = latam_immig.immigrant_languages(imm, SPANISH, "ec", drop_roots=latam_immig.DROP_ROOTS)
    out, _ = latam_immig.fold_into(df, rows, SPANISH)
    return out


ENTRY = dict(
    name="Ecuador",
    source="VIII Censo de Población y VII de Vivienda 2022 (INEC), tables 5.1, 7.1 and 10.1 of "
           "the tabulado on self-identification and culture: languages spoken per canton, by "
           "combination and by indigenous language; the migration tabulado's tables 1.1 and "
           "7 for people born abroad by canton and by province and country of birth; France's "
           "TeO2 survey for how many immigrants keep their language",
    how="census, 2022, languages spoken, several allowed, each person shared across the "
        "languages they named; foreign languages drawn as Spanish; people born abroad drawn "
        "on their birth country's languages",
    parts=[
        dict(covers="Indigenous languages and Ecuadorian Sign Language",
             source="2022 census, languages spoken, aged 1 and over, each person shared across "
                    "the languages they named",
             people=395_796),
        dict(covers="People born abroad, languages other than Spanish",
             source="2022 census, country of birth, drawn on that country's languages; about a "
                    "quarter moved to Spanish by France's TeO2 survey",
             people=23_114),
        dict(covers="Everyone else", source="2022 census, Spanish and foreign-language "
             "speakers, drawn as Spanish", rest=True),
    ],
    grain="221 cantons, 76,600 people on average",
    gap="327,910 people (1.9%): 241,750 children under 1, not asked, and 86,160 who do not "
        "speak",
    view=[-81.3, -5.2, -75.0, 1.7],
    counts=_counts,
    mappings=["ec2022"],
    place=GEO / "ec" / "ec_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2022 census asked everyone aged one and over which languages they speak or "
        "communicate in, as many as apply, so it counts what people can speak rather than a "
        "first language. Each person is drawn shared between the languages they named: someone "
        "who speaks Kichwa and Spanish counts half to each. That is why Kichwa, spoken by "
        "538,449 people, is drawn as about 307,000. About 472,000 people named a foreign "
        "language without saying which, nearly all of them also Spanish speakers, so that share "
        "is drawn as Spanish. Instead the 425,000 people born abroad are drawn by the languages "
        "of their birth country; most were born in Venezuela or Colombia. Inside a canton every "
        "language is spread by population, so an Amazonian language may be drawn in the canton "
        "town."),
)
