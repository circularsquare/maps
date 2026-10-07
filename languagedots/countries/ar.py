# Argentina. Censo 2022, people who speak or understand the language of their own indigenous
# pueblo, per departamento (sources/ar_censo.py); everyone else drawn as Spanish (spec §3.5).
# Placed on Kontur hexes keyed to IGN's departamento polygons (sources/ar_geo.py).
from _shared import *  # noqa: F401,F403

SPANISH = "indoeuropean.romance.spanish"
ANTARCTICA = "94028"     # 81 people, all at the bases (collective dwellings); no placement


def _counts():
    import ar2022
    df = pd.read_csv(NORM / "ar.csv", dtype={"geo_id": str})
    df = df[df["geo_id"] != ANTARCTICA].copy()
    df["node"] = [ar2022.resolve(c, g) for c, g in zip(df["code"], df["geo_id"])]
    df = df[df["node"].notna()].copy()                # code 2, P24 ignorado: in `gap`
    df["tier"] = df["node"].map(lambda n: "derived" if n == SPANISH else "measured")
    df = df.rename(columns={"geo_id": "unit"})
    df = df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    return _immigrants(df)


# 2026-10-05 (session edd42a8c-lats, sources/ar.md "Immigrant languages"): the foreign-born by
# country of birth (sources/ar_immig.py) on their origin's languages (origin_mix), retained at
# France's TeO2 rate (sources/latam_immig.py), taken out of the Spanish remainder. Paraguayan
# Guarani, Quechua and Aymara are already counted among self-identified indigenous speakers,
# many of them the same immigrants: only the estimate above the measured count is added.
DEDUPE = {"tupian.tupiguarani.guarani.paraguayan", "quechuan.quechua", "aymaran.aymara"}


def _immigrants(df):
    sys.path.insert(0, str(ROOT / "sources"))
    import latam_immig
    if latam_immig.active():          # an origin's home mix for another country's build
        return df
    imm = pd.read_csv(NORM / "ar_immig.csv", dtype={"unit": str}, keep_default_na=False)
    imm = imm[imm["unit"] != ANTARCTICA]
    rows = latam_immig.immigrant_languages(imm, SPANISH, "ar")
    out, rep = latam_immig.fold_into(df, rows, SPANISH, dedupe=DEDUPE)
    return _welsh(out)


# 2026-10-06 (session 5d7dac7e-wl, sources/ar.md "Welsh in Chubut"): no census or survey counts
# Welsh speakers, so the estimate route (ask 019). Chubut province's own figure of about 1,500
# speakers (Western Mail, 27 Dec 2004, via Wikipedia "Y Wladfa"), the low end of the published
# 1,500-5,000 and the one least swollen by course learners. Split over the three departamentos
# where the community lives by the Welsh Language Project's 2019 class count per area (British
# Council annual report 2019, p. 8): Gaiman incl. Dolavon 68, Trelew 23 (drawn in Rawson, the
# departamento that holds Trelew), the Andes 23 (Futaleufú: Esquel and Trevelin). Taken out of
# each departamento's Spanish, so totals are unchanged.
WELSH = "indoeuropean.celtic.welsh"
WELSH_SPEAKERS = 1_500
WELSH_SPLIT = {"26042": 68, "26077": 23, "26035": 23}   # Gaiman, Rawson, Futaleufú


def _welsh(df):
    tot = sum(WELSH_SPLIT.values())
    df = df.copy()
    add = []
    for unit, w in WELSH_SPLIT.items():
        n = WELSH_SPEAKERS * w / tot
        sel = (df["unit"] == unit) & (df["node"] == SPANISH)
        have = df.loc[sel, "count"].sum()
        assert have > n * 10, (unit, have, n)
        df.loc[sel, "count"] = df.loc[sel, "count"] - n * df.loc[sel, "count"] / have
        add.append((unit, WELSH, "modelled", n))
    out = pd.concat([df, pd.DataFrame(add, columns=["unit", "node", "tier", "count"])],
                    ignore_index=True)
    return out


ENTRY = dict(
    name="Argentina",
    source="Censo Nacional de Población, Hogares y Viviendas 2022 (INDEC), tabulated on "
           "INDEC's REDATAM server; departamento boundaries from the Instituto Geográfico "
           "Nacional",
    how="census, 2022, speaks or understands the language of their own indigenous people; "
        "people born abroad drawn by their birth country's languages, less the share France's "
        "TeO2 survey finds speaking only the host language at home; Welsh in Chubut from the "
        "province's estimate of about 1,500 speakers, split by Welsh classes per area; "
        "everyone else drawn as Spanish",
    parts=[
        dict(covers="Indigenous languages",
             source="2022 census, indigenous people who speak or understand their people's "
                    "language",
             people=382_697),
        dict(covers="People born abroad, languages other than Spanish",
             source="2022 census, country of birth, drawn on that country's languages; a "
                    "quarter to a third moved to Spanish by France's TeO2 survey",
             people=461_254),
        dict(covers="Welsh in Chubut",
             source="Chubut province's estimate of about 1,500 speakers (2004), split over "
                    "Gaiman, Rawson (Trelew) and Futaleufú (Esquel, Trevelin) by the Welsh "
                    "Language Project's 2019 classes per area",
             people=1_500),
        dict(covers="Everyone else", source="drawn as Spanish", rest=True),
    ],
    grain="527 departamentos (the capital's 15 comunas among them), 87,000 people on average",
    gap="155,542 indigenous people who did not say whether they speak their people's "
        "language, and 81 people at the Antarctic bases",
    view=[-73.6, -55.1, -53.6, -21.8],
    counts=_counts,
    mappings=["ar2022"],
    place=GEO / "ar" / "ar_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2022 census asked about language only of people who identify as indigenous, and "
        "asked whether they speak or understand the language of their own people, without "
        "naming it. So each speaker is drawn on the language of the people they named, and "
        "everyone else as a Spanish speaker. People born abroad (1.9 million) are drawn by the "
        "languages of their birth country instead, so Paraguayans mostly as Guarani speakers, "
        "Bolivians and Peruvians partly as Quechua and Aymara speakers, Brazilians as "
        "Portuguese and Italians as Italian. Of those whose birth country speaks another "
        "language, a quarter to a third are drawn as Spanish speakers, the share of "
        "immigrants from the same part of the world who speak only French with their "
        "children in France, since no Argentine survey measures it. Where more indigenous speakers of Guarani, Quechua or "
        "Aymara were counted in a departamento than this estimate gives, only the census "
        "count is drawn. Argentine-born children of immigrants are drawn as Spanish, and so "
        "are other long-settled communities, which no census or survey counts. Welsh, spoken "
        "in Chubut since the settlement of 1865, is drawn from the province's own estimate of "
        "about 1,500 speakers (others say up to 5,000, counting learners), around Gaiman, "
        "Trelew, Esquel and Trevelin. People living "
        "in collective dwellings or on the street were not asked and are drawn as Spanish too. "
        "It is a question about speaking or understanding, not about which language came "
        "first. Many peoples whose ancestral language is no longer in everyday use also "
        "answered yes: 14,947 Diaguita, 4,954 Omaguaca, 4,834 Tonokoté, 2,034 Huarpe and "
        "1,369 Comechingón among them. They are drawn as the census recorded them, but they "
        "probably know some of a heritage or revived language, or speak another indigenous "
        "language; in Santiago del Estero that is most likely Santiago del Estero Quichua. "
        "82,156 speakers whose people was not recorded are drawn as an unnamed indigenous "
        "language. In Salta and Jujuy the Guarani people are the Chiriguano, so their "
        "speakers there are drawn as Ava Guaraní; elsewhere Guarani is drawn as Paraguayan "
        "Guarani. Language is published per departamento, and inside each one the dots "
        "follow where people live, not where indigenous communities are."),
)
