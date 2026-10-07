# Brazil. Censo 2022, indigenous languages per município (sources/br_censo.py); everyone else
# drawn as Portuguese (spec §3.5). Placed on religiondots' 2022 census setores, read-only.
from _shared import *  # noqa: F401,F403
import numpy as np

PORTUGUESE = "indoeuropean.romance.portuguese"
SETORES = RD_GEO / "br" / "br_setores_2022.gpkg"   # setor, unit (município code), pop
NOT_LANG = {"Não determinada", "Mal definida", "Não sabe"}


def _status():
    s = pd.read_csv(NORM / "br_status.csv", dtype={"geo_id": str}).set_index("geo_id")
    s["speakers"] = s["total"] - s["none"]                 # incl. the 2,495 unnamed
    s["named"] = s["one"] + s["two"] + s["three"]
    s["mentions"] = s["one"] + 2 * s["two"] + 3 * s["three"]
    return s


def _counts():
    import br2022
    import pyogrio
    s = _status()
    df = pd.read_csv(NORM / "br.csv", dtype={"geo_id": str, "code": str})
    df = df[~df["source_category"].isin(NOT_LANG)].copy()
    df["node"] = df["source_category"].map(br2022.resolve)
    # Up to three languages a person: Tabela 26 counts each person once per language named.
    # Scale each município's mentions to its people (spec §3.6). 98% of speakers named one
    # language, but one bilingual person anywhere puts a município in the scaled set (491 of
    # them, 83% of speakers), and a derived row may not ring, which hid 79 small languages.
    # So a row is `derived` only where the scaling moved its município by more than 5%.
    f = (s["named"] / s["mentions"]).where(s["mentions"] > 0, 1.0)
    df["count"] = df["count"] * df["geo_id"].map(f)
    df["tier"] = df["geo_id"].map(f < 0.95).map({True: "derived", False: "measured"})
    df = df[df["node"].notna() & (df["count"] > 0)]

    # Everyone else, Portuguese: the município's whole 2022 population (the setores' own
    # v0001, as religiondots reads it) less its indigenous-language speakers aged 2+,
    # including the 2,495 whose language the census could not name.
    pop = pyogrio.read_dataframe(SETORES, read_geometry=False, columns=["unit", "pop"])
    pop = pop.groupby("unit")["pop"].sum()
    rest = pop - s["speakers"].reindex(pop.index).fillna(0)
    if (rest < 0).any():
        raise SystemExit(f"br: speakers exceed population in {list(rest[rest < 0].index[:5])}")
    pt = pd.DataFrame({"geo_id": rest.index, "node": PORTUGUESE, "count": rest.values,
                       "tier": "derived"})
    out = pd.concat([df[["geo_id", "node", "count", "tier"]], pt], ignore_index=True)
    out = out.rename(columns={"geo_id": "unit"})
    out = out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()
    return _immigrants(_settlers(out))


# Settled communities (sources/br_settlers.py): Hunsrik, Talian and Pomerano from published
# regional estimates placed by homeland, `modelled`, taken out of the Portuguese remainder.
SETTLER_NODES = {"indoeuropean.germanic.continental.hunsrik",
                 "indoeuropean.germanic.continental.lowgerman.pomeranian",
                 "indoeuropean.romance.talian"}


def _settlers(df):
    s = pd.read_csv(NORM / "br_settlers.csv", dtype={"unit": str})
    take = s.groupby("unit")["count"].sum()
    pt = (df["node"] == PORTUGUESE)
    df = df.copy()
    df.loc[pt, "count"] = df.loc[pt, "count"] - df.loc[pt, "unit"].map(take).fillna(0)
    if (df.loc[pt, "count"] < -0.5).any():
        raise SystemExit("br: settler languages exceed a município's Portuguese")
    s["tier"] = "modelled"
    out = pd.concat([df, s[["unit", "node", "count", "tier"]]], ignore_index=True)
    return out[out["count"] > 0]


# Immigrants: the census's naturalised and foreign residents per município, split by the
# nationality mix of the UF's active Polícia Federal registrations (sources/br_immig.py), on
# their origin's languages (origin_mix), retained at France's TeO2 rate (sources/latam_immig.py),
# taken out of the Portuguese remainder. Quechua and Aymara are already measured among
# Bolivians and Peruvians who declared themselves indigenous: only the estimate above that.
DEDUPE = {"quechuan.quechua", "aymaran.aymara"}


def _immigrants(df):
    sys.path.insert(0, str(ROOT / "sources"))
    import latam_immig
    if latam_immig.active():          # an origin's home mix for another country's build
        return df
    imm = pd.read_csv(NORM / "br_immig.csv", dtype={"unit": str}, keep_default_na=False)
    imm = imm[imm["count"] > 0]
    rows = latam_immig.immigrant_languages(imm, PORTUGUESE, "br")
    out, _ = latam_immig.fold_into(df, rows, PORTUGUESE, dedupe=DEDUPE, from_tier="derived")
    return out


class _BrWeighter:
    """Inside a município, indigenous-language dots go where its indigenous people live, and
    Portuguese dots where everyone else does. A placement weight only: every município's
    totals are the census's either way.

    Indigenous people per setor are V01690 of the setor aggregates. IBGE suppresses small
    setor counts ("X", 80,369 setores), so a município's setores can hold fewer indigenous
    people than SIDRA 10392 says it has (88,884 people nationally); that shortfall is spread
    over the município's setores by population. Portuguese weight per setor is its population
    less its expected speakers: pop - indig_est * speakers / indigenous in the município.
    """

    def __init__(self, place):
        self.pop = place["pop"].to_numpy(dtype=float)
        ind = pd.read_csv(ROOT / "data" / "processed" / "br_setor_indigenous.csv",
                          dtype={"setor": str}).set_index("setor")["indig"]
        indig = place["setor"].astype(str).map(ind).fillna(0).to_numpy(dtype=float)
        s = _status()
        unit = place["unit"].astype(str)
        u_pop = pd.Series(self.pop).groupby(unit.to_numpy()).transform("sum").to_numpy()
        u_ind = pd.Series(indig).groupby(unit.to_numpy()).transform("sum").to_numpy()
        want = unit.map(s["total"]).fillna(0).to_numpy(dtype=float)       # indigenous, 2+
        short = (want - u_ind).clip(min=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            est = indig + np.where(u_pop > 0, short * self.pop / u_pop, 0)
            u_est = pd.Series(est).groupby(unit.to_numpy()).transform("sum").to_numpy()
            share = np.where(u_est > 0, unit.map(s["speakers"]).fillna(0).to_numpy() / u_est, 0)
        self.indig = est
        self.rest = (self.pop - est * np.minimum(share, 1.0)).clip(min=0)
        self.n = {"indigenous": 0, "rest": 0, "fallback": 0}

    def weights(self, node, idx, count, plain=False):
        import br2022
        # indigenous languages on indigenous people; Portuguese, settler and immigrant
        # languages on everyone else
        rest = node not in set(br2022.NAMES.values())
        w = (self.rest if rest else self.indig)[idx]
        if w.sum() > 0:
            self.n["rest" if rest else "indigenous"] += 1
            return w
        p = self.pop[idx]
        self.n["fallback"] += 1
        return p if p.sum() > 0 else None

    def summary(self):
        return (f"{self.n['indigenous']:,} indigenous-language rows placed on setor indigenous "
                f"people, {self.n['rest']:,} Portuguese, settler and immigrant rows on the rest, "
                f"{self.n['fallback']:,} on plain population")


def _weight(place):
    return _BrWeighter(place)


ENTRY = dict(
    name="Brazil",
    source="Censo Demográfico 2022, Tabela complementar 26 and SIDRA tables 10392 and 10157 "
           "(IBGE); SISMIGRA active registrations (Polícia Federal, via OBMigra); France's "
           "TeO2 survey for how many immigrants keep their language; Altenhofen, Morello et "
           "al., Hunsrückisch: inventário de uma língua do Brasil (2018), for Hunsrik and "
           "Talian; IPOL (2014) for Pomerano; Censo 2010 Lutherans per município for placing "
           "German varieties",
    how="census, 2022, indigenous languages spoken at home, several allowed; immigrants by "
        "their state's mix of nationalities; Hunsrik, Talian and Pomerano from published "
        "speaker estimates; everyone else drawn as Portuguese",
    parts=[
        dict(covers="Indigenous languages",
             source="2022 census, indigenous languages spoken at home, each person shared "
                    "across the languages they named", people=472_360),
        dict(covers="People born abroad, languages other than Portuguese",
             source="2022 census count per município, by the state's federal police "
                    "registrations by nationality; some moved to Portuguese by France's TeO2 "
                    "survey", people=690_173),
        dict(covers="Hunsrik, Talian and Pomerano",
             source="published speaker estimates (Altenhofen, Morello et al. 2018; IPOL 2014), "
                    "placed by where Lutherans lived in 2010 and in the Italian colonies",
             nodes=["indoeuropean.germanic.continental.hunsrik",
                    "indoeuropean.romance.talian",
                    "indoeuropean.germanic.continental.lowgerman.pomeranian"]),
        dict(covers="Everyone else", source="drawn as Portuguese", rest=True),
    ],
    grain="5,570 municípios, 36,000 people on average; indigenous languages in 1,990 of them; "
          "immigrant mixes by state",
    gap="2,495 indigenous people who speak an indigenous language the census could not name",
    # mainland only: Martim Vaz and Trindade belong to Vitória (religiondots' reasoning)
    view=[-74.2, -34.0, -34.2, 5.5],
    counts=_counts,
    mappings=["br2022"],
    place=SETORES,
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=_weight,
    note_public=(
        "The 2022 census asked about language only of indigenous people aged 2 or over. "
        "Everyone else is drawn as a Portuguese speaker, except in three groups. People "
        "born abroad (about a million) are drawn by the languages of their nationality: the "
        "census counts them per município, and the federal police register gives each "
        "state's mix of nationalities. Since no Brazilian survey asks which language "
        "immigrants speak at home, a share of them is drawn as Portuguese, the share of "
        "immigrants in France who speak only French with their children. Hunsrik, a German "
        "language, is drawn at about 1.2 million speakers in Rio Grande do Sul, Santa Catarina "
        "and southwestern Paraná, Talian, the Venetian of the Italian colonies, at 587,000 in "
        "Rio Grande do Sul, and Pomerano at 120,000 in Espírito Santo. These are rough "
        "estimates, partly from a 1990 survey, and their placement inside the region is "
        "modelled. Pomerano and Hunsrik elsewhere, Japanese among Brazilian-born Japanese "
        "descendants and other heritage languages have no usable figure and are drawn as "
        "Portuguese. Inside a município, indigenous dots are placed where the census counted "
        "indigenous people."),
)
