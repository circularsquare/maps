# Madagascar. RGPH-3 2018 (INSTAT), "Compétences linguistiques et scolarisation", Tableau 2.2:
# per region, the share of people aged 3 and over able to speak Malagasy, French, English and
# other (foreign) languages, several allowed (sources/mg_rgph.py). Each person is shared across
# the languages they speak (spec §3.6), and the shares of French, English and the other languages,
# learned second languages here, are drawn as Malagasy (AGENT_BRIEF §2, Anita 2026-10-05), so
# every region is drawn wholly Malagasy. Placed on religiondots' Kontur hexes for the same 22
# regions. Record: sources/mg.md.
from _shared import *  # noqa: F401,F403

MALAGASY = "austronesian.malagasy"
LANGS = ("Malagasy", "Français", "Anglais", "Autres langues")
SECOND = {"Français", "Anglais", "Autres langues"}     # drawn as Malagasy, see _counts()


def split():
    """Each region's people aged 3+ shared across the languages they speak, before any folding:
    count = mentions * pop3 / sum(mentions). sources/mg.md tables what this gives."""
    df = pd.read_csv(NORM / "mg.csv")
    w = df.pivot_table(index="geo_id", columns="source_category", values="count", aggfunc="sum")
    if len(w) != 22:
        raise SystemExit(f"mg: {len(w)} regions, expected 22")
    m = w[list(LANGS)]
    return w["Population"], w["Population 3+"], m.mul(w["Population 3+"] / m.sum(axis=1), axis=0)


def _counts():
    import mg2018
    pop, pop3, sh = split()
    rows = []
    for label in LANGS:
        node = MALAGASY if label in SECOND else mg2018.resolve(label)
        rows.append(pd.DataFrame({"unit": sh.index, "node": node, "count": sh[label].to_numpy()}))
    # Children under 3 and people in collective households (Tableau 6's population less Tableau
    # 2.2's base, 8.4%) were not tabulated; drawn as Malagasy, the language 99.9% of every
    # region's people aged 3 and over speak.
    rows.append(pd.DataFrame({"unit": pop.index, "node": MALAGASY,
                              "count": (pop - pop3).to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    per = out.groupby("unit")["count"].sum().reindex(pop.index)
    if ((per - pop).abs() > 1e-6).any():
        raise SystemExit("mg: the shares do not add back to each region's population")
    out["tier"] = "derived"                      # every row, spec §3.6
    return out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Madagascar",
    source="Troisième Recensement Général de la Population et de l'Habitation 2018 (INSTAT), "
           "Compétences linguistiques et scolarisation, Tableau 2.2, and Tome 1, Tableau 6",
    how="census, 2018, languages spoken, several allowed; foreign languages spoken as second "
        "languages drawn as Malagasy",
    parts=[
        dict(covers="People aged 3 and over",
             source="2018 census, languages spoken, several allowed", people=23_507_970),
        dict(covers="Children under 3 and people in collective households",
             source="not asked, drawn as Malagasy", rest=True),
    ],
    grain="22 regions, 1.2 million people on average",
    view=[43.0, -25.8, 50.7, -11.8],
    counts=_counts,
    mappings=["mg2018"],
    place=RD_GEO / "mg" / "mg_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2018 census asked everyone aged three and over whether they can speak Malagasy, "
        "French, English or another language, and 99.9% of them in every region speak Malagasy. "
        "It did not ask which language anyone learned first, so everyone is drawn as a Malagasy "
        "speaker. French (23.6% of those asked), English (8.2%) and other foreign languages "
        "(0.6%) are second languages for nearly all who speak them, so they are not drawn. The "
        "0.1% who speak no Malagasy are drawn as Malagasy too, since the census does not say "
        "what they speak. Malagasy has a regional variety for each of the island's peoples, "
        "and the census does not ask about them."),
)
