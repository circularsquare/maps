# El Salvador. Censo 2024 (ONEC/BCR), "Habla otro idioma aparte del espanol? Cuales?", people of
# 3 and over, several answers allowed, per distrito, from the BCR's census GeoPortal
# (sources/sv_censo.py). Each person who named another language is shared between Spanish and
# the languages named (spec §3.6); everyone else, under-3s included, is drawn as Spanish, and so
# are the foreign-language shares (Anita, 2026-10-05, consistent with the neighbours). Placed
# on Kontur hexes re-keyed from religiondots' department layer to the BCR's own 262 distrito
# polygons (sources/sv_geo.py). Record: sources/sv.md.
from _shared import *  # noqa: F401,F403

SPANISH = "indoeuropean.romance.spanish"
FOREIGN = {"Inglés", "Francés", "Italiano", "Otro"}   # drawn as Spanish, see _counts()


def _counts():
    import sv2024
    df = pd.read_csv(NORM / "sv.csv", dtype={"geo_id": str})
    w = df.pivot_table(index="geo_id", columns="source_category", values="count", aggfunc="sum")
    if len(w) != 262:
        raise SystemExit(f"sv: {len(w)} distritos, expected 262")
    P, S = w.pop("Población").astype(float), w.pop("Sí").astype(float)
    w = w.drop(columns="Español")          # already counted: the question presupposes Spanish
    M = w.sum(axis=1)
    if not FOREIGN <= set(w.columns):
        raise SystemExit(f"sv: foreign labels missing from sv.csv: {FOREIGN - set(w.columns)}")
    # The people who said yes (S) each speak Spanish plus at least one other language. Within
    # that group, mentions are Spanish once per person plus the M other mentions, and each
    # person is shared across their languages by scaling the group's mentions to its size
    # (spec §3.6 applied to the group the census separates; a yes naming one other language
    # counts half to each, exactly). Everyone outside the group is Spanish.
    k = (S / (S + M)).where(S > 0, 0.0)
    rows = [pd.DataFrame({"unit": w.index, "node": SPANISH, "count": (P - S + S * k).to_numpy()})]
    for label in w.columns:
        # Foreign languages here are second languages; the neighbours (gt, ni, cr, mx:
        # spec §3.5) draw only indigenous languages as measured and everyone else as Spanish
        # (Anita, 2026-10-05). So a foreign share, and Otro's (mostly foreign: the census names
        # all three indigenous languages), goes to Spanish; the indigenous and LESSA shares stay
        # exactly as the split gives them. sources/sv.md keeps the counts.
        node = SPANISH if label in FOREIGN else sv2024.resolve(label)
        rows.append(pd.DataFrame({"unit": w.index, "node": node,
                                  "count": (w[label] * k).to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    per_unit = out.groupby("unit")["count"].sum().reindex(P.index)
    if ((per_unit - P).abs() > 1e-6).any():
        raise SystemExit("sv: the shares do not add back to the population in every distrito")
    out = out[out["count"] > 0]
    out["tier"] = "derived"                  # every row, spec §3.6
    return out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="El Salvador",
    source="VII Censo de Población y VI de Vivienda 2024 (Oficina Nacional de Estadística y "
           "Censos, Banco Central de Reserva), languages spoken besides Spanish and population "
           "per distrito, from the BCR's census GeoPortal",
    how="census, 2024, languages spoken besides Spanish, several allowed; indigenous and sign "
        "language speakers shared between Spanish and those languages, everyone else drawn as "
        "Spanish",
    parts=[
        dict(covers="Indigenous and sign languages",
             source="2024 census, languages spoken besides Spanish, each speaker shared half "
                    "with Spanish",
             nodes=["signlanguage.lessa", "utoaztecan.pipil", "isolate.lenca_salvador",
                    "misumalpan.cacaopera"]),
        dict(covers="Everyone else",
             source="2024 census population, drawn as Spanish (English and other foreign "
                    "second languages included)",
             rest=True),
    ],
    grain="262 distritos, 22,600 people on average",
    gap="107,055 people (1.8%) of the census's headline count, whom the per-distrito "
        "population table does not include",
    view=[-90.2, 13.1, -87.6, 14.5],
    counts=_counts,
    mappings=["sv2024"],
    place=GEO / "sv" / "sv_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2024 census asked everyone aged three and over whether they speak a language "
        "besides Spanish, and which. It did not ask which language anyone learned first, so "
        "everyone is drawn as a Spanish speaker, and each person who named an indigenous or "
        "sign language is shared between Spanish and the languages they named: someone who "
        "speaks Spanish and Náhuat counts half to each. 414,887 people named English, but for "
        "nearly all of them it is a second language, so they are drawn as Spanish, as are "
        "speakers of French, Italian and other foreign languages. 1,135 people named Náhuat, "
        "the Nahua language of western El Salvador, which drawn shared comes to fewer than one "
        "dot; most live in Sonsonate and San Salvador. Children under three were not asked "
        "and are drawn as Spanish. The 262 districts are the municipalities before the 2024 "
        "reform."),
)
