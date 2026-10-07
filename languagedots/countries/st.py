# São Tomé and Príncipe. IV RGPH 2012 (INE), Quadro 10 "língua falada", people of 1 and over,
# several answers allowed, per district, from the seven district reports (sources/st_rgph.py).
# Each person is shared across the languages they named (spec §3.6); French and English, school
# languages here, are drawn as Portuguese (AGENT_BRIEF §2). Placed on religiondots' Kontur hexes
# for the same seven districts. Record: sources/st.md.
from _shared import *  # noqa: F401,F403

PORTUGUESE = "indoeuropean.romance.portuguese"
FOLDED = {"Francês", "Inglês"}          # drawn as Portuguese, see _counts()
DENOM = "População 1+"


def _counts():
    import st2012
    df = pd.read_csv(NORM / "st.csv", dtype={"geo_id": str})
    w = df.pivot_table(index="geo_id", columns="source_category", values="count", aggfunc="sum")
    if sorted(w.index) != ["ST11", "ST21", "ST22", "ST23", "ST24", "ST25", "ST26"]:
        raise SystemExit(f"st: districts {sorted(w.index)}")
    P = w.pop(DENOM).astype(float)
    if set(w.columns) != {k for k, v in st2012.NAMES.items() if v}:
        raise SystemExit(f"st: labels {sorted(w.columns)} do not match st2012.NAMES")
    # Every person aged 1+ named at least Portuguese or something else; each is shared across
    # the languages named by scaling the unit's mentions to its population (spec §3.6).
    k = P / w.sum(axis=1)
    rows = []
    for label in w.columns:
        node = PORTUGUESE if label in FOLDED else st2012.resolve(label)
        rows.append(pd.DataFrame({"unit": w.index, "node": node,
                                  "count": (w[label] * k).to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    per_unit = out.groupby("unit")["count"].sum().reindex(P.index)
    if ((per_unit - P).abs() > 1e-6).any():
        raise SystemExit("st: the shares do not add back to the 1+ population in every district")
    out = out[out["count"] > 0]
    out["tier"] = "derived"                  # every row, spec §3.6
    return out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="São Tomé and Príncipe",
    source="IV Recenseamento Geral da População e da Habitação 2012 (Instituto Nacional de "
           "Estatística), Quadro 10, languages spoken, from the seven district reports",
    how="census, 2012, languages spoken, several allowed; each person shared across the "
        "languages they named; French and English, learned at school, drawn as Portuguese",
    parts=[dict(covers="Everyone aged 1 and over",
                source="2012 census, languages spoken, each person shared across the languages "
                       "they named",
                rest=True)],
    grain="7 districts, 25,000 people on average",
    gap="5,724 children under one (3.2%), whom the language table does not include",
    view=[6.4, -0.1, 7.55, 1.75],
    counts=_counts,
    mappings=["st2012"],
    place=RD_GEO / "st" / "st_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2012 census asked everyone aged one and over which languages they speak, and "
        "allowed several. It did not ask which one anyone learned first. 98.4% named "
        "Portuguese, 36.2% Forro, the creole of São Tomé island, 8.5% Kabuverdianu, 6.6% "
        "Angolar and 1.0% Lung'ie, the creole of Príncipe. Each person is shared across the "
        "languages they named, so someone who speaks Portuguese and Forro counts half to "
        "each, which is why Portuguese comes out at about two thirds of the dots. French and "
        "English are taught at school and are drawn as Portuguese. Kabuverdianu came with "
        "Cape Verdean plantation workers and is named more often on Príncipe than Lung'ie; "
        "most of those who named Lung'ie live on São Tomé. Inside each district the dots "
        "follow where people lived in 2023."),
)
