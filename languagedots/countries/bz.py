# Belize. 2022 Population and Housing Census (SIB), Table 7: languages spoken well enough to
# hold a conversation, people aged 4 and over, several answers allowed, per district
# (sources/bz_census.py). Drawn under spec §3.6: inside each district the mentions are scaled to
# the people aged 4+ who named a language. On religiondots' Kontur hexes for the same six
# districts (COD-AB pcodes), read-only. Record: sources/bz.md.
from _shared import *  # noqa: F401,F403

POP4 = "Population aged 4 and over"     # all ages less under-4s from the age tables
CANNOT = "Cannot Speak"


def _counts():
    import bz2022
    df = pd.read_csv(NORM / "bz.csv", dtype={"geo_id": str})
    if df["geo_id"].nunique() != 6:
        raise SystemExit(f"bz: {df['geo_id'].nunique()} districts, expected 6")
    w = df.pivot_table(index="geo_id", columns="source_category", values="count", aggfunc="sum")
    labels = [l for l in bz2022.NAMES if bz2022.resolve(l) is not None]
    missing = set(bz2022.NAMES) - set(w.columns)
    if missing:
        raise SystemExit(f"bz: labels missing from bz.csv: {missing}")
    # The people who named at least one language: the 4+ population less Cannot Speak. SIB
    # does not publish DK/NS, so it is inside this figure and shared like everyone else.
    P = w[POP4] - w[CANNOT]
    M = w[labels].sum(axis=1)
    rows = []
    for label in labels:
        rows.append(pd.DataFrame({"unit": w.index, "node": bz2022.resolve(label),
                                  "count": (w[label] * P / M).to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    per_unit = out.groupby("unit")["count"].sum().reindex(P.index)
    if ((per_unit - P).abs() > 1e-6).any():
        raise SystemExit("bz: the shares do not add back to the 4+ population in every district")
    out = out[out["count"] > 0]
    out["tier"] = "derived"                  # every row, spec §3.6
    return out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="Belize",
    source="2022 Population and Housing Census (Statistical Institute of Belize), Table 7, "
           "languages spoken by people aged 4 and over, per district",
    how="census, 2022, languages spoken, several allowed; each person shared across the "
        "languages they named",
    parts=[dict(covers="Everyone aged 4 and over",
                source="2022 census, languages spoken, each person shared across their answers",
                rest=True)],
    grain="6 districts, 66,000 people on average",
    gap="28,559 children under 4 (7.2%), whom the census does not ask, and 716 people "
        "recorded as unable to speak",
    view=[-89.30, 15.80, -87.35, 18.55],
    counts=_counts,
    mappings=["bz2022"],
    place=RD_GEO / "bz" / "bz_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2022 census asked everyone aged four and over which languages they speak well "
        "enough to hold a conversation, several allowed. It did not ask which language anyone "
        "learned first, and most Belizeans named two or three: 368,924 people gave about "
        "721,000 answers. Each person is shared across the languages they named, so someone who "
        "speaks English, Kriol and Spanish counts a third to each. English is the official "
        "language and the language of school, and 75.5% named it; for many it is not the "
        "language of home, so English is larger here than on a map of first languages, and "
        "Kriol and Spanish smaller. German is mostly the Mennonite settlements, whose everyday "
        "language is Plautdietsch, a Low German. Languages are published for the six districts "
        "only, and inside each district the dots follow population, so Q'eqchi' in Toledo is "
        "drawn in Punta Gorda town as well as in the Maya villages."),
)
