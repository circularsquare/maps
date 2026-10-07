# New Caledonia. Recensement 2019 (INSEE-ISEE), Kanak languages spoken by people aged 15+, per
# commune (sources/nc_rp2019.py); everyone else aged 15+ drawn as French (spec 3.5). On
# religiondots' Kontur hexes for the 33 communes. Record: sources/nc.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import nc2019
    df = pd.read_csv(NORM / "nc.csv", dtype={"geo_id": str})
    w = df.pivot_table(index="geo_id", columns="source_category", values="count", aggfunc="sum")
    if len(w) != 33:
        raise SystemExit(f"nc: {len(w)} communes, expected 33")
    langs = [c for c in w.columns if c not in nc2019.EXCLUDED]
    if len(langs) != 29:
        raise SystemExit(f"nc: {len(langs)} languages, expected 29")
    M = w[langs].sum(axis=1)
    speak, p15 = w["Parle"], w["Population 15+"]
    if (M > speak).any():
        # never so in 2019 (sources/nc_rp2019.py check 5); spec 3.6 would then scale the
        # mentions down to the speakers
        raise SystemExit("nc: a commune has more mentions than speakers; scale them (spec 3.6)")
    rows = []
    # Each language's mentions are the census's count of its speakers in the commune, drawn as
    # published. A speaker of two languages counts in both, and the unnamed remainder below is
    # smaller by as much, so every commune's speakers add back exactly to P21's Parle.
    for label in langs:
        rows.append(pd.DataFrame({"unit": w.index, "node": nc2019.resolve(label),
                                  "count": w[label].to_numpy(), "tier": "measured"}))
    # Speakers who named no language (and the overlap of those who named two): spec 3.2.
    rows.append(pd.DataFrame({"unit": w.index, "node": nc2019.KANAK,
                              "count": (speak - M).to_numpy(), "tier": "derived"}))
    # Everyone else aged 15+: only understands a Kanak language, or neither (spec 3.5).
    rows.append(pd.DataFrame({"unit": w.index, "node": nc2019.FRENCH,
                              "count": (p15 - speak).to_numpy(), "tier": "derived"}))
    out = pd.concat(rows, ignore_index=True)
    per_unit = out.groupby("unit")["count"].sum().reindex(p15.index)
    if (per_unit != p15).any():
        raise SystemExit("nc: rows do not add back to the population aged 15+ in every commune")
    lut = pd.read_csv(RD_GEO / "nc" / "nc_lookup.csv", dtype=str)
    out["unit"] = out["unit"].map(dict(zip(lut["geo_id"], lut["unit"])))
    if out["unit"].isna().any():
        raise SystemExit("nc: communes missing from religiondots' nc_lookup.csv")
    out = out[out["count"] > 0]
    return out.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


ENTRY = dict(
    name="New Caledonia",
    source="Recensement de la population 2019 (INSEE-ISEE): speakers of each Kanak language "
           "aged 15 and over by commune (langues-vernaculaires-locuteurs.xls) and table P21, "
           "knowledge of a Kanak language by commune",
    how="census, 2019, Kanak languages spoken, aged 15 and over; everyone else drawn as French",
    parts=[
        dict(covers="Kanak-language speakers",
             source="2019 census, Kanak languages spoken, aged 15 and over",
             people=75_853),
        dict(covers="Everyone else aged 15 and over",
             source="2019 census, drawn as French", rest=True),
    ],
    grain="33 communes, 6,400 people aged 15 and over on average",
    gap="children under 15, 60,426 (22.3%), whom the census does not ask",
    view=[163.5, -22.8, 168.2, -19.5],
    counts=_counts,
    mappings=["nc2019"],
    place=RD_GEO / "nc" / "nc_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The 2019 census asked everyone aged 15 or over whether they speak a Kanak language, "
        "and which, and asked about no other language, so everyone else is drawn as a French "
        "speaker. That hides Wallisian and Futunian, the first language of many of the 22,520 "
        "people of Wallisian or Futunian community, and Tahitian, Vietnamese, Javanese and "
        "Bislama. A Kanak-language speaker is drawn on that language even if they use French "
        "more. Speakers of two languages are counted under each, and at least 10,668 speakers "
        "did not say which language; they are drawn as Kanak languages, language not named. "
        "The census counts Tayo, the French-based creole of Saint-Louis, among the Kanak "
        "languages; here it is drawn with the creoles. Inside each commune the dots follow "
        "where people live, so in Nouméa the Kanak-language dots are spread across the whole "
        "city rather than the neighbourhoods where Kanak families live."),
)
