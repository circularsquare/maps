# Taiwan. 2020 Population and Housing Census, language learned earliest in childhood, for the 368
# townships and districts (sources/tw_census.py), on Kontur hexes keyed to MOI's township
# polygons (sources/tw_geo.py). The queue calls Taiwan `cn-tw` (Natural Earth's ISO_A2); the
# loader takes two-letter files only, and religiondots draws it as `tw`.
from _shared import *  # noqa: F401,F403


def _counts():
    import tw2020
    df = pd.read_csv(NORM / "tw.csv")
    df = df[(df["geo_level"] == "town") & (df["question"] == "earliest")].copy()
    df["node"] = df["source_category"].map(tw2020.resolve)
    unmapped = set(df.loc[df["node"].isna(), "source_category"]) - tw2020.EXCLUDED
    if unmapped:
        raise SystemExit(f"tw: categories with no node: {sorted(unmapped)}")
    df = df[df["node"].notna() & (df["count"] > 0)]
    df["tier"] = "measured"
    df = _share_indigenous(df, tw2020)
    lut = pd.read_csv(GEO / "tw" / "tw_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["key"], lut["unit"])))
    if df["unit"].isna().any():
        raise SystemExit(f"tw: {df.loc[df['unit'].isna(), 'geo_id'].nunique()} townships with no "
                         "polygon; re-run sources/tw_geo.py")
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


def _share_indigenous(df, tw2020):
    """Ask 012 (Anita, 2026-10-05): share each township's census 原住民族語 across the indigenous
    peoples registered there at the end of October 2020 (sources/tw_cip.py), in proportion, rows
    tier derived. The census count per township is unchanged; 尚未申報 (people not declared)
    stays on the group node. A township with no registered indigenous resident keeps its figure
    on the group, as measured."""
    reg = pd.read_csv(NORM / "tw_peoples.csv")
    unknown = set(reg["people"]) - set(tw2020.PEOPLES)
    if unknown:
        raise SystemExit(f"tw: registered peoples with no node: {sorted(unknown)}")
    reg["pnode"] = reg["people"].map(tw2020.PEOPLES)
    reg = reg[reg["count"] > 0]
    reg["w"] = reg["count"] / reg.groupby("geo_id")["count"].transform("sum")
    ind = df[df["node"] == tw2020.TW]
    split = ind.merge(reg[["geo_id", "pnode", "w"]], on="geo_id", how="inner")
    split["count"] = split["count"] * split["w"]
    split["node"] = split["pnode"]
    split["tier"] = "derived"
    keep = ind[~ind["geo_id"].isin(split["geo_id"])]
    out = pd.concat([df[df["node"] != tw2020.TW], keep, split[df.columns]], ignore_index=True)
    assert abs(out["count"].sum() - df["count"].sum()) < 1e-6 * df["count"].sum()
    return out


ENTRY = dict(
    name="Taiwan",
    source="2020 Population and Housing Census, county reports, table 7 (Directorate-General of "
           "Budget, Accounting and Statistics)",
    how="census, 2020, language learned first in childhood; indigenous languages split by the "
        "household register's indigenous peoples in each township",
    parts=[
        dict(covers="Everyone of Taiwanese nationality aged 6 and over",
             source="2020 census, language learned first in childhood", people=21_555_603),
        dict(covers="Indigenous languages",
             source="2020 census's one indigenous answer, split by the indigenous peoples in "
                    "each township's household register (October 2020)",
             rest=True),
    ],
    grain="368 townships and districts, 59,000 people on average",
    gap="children under 6 and residents without Taiwanese nationality, whom the question does "
        "not cover, and 60,672 people (0.3%) who answered that they do not know or have none",
    view=[119.3, 21.8, 122.2, 25.4],
    counts=_counts,
    mappings=["tw2020"],
    place=GEO / "tw" / "tw_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "The census asked everyone of Taiwanese nationality aged 6 and over which language they "
        "learned first as a child, and separately which they use most now. This map shows the "
        "first: Taiwanese (Hokkien) 53%, Mandarin 40%, Hakka 4.6%. Asked what they use most "
        "now, 66% answered Mandarin and 32% Taiwanese. The census counts the indigenous languages "
        "as one answer (169,000 people). In each township they are shared across the indigenous "
        "peoples in the household register there, each read as speaking its own language, so "
        "which indigenous language a dot shows is an estimate. \"Other\" includes Taiwan Sign "
        "Language and foreign languages. The language questions were asked in a sample of "
        "about 16% of census areas, so the shares in each township are estimates."),
)
