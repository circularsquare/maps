# Latvia. Tautas skaitīšana 2011, language mostly spoken at home, by municipality
# (sources/lv_census.py), on Kontur hexes keyed to religiondots' 119 LAUs (sources/lv_geo.py).
# The record is sources/lv.md.
from _shared import *  # noqa: F401,F403


def _counts():
    import lv2011
    df = pd.read_csv(NORM / "lv.csv", dtype={"geo_id": str})
    df = df[df["geo_level"] == "municipality"].copy()
    if df["geo_id"].nunique() != 119:
        raise SystemExit(f"lv.csv: {df['geo_id'].nunique()} municipalities, expected 119")
    # census code "LV" + ATVK; the hexes carry the bare seven-digit ATVK (GISCO's LAU_ID)
    df["unit"] = df["geo_id"].str.removeprefix("LV")
    df["node"] = df["source_category"].map(lv2011.resolve)
    df = df[df["node"].notna() & (df["count"] > 0)].copy()
    df["tier"] = "measured"
    df = _latgalian(df)
    return df.groupby(["unit", "node", "tier"], as_index=False)["count"].sum()


# Latgale statistical region 2011 (LV005): 21 of its municipalities by ATVK; asserted against the
# region's own TSG11-08 row in _latgalian. (Varakļāni county is Vidzeme, not Latgale.)
_LATGALE = ["0050000", "0210000", "0604300", "0384400", "0380200", "0684901", "0601000",
            "0440200", "0440801", "0681000", "0600202", "0761201", "0680200", "0760202",
            "0780200", "0766300", "0387500", "0769101", "0381600", "0781800", "0681801"]
_LATGALIAN_HOME_SHARE = 0.088   # CSB Adult Education Survey 2022: language used at home, Latgale


def _latgalian(df):
    """Move Latgalian out of Latvian in Latgale (audit 2026-10-13, sources/lv.md).

    The 2011 home-language question had no Latgalian answer, so its speakers said Latvian. The
    one-answer home-language question of the 2022 Adult Education Survey (18-69) gives Latgalian
    1.2% nationally and 8.8% in Latgale. 8.8% of Latgale's 274,917 people with a home language is
    moved from Latvian onto Latgalian, shared over the municipalities by the number of people who
    say Latvian at home and use Latgalian daily (TSG11-08). Latgale's population is unchanged."""
    import json
    raw = json.load(open(NORM.parent / "raw" / "lv" / "TSG11-08.json", encoding="utf8"))
    cell = {tuple(r["key"]): r["values"][0] for r in raw["data"]}
    reg_total = int(cell[("LV005", "TOTAL", "TOTAL", "NUMB", "TOTAL")])
    reg_daily = int(cell[("LV005", "TOTAL", "Y", "NUMB", "LAV")])
    daily = {m: int(cell[("LV" + m, "TOTAL", "Y", "NUMB", "LAV")]) for m in _LATGALE}
    tot = {m: int(cell[("LV" + m, "TOTAL", "TOTAL", "NUMB", "TOTAL")]) for m in _LATGALE}
    assert sum(tot.values()) == reg_total == 274_917 and sum(daily.values()) == reg_daily
    target = round(_LATGALIAN_HOME_SHARE * reg_total)
    raw_alloc = {m: target * daily[m] / reg_daily for m in _LATGALE}
    alloc = {m: int(v) for m, v in raw_alloc.items()}
    for m in sorted(_LATGALE, key=lambda m: raw_alloc[m] - alloc[m], reverse=True)[
            :target - sum(alloc.values())]:
        alloc[m] += 1
    lav = "indoeuropean.baltic.latvian"
    rows = []
    for m, k in alloc.items():
        mask = (df["unit"] == m) & (df["node"] == lav)
        have = int(df.loc[mask, "count"].sum())
        assert 0 < k <= have, (m, k, have)
        df.loc[mask, "count"] = have - k
        rows.append(dict(unit=m, node="indoeuropean.baltic.latgalian", count=k, tier="derived"))
    return pd.concat([df[["unit", "node", "count", "tier"]], pd.DataFrame(rows)])


ENTRY = dict(
    name="Latvia",
    source="Tautas skaitīšana 2011, resident population by language mostly spoken at home, by "
           "municipality (Central Statistical Bureau, table TSG11-07)",
    how="census, 2011, language mostly spoken at home; Latgalian in Latgale taken out of Latvian "
        "at the 2022 survey's home-language share",
    parts=[dict(covers="Latgalian speakers in Latgale",
                source="8.8% of Latgale's people, the Adult Education Survey 2022 share naming "
                       "Latgalian as the language used at home, moved from Latvian and shared "
                       "over the region's municipalities by daily Latgalian use among "
                       "Latvian-at-home people (2011 census, table TSG11-08)",
                people=24_193),
           dict(covers="Everyone else", source="2011 census, language mostly spoken at home",
                rest=True)],
    grain="119 municipalities, 17,400 people on average",
    gap="193,559 people (9.3%) whose home language the census did not record.",
    view=[20.8, 55.6, 28.3, 58.1],
    counts=_counts,
    mappings=["lv2011"],
    place=GEO / "lv" / "lv_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "These figures are from 2011, the last census to ask about language. The 2021 census "
        "was compiled from registers, which do not record language. Latvia has lost about a "
        "tenth of its people since 2011, and the Ukrainians who arrived after 2022 are not "
        "counted here. "
        "Latgalian, which the law treats as a variety of Latvian, was not an answer to the home "
        "language question, so its speakers answered Latvian. In a 2022 survey 8.8% of people "
        "in Latgale named it as the language they use at home, and 35.5% there said in 2011 that "
        "they use it every day. That 8.8%, 24,000 people, is drawn as Latgalian and taken from "
        "Latvian, spread over the region's municipalities by who used it daily in 2011."),
)
