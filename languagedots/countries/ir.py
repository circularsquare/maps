# Iran. No census asks language. World Values Survey waves 5 (2005) and 7 (2020), language at
# home, pooled per province, applied to the 1395 (2016) census province populations
# (sources/ir_wvs.py). Placed on religiondots' Kontur 400 m hexes, calibrated there to the census's
# 429 county totals. Record: sources/ir.md.
from _shared import *  # noqa: F401,F403

POP_2016 = 79_926_270


def _counts():
    import ir2020
    df = pd.read_csv(NORM / "ir.csv")
    if df["geo_id"].nunique() != 31:
        raise SystemExit(f"ir.csv: {df['geo_id'].nunique()} provinces, expected 31")
    if int(df["count"].sum()) != POP_2016:
        raise SystemExit(f"ir.csv sums to {df['count'].sum():,}, expected {POP_2016:,}")
    df["node"] = df["source_category"].map(ir2020.resolve)
    df = df[df["node"].notna()]
    df["unit"] = df["geo_id"]
    out = by_unit(df)
    out["tier"] = "modelled"
    return out


ENTRY = dict(
    name="Iran",
    source=("World Values Survey, Iran waves 5 (2005) and 7 (2020), read through the WVS online "
            "analysis tool; 1395 (2016) census province populations (Statistical Centre of Iran)"),
    how=("a survey, 2005 and 2020 pooled, language spoken at home by adults; shares per province "
         "applied to each province's whole 2016 population"),
    parts=[dict(covers="Everyone", source="World Values Survey 2005 and 2020 pooled, language "
                "spoken at home by adults", rest=True)],
    grain="31 provinces, 2.6 million people on average",
    gap="the survey's 25 people who gave no language, left out before the shares",
    view=[44.0, 25.0, 63.4, 39.8],
    counts=_counts,
    mappings=["ir2020"],
    place=RD_GEO / "ir" / "ir_hexes.gpkg",
    place_unit=lambda g: g["unit"].astype(str),
    place_weight=pop_weight,
    note_public=(
        "Iran's census asks no language question, so these are survey shares of about 4,200 "
        "adults, applied to each province's 2016 population, children included. Many provinces "
        "had 10 to 50 interviews, so a province's mix can be well off. Kurdish is one answer, "
        "Luri includes Bakhtiari, and Azerbaijani includes Iran's other Turkic languages such "
        "as Qashqai. Persian in Mazandaran is undercounted. Turkmen was not on the 2020 card, "
        "so half of Golestan is drawn as other."),
)
