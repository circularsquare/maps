# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _jp_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer, keyed to the 47 prefectures by
    sources/jp_grid.py. A population weight only: it says where in a prefecture people live
    and nothing about which of them name a religion."""
    return _kontur_place_weight(place, "jp_hexes.gpkg", "sources/jp_grid.py")


def _jp_counts():
    """JGSS 2021-2024 national totals spread over 47 prefectures; every row `modelled`.

    sources/jp_alloc.py does the arithmetic and its docstring says why. Three sources, each
    deciding one thing: how many (JGSS 2021H-2024N pooled, national), how religious each of
    JGSS's six sampling blocks is (JGSS-2015, relative only), and where inside a block each
    religion sits (NHK's 1996 prefecture survey, Anita's call in ask 014). Nothing is measured
    at prefecture level, so `inferred dots: hidden` empties the country.
    """
    from jp2024 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "jp_prefecture_allocated.csv",
                     dtype={"geo_id": str}, low_memory=False)
    df["unit"] = df["geo_id"]
    if df["unit"].nunique() != 47:
        raise SystemExit(f"jp: {df['unit'].nunique()} prefectures, expected 47; run "
                         "sources/jp_alloc.py")
    df["node"] = df["source_category"].map(resolve)
    df = df[df["node"].notna() & (df["count"] > 0)]
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["congregations"] = 0
    df["tier"] = "modelled"
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    # ---- JAPAN (sources/jp.py) ----------------------------------------------------------
    # Anita's baseline, 2026-09-14 (ask/answered/011): national self-identification first.
    # One unit on Kontur hexes like the microstate tier, but a survey, so every row is
    # `modelled` rather than the tier's `measured`.
    "jp": dict(
        name="Japan",
        source="Japanese General Social Surveys 2021 to 2024 and 2015 (JGSS Research Center, "
               "Osaka University of Commerce); NHK prefecture survey 1996",
        basis="self-identification, sample survey",
        view=[122.9, 24.0, 146.2, 45.6],
        how="survey, 10,612 people, 2021 to 2024; spread over prefectures by a 1996 survey",
        grain="47 prefectures, 2.6m people each",
        fill=("from NHK's 1996 survey of every prefecture, scaled to the 2021 to 2024 national "
              "figures"),
        # gap_share.py computes 3.60% and does not write it, as a "rows only" case. Written by
        # hand because the universe IS stated: sources/jp.py proves the pooled rows close on the
        # page's own 計 rows in every wave, so 382 / 10,612 is exact.
        gap_share=0.036,
        gap="the 3.6% of respondents who left the religion questions blank or said they did "
            "not know",
        note_public=(
            "**Most people in Japan say they have no religion.** Across four rounds of the "
            "Japanese General Social Survey from 2021 to 2024, **71.7%** of adults did. The "
            "survey asks whether you believe in a religion and offers a middle answer for "
            "people who do not believe personally but whose family has one; both are then "
            "asked to name it, and both are drawn under the religion they named. Buddhism is "
            "**18.8%**: the Pure Land schools 6.8%, Zen 1.8%, Shingon 1.5%, Nichiren 1.4%, "
            "Tendai 0.2%, and 7.0% who named no school. Soka Gakkai, drawn with the Japanese "
            "new religions, is 1.8%. "
            "**Where each religion is drawn comes from a survey taken in 1996.** No survey "
            "since has published religion by prefecture. NHK asked about 600 people in every "
            "prefecture in 1996 which religion or school they followed, and those answers "
            "decide which prefectures get more of each religion; how religious each of six "
            "regions is comes from the same JGSS in 2015, and the totals stay at their 2021 to "
            "2024 figures. In 1996 the Pure Land schools were named by **41%** of people in "
            "Toyama and Fukui and by almost nobody in Okinawa, and Shingon and Tendai were "
            "strongest in Tokushima, Okayama and Kagawa. Anything that has moved between "
            "prefectures since then is not on this map, and a small religion's prefecture "
            "detail rests on a handful of answers in each. "
            "**The government's count of believers is not used.** The Agency for Cultural "
            "Affairs reports 175 million believers in a country of 124 million, because "
            "shrines report everyone who lives in their parish and temples report the "
            "households whose family grave they keep. Its count of Christians files each "
            "church body's members where the body is registered, which puts nearly half of "
            "them in Tokyo."),
        counts=_jp_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "jp" / "jp_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_jp_place_weight,
        note="**THE SECOND BUILD, 2026-09-14, on Anita's answers in ask 014**: JGSS 2021H-2024N "
             "national totals, JGSS-2015's six-block level and NHK 1996's prefecture pattern, "
             "combined by sources/jp_alloc.py. Christians follow NHK 1996 rather than the "
             "Agency roll (sources/jp.md §7); `python sources/jp_alloc.py --christians roll` "
             "builds the other version. Placement is COD-AB ADM1 2019 on Kontur hexes "
             "(sources/jp_grid.py).",
    ),
}
