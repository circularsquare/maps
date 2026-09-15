# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ye_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    Hadramawt and Al Maharah are half of Yemen's area and 5.6% of its people; drawn flat, their
    dots would sit on the edge of the Empty Quarter instead of in the Wadi and on the coast
    (sources/ye_grid.py).
    """
    return _kontur_place_weight(place, "ye_hexes.gpkg", "sources/ye_grid.py")


def _ye_counts():
    """Arab Barometer wave V (2018-2019) at governorate: 5 categories, 21 units.

    A SURVEY ON A PROJECTION, AND EVERY ROW IS `modelled` (§7b). Wave V's governorate shares of
    `Q1012` crossed with the logged sect item, applied to the Population Task Force's 2025
    estimate (CSO, UNFPA, IOM, OCHA), which projects the 2004 census and corrects it for
    displacement. Nobody in Yemen has published a count of religion or sect.

    ONE WAVE IS DRAWN AND THE OTHER IS THE TEST. Wave III (2013) offered only Sunni or Shia, so
    it cannot be pooled with wave V's `Just a Muslim` box; it replicates wave V's Zaydi
    geography instead (Spearman +0.844 over 21 governorates, exact bar +0.370), and wave V's 240
    PSUs give the split-half (Sunni +0.947, Zaydi +0.930, Just a Muslim +0.846).

    CODE 14 `Alawi` IS ZAYDI, read on its geography and asserted in sources/ye.py.

    DRAWN AT GOVERNORATE under ask 021, whose ruling on Iraq's Sunni and Shia geography said
    Yemen is built the same way.

    SOCOTRA (75,725) IS NOT DRAWN: a governorate since 2013, in neither wave's list.
    """
    from ye2019 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "ye.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "ye" / "ye_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"ye.csv governorates with no polygon: {missing} -- re-run "
                         "sources/ye_geo.py, the lookup is stale")
    if df["unit"].nunique() != 21 or "YE32" in set(df["unit"]):
        raise SystemExit(f"{df['unit'].nunique()} governorates, expected the 21 without Socotra")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"ye.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "ye": dict(
        name="Yemen",
        source="Arab Barometer, wave V 2018 to 2019 (Arab Barometer, Princeton University), "
               "against the 2025 population estimate of the Population Task Force (Central "
               "Statistical Organization, UNFPA, IOM and OCHA)",
        basis="self-identification, adults 18 and over",
        note_public=(
            "**Nobody has counted religion in Yemen, and there has been no census since "
            "2004.** This map is 2,400 people interviewed by the Arab Barometer in 2018 and "
            "2019, with each governorate's answers applied to that governorate's 2025 "
            "population as the humanitarian Population Task Force estimates it: **34,879,018** "
            "people, projected from the 2004 census and corrected for displacement. None of "
            "the dots was counted, and they are drawn desaturated to say so. "
            "**Sunnis are 58.0% of the country as drawn and Zaydis 17.7%, and the Zaydis are "
            "in the northern highlands.** Sa'dah is **62.5%** Zaydi, Amran 51.3%, Dhamar 39.4%, "
            "Hajjah 34.6% and Sana'a City 28.1%. Ta'iz, Aden, Lahj, Abyan, Al Bayda, Ad Dali', "
            "Shabwah and Al Maharah have none drawn. The survey's answer list had no box for "
            "Zaydi, and the interviewers logged those answers under a code the data file "
            "labels Alawi; they are read as Zaydi because of where they fall, 1 interview in "
            "950 across the south and east and 62 in 100 in Sa'dah. An earlier round in 2013, "
            "which offered only Sunni or Shia, put its Shia answers in the same places, 39 of "
            "40 in Sa'dah. "
            "**The Zaydi share is lower than the usual estimates.** Of the people drawn with a "
            "branch, **23.4%** are Zaydi, where the US government estimates 35% of the "
            "population and the conflict monitor ACLED 45% of Muslims; neither is a count. "
            "The plain answer just a Muslim was most common in Sana'a (46.8%), Sana'a City "
            "(43.9%) and Hajjah (36.2%), so some of the highlands' Zaydis are probably on the "
            "plain Islam colour, and the Zaydi share here is best read as a floor. "
            "**Nearly a quarter of the country is drawn as Muslim with no branch, and part of "
            "that is how the answers were written down.** The question was asked without "
            "reading out the options, and the interviewer logged what was said. **8.45 million** "
            "people are on the plain Islam colour. In eight governorates nobody at all was "
            "logged as just a Muslim, among them Ta'iz (0 of 260 interviews) and Lahj (0 of "
            "100), which looks more like the habit of the teams working there than a fact "
            "about those places, and the Sunni share drawn in those eight is higher for it. "
            "**Ma'rib's population is mostly people who moved there during the war.** The "
            "task force puts it at **1.77 million**, 1.64 million of them displaced, and its "
            "dots follow that figure. "
            "**Socotra is not drawn.** It became a governorate in 2013 and neither survey round "
            "sampled it, so its 75,725 people are left off rather than given Hadramawt's "
            "answers. "
            "**No religious minority is drawn.** The State Department puts Yemen's Hindus, "
            "Bahá'ís, Christians and Jews together at under 1% of the population, and none of "
            "the 2,400 respondents named any of them; the one who said atheist is left out. "
            "The survey interviewed adults and the shares are applied to everyone, and 43.2% "
            "of Yemenis are under 15."),
        how="survey, one round 2018 to 2019",
        grain="governorates, 1.66 million people on average",
        gap="Socotra, 0.2% of Yemen, which no survey round sampled",
        gap_share=0.0022,
        counts=_ye_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ye" / "ye_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ye_place_weight,
        note="WAVE V IS DRAWN AND WAVE III IS THE WITNESS. The files offer Yemen religion "
             "answers in waves III (2013, 1,200, a Sunni/Shia card) and V (2018-19, 2,400, an "
             "interviewer-logged list); wave II has Yemenis and an empty q1012. Pooling III "
             "would put forced choices beside a card with a Just a Muslim box, Iraq's card rule. "
             "THE SPLIT-HALF IS REPLACED BY TWO TESTS: wave III against wave V on the Zaydi share "
             "of branch-namers (Spearman +0.844, exact bar +0.370, permutation p 5e-05, "
             "leave-one-out never under +0.818), and lits.stability on wave V's 240 PSUs. "
             "CODE 14 `Alawi` IS ZAYDI on its geography (sources/ye.py::zaydi_geography). "
             "WAVE III'S Q1 LABELS NAME TWO GOVERNORATES `Sana'a` (10503 and 10513), so wave "
             "III is decoded on the CSO-order code and wave II's labels witness it. "
             "32 WAVE V WEIGHTS ARE BLANK, all respondents with no recorded gender, 22 of them "
             "Zaydi; they take their PSU's mean weight (arabbarometer.py::_fill_blank_weights). "
             "BOUNDARIES are COD-AB (22 governorates); POPULATION is the Task Force's 2025 "
             "district table, whose governorate sums equal its methodology note's printed "
             "totals to the person, and Kontur's governorate totals pin the p-code pairing "
             "(rho +0.870, 0 of 5,000).",
    ),
}
