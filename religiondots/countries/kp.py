# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _kp_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Jagang, Ryanggang and the Hamgyong provinces are mostly mountain. Drawn flat, their dots would
    sit on the Kaema plateau (sources/kp_grid.py).
    """
    return _kontur_place_weight(place, "kp_hexes.gpkg", "sources/kp_grid.py")


def _kp_counts():
    """Pew 2020's national mix (the World Religion Database) on the 2008 census: 11 provinces.

    EVERY ROW IS `modelled` (§7b). No census or survey has ever asked religion, so each province's
    people take one national mix. sources/kp.py and sources/kp.md.
    """
    from kp2020 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "kp.csv",
                     dtype={"geo_id": str}, keep_default_na=False, na_values=[""])
    lut = pd.read_csv(HERE / "data" / "geo" / "kp" / "kp_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"kp.csv provinces with no polygon: {missing}; re-run sources/kp_geo.py")
    if df["unit"].nunique() != 11:
        raise SystemExit(f"{df['unit'].nunique()} provinces, expected 11")
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"kp.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df["tier"] = "modelled"
    df["congregations"] = 0
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "kp": dict(
        name="North Korea",
        source="No census or survey asks; Pew Research Center's 2020 estimate for North Korea "
               "(from the World Religion Database), against each province's people in the 2008 "
               "census (Central Bureau of Statistics, with the UN Population Fund)",
        basis="a compiler's national estimate, the same in every province",
        note_public=(
            "**No North Korean census or survey has ever asked about religion.** The 1993 and "
            "2008 censuses did not ask, and no independent survey of people living in the "
            "country has been possible. This map uses Pew Research Center's estimate for 2020, "
            "which for North Korea comes from the World Religion Database rather than from a "
            "survey; Pew lists it as the only source there is. It puts **72.9%** with no "
            "religion, **12.9%** following Cheondogyo, **12.3%** Korean folk religion, 1.5% "
            "Buddhist and 0.4% Christian, and the map draws those shares in every province. Nobody counted these dots, so they "
            "disappear when inferred dots are turned off. "
            "**These are outside estimates of belief in a country that punishes religious "
            "practice.** The state keeps a few official churches and temples, and in 2002 it "
            "told the UN Human Rights Committee there were 12,000 Protestants, 10,000 "
            "Buddhists, 800 Catholics and 15,000 followers of Cheondogyo. Outside estimates of "
            "the number of Christians run from the World Religion Database's 100,000 to Open "
            "Doors' 400,000. The 72.9% with no religion are the database's agnostics and "
            "atheists. "
            "**The Cheondogyo and Korean folk religion figures are the World Religion "
            "Database's estimates.** Pew puts both under other religions. The database counts "
            "12.9% of North Koreans as followers of new religions, which in North Korea means "
            "chiefly Cheondogyo, the religion of the 19th-century Donghak movement, and 12.3% as "
            "followers of an ethnic religion, Korean shamanism; the map draws them as those two "
            "religions. Neither figure comes from a count. The database's share comes to about "
            "3 million followers of Cheondogyo, where the state reports about 15,000. Another "
            "0.06%, about 14,000 people, are drawn as Chinese folk religion. "
            "**The map does not show where any religious community lives.** No source gives "
            "religion by province, so every province is drawn at the national mix, and every "
            "religion's dots stand wherever people live, not where any church, temple or "
            "shrine is. "
            "**The dots stand where people lived in 2008.** They follow the 2008 census, taken "
            "with the UN Population Fund's support, which counted **23,349,859** people by "
            "province; Kontur's population grid places them within each province. The census "
            "counted another 702,372 people nationally and in no province (its national tables "
            "include military camps), and they are not drawn. Nor are about 2,700 Muslims and "
            "Hindus in Pew's estimate, whom nothing places."),
        how="no source asks; a compiler's national estimate, one mix everywhere",
        grain="provinces, 2.1 million people on average",
        gap="the 702,372 people the 2008 census counts nationally but in no province (its "
            "national tables include military camps), and about 2,700 Muslims and Hindus; 2.93% "
            "in all",
        gap_share=0.029314,
        counts=_kp_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "kp" / "kp_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_kp_place_weight,
        note="BUILT ON ANITA'S RULINGS OF 2026-09-15 (a §14 case files an ask and carries on) AND "
             "2026-09-16 (a compiler's figure where nothing asks); ASK 052 holds whether North "
             "Korea is drawn at all and whether its Christians are. sources/kp.md is the record. "
             "LEVEL: Pew 2020 North Korea row, which Pew's Appendix A sources to the World "
             "Religion Database and calls the only source (UN WPP 2024 population): unaffiliated "
             "72.868, other 25.220, Buddhist 1.517, Christian 0.384; Muslims and Hindus 3,028 "
             "not drawn (gap). OTHER: WRD (ARDA 2025) new religionists 12.88 + ethnic "
             "religionists 12.28 + Chinese folk 0.06, Pew's other split in those proportions onto "
             "eastasiannew.korean.cheondogyo, indigenous.korean and chinesefolk (Anita, ask 052, "
             "2026-10-03; other.kp retired). POPULATION: "
             "2008 census national report Table 2 (209 units, 23,349,859), re-cut to COD-AB's 11 "
             "units (Nampo out of South Pyongan; Kangnam, Junghwa, Sangwon from Pyongyang to "
             "North Hwanghae); COD-PS's PRK admin1 is the same table with Samchon missing and "
             "Sindo counted twice, not used. 702,372 in Table 1 and not Table 2 (military camps) "
             "in gap. GEOGRAPHY: COD-AB cod-ab-prk v01 admin1. PLACEMENT: Kontur KP 2023, ratio "
             "1.127, rank witness +0.982. NOT DRAWN: any placement of any community (none exists).",
    ),
}
