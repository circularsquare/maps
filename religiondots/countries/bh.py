# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _bh_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Kontur decides where people are inside each governorate, uncalibrated, since nothing finer
    than the governorate is used (sources/bh_grid.py; per governorate 0.82-1.18 of the census).
    """
    return _kontur_place_weight(place, "bh_hexes.gpkg", "sources/bh_grid.py")


def _bh_counts():
    """2020 census: Muslim / Others by nationality and sex, national, carried to the four
    governorates by their nationality groups and sex (sources/bh.py, sources/bh.md).

    Bahrainis and the non-Bahraini Muslim / Others totals are `derived`: counted for the country,
    placed by a proxy. The split of the non-Bahraini `Others` into religions is `modelled` (UN DESA
    origins through Pew 2020, Christians against Hindus by the Gulf rule), and so is the split of
    Bahraini Muslims into Shia and Sunni (mosque counts at Arab Barometer I's level; ask 055). No
    row rolls up (below), the sect rows included: rolling them to `islam` would show Bahraini
    Muslims by governorate under `inferred dots: not shown`, a count nobody made at that unit.
    """
    from bh2020 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "bh.csv", keep_default_na=False,
                     na_values=[""])
    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"bh.csv categories with no node: {unmapped}")
    lut = pd.read_csv(HERE / "data" / "geo" / "bh" / "bh_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"bh rows with no unit: {missing}; re-run sources/bh_geo.py")
    if df["unit"].nunique() != 4:
        raise SystemExit(f"{df['unit'].nunique()} governorates, expected 4")
    bad = sorted(set(df["tier"]) - {"derived", "modelled"})
    if bad:
        raise SystemExit(f"bh.csv tiers {bad}")
    df = df[df["count"] > 0]
    # the weakest tier wins on a (unit, node) pair: other.bh holds derived Bahrainis and the
    # modelled residual of the foreign split, so it is modelled
    df = df.groupby(["unit", "node"], as_index=False).agg(
        count=("count", "sum"),
        tier=("tier", lambda t: "modelled" if (t == "modelled").any() else "derived"))
    df["congregations"] = 0
    # spec §3.10: a carried or modelled count cannot establish that anyone is present
    df["may_ring"] = False
    # spec §7a-i-1: the census's columns (Muslim, Others) were counted for the whole country only,
    # so nothing rolls and Bahrain empties under `inferred dots: not shown`, as Kuwait and the UAE.
    from rollup import NOWHERE
    df["roll"] = NOWHERE
    return df[["unit", "node", "count", "congregations", "tier", "roll", "may_ring"]]


ENTRY = {
    "bh": dict(
        name="Bahrain",
        source="2020 census (Information & eGovernment Authority, open data portal): religion by "
               "nationality and sex, and population by governorate, nationality group and sex; "
               "the foreign residents' other religions through UN DESA's 2024 migrant stock by "
               "origin and Pew Research Center's 2020 estimates; Bahraini Muslims' sect from the "
               "Ja'fari and Sunni Endowments' mosque counts and Arab Barometer's 2009 survey",
        basis="Religion as the census counted it for the whole country, placed by nationality "
              "group and sex; Bahraini Muslims' sect estimated from mosque counts",
        note_public=(
            "**Bahrain's 2020 census counted everyone's religion, but its published table gives "
            "only Muslim or other, for Bahrainis and for foreign residents by sex, for the whole "
            "country.** The 2010 census form also coded Christian and Jewish, and neither census "
            "printed them apart. No table found crosses religion with a governorate, so each governorate's "
            "Bahrainis, and its foreign residents of each nationality group and sex, are drawn "
            "at shares that add up to the census's national counts. These dots disappear when "
            "inferred dots are turned off. "
            "**The 712,362 Bahrainis are 99.7% Muslim in the census, and their split into Shia "
            "and Sunni is an estimate.** Nobody has counted sect in Bahrain. Arab Barometer's 2009 "
            "survey of Bahraini citizens found 249 Shia and 183 Sunni among 435 people, so "
            "**57%** of Bahraini Muslims are drawn as Shia. Where they live comes from two "
            "counts of mosques by governorate: the Ja'fari Endowments' count of Shia mosques in "
            "2016 and the Sunni Endowments' count of about 2022. Shia mosques are about four in "
            "five in the Capital and Northern governorates and one in five in Muharraq and the "
            "Southern, and Bahraini Muslims are drawn about 80% and 21% Shia there. A mosque "
            "count is not a count of people, so these shares are rough. Foreign residents' Muslims "
            "have no sect drawn, since nothing measures it. The 2,295 Bahrainis counted as other "
            "than Muslim are drawn as another religion, since nothing says which. "
            "**Nearly half of the 789,273 foreign residents are not Muslim: 387,807 in the "
            "census, 46% of the men and 58% of the women.** How that splits between nationality "
            "groups is a model. Each group's share comes from the UN's 2024 estimate of where "
            "Bahrain's migrants come from and Pew Research Center's estimate for each home "
            "country, with the Asian and African residents' shares adjusted so the totals match "
            "the census. Other Arab residents are drawn about 96% Muslim, Asian men 48% "
            "non-Muslim and Asian women 69%. "
            "**The non-Muslims' religions come from the same home-country figures, with one "
            "change.** Taken alone, those would make Hindus outnumber Christians nearly three to "
            "one, while Pew Research Center's 2020 estimate for Bahrain has more Christians than "
            "Hindus. The map follows Pew on that one balance by drawing more of the Indian "
            "residents as Christian, as for the UAE and Oman, and leaves the other religions as "
            "the home countries give them. That puts **193,586** people on Christianity, "
            "**158,651** on Hinduism, **12,404** on Buddhism, **10,952** on no religion and "
            "**6,750** on Sikhism. The 2001 census, the last that printed Christians, found 48% "
            "of non-Muslims Christian, close to the 50% drawn. Christians are not split into "
            "churches. "
            "**The Capital governorate, home to nearly half the foreign residents, has the most "
            "non-Muslims.** The share drawn as Muslim is 64% there, 74% in the Southern "
            "governorate, 78% in Muharraq and 86% in the Northern."),
        how="census, 2020, religion counted for the country by nationality and sex; Bahraini "
            "Muslims' sect estimated",
        fill="from the census's national count for Bahrainis and foreign residents by sex, by "
             "each governorate's nationality groups",
        grain="governorates, 375,000 people on average",
        counts=_bh_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "bh" / "bh_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_bh_place_weight,
        note="REOPENED 2026-10-03 from sources.md §scout-2026-10-03-negatives (closed in §11ao "
             "because religion never meets governorate). sources/bh.md is the record. RELIGION: "
             "data.gov.bh census 2020, Muslim / Others x Bahraini / non-Bahraini x sex, national "
             "(1,501,635; UNSD's 2020 row). POPULATION: governorate x nationality x sex and "
             "governorate x 8 nationality groups x sex, both closing on it. BAHRAINIS: national "
             "shares by sex; 2,295 Others on other.bh. SECT (ask 055, 2026-10-03; "
             "sources.md §bh-2026-10-03c): Bahraini Muslims split islam.shia / islam.sunni "
             "by governorate, modelled: Shia share of mosques (Ja'fari Endowments 2016 against "
             "Sunni Endowments c.2022) moved by one logit shift (+0.055) to Arab Barometer I's "
             "249 / 183 / 3 of 435; the 3 on bare islam; drawn Shia 79% Capital, 80% Northern, "
             "22% Muharraq, 21% Southern; witnesses OSM Shia-tagged mosques (0.66-0.88 of the "
             "register per governorate) and the 2017 Washington Institute poll (62%). Rolls "
             "NOWHERE like the rest: Muslims were counted for the country only. NON-BAHRAINIS: each group-sex's non-Muslim share from UN DESA 2024 "
             "origins x Pew 2020, Asian and African shifted by one logit per sex so the census's "
             "Others close exactly (men -0.359, women +0.505); Others split by family from the "
             "same origins, then the Gulf rule as ae and om (Christians / (Christians + Hindus) "
             "raked to Pew 2020's Bahrain 0.550 by moving 102,870 Indians from Hindu to "
             "Christian, other families as the origins give; sources.md §bh-2026-10-03b), "
             "Other_religions via origin_religion.OTHER. TIERS: Muslim/Others derived, the "
             "split modelled. ROLL: NOWHERE. WITNESS: Christians 49.6% of non-Muslims against "
             "the 2001 census's 47.7% (band 10 points). GEOGRAPHY: geoBoundaries BHR ADM1 (OSM "
             "2017, the four post-2014 governorates; COD-AB is GAUL 2008), 100% inside Kontur "
             "Boundaries' 2023 OSM lines. PLACEMENT: Kontur BH uncalibrated, 0.989 of the census; "
             "0.82 Northern to 1.18 Southern.",
    ),
}
