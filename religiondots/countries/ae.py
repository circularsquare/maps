# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _ae_place_weight(place):
    """countries.py hook. `place` is the 400 m hex layer scatter.py has read.

    Kontur decides where people are inside each emirate, uncalibrated, since nothing finer than
    the emirate is used (sources/ae_grid.py).
    """
    return _kontur_place_weight(place, "ae_hexes.gpkg", "sources/ae_grid.py")


def _ae_counts():
    """2024: Emiratis on Islam, everyone else at one national mix of origins, seven emirates.

    EVERY ROW IS `modelled` (§7b). No source asks religion in the UAE, so every Emirati is drawn on
    Islam; each emirate's non-Emiratis take UN DESA 2024's national origin mix through Pew 2020,
    with India's Hindu share set by Pew's UAE estimate and its Christian share by the Gulf rule. Both halves come from the emirates' own
    counts, scaled to FCSC's 2024 total, so they partition each emirate. sources/ae.py, ae.md.
    """
    from ae2024 import resolve

    nat = pd.read_csv(HERE / "data" / "normalized" / "ae.csv", dtype={"geo_id": str},
                      keep_default_na=False, na_values=[""])
    nat["node"] = nat["source_category"].map(resolve)
    unmapped = sorted(nat.loc[nat["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"ae.csv categories with no node: {unmapped}")
    ext = pd.read_csv(HERE / "data" / "normalized" / "ae_foreign.csv", dtype={"geo_id": str})
    df = pd.concat([nat[["geo_id", "node", "count"]], ext[["geo_id", "node", "count"]]],
                   ignore_index=True)
    lut = pd.read_csv(HERE / "data" / "geo" / "ae" / "ae_lookup.csv", dtype=str)
    df["unit"] = df["geo_id"].map(dict(zip(lut["geo_id"], lut["unit"])))
    missing = sorted(df.loc[df["unit"].isna(), "geo_id"].unique())
    if missing:
        raise SystemExit(f"ae rows with no unit: {missing}; re-run sources/ae_geo.py")
    if df["unit"].nunique() != 7:
        raise SystemExit(f"{df['unit'].nunique()} emirates, expected 7")
    df = df[df["count"] > 0]
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["tier"] = "modelled"
    df["congregations"] = 0
    # spec §3.10: a modelled count cannot establish that anyone is present
    df["may_ring"] = False
    # spec §7a-i-1: nothing was measured at an emirate (or anywhere), so nothing rolls and the UAE
    # empties under `inferred dots: not shown`, as Saudi Arabia, Oman and Kuwait do. NOWHERE as
    # Kuwait, Angola and Uganda (supervisor note from the kw review, 2026-10-03).
    from rollup import NOWHERE
    df["roll"] = NOWHERE
    return df[["unit", "node", "count", "congregations", "tier", "roll", "may_ring"]]


ENTRY = {
    "ae": dict(
        name="United Arab Emirates",
        name_in="the United Arab Emirates",
        source="No census or survey asks; each emirate's statistics office for its population and "
               "its citizens (latest years 2005 to 2024), the Federal Competitiveness and "
               "Statistics Centre's 2024 total, and UN DESA's 2024 migrant stock by origin; "
               "through Pew Research Center's 2020 estimates",
        basis="Emiratis drawn as Muslim, which nobody asked; foreign residents by country of origin",
        note_public=(
            "**Nobody in the United Arab Emirates is asked their religion.** The last federal "
            "census, in 2005, had no question on it, and none of the emirates' own censuses since "
            "has published one. No source counts an Emirati who is not Muslim, so the "
            "**1,519,227** Emiratis are all drawn as Muslim. That figure is each emirate's latest "
            "count of its citizens, from 2005 in Ajman to 2022 in Sharjah, grown to 2024 at the "
            "national rate; it is 9% more than a federal series built from births and deaths, so "
            "it may be high. The map does not split them into Sunni and Shia, since no figure "
            "places either. Nobody counted these dots, so they disappear when inferred dots are "
            "turned off. "
            "**Every non-Muslim drawn is a foreign resident.** The Federal Competitiveness and "
            "Statistics Centre counts **11,294,243** people in 2024. Abu Dhabi, Dubai and "
            "Fujairah published 2024 totals; the latest for Sharjah is its 2022 census, for "
            "Ajman 2017, for Ras Al Khaimah 2015 and for Umm Al Quwain 2005, and those four are "
            "scaled up together so the seven add up to the national figure. Nothing published "
            "says which nationalities live in which emirate, so the non-Emiratis of every "
            "emirate are drawn at one mix: the UN's 2024 estimate of where the UAE's migrants "
            "come from, India 40% of them, then Bangladesh, Pakistan, Egypt and the Philippines. "
            "Each origin is drawn at Pew Research Center's 2020 estimate for that country, which "
            "cannot see anyone who converted or stopped practising, except that migrants from "
            "the other Gulf states are drawn as Muslim. "
            "**Indians are not drawn at India's own figure.** Pew's migration estimates count "
            "more of the Indians in Muslim-majority Middle Eastern countries as Muslim than "
            "India's own share, taking Egypt's census of its Indian residents as the guide. So "
            "Indians are drawn at the Hindu share that gives Pew's figure for Hindus in the UAE "
            "(11.8% of everyone), with the difference drawn as Muslim. Figures by home country "
            "also give the Gulf too few Christians: Kuwait, the one Gulf state that records "
            "religion by region of origin, counts about ten times as many Christians among its Asian "
            "residents as they give. So some of those Indian Hindus are then drawn as Christian, "
            "until Christians and Hindus stand in the same proportion as in Pew's UAE estimate. "
            "Indians end up 66% Muslim, 21% Hindu and 10% Christian, against India's own 15%, "
            "79% and 2%. That puts **2,556,807** people on religions other than Islam: 1,224,962 "
            "Christians, 1,005,866 Hindus, 152,171 Buddhists and 86,923 Sikhs. Pew's estimate "
            "for everyone living in the UAE is 72.9% Muslim and 14.3% Christian; this map draws "
            "77.4% and 10.8%. Pew's figures here are estimates too, not counts, and give the "
            "non-Muslims of Bahrain, Kuwait, Qatar and the UAE almost the same split. The "
            "Buddhists are about eight times Pew's figure, most of them Sri Lankans."),
        how="no source asks; Emiratis drawn as Muslim, foreign residents by country of origin",
        grain="emirates, 1.6 million people on average",
        gap="Emiratis who are not Muslim, whom no source counts",
        counts=_ae_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ae" / "ae_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_ae_place_weight,
        note="BUILT ON ANITA'S PRIORITY LINE AND THE MAGHREB AND MAURITANIA RULINGS (ask/RULINGS.md "
             "2026-09-15 and 2026-09-16), as sa, om. sources/ae.md is the record. EMIRATIS: nothing "
             "asks (2005 form no item; no emirate census publishes religion; not in the Arab "
             "Barometer; no PACI-style table found, Wayback blocked this host on 2026-10-03); "
             "1,519,227 on islam, each emirate's newest count (AD 2016, Dubai 2016, Sharjah 2022, "
             "Ajman 2005, UAQ 2010, RAK 2015, Fujairah 2016) grown to 2024 at GLMM's FCSC-based "
             "national rate, 1.093x that series, not forced; no Sunni/Shia split (asks 040, 043). "
             "TOTALS: AD 4,135,985, Dubai 3,863,600, Fujairah 314,829 (2024 as printed); Sharjah, "
             "Ajman, RAK, UAQ scaled by 1.1040 to close on FCSC's 11,294,243. NON-EMIRATIS: one "
             "national mix, UN DESA 2024 (33 origins, Others 3% at the named mix); by sex tried "
             "and dropped (Christians 9.0% men, 9.7% women). Pew 2020 per origin, Muslim branches "
             "folded; Gulf origins (BH, KW, QA, SA; 73,719 in DESA) on islam; India's Hindu share "
             "28.63% so the layer's Hindus equal Pew's UAE 11.754%, then the Gulf rule "
             "(sources.md §gulf-2026-10-03) moves 321,606 Indians Hindu to Christian so C/(C+H) is "
             "Pew's 0.549 (Indians 20.6% H, 10.2% C). WITNESS: Christians 0.76 of Pew's UAE share "
             "(band 0.2-2.0; 0.56 before the rule); Buddhists 8x, not corrected. GEOGRAPHY: COD-AB ARE "
             "ADM1 (7, pcodes), geoBoundaries ADM1 as witness (IoU 0.76-0.99). PLACEMENT: Kontur "
             "AE uncalibrated (0.843 of FCSC; 0.74 Abu Dhabi to 1.62 Umm Al Quwain); hexes in "
             "Oman dropped; no block at the cap. ROLL: NOWHERE.",
    ),
}
