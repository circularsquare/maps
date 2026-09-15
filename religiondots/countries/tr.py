# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


def _tr_place_weight(place):
    """countries.py hook. `place` is the 400m hex layer scatter.py has read.

    TWELVE UNITS FOR 85 MILLION PEOPLE, and they are enormous in area as well: TR7 Orta
    Anadolu is 90,000 km² of which most is the Konya-Aksaray steppe, and TRB runs from
    Malatya to the Iranian border over a plateau whose people are in a handful of basins.
    An equal share per polygon would put a large part of Türkiye's dots on empty high
    ground, and at this unit size that is most of what a reader would see
    (sources/tr_grid.py). §8.2's emptiness case at the largest scale it has come up.
    """
    return _kontur_place_weight(place, "tr_hexes.gpkg", "sources/tr_grid.py")


def _tr_counts():
    """Diyanet + TÜİK 2014 at İBBS-1: 8 drawn nodes on 12 regions, every row `modelled`.

    THE SOURCE IS THE RELIGIOUS AFFAIRS DIRECTORATE, NOT THE STATISTICAL OFFICE, and that is
    why this country was closed for so long. sources.md §11r asked TÜİK, asked the census,
    found nothing and wrote *"Türkiye's own publication of religion is nothing since 1965"*.
    The Diyanet publishes it, TÜİK ran the fieldwork, and §11ac is the correction.

    TWELVE UNITS IS THE SOURCE'S OWN ESTIMATION LEVEL. The report's methodology says the
    sample was sized to produce estimates for *"Türkiye total, Türkiye urban/rural and İBBS-1
    region totals"*, so this is not a coarse reading of a finer table — it is the whole of
    what exists. §14.4's rule 2 is therefore satisfied by construction: no resolution finer
    than the state's own publication, because this IS the state's own publication.

    `modelled` ON EVERY ROW, ON §7b'S TEST, which Guatemala settled: the tiers are about
    whether anybody was COUNTED, and nobody counted religion in Türkiye. 21,632 respondents
    cut twelve ways against a register population. `inferred dots: hidden` empties the
    country, and that is the honest picture of it.

    THE CONSTRUCTION INVENTS NO MAGNITUDE. A region's shares come from the survey; the number
    of people they apply to comes from OCHA COD-PS 2022, which is ADNKS's own resident
    population. Every person drawn is a person the register counts in that region, and the
    survey only decides the column — §14.4 rule 1, the same shape as Guatemala and Kazakhstan.

    FOUR SOURCE COLUMNS COLLAPSE ONTO `islam`, 10.4% of Turkish Muslims: `Hiçbiri`,
    `Bilmiyorum`, `Diğer` and `Cevap vermeyen` at question 11. All four answered *Islam* at
    question 10, so their religion is measured and only their school is not. **It is also
    where Türkiye's Alevis are**, because the questionnaire offers no Alevi option — see
    taxonomy/tr2014.py, which argues at length why nothing may be modelled out of it.
    """
    from tr2014 import resolve

    df = pd.read_csv(HERE / "data" / "normalized" / "tr.csv",
                     dtype={"geo_id": str}, low_memory=False,
                     keep_default_na=False, na_values=[""])
    df = df[df["geo_level"] == "region"].copy()

    # No lookup file: the Diyanet's own row labels ARE the NUTS codes GISCO ships, so
    # `geo_id` is the unit and §12's join failures cannot arise (sources/tr_geo.py).
    df["unit"] = df["geo_id"]
    if df["unit"].nunique() != 12:
        raise SystemExit(f"{df['unit'].nunique()} İBBS-1 regions, expected 12")

    df["node"] = df["source_category"].map(resolve)
    unmapped = sorted(df.loc[df["node"].isna(), "source_category"].unique())
    if unmapped:
        raise SystemExit(f"tr.csv categories with no node: {unmapped}")
    df = df[df["count"] > 0]
    df = df.groupby(["unit", "node"], as_index=False)["count"].sum()
    df["congregations"] = 0
    # EVERY row, without exception — there is no measured tier in this country (§7).
    df["tier"] = "modelled"
    return df[["unit", "node", "count", "congregations", "tier"]]


ENTRY = {
    "tr": dict(
        name="Türkiye",
        source="Türkiye'de Dinî Hayat Araştırması 2014, Table 4 (Diyanet İşleri Başkanlığı "
               "with TÜİK)",
        basis="self-identified school of law, sample survey; asked only of Muslims",
        note_public=(
            "**Türkiye's religion figures are not published by its statistical office, and "
            "that is why this map went without the country for so long.** No census has "
            "printed a religion table since 1965, the censuses after that asked and the "
            "answers were never released, and TÜİK publishes nothing. What does exist is a "
            "survey by the **Presidency of Religious Affairs**, with TÜİK designing and "
            "running the fieldwork: 21,632 people interviewed face to face in 2013, weighted "
            "to the address register, and sized to produce estimates for twelve statistical "
            "regions. Those twelve regions are the whole of what anybody publishes, so this "
            "is the coarsest country on the map, at 7.1 million people each. "
            "**What it buys is the one question almost no census anywhere asks: which "
            "school.** A census that asks about religion at all normally stops at Muslim. "
            "This one asks *which school of law do you feel you belong to*, and the answer "
            "is regional rather than doctrinal. Hanafi is **77.5%** of Turkish Muslims and "
            "runs above 90% along the Black Sea and through central Anatolia. Shafi'i is "
            "**11.1%** nationally and **48.7% in Ortadoğu Anadolu**, the only region of the "
            "country where it leads, with 42.0% in the southeast and 35.2% in the northeast. "
            "That band is the Kurdish provinces, and it is drawn here from the state's own "
            "survey rather than inferred from who lives there. The Ja'fari figure does the "
            "same thing at a tenth of the size: 1.0% nationally, **4.6% in Kuzeydoğu "
            "Anadolu**, which is Iğdır and Kars on the Azerbaijani border. "
            "**There is no Alevi answer on the card, and this map cannot draw one.** The "
            "report reprints its own questionnaire: the options are Hanafi, Shafi'i, Maliki, "
            "Hanbali, Ja'fari, Nusayri, don't know, other, none, and refuse. The word Alevi "
            "does not appear once in 293 pages. So Türkiye's Alevis are somewhere inside the "
            "**10.4% drawn here as Muslim with no school given**, alongside everybody else "
            "who never thought about the question, and nothing in the source separates them. "
            "Independent surveys put Alevis at four to six percent of the country and Alevi "
            "organisations put them much higher; **no source published anywhere gives the "
            "figure by region**, so this map states the absence rather than filling it. Read "
            "the grey as the space where a question was not asked. "
            "**The non-Muslim minorities are one number for the whole country.** 0.4% "
            "answered that they belong to another religion or to none, in a single cell that "
            "holds Türkiye's Christians, its Jews and its irreligious together; the report "
            "publishes it nationally and nowhere finer, so it sits at the same rate in all "
            "twelve regions here. Almost all of those people are in Istanbul in reality. "
            "**And nothing here was counted.** Every dot is a survey share applied to the "
            "population the address register puts in that region, so hiding inferred dots "
            "empties the country, which is the fair test of it."),
        how="survey, 21,632 people, 2013; no census has asked since 1965",
        grain="twelve statistical regions, 7.1m people each; the source's own limit",
        fill=("from the survey's national figures, for the religion split; the school split "
              "is regional"),
        gap=("Alevis, who have no option on the questionnaire; and any breakdown of the "
             "0.4% who are not Muslim"),
        counts=_tr_counts,
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "tr" / "tr_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_tr_place_weight,
        note="THE FINDING IS WHERE THE DATA LIVES, not what it says (sources.md §11ac). "
             "§11r closed Türkiye in September 2026 after asking TÜİK and asking the census; "
             "religion here is published by the Diyanet, with TÜİK's own fieldwork behind "
             "it. Before closing any country on `the office does not publish it`, ask which "
             "MINISTRY would. "
             "Shares are Table 4, page 42, parsed off the PDF by word position and asserted "
             "against two values read by eye; they are shares OF MUSLIMS, so sources/tr.py "
             "scales them into Table 1's national 99.2% and lays the other 0.9% on at the "
             "national rate. Magnitude is OCHA COD-PS 2022 at province, summed to İBBS-1 "
             "through a crosswalk written out in full and asserted exhaustive both ways. "
             "Boundaries are GISCO NUTS 2024 level 1, whose codes ARE the Diyanet's row "
             "labels, so the join is the identity function and 12 = 12 with nothing left "
             "over. Placement is Kontur's 400 m hexes, 454,587 of them, reproducing the "
             "register to 0.997x nationally and staying inside 0.82-1.15x in all twelve "
             "regions against 7 of 12 outside a factor of two when the labels are shuffled. "
             "THE ALEVI QUESTION IS THE OPEN ONE and it is a §14.2 problem rather than a "
             "sourcing one: KONDA's entire archived library, the World Values Survey (whose "
             "Türkiye denomination list is Islam / Orthodox / other / none) and the ESS were "
             "all checked, and none publishes an Alevi share by region. Nişanyan's "
             "settlement inventory has the geography and is out on Anita's call: its terms "
             "forbid systematic retrieval, and its coverage runs inverse to sect anyway.",
    ),
}
