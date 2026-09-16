# One country's COUNTRIES entry and the helpers only it uses. countries.py loads this file;
# helpers that two or more countries use are in countries/_shared.py.
from countries._shared import *  # noqa: F401,F403


ENTRY = {
    # ---- ANDORRA (sources/ad.py, sources/ad.md) --------------------------------------------
    # The microstate tier's shape (one unit, Kontur hexes) from survey shares. _terr_counts with
    # tier="modelled", because _micro_counts would mark a survey share `measured`.
    "ad": dict(
        name="Andorra",
        source="World Values Survey wave 7, 2018 (Institut d'Estudis Andorrans), against the "
               "Department of Statistics' 2018 population estimate",
        basis="self-identification, adults 18 and over",
        view=[1.40, 42.42, 1.80, 42.66],
        how="survey, one round of 1,004 interviews in 2018",
        grain="the country as one unit, 76,200 people",
        gap="the 0.2% who did not answer",
        gap_share=0.00199,
        note_public=(
            "Andorra has never asked about religion in a census. This map is drawn from the "
            "2018 World Values Survey, which interviewed **1,004** residents aged 18 and over "
            "in all seven parishes, with quotas for sex, age and nationality, and its shares "
            "are applied to the 76,177 people the Department of Statistics estimated were "
            "living in Andorra in 2018. "
            "**63.8%** said they were Catholic and **30.1%** that they belonged to no "
            "religion. Orthodox Christians were 1.7%, Muslims 1.1%, Protestants 1.0%, Hindus "
            "0.9%, Buddhists 0.6% and Jews 0.1%, which is between 1 and 17 people each in the "
            "sample, so these small shares are rough. "
            "The country is drawn as one unit, so the dots follow where people live."),
        counts=lambda: _terr_counts("ad", "ad2018", 1, tier="modelled"),
        units=None,
        unit_key=None,
        place=HERE / "data" / "geo" / "ad" / "ad_hexes.gpkg",
        place_unit=lambda g: g["unit"].astype(str),
        place_weight=_micro_place_weight("ad", "sources/ad.py"),
        note="WVS-7 ANDORRA 2018 FROM THE IHSN CATALOGUE'S FREQUENCIES, NO DOWNLOAD FORM. "
             "catalog.ihsn.org/catalog/11550 prints each variable's unweighted counts for the "
             "same file the WVS form guards; Q289 is checked one to one against Q289CS9, "
             "W_WEIGHT is one value for all 1,004, and the survey team says quotas made "
             "weighting unnecessary. Shares laid on the Department of Statistics' 2018 "
             "estimate, 76,177 (the fieldwork year; 89,058 by 2025, mostly nationalities the "
             "quotas barely reached). Drawn as one unit: the catalogue has no religion by "
             "parish, and the parishes were not sampled in proportion (La Massana and Ordino "
             "8.2% of interviews, 20.0% of residents). Other is a write-in coded Other; nfd, on "
             "other.ad. sources/ad.md.",
    ),
}
