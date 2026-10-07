"""Caribbean Netherlands, CBS Omnibus survey 2021 (StatLine 82867NED), main language (voertaal) of
persons aged 15 and over -> node. sources/bq_survey.py; keyed by the table's own Dutch labels.

CALLS (sources/bq.md says more):
  Papiaments: `creole.portuguese_based.papiamento`, as cw, aw and sx.
  Engels: English. On Sint Eustatius and Saba this is largely the islands' own English-lexifier
    vernacular, which the survey offers no separate answer for; the map draws the answer given.
  Anders (other, 1.2-2.5%): `other`. The survey offered only five answers, so this holds every
    other language (Haitian Creole, Portuguese, Chinese...) unnamed.
  Niet gepubliceerd: CBS withheld Saba's Papiamentu share as too unreliable; the remainder
    (4 people) is not drawn and goes in gap.
"""
NAMES = {
    "Papiaments": "creole.portuguese_based.papiamento",
    "Engels": "indoeuropean.germanic.english",
    "Nederlands": "indoeuropean.germanic.continental.dutch",
    "Spaans": "indoeuropean.romance.spanish",
    "Anders": "other",
}
SKIP = {"Niet gepubliceerd"}


def resolve(label):
    if label in SKIP:
        return None
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"bq2021: unmapped label {label!r}")
