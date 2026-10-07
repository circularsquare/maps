"""North Macedonia, Popis 2021, mother tongue (majchin jazik) -> node. sources/mk_census.py.

Keyed by the English labels of SSO's PxWeb table T1015P21, exactly as data/normalized/mk.csv
carries them. The municipal table prints seven named languages, sign language, "Other languages
not mentioned", "Unknown" and code 88, persons whose data were taken from administrative sources.
The national table T1013P21 names 28 more languages that the municipal table folds into "Other";
sources/mk.md lists them.

CALLS (sources/mk.md says more):
  Bosniak (Boshnjachki, 15,615): the existing `bosnian` leaf. "Boshnjachki jazik" is the name
    Macedonian usage and SSO give the language of the Bosniaks, and the census offers no separate
    "Bosnian" answer, so this is the one answer for that language here, not a rival name beside
    it. Bosnia's census, which prints both Bosanski and Boshnjachki, keeps them apart (ba2013);
    nothing here needs that.
  Vlachs (Vlashki, 3,151): `romance.aromanian` (au.txt, al2011). The Vlachs of North Macedonia
    are Aromanians (Shtip, Bitola, Krushevo, Skopje); the census's label also takes in the few
    hundred Megleno-Romanian speakers of Huma and the villages of Gevgelija, whom SSO does not
    count apart. Not Serbia's Vlach (rs2022), which is Romanian.
  Romani (31,721): `romani.romani`, variety not stated, as rs2022 and ba2013.
  Macedonian, Albanian, Turkish, Serbian: as printed, on the existing leaves.
  Sign language (27): `signlanguage`; SSO does not say which.
  "Other languages not mentioned" (5,167): `other`. Nationally (T1013P21) it is Bulgarian 1,519,
    Croatian 958, Russian 437 and 25 more, with 106 left unnamed; there is no indigenous
    remainder to keep apart, and the split is not published by municipality.
  NOT_STATED: "Unknown" (402) and "Persons for whom data are taken from administrative sources"
    (132,260, 7.2%: residents counted from registers, who were never asked). Not drawn; the
    entry's gap. "Total" is the universe row.
"""
IE = "indoeuropean"
SL = f"{IE}.slavic.south"

NAMES = {
    "Macedonian": f"{SL}.macedonian",
    "Albanian": f"{IE}.albanian.albanian",
    "Turkish": "turkic.turkish",
    "Romani": f"{IE}.indoaryan.romani.romani",
    "Vlachs": f"{IE}.romance.aromanian",
    "Serbian": f"{SL}.serbian",
    "Bosniak": f"{SL}.bosnian",
    "Sign language": "signlanguage",
    "Other languages not mentioned": "other",
}
NOT_STATED = {"Unknown", "Persons for whom data are taken from administrative sources", "Total"}


def resolve(label):
    if label in NOT_STATED:
        return None
    if label not in NAMES:
        raise KeyError(f"mk2021: unmapped label {label!r}")
    return NAMES[label]
