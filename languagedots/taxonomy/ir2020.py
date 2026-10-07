"""Iran, World Values Survey waves 5 (2005) and 7 (2020), language at home, pooled by province
(sources/ir_wvs.py) -> node. Keyed by the labels the WVS online tool prints, with the place or wave
tags ir_wvs.py adds where one label means two things.

CALLS (sources/ir.md says more):
  "Azerbaijani; Azeri" (wave 7) and "Turkish [2005]" (wave 5's file code "Azari", which the online
    tool prints as Turkish): `turkic.azerbaijani`. Iran's other Turkic speech (Qashqai in Fars,
    Khorasani Turkic, Afshar) had no answer of its own in either wave and is inside these.
  "Turkmen [2005]": wave 5's file code "Turkish" (7 answers), which the online tool harmonises to
    Turkmen; drawn as the tool labels it. 2 of them are in North Khorasan.
  "Kurdish; Yezidi": `indoeuropean.iranian.kurdish`, the leaf other countries draw. One answer in
    both waves: Central (Sorani), Southern and Northern (Kurmanji) Kurdish are not told apart, and
    no variety is guessed from the province.
  "Lurish; Luri; Bakhtiari": one answer for the Luric languages (Glottolog luri1252: Northern
    Luri, Bakhtiari, Southern Luri); a new leaf `luri`, not a group, so it is not drawn as
    "language not named". Laki had no answer.
  "Gilaki": wave 7's answer took Mazandarani speakers too (all 41 Mazandaran interviews; the
    ethnic card's "Gilak/Mazani/Shomali" is one group). Split by the survey's own province, as
    India's Pahari: in Mazandaran and Golestan ("Gilaki [Mazandaran, Golestan]") it is
    Mazandarani (maza1291), everywhere else Gilaki (gila1241). Wave 5 had Gilaki and no Mazandarani
    answer; ir_wvs.py leaves wave 5 out of those two provinces.
  "Balochi" (wave 5) and "Other (ethnic group Baluch)" (wave 7's "Other" from respondents whose
    ethnic group is Baluch, 24 in Sistan and Baluchestan): `indoeuropean.iranian.balochi`.
  "Zoroastrian [2005]": one answer in Yazd, the Zoroastrians' own language (Zoroastrian Dari,
    zoro1242, gbz); a new leaf.
  "Other": wave 7's bare other (Golestan 25, Qazvin 10, Hormozgan 3...): `other`. Turkmen and
    Tati were not on the card and nothing in the file names them.
  "No answer": not drawn (ir_wvs.py drops it before the shares).
"""
NAMES = {
    "Persian; Farsi; Dari": "indoeuropean.iranian.persian",
    "Azerbaijani;  Azeri": "turkic.azerbaijani",
    "Turkish [2005]": "turkic.azerbaijani",
    "Turkmen [2005]": "turkic.turkmen",
    "Kurdish; Yezidi": "indoeuropean.iranian.kurdish",
    "Lurish; Luri; Bakhtiari": "indoeuropean.iranian.luri",
    "Gilaki": "indoeuropean.iranian.gilaki",
    "Gilaki [Mazandaran, Golestan]": "indoeuropean.iranian.mazandarani",
    "Arabic": "afroasiatic.arabic",
    "Balochi": "indoeuropean.iranian.balochi",
    "Other (ethnic group Baluch)": "indoeuropean.iranian.balochi",
    "Armenian; Hayeren": "indoeuropean.armenian.armenian",
    "Assyrian Neo-Aramaic": "afroasiatic.assyrian",
    "Zoroastrian [2005]": "indoeuropean.iranian.zoroastrian_dari",
    "Other": "other",
}
SKIP = {"No answer"}


def resolve(label):
    if label in SKIP:
        return None
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"ir2020: unmapped label {label!r}")
