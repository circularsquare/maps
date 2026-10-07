"""Israel, CBS Social Survey 2021 native language (sources/il_social.py) -> node.

Labels are "<generator label> (<population group>)": the survey table is split by population
group so that Arabic can be drawn as two things. Arabic among Arabs is Levantine Arabic (sa.txt's
node, nort3139; Palestinian Arabic in Israel is South Levantine, a dialect of it in Glottolog), as
Palestine (ps2017). Arabic among Jews and others is mostly the Arabic of Jews from Iraq, Morocco
and Yemen and is not Levantine, so it sits on plain Arabic, as countries without a variety do.
"Another Language" (every language the 2021 generator does not name) is a bare other: `other`.
"""
AA = "afroasiatic"
IE = "indoeuropean"

LANG = {
    "Hebrew": f"{AA}.hebrew",
    "Russian": f"{IE}.slavic.east.russian",
    "English": f"{IE}.germanic.english",
    "French": f"{IE}.romance.french",
    "Spanish": f"{IE}.romance.spanish",
    "Yiddish": f"{IE}.germanic.continental.yiddish",
    "Amharic": f"{AA}.ethiosemitic.amharic",
    "AnotherLanguage": "other",
}
NAMES = {}
for _g in ("Arabs", "Jews and others"):
    for _l, _n in LANG.items():
        NAMES[f"{_l} ({_g})"] = _n
NAMES["Arabic (Arabs)"] = f"{AA}.levantine_arabic"
NAMES["Arabic (Jews and others)"] = f"{AA}.arabic"


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"il2021: unmapped label {label!r}")
