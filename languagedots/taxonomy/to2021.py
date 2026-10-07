"""Tonga, 2021 census, language used at home (Table G 48; sources/to_census.py, sources/to.md).

  Tongan language only: Tongan.
  Tongan and other language(s): Tongan, the one language the answer names; the other is not
    recorded (mostly English in practice). Drawn on Tongan rather than shared with an unnamed
    remainder, since Tongan is the language these people grew up with in nearly every case.
  Tongan language is not used at home: `other`; the census does not record which language.
"""
NAMES = {
    "Tongan language only": "austronesian.oceanic.tongan",
    "Tongan and other language(s)": "austronesian.oceanic.tongan",
    "Tongan language is not used at home": "other",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"to2021: unmapped label {label!r}")
