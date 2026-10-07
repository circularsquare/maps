"""Jamaica, JLU Language Competence Survey 2006 -> node (sources/jm_jlu.py, sources/jm.md).

  Jamaican (Patwa): Jamaican Creole, the node us, ca, uk, fr and pt already use.
  English: English, for the share who spoke only English in the survey.
"""
NAMES = {
    "Jamaican (Patwa)": "creole.english_based.jamaican",
    "English": "indoeuropean.germanic.english",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"jm2006: unmapped label {label!r}")
