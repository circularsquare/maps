"""Cape Verde: no language question; one label, everyone on Kabuverdianu (sources/cv_pop.py,
sources/cv.md). Portuguese is a learned second language (AGENT_BRIEF §2) and is not drawn.
"""
NAMES = {"Kabuverdianu": "creole.portuguese_based.kabuverdianu"}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"cv2021: unmapped label {label!r}")
