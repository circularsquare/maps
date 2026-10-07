"""Puerto Rico, Puerto Rico Community Survey 2020-2024 5-year: language spoken at home -> node.

The PRCS is the American Community Survey run in Puerto Rico, with the same PUMS language codes
(ACSPUMS2020_2024CodeLists.xlsx). So every label here is mapped exactly as taxonomy/us2024.py
maps it, and that module's docstring holds the reasoning (remainders, "Chinese" on the Sinitic
group, "Filipino" apart from Tagalog). NAMES is us2024's mapping cut to the labels Puerto Rico's
rows carry (sources/pr_acs.py); a label not listed here fails resolve() loudly, so a new code in
a later release is noticed instead of passing through on the US mapping unseen.

Remainders that occur in Puerto Rico: "India N.E.C." (42 people) and "Other and unspecified
languages" (77) on `other`; "Other Indo-European languages" (78) on Indo-European; "Other Bantu
languages" (32) on Bantu; "Chinese" (580, variety not named) on the Sinitic group.
"""
from us2024 import NAMES as _US

LABELS = (
    "Speak only English", "Spanish", "French", "Haitian Creole", "Italian", "Portuguese",
    "Romanian", "German", "Dutch", "Norwegian", "Russian", "Hindi", "Nepali", "India N.E.C.",
    "Other Indo-European languages", "Telugu", "Chinese", "Mandarin", "Cantonese", "Japanese",
    "Korean", "Vietnamese", "Thai", "Filipino", "Arabic", "Hebrew", "Other Bantu languages",
    "Other and unspecified languages",
)
NAMES = {k: _US[k] for k in LABELS}


def resolve(label):
    return NAMES[label]
