"""Saint Vincent and the Grenadines, 2012 census ethnicity by ED (sources/vc_census.py,
sources/vc.md). No language question: white Vincentians on English, everyone else on Vincentian
Creole (Glottolog vinc1243; node defined in tree.d/bb.txt, repeated and coloured in
tree.d/vc.txt). Garifuna (the census's Indigenous, 3,280) is no longer spoken in St Vincent."""
NAMES = {
    "Everyone else": "creole.english_based.vincentian",
    "White": "indoeuropean.germanic.english",
}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"vc2012: unmapped label {label!r}")
