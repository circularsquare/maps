"""North Korea: no language question; one label, everyone on Korean (sources/kp_pop.py,
sources/kp.md).
"""
NAMES = {"Korean": "koreanic.korean"}


def resolve(label):
    if label in NAMES:
        return NAMES[label]
    raise KeyError(f"kp2008: unmapped label {label!r}")
