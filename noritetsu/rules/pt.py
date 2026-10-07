"""Portugal's rules for build_model.py (build_model.country_rules lists what it reads)."""
import re

PT_TRAIN = re.compile(r"^(?:CP )?(?:Alfa Pendular|Intercidades)\b|^Comboio Celta|^Train IN\b")


def looks_like_service(tags, name, name_en):
    # CP's long-distance products (Alfa Pendular, Intercidades) and the Celta to Vigo are
    # named trains; Regional, InterRegional and the Urbanos of Lisbon and Porto are lines.
    return bool(PT_TRAIN.match(name))
