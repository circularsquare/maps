"""Poland's rules for build_model.py (build_model.country_rules lists what it reads)."""
import re

# "EIP:" is written with a colon straight after the brand.
PL_TRAIN = re.compile(r"^(?:EIC|EIP|IC|TLK|EC|EN|ICE|RJX?|NJ)(?:[\s\d:]|$)"
                      r"|\bEuro(?:City|Night)\b|\bRailjet\b")


def looks_like_service(tags, name, name_en):
    # PKP Intercity maps each train as its own relation ("IC1213 Czechowicz: Warszawa
    # Wschodnia => Lublin Główny", "EIP: Kraków Główny <=> Gdynia Główna", "TLK 38190
    # Bursztyn", "EC 57 Wawel"). Polregio's IR and the regional/agglomeration lines
    # (Linia K5, S1, RE, ŁKA, SKM) carry no such brand: lines.
    return bool(PL_TRAIN.search(name))
