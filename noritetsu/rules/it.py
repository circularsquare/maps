"""Italy's rules for build_model.py (build_model.country_rules lists what it reads)."""
import re

from rules.shared import EU_TRAIN

IT_TRAIN = re.compile(r"^(?:Treno\s+|Train\s+)?(?:Freccia(?:rossa|argento|bianca)|\.?[Ii]talo\b"
                      r"|Inter[Cc]ity\b|ICN?\s?\d)")


def looks_like_service(tags, name, name_en):
    # OSM Italy maps Trenitalia's and Italo's long-distance brands route by route, with
    # no train number ("Frecciarossa (Milano Centrale → Napoli Centrale)", ".italo (Roma
    # Ostiense → Milano Porta Garibaldi)", "Frecciabianca (Roma Termini - Genova Piazza
    # Principe)", "InterCity Milano-Ventimiglia"), and the EuroCity, Nightjet, TGV and
    # European Sleeper trains one relation each. Regionale (R), Regionale Veloce (RV),
    # RegioExpress (RE), the Leonardo Express and the suburban S, FL, SFM and FM lines
    # are lines.
    return bool(IT_TRAIN.search(name) or EU_TRAIN.search(name))
