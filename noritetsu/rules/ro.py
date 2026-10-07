"""Romania's rules for build_model.py (build_model.country_rules lists what it reads)."""
import re

RO_TRAIN = re.compile(r"\b(?:R|R-E|RE|IR|IRN|IC|INT|EC|EN|ICN)\s?-?\s?\d{2,5}\b")


def looks_like_service(tags, name, name_en):
    # OSM Romania maps CFR Călători's, Regio's and Transferoviar's trains one relation per
    # train ("IR 1582 Constanța => București Nord", "R 3127 Arad => Brad", "Tren R9132/4:
    # Calafat - Craiova", or no name and ref "R-E 9263"); Romanian trains are known by
    # number and none runs as a branded interval line. The airport and Obor shuttles
    # (service=commuter), MÁV's "Sz: Debrecen => Valea lui Mihai" patterns and unnumbered
    # relations stay lines.
    if tags.get("service") == "commuter":
        return False
    return bool(RO_TRAIN.search(name) or RO_TRAIN.search(tags.get("ref") or ""))
