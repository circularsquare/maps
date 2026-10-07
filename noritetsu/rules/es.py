"""Spain's rules for build_model.py (build_model.country_rules lists what it reads)."""
import re

from rules.shared import EU_TRAIN

ES_TRAIN = re.compile(r"^(?:Train\s+|Tren\s+)?(?:AVE|AV City|Alvia|ALVIA|Avlo|AVLO|Euromed|"
                      r"Intercity|InterCity|Intercités|Iryo|IRYO|Ouigo|OUIGO|Trenhotel|Talgo|TLG|"
                      r"Renfe-SNCF|TGV|IN)(?=[\s\d:]|$)")
ES_TRAIN_NETWORKS = {"Renfe AVE", "Iryo", "Ouigo España", "OUIGO España", "Renfe InterCity",
                     "TGV Europe"}


def looks_like_service(tags, name, name_en):
    # OSM Spain maps Renfe's long-distance products and the open-access operators one
    # relation per train or train pair ("Alvia 00194 Madrid → Badajoz", "AVE Madrid -
    # Sevilla", "Train Iryo ...", "Intercity 00283 Irun → A Coruña", "Renfe-SNCF 9736",
    # "Train IN: Porto - Campanhã → Vigo-Guixar"). Cercanías and Rodalies (C-1, R2),
    # Media Distancia, Regional, Avant and the FGC, Euskotren, FGV and SFM lines are lines.
    # The network decides too, but only networks that hold nothing else: "Renfe Alvia" is
    # also on a Bilbao Cercanías C-3 relation, bare "TGV" on a liO TER one. 80 of 527 train
    # relations (2026-10-02 extract).
    return bool(ES_TRAIN.match(name) or EU_TRAIN.search(name)
                or tags.get("network", "") in ES_TRAIN_NETWORKS)
