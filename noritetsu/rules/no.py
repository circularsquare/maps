"""Norway's rules for build_model.py (country_rules): which route=train relations are named
trains rather than lines.

What OSM Norway has (tags of all 105 route=train relations in the 2026-10-03 extract): Vy's,
SJ Norge's and Go-Ahead's products by Bane NOR's line number, as route_masters with variants
("F4 Oslo S - Bergen, Bergensbanen", "F5 Oslo S - Stavanger S, Sørlandsbanen", "F6 Oslo S -
Trondheim S, Dovrebanen", "F7 Trondheim S - Bodø, Nordlandsbanen", "RE11 Skien - Eidsvoll",
"R40 Bergen - Myrdal", "L1 Spikkestad - Lillestrøm", Flytoget's "FLY1"/"FLY2", unnamed ones with
ref only: R14, R31, RE30, "52", "50", "L4"); the cross-border "F1;70" Oslo - Stockholm (Vy Tåg
and SJ, two or three a day), Norrtåg's "R71 Trondheim S => Storlien" and Vy Tåg's "F8 / 30"
Luleå - Narvik (daily); Flåmsbana as "R45 Myrdal→Flåm" tagged service=tourism (it runs every
day, all year: a line); and SJ's "Nattåg 93: Stockholm => Narvik" (service=night).

The F-lines are interval products running several times a day, each night train among them
filed as a variant of the same route_master (OSM has no separate Norwegian night-train
relation), so they are lines, as are the cross-border day trains (Sweden's rule keeps SJ's and
Vy Tåg's tables as lines too, and the relations are the same ones).

Named trains: anything tagged service=night or car, the night trains by name (Nattåg, Nattog,
Snälltåget, EuroNight), and two relations that are no scheduled passenger service at all:
"Kirkenes–Bjørnevatnbanen" (service=industrial, the ore railway) and "Reli Safari"
(service=tourism with no line number, an amusement-park train). Flagged, they have no
percentage of their own and the totals leave them out.
"""
import re

NO_TRAIN = re.compile(
    r"\bNatt(?:åg|og)\b|\bSnälltåget\b|\bEuro(?:City|Night)\b|\bNightjet\b"
    r"|\bEuropean Sleeper\b|^(?:Train\s+)?(?:EC|EN)\s?\d")
NO_TRAIN_OPERATORS = {"Snälltåget", "Snälltåget AB"}


def looks_like_service(tags, name, name_en):
    service = set((tags.get("service") or "").split(";"))
    if service & {"night", "car", "industrial"}:
        return True
    if "tourism" in service and not tags.get("ref"):
        return True
    if tags.get("operator") in NO_TRAIN_OPERATORS:
        return True
    return bool(NO_TRAIN.search(name or "") or NO_TRAIN.search(tags.get("ref") or ""))
