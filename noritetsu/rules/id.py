"""Indonesia's rules for build_model.py (read by build_model.country_rules): which route=train
relations are named trains rather than lines.

What OSM Indonesia has (data/proc/id, 2026-10-03: 268 route=train relations in 90 route
masters). KAI's trains are mapped one relation per train and direction, named for the train:
"Argo Bromo Anggrek: Gambir → Surabaya Pasarturi", "Taksaka: Gambir → Yogyakarta", network
"KAI". The commuter networks are mapped as lines: "Lin Bogor: Jakarta Kota → Bogor", "Commuter
Line Prameks: Kutoarjo → Yogyakarta", network "KAI Commuter"; the airport trains under "KAI
Bandara" ("Lin Srilelawangsa: Medan → Kualanamu") or KAI Commuter ("Kereta Api Bandara YIA");
Whoosh under network "Whoosh".

THE RULE (the project's: a single long-distance train is a named train; an interval product a
rider uses as a line stays a line), read through KAI's own two classes of train:

- KA antarkota (intercity), long and medium distance: each is one named train running once to
  a few times a day on a fixed timetable (Argo Bromo Anggrek twice each way, Taksaka twice,
  Gajayana once; Kaligung, Kamandaka, Joglosemarkerto, Sribilah Utama and Putri Deli a few times
  a day). Named trains. Their track counts through KAI's register lines, which cover every
  kilometre they run.
- KA lokal, komuter and bandara (local, commuter, airport): KAI Commuter's lines, the airport
  trains, and the local trains KAI files as "KA lokal" (Pangrango Bogor - Sukabumi, Siliwangi
  Cipatat - Sukabumi, Batara Kresna Purwosari - Wonogiri, Kedungsepur Semarang - Ngrombo, the
  Whoosh feeder Padalarang - Bandung, Medan's Sri Lelawangsa, West Sumatra's and Aceh's locals).
  These are how a rider travels the line, several times a day as the only service on most of
  them: lines. Whoosh (over 60 trains a day) is a line.

Relations mistagged route=train for a railway line itself ("Kertosono–Bangil railway",
"Padalarang–Kasugihan railway") are flagged named trains so they never stand as lines of their
own beside the register line.
"""
import re

LINE_NETWORKS = {"KAI Commuter", "KAI Bandara", "Whoosh", "KCIC", "Kereta Cepat Indonesia China",
                 "Railink", "LRT Jabodebek", "LRT Jakarta", "LRT Palembang", "MRT Jakarta"}
# KAI's "KA lokal" products and the airport trains, by name or ref.
LOCAL = re.compile(r"^(?:Pangrango|Siliwangi|Batara Kresna|Kedungsepur|Feeder KCJB|Feeder Whoosh|"
                   r"Kereta Api BIAS|BIAS|Bandara|Kereta Api Bandara|Sri ?Lelawangsa|Lin |KRL|"
                   r"Commuter Line|Cut Meutia|Lembah Anai|Minangkabau Ekspres|Sibinuang|"
                   r"Pariaman Ekspres|Kuala Stabas|Lokal|Whoosh)", re.I)
LOCAL_REF = {"PG", "SW", "BK", "KS", "KC", "AS"}
NOT_A_TRAIN = re.compile(r"\brailway$|^Jalur ", re.I)
# A route master with no tags of its own ("Cikuray": name and service only) is a named train
# when every one of its routes is.
SERVICE_IF_ALL_ROUTES_ARE = True


def looks_like_service(tags, name, name_en):
    name = name or ""
    if NOT_A_TRAIN.search(name):
        return True
    if tags.get("service") in ("long_distance", "night"):
        return True
    net = " ".join(tags.get(k) or "" for k in ("network", "operator"))
    if any(n in net for n in LINE_NETWORKS):
        return False
    if LOCAL.search(name) or (tags.get("ref") or "") in LOCAL_REF and tags.get("network") == "KAI":
        return False
    # every other KAI train is an intercity named train
    return "KAI" in net or "Kereta Api Indonesia" in net
