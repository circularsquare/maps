"""India's rules for build_model.py (country_rules): which route=train relations are named
trains rather than lines.

What OSM India has (single relations read through the OSM API on 2026-10-03, about 20 calls;
in_sources.md lists them). Every Indian Railways train has a five-digit number, and OSM maps
trains one relation per train and direction, under three naming habits:
  "Train Rajdhani Express 22222: Hazrat Nizamuddin → Mumbai CSMT"  (ref 22222, network
      "Rajdhani Express"; likewise "Train SuperFast Express 12701: Mumbai CSMT → Hyderabad",
      "Train Express 15017: Mumbai LTT → Gorakhpur", network "Express")
  "Train 11021 Chalukya Express: Dadar → Tirunelveli"  (ref 11021, no network)
  "20705 Mumbai CSMT Vande Bharat Express"  (ref 20705, network "Vande Bharat Express")
  "Kalka Shatabdi Express: New Delhi -> Kalka"  (ref 12011; stop members only, no ways)
These are named trains. The lines a rider uses as lines are the suburban networks:
  Mumbai: "Central Line: Mumbai CSMT → Kalyan Junction (Fast)", ref C, network "Mumbai
      Suburban Railway", passenger=suburban, in a route_master "Central Line"; "Vasai Road-Diva
      Line (main): Vasai Road => Diva", ref MEMU, same network
  Chennai: "Chennai Beach - Tambaram suburban line", ref "South Line", network "Chennai
      Suburban Railway", service=commuter
  Hyderabad: "Hyderabad MMTS: Falaknuma → Lingampalli", ref MMTS, network "Hyderabad MMTS",
      passenger=suburban; it also carries the train's number in nat_ref (47218), so the
      suburban test comes first
and they stay lines whatever number they carry. Kolkata's, Bengaluru's and Pune's locals and
the RRTS (Namo Bharat) are caught by the same words. Anything else with a five-digit train
number, a train brand (Rajdhani, Shatabdi, Vande Bharat, Duronto...), "Train " in front, or a
luxury or international train's name is a named train; a relation with none of these is a
line.

Indian Railways' own lines are register lines (in_register.py), so a named train's track
counts through them.
"""
import re

SUBURBAN = re.compile(r"suburban|\bMMTS\b|\blocal\b|circular railway|\bRRTS\b|Namo Bharat|"
                      r"\bEMU\b(?!.*\d{5})", re.I)
TRAIN_NO = re.compile(r"(?<![\d/])\d{5}(?![\d/])")
TRAIN_BRAND = re.compile(
    r"^Train\b|\b(?:Rajdhani|Shatabdi|Duronto|Vande Bharat|Tejas|Humsafar|Garib Rath|"
    r"Gatimaan|Antyodaya|Amrit Bharat|Sampark Kranti|Double Decker|Uday|Yuva|Suvidha|"
    r"Jan Sadharan|Mahamana|Kavi Guru|Vivek|Rajya Rani|Intercity|Superfast|Express|Mail|"
    r"Passenger|MEMU|DEMU)\b"
    r"|Palace on Wheels|Maharajas'? Express|Deccan Odyssey|Golden Chariot|"
    r"Royal Rajasthan on Wheels|Buddhist Circuit|Bharat Gaurav|"
    r"\b(?:Maitree|Bandhan|Mitali|Samjhauta|Thar) Express\b", re.I)


def looks_like_service(tags, name, name_en):
    text = " ".join(t for t in (name, name_en, tags.get("network"), tags.get("ref")) if t)
    if SUBURBAN.search(text):
        return False
    if TRAIN_NO.search(tags.get("ref") or "") or TRAIN_NO.search(name or ""):
        # a train by its number, even where OSM tags it passenger=suburban (Bihar's DEMUs
        # "Train 75721", "Train 55749"): only a suburban network's name above keeps a
        # numbered relation a line (Hyderabad's MMTS carries its number in nat_ref)
        return True
    if tags.get("passenger") == "suburban" or tags.get("service") == "commuter":
        return False
    return bool(TRAIN_BRAND.search(name or "") or TRAIN_BRAND.search(tags.get("network") or ""))
