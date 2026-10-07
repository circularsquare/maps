"""South Africa: a hand-written list of the passenger lines, written into rinf.py's input format
(as balkans_register.py does for the Balkans), so rinf.py traces it over OSM track, matches
stops, merges sections and names lines.

    python za_register.py --dry          # convert without writing: traced km per line, the
                                          # OSM stations each trace passes, ways two lines share
    python za_register.py --convert      # write data/raw/rinf/za/{sections,points,names}.json
    python build_model.py --region za --register za_register:data/raw/rinf/za

`--register za_register:data/raw/rinf/za` converts and then runs rinf.build on the result, then
sets each line's network, operator and colour (LINES). The settings rinf.py reads are in
rinf_countries/za.py (`country_conf` below); za_sources.md has the sources, what runs and the
checks.

WHY A HAND LIST. South Africa has no open line register. OSM names only 22% of its main-line
track (`python probe_kr_ways.py --region za`), so Korea's named-track recipe does not work; it
does have 190 infrastructure relations ("Cape Town–De Aar", "Salt River–Simon's Town",
"Kaalfontein–Leralla"), the Transnet and PRASA lines by their ends, and complete Metrorail and
Gautrain route relations (133, with stops). So each register line here is a run of track
between two places: its ends and enough stations between to pin the path, traced by rinf.py
over OSM track, preferring the ways of the infrastructure relation(s) it names (`own`). Every
OSM station a passenger route stops at that lies on a traced section becomes a stop
(rinf.py's `osm_stops`).

THE LINE UNIT is the track, cut so that every piece of passenger track is on exactly one line
and each line is wholly running or wholly not: the OSM infrastructure relation where its
passenger part is one piece ("Salt River–Simon's Town", "Kraaifontein–Malmesbury"), else that
relation cut where service stops ("Eerste River–Du Toit" running and "Du Toit–Muldersvlei"
not, both on OSM's "Eerste River–Muldersvlei"). Metrorail's own lines (Southern Line,
Northern Line, the Gauteng colour lines) are services over several of these and stay OSM
lines, as JR's services do over Japan's legal lines.

WHAT IS BUILT (za_sources.md has the evidence for each):
  - Metrorail's four regions, every corridor its OSM routes cover. Running, by PRASA's 2026
    corridor table (Parliament's ATC report of 10 June 2026), its timetables and the press:
    greyed (`suspended`) where no train runs today (Springs, Nigel, Daveyton, Oberholzer via
    Midway, both Vereeniging lines south of Lenasia and Elsburg, the Jikeleza loop, Durban's
    North Coast and Bluff lines, the South Coast beyond Winklespruit, Du Toit - Muldersvlei).
  - Gautrain is left to its OSM route relations (OSM lines): its Park station's OSM record
    sits on Metrorail's platforms, 120 m off Gautrain's track, so no trace starts from it.
  - The main line Pretoria - Johannesburg - Kimberley - De Aar - Cape Town: the Blue Train
    (about weekly each way) and Rovos Rail run it, so it counts (named trains; their track
    counts through these lines). Shosholoza Meyl's other routes, suspended since October
    2024 with a 2027 return announced (Johannesburg - Durban, - East London, - Musina), are
    greyed. Its Gqeberha and Komatipoort routes have not run since 2020 and are not built.
  - No passenger train crosses a border (Komatipoort - Ressano Garcia: CFM's Maputo trains
    turn at Ressano Garcia; nothing from Musina, Mahikeng or the Lesotho branches), so no
    border points.
"""
import argparse
import json
import math
import os
import pickle
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
CC = "za"
OUT = ROOT / "data" / "raw" / "rinf" / CC

PRASA = "PRASA"
TFR = "Transnet Freight Rail"
WC, GP, KZN, EC = ("Metrorail Western Cape", "Metrorail Gauteng", "Metrorail KwaZulu-Natal",
                   "Metrorail Eastern Cape")


def key(s):
    """One key for a station name: accents and case off, "Station" off ("Polokwane Train
    Station", "Johannesburg Park Station")."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    s = re.sub(r"\b(train\s+)?station\b", " ", s)
    return re.sub(r"[^0-9a-z]", "", s)


def dist_m(lon1, lat1, lon2, lat2):
    dx = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2)) * 111320
    return math.hypot(dx, (lat2 - lat1) * 110570)


# ============================================================== the line list
#
# A point is "Name@lon,lat": the OSM rail station of that name nearest the coordinate (within
# PLACE_M), or with none there a junction at the coordinate under that name. Coordinates are
# the stations' own (OSM, 2026-10 extract), so a name two places share (Centurion, Pretoria,
# Rosebank, Philippi, Rhodesfield) finds the right one.

PLACE_M = 1500


def L(lid, name, pts, network, operator, colour, own=(), suspended=False, name_en="", note=""):
    return {"id": lid, "name": name, "name_en": name_en, "pts": list(pts), "network": network,
            "im": operator, "colour": colour, "own": list(own), "suspended": suspended,
            "note": note}


LINES = [
    # ---------------------------------------------------------- Metrorail Western Cape
    # Running (cttrains.co.za and Metrorail's 2026 timetables): Southern Line to Simon's Town,
    # Cape Flats Line, Central Line to Kapteinsklip, Chris Hani and Bellville, Northern Line to
    # Bellville (both ways), Strand, Wellington and Stellenbosch (trains turn at Du Toit;
    # Koelenhof has been closed since 2020), Malmesbury (peak only), Worcester (one train a
    # day each way, weekdays).
    L("ct-bv-mutual", "Cape Town–Bellville (via Mutual)",
      ["Cape Town@18.4259,-33.9224", "Salt River@18.4670,-33.9273", "Maitland@18.4884,-33.9246",
       "Mutual@18.5142,-33.9212", "Goodwood@18.5479,-33.9144", "Parow@18.5862,-33.9097",
       "Bellville@18.6273,-33.9068"], WC, PRASA, "#00a650",
      own=["Cape Town–Bellville (Old Main)"]),
    L("ct-deaar", "Cape Town–De Aar",
      ["Cape Town@18.4266,-33.9217", "Ysterplaat@18.4768,-33.9200",
       "Century City@18.5123,-33.9009", "Monte Vista@18.5476,-33.8918",
       "Bellville@18.6258,-33.9059", "Kraaifontein@18.7249,-33.8464",
       "Muldersvlei@18.8284,-33.8304", "Paarl@18.9647,-33.7648", "Wellington@18.9920,-33.6345",
       "Gouda@19.0416,-33.3015", "Wolseley@19.1961,-33.4130", "Worcester@19.4411,-33.6396",
       "Touwsrivier@20.0366,-33.3382", "Matjiesfontein@20.5809,-33.2315",
       "Laingsburg@20.8603,-33.1979", "Prince Albert Road@21.6867,-32.9850",
       "Beaufort West@22.5769,-32.3518", "Hutchinson@23.1888,-31.4983",
       "De Aar@24.0139,-30.6504"], WC, TFR, "#00a650", own=["Cape Town–De Aar"],
      note="Metrorail to Worcester; the Blue Train and Rovos Rail beyond"),
    L("sr-simonstown", "Salt River–Simon's Town",
      ["Salt River@18.4638,-33.9272", "Observatory@18.4718,-33.9389",
       "Wynberg@18.4712,-34.0050", "Heathfield@18.4656,-34.0460", "Retreat@18.4630,-34.0602",
       "Muizenberg@18.4678,-34.1098", "Fish Hoek@18.4323,-34.1378",
       "Simon's Town@18.4254,-34.1862"], WC, PRASA, "#ed1c24", own=["Salt River–Simon's Town"],
     ),
    L("maitland-retreat", "Maitland–Heathfield",
      ["Maitland@18.4885,-33.9245", "Pinelands@18.4907,-33.9408", "Athlone@18.5017,-33.9641",
       "Ottery@18.4947,-34.0145", "Southfield@18.4808,-34.0333", "Heathfield@18.4656,-34.0460"],
      WC, PRASA, "#8c493a", own=["Maitland–Retreat"]),
    L("ysterplaat-langa", "Ysterplaat–Langa",
      ["Ysterplaat@18.4769,-33.9201", "Mutual@18.5146,-33.9220", "Langa@18.5313,-33.9393"],
      WC, PRASA, "#33bef3", own=["Ysterplaat–Langa"]),
    L("pinelands-bellville", "Pinelands–Bellville",
      ["Pinelands@18.4907,-33.9397", "Langa@18.5286,-33.9387", "Bonteheuwel@18.5502,-33.9419",
       "Lavistown@18.5840,-33.9434", "Belhar@18.6092,-33.9395", "Unibell@18.6285,-33.9371",
       "Pentech@18.6462,-33.9348", "Sarepta@18.6612,-33.9265", "Bellville@18.6273,-33.9068"],
      WC, PRASA, "#33bef3", own=["Pinelands–Bellville"]),
    L("bonteheuwel-kapteinsklip", "Bonteheuwel–Kapteinsklip",
      ["Bonteheuwel@18.5502,-33.9421", "Netreg@18.5635,-33.9527", "Nyanga@18.5599,-33.9938",
       "Philippi@18.5875,-34.0132", "Mitchell's Plain@18.6191,-34.0503",
       "Kapteinsklip@18.6208,-34.0670"], WC, PRASA, "#33bef3",
      own=["Bonteheuwel–Mitchell's Plain"]),
    L("philippi-chrishani", "Philippi–Chris Hani",
      ["Philippi@18.5845,-34.0134", "Stock Road@18.6061,-34.0142", "Nolungile@18.6500,-34.0169",
       "Khayelitsha@18.6711,-34.0487", "Chris Hani@18.7101,-34.0547"], WC, PRASA, "#33bef3",
      own=["Philippi–Khayelitsha"]),
    L("bellville-strand", "Bellville–Strand",
      ["Bellville@18.6273,-33.9068", "Kuils River@18.6776,-33.9336",
       "Eerste River@18.7308,-34.0005", "Faure@18.7475,-34.0263", "Firgrove@18.7927,-34.0556",
       "Somerset West@18.8415,-34.0843", "Van der Stel@18.8524,-34.0950",
       "Strand@18.8319,-34.1154"], WC, PRASA, "#00a650",
      own=["Bellville–Protem", "Van der Stel–Strand"]),
    L("eersteriver-dutoit", "Eerste River–Du Toit",
      ["Eerste River@18.7308,-34.0005", "Lynedoch@18.7713,-33.9813",
       "Vlottenburg@18.8000,-33.9620", "Stellenbosch@18.8496,-33.9388",
       "Du Toit@18.8541,-33.9239"], WC, PRASA, "#00a650", own=["Eerste River–Muldersvlei"]),
    L("dutoit-muldersvlei", "Du Toit–Muldersvlei",
      ["Du Toit@18.8541,-33.9239", "Koelenhof@18.8196,-33.8739",
       "Muldersvlei@18.8278,-33.8317"], WC, PRASA, "#00a650", own=["Eerste River–Muldersvlei"],
      suspended=True, note="trains turn at Du Toit; Koelenhof closed since 2020"),
    L("kraaifontein-malmesbury", "Kraaifontein–Malmesbury",
      ["Kraaifontein@18.7233,-33.8476", "Klipheuwel@18.7007,-33.6993",
       "Kalbaskraal@18.6475,-33.5723", "Malmesbury@18.7233,-33.4677"], WC, PRASA, "#99d420",
      own=["Kraaifontein–Malmesbury"]),

    # ---------------------------------------------------------- Metrorail Gauteng
    # PRASA's Gauteng corridor table (ATC, 10 June 2026): running Mabopane - Pretoria,
    # Saulsville - Pretoria, Pienaarspoort - Pretoria, Germiston - Leralla, Germiston -
    # Johannesburg, Naledi - Johannesburg, Pretoria - Kempton Park, De Wildt - Pretoria,
    # Mabopane - Belle Ombre, Hercules - Koedoespoort, Johannesburg - Midway (Lenasia since
    # June 2026), Germiston - Kwesine, Johannesburg - Randfontein; Vereeniging - Union "no
    # service"; and "no train services on the Daveyton, Springs, Nigel, Oberholzer and
    # Jikeleza lines, including Midway to Vereeniging".
    L("pta-saulsville", "Pretoria–Saulsville",
      ["Pretoria@28.1906,-25.7604", "Pretoria-Wes@28.1666,-25.7562", "Rebecca@28.1523,-25.7589",
       "Kalafong@28.0874,-25.7602", "Saulsville@28.0606,-25.7637"], GP, PRASA, "#00BAF1",
      own=["Pretoria–Saulsville"]),
    L("pta-dewildt", "Pretoria–De Wildt",
      ["Pretoria@28.1906,-25.7604", "Pretoria-Wes@28.1666,-25.7562", "Golf@28.1608,-25.7405",
       "Hercules@28.1672,-25.7246", "Wonderboom@28.1823,-25.6800",
       "Pretoria-Noord@28.1820,-25.6715", "Winternest@28.1286,-25.6468",
       "Rosslyn@28.0951,-25.6378", "Ga-Rankuwa@27.9922,-25.6195", "De Wildt@27.9445,-25.6247"],
      GP, PRASA, "#F58220", own=["Pretoria–De Wildt"]),
    L("winternest-mabopane", "Winternest–Mabopane",
      ["Winternest@28.1286,-25.6468", "Akasiaboom@28.1071,-25.6238",
       "Kopanong@28.0905,-25.5810", "Soshanguve@28.0829,-25.5199", "Mabopane@28.0899,-25.4955"],
      GP, PRASA, "#F58220", own=["Winternest–Mabopane"]),
    L("belleombre-technikonrant", "Belle Ombre–Technikon Rant",
      ["Belle Ombre@28.1787,-25.7374", "Technikon Rant@28.16714,-25.73277"], GP, PRASA, "#F58220",
      own=["Pretoria–Beit Bridge"]),
    L("hercules-koedoespoort", "Hercules–Koedoespoort",
      ["Hercules@28.1675,-25.7243", "Capital Park@28.1896,-25.7220", "Villeria@28.2298,-25.7198",
       "Queenswood@28.2531,-25.7223", "Koedoespoort@28.2798,-25.7264"], GP, PRASA, "#ED038C",
      own=["Hercules–Koedoespoort"]),
    L("pta-pienaarspoort", "Pretoria–Pienaarspoort",
      ["Pretoria@28.1906,-25.7604", "Walker Street@28.2126,-25.7605",
       "Hartbeesspruit@28.2411,-25.7466", "Koedoespoort@28.2798,-25.7264",
       "Silverton@28.2970,-25.7273", "Eerste Fabrieke@28.3612,-25.7219",
       "Pienaarspoort@28.4278,-25.7352", "Panpoort@28.4428,-25.7345"], GP, PRASA, "#6E2527",
      own=["Pretoria–Komatipoort"]),
    L("germiston-pta", "Germiston–Pretoria",
      ["Germiston@28.1679,-26.2101", "Elandsfontein@28.2050,-26.1674",
       "Kempton Park@28.2272,-26.1083", "Kaalfontein@28.2547,-26.0355",
       "Olifantsfontein@28.2357,-25.9641", "Irene@28.2243,-25.8748",
       "Centurion@28.2109,-25.8350", "Fonteine@28.1929,-25.7834", "Pretoria@28.1906,-25.7604"],
      GP, TFR, "#0072BC", own=["Germiston–Pretoria"]),
    L("kaalfontein-leralla", "Kaalfontein–Leralla",
      ["Kaalfontein@28.2547,-26.0355", "Tembisa@28.2314,-26.0096", "Leralla@28.1963,-26.0293"],
      GP, PRASA, "#0072BC", own=["Kaalfontein–Leralla"]),
    L("germiston-kimberley", "Germiston–Kimberley",
      ["Germiston@28.1679,-26.2101", "Driehoek@28.1493,-26.2136",
       "George Goch@28.0800,-26.2079", "Johannesburg Park@28.0422,-26.1981",
       "Braamfontein@28.0235,-26.1978", "Langlaagte@27.9916,-26.2017",
       "Florida@27.9144,-26.1768", "Roodepoort@27.8702,-26.1592",
       "Krugersdorp@27.7708,-26.1090", "Randfontein@27.6980,-26.1814", "Bank@27.5129,-26.3112",
       "Oberholzer@27.3933,-26.3426", "Potchefstroom@27.0847,-26.7117",
       "Klerksdorp@26.6704,-26.8697", "Bloemhof@25.6004,-27.6429",
       "Christiana@25.1632,-27.9009", "Warrenton@24.8676,-28.1156",
       "Kimberley@24.7699,-28.7353"], GP, TFR, "#00A54F",
      own=["Fourteen Streams–Germiston", "De Aar–Fourteen Streams"],
      note="Metrorail Germiston - Johannesburg - Randfontein; the Blue Train and Rovos Rail. "
           "The Kimberley line leaves the Mafikeng line short of Fourteen Streams (Veertien "
           "Strome), which no passenger train calls at"),
    L("kimberley-deaar", "Kimberley–De Aar",
      ["Kimberley@24.7699,-28.7353", "Orange River@24.2072,-29.6689", "De Aar@24.0139,-30.6504"],
      "", TFR, "#1c3f94", own=["De Aar–Fourteen Streams"],
      note="the Blue Train and Rovos Rail"),
    L("langlaagte-lenasia", "Langlaagte–Lenasia",
      ["Langlaagte@27.9916,-26.2017", "Croesus@27.9713,-26.2014",
       "New Canada@27.9421,-26.2147", "Orlando@27.9168,-26.2389",
       "Nancefield@27.9064,-26.2516", "Midway@27.8508,-26.2933", "Lenasia@27.8233,-26.3194"],
      GP, PRASA, "#00652E", own=["Langlaagte–Vereeniging"]),
    L("lenasia-vereeniging", "Lenasia–Vereeniging",
      ["Lenasia@27.8233,-26.3194", "Lawley@27.8308,-26.3551", "Grasmere@27.8627,-26.4265",
       "Residensia@27.8881,-26.5377", "Houtheuwel@27.8534,-26.6032",
       "Kleigrond@27.8714,-26.6236", "Leeuhof@27.9061,-26.6550",
       "Vereeniging@27.9351,-26.6735"], GP, PRASA, "#00652E", own=["Langlaagte–Vereeniging"],
      suspended=True, note="Lawley - Vereeniging closed; Houtheuwel expected June 2027"),
    L("newcanada-naledi", "New Canada–Naledi",
      ["New Canada@27.9435,-26.2139", "Mzimhlope@27.9230,-26.2232", "Dube@27.8928,-26.2338",
       "Inhlazane@27.8635,-26.2496", "Naledi@27.8233,-26.2584"], GP, PRASA, "#00652E",
      own=["New Canada–Naledi"]),
    L("midway-bank", "Midway–Bank",
      ["Midway@27.8508,-26.2933", "Waterworks@27.8160,-26.3024",
       "Westonaria@27.6518,-26.3128", "Bank@27.5129,-26.3112"], GP, PRASA, "#F9CC3E",
      own=["Midway–Bank"], suspended=True, note="the Oberholzer line: no service"),
    L("germiston-elsburg", "Germiston–Elsburg",
      ["Germiston@28.1679,-26.2101", "Kutalo@28.1894,-26.2175", "Elsburg@28.1920,-26.2430"],
      GP, PRASA, "#EE1D23"),
    L("elsburg-kwesine", "Elsburg–Kwesine",
      ["Elsburg@28.1920,-26.2430", "Wadeville@28.1817,-26.2605", "Katlehong@28.1614,-26.3075",
       "Pilot@28.1524,-26.3409", "Kwesine@28.1527,-26.3651"], GP, PRASA, "#EE1D23",
      own=["Elsburg–Kwesine"]),
    L("president-elsburg", "President–Elsburg",
      ["President@28.1594,-26.2139", "Germiston West@28.1622,-26.2212",
       "Germiston South@28.1648,-26.2276", "Webber@28.1760,-26.2338",
       "Park Hill@28.1866,-26.2358", "Elsburg@28.1920,-26.2430"], GP, PRASA, "#EE1D23",
      suspended=True, note="Kwesine trains run via Kutalo (15.5 km in PRASA's table)"),
    L("elsburg-vereeniging", "Elsburg–Vereeniging",
      ["Elsburg@28.1932,-26.2412", "Dallas@28.1844,-26.2482", "Union@28.1627,-26.2705",
       "Natalspruit@28.1431,-26.3044", "Angus@28.1264,-26.3503", "Kliprivier@28.0869,-26.4205",
       "Meyerton@28.0096,-26.5556", "Redan@27.9670,-26.6274", "Vereeniging@27.9352,-26.6742"],
      GP, PRASA, "#90268F", suspended=True,
      note="Vereeniging - Union: no service (PRASA, 2026); Shosholoza Meyl's East London "
           "trains also ran this way"),
    L("germiston-springs", "Germiston–Springs",
      ["Germiston@28.1679,-26.2101", "Delmore@28.1989,-26.2064", "Boksburg@28.2412,-26.2191",
       "Dunswart@28.2855,-26.2096", "Benoni@28.3111,-26.1977", "Brakpan@28.3611,-26.2405",
       "Springs@28.4378,-26.2500"], GP, PRASA, "#00A54F", own=["Germiston–Ermelo"],
      suspended=True),
    L("dunswart-daveyton", "Dunswart–Daveyton",
      ["Dunswart@28.2855,-26.2096", "Northmead@28.3181,-26.1810", "Van Ryn@28.3445,-26.1730",
       "Daveyton@28.4246,-26.1564"], GP, PRASA, "#555555",
      own=["Dunswart–Daveyton", "Daveyton Branch Line"], suspended=True),
    L("springs-nigel", "Springs–Nigel",
      ["Springs@28.4378,-26.2500", "Daggafontein@28.4694,-26.2922",
       "Dunnottar@28.4416,-26.3493", "Nigel@28.4485,-26.4226"], GP, PRASA, "#6E2527",
      own=["Springs - Nigel"], suspended=True),
    L("germiston-newcanada", "Germiston–New Canada (via Booysens)",
      ["Germiston@28.1677,-26.2098", "India@28.1608,-26.2184", "Refinery@28.1503,-26.2213",
       "Jupiter@28.1187,-26.2200", "Kaserne West@28.0703,-26.2201",
       "Booysens@28.0402,-26.2252", "Crown@28.0138,-26.2225", "New Canada@27.9426,-26.2146"],
      GP, PRASA, "#a0a0a0", own=["Driehoek–New Canada"], suspended=True,
      note="part of the Jikeleza loop: no service"),
    L("georgegoch-kasernewest", "George Goch–Kaserne West",
      ["George Goch@28.0790,-26.2081", "Benrose@28.0859,-26.2173",
       "Kaserne West@28.0687,-26.2205"], GP, PRASA, "#00652E", suspended=True),
    L("booysens-faraday", "Booysens–Faraday",
      ["Booysens@28.0389,-26.2247", "Village Main@28.0449,-26.2190",
       "Faraday@28.0445,-26.2121"], GP, PRASA, "#00652E", suspended=True),
    L("crown-westgate", "Crown–Westgate",
      ["Crown@28.0147,-26.2227", "Westgate@28.0337,-26.2116"], GP, PRASA, "#00652E",
      suspended=True),

    # ---------------------------------------------------------- Metrorail KwaZulu-Natal
    # GroundUp, 1 September 2026: KwaMashu - Durban - Umlazi runs (17 return trips a day);
    # the South Coast line to Winklespruit (the rest waits for the Illovo bridge); Chatsworth
    # (Merebank - Crossmoor) and the Old Main (Rossburgh - Pinetown) on one track; the Bluff
    # line closed; only sections of the North Coast and KwaMashu lines run. nexttrain.co.za
    # lists Berea Road - Bridge City, Durban - Cato Ridge, - Crossmoor, - Pinetown, - Umlazi,
    # - Winklespruit.
    L("durban-catoridge", "Durban–Cato Ridge",
      ["Durban@31.0225,-29.8448", "Dalbridge@31.0064,-29.8680", "Rossburgh@30.9809,-29.8986",
       "Mount Vernon@30.9336,-29.9010", "Shallcross@30.8791,-29.8815",
       "Mariannhill@30.8298,-29.8680", "Hammarsdale@30.6580,-29.8017",
       "Cato Ridge@30.5881,-29.7318"], KZN, PRASA, "#00BAF1", own=["Durban–Ladysmith"],
     ),
    L("rossburgh-pinetown", "Rossburgh–Pinetown",
      ["Rossburgh@30.9809,-29.8986", "Sea View@30.9612,-29.9003", "Malvern@30.9193,-29.8796",
       "Northdene@30.8856,-29.8633", "Pinetown@30.8578,-29.8179"], KZN, PRASA, "#6E2527",
      own=["Rossburgh–Cato Ridge (Old Main)"]),
    L("rossburgh-winklespruit", "Rossburgh–Winklespruit",
      ["Rossburgh@30.9809,-29.8991", "Clairwood@30.9748,-29.9118", "Merebank@30.9589,-29.9436",
       "Reunion@30.9410,-29.9636", "Isipingo@30.9285,-29.9841",
       "Amanzimtoti@30.8821,-30.0569", "Winklespruit@30.8567,-30.0986"], KZN, PRASA, "#0072BC",
      own=["Durban–Port Shepstone"]),
    L("winklespruit-kelso", "Winklespruit–Kelso",
      ["Winklespruit@30.8567,-30.0986", "Umkomaas@30.8026,-30.2035",
       "Scottburgh@30.7590,-30.2848", "Kelso@30.7142,-30.3627"], KZN, PRASA, "#0072BC",
      own=["Durban–Port Shepstone"], suspended=True,
      note="repairs wait on the Illovo bridge"),
    L("reunion-umlazi", "Reunion–Umlazi",
      ["Reunion@30.9410,-29.9636", "Zwelethu@30.9165,-29.9634", "Lindokuhle@30.8881,-29.9591",
       "Umlazi@30.8659,-29.9539"], KZN, PRASA, "#EE1D23", own=["Reunion–Umlazi"]),
    L("merebank-crossmoor", "Merebank–Crossmoor",
      ["Merebank@30.9593,-29.9433", "Havenside@30.9365,-29.9261", "Chatsglen@30.8839,-29.9070",
       "Crossmoor@30.8608,-29.8983"], KZN, PRASA, "#F9CC3E", own=["Merebank–Crossmoor"],
     ),
    L("clairwood-wests", "Clairwood–Wests",
      ["Clairwood@30.9748,-29.9118", "Jacobs@30.9859,-29.9241", "Wentworth@30.9964,-29.9180",
       "Fynnlands@31.0282,-29.8950", "Wests@31.0537,-29.8787"], KZN, PRASA, "#A6CE39",
      own=["Clairwood–Wests"], suspended=True),
    L("durban-bridgecity", "Durban–Bridge City",
      ["Durban@31.0225,-29.8448", "Moses Mabhida@31.0283,-29.8267", "Umgeni@31.0273,-29.8163",
       "Temple@31.0060,-29.8037", "Effingham@31.0023,-29.7721", "Duff's Road@31.0045,-29.7429",
       "kwaMashu@30.9732,-29.7505", "Bridge City@30.9870,-29.7272"], KZN, PRASA, "#EE1D23",
      own=["Umgeni–Duff's Road", "Duff's Road–KwaMashu"]),
    L("umgeni-kwadukuza", "Umgeni–KwaDukuza",
      ["Umgeni@31.0273,-29.8163", "Briardene@31.0140,-29.7969", "Avoca@31.0194,-29.7591",
       "Duff's Road@31.0053,-29.7420", "Phoenix@31.0094,-29.7189", "Verulam@31.0490,-29.6471",
       "Tongaat@31.1265,-29.5598", "Umhlali@31.2195,-29.4765", "KwaDukuza@31.2951,-29.3429"],
      KZN, PRASA, "#90268F", own=["Durban–Golela"], suspended=True,
      note="no service since the 2022 floods; a Transnet agreement pending"),

    # ---------------------------------------------------------- Metrorail Eastern Cape
    L("kugompo-ntabozuko", "KuGompo City–Ntabozuko",
      ["KuGompo City@27.9076,-33.0167", "Mdantsane@27.7816,-32.9316",
       "Ntabozuko@27.5829,-32.8825"], EC, PRASA, "#F58220", own=["East London–Springfontein"],
      name_en="East London–Berlin"),
    L("gqeberha-kariega", "Gqeberha–Kariega",
      ["Gqeberha@25.6246,-33.9604", "North End@25.6106,-33.9474", "Swartkops@25.6015,-33.8688",
       "Despatch@25.4704,-33.7979", "Kariega@25.4087,-33.7716"], EC, PRASA, "#F9CC3E",
      own=["Port Elizabeth–De Aar", "Swartkops–Klipplaat"],
      name_en="Port Elizabeth–Uitenhage", note="one train each way on weekdays"),

    # ---------------------------------------------------------- Shosholoza Meyl (suspended)
    # The four routes that ran in 2022-2024; suspended since October 2024 (GroundUp, 23 March
    # 2026; PRASA plans their return in 2027).
    L("union-umnambithi", "Union–uMnambithi",
      ["Union@28.16271,-26.27045", "Heidelberg@28.37026,-26.50958", "Standerton@29.2347,-26.9541",
       "Newcastle@29.9814,-27.7607", "uMnambithi@29.7861,-28.5580"], "", TFR, "#8a6d3b",
      own=["Ladysmith–Germiston"], suspended=True, name_en="Union–Ladysmith"),
    L("umnambithi-catoridge", "uMnambithi–Cato Ridge",
      ["uMnambithi@29.7861,-28.5580", "Pietermaritzburg@30.3684,-29.6107",
       "Cato Ridge@30.5881,-29.7318"], "", TFR, "#8a6d3b", own=["Durban–Ladysmith"],
      suspended=True, name_en="Ladysmith–Cato Ridge"),
    L("vereeniging-springfontein", "Vereeniging–Springfontein",
      ["Vereeniging@27.9355,-26.6743", "Sasolburg@27.8665,-26.8546", "Koppies@27.5692,-27.2376",
       "Kroonstad@27.2343,-27.6594", "Hennenman@27.0243,-27.9750",
       "Virginia@26.8927,-28.1288", "Theunissen@26.7191,-28.4046",
       "Brandfort@26.4720,-28.6986", "Bloemfontein@26.2265,-29.1190",
       "Edenburg@25.9302,-29.7354", "Springfontein@25.7076,-30.2670"], "", TFR, "#8a6d3b",
      own=["Noupoort–Germiston"], suspended=True),
    L("springfontein-ntabozuko", "Springfontein–Ntabozuko",
      ["Springfontein@25.7076,-30.2670", "Bethulie@25.9986,-30.4910",
       "Burgersdorp@26.3234,-31.0004", "Molteno@26.3656,-31.3952",
       "Sterkstroom@26.5502,-31.5666", "Komani@26.8777,-31.8928",
       "Cathcart@27.1524,-32.2983", "Stutterheim@27.4344,-32.5641",
       "Blaney@27.5239,-32.8666", "Ntabozuko@27.5829,-32.8825"], "", TFR, "#8a6d3b",
      own=["East London–Springfontein"], suspended=True, name_en="Springfontein–Berlin"),
    L("ptanoord-musina", "Pretoria-Noord–Musina",
      ["Pretoria-Noord@28.1820,-25.6715", "Polokwane@29.4458,-23.8965",
       "Musina@30.0407,-22.3432"], "", TFR, "#8a6d3b", own=["Pretoria–Beit Bridge"],
      suspended=True),
]

BY_ID = {l["id"]: l for l in LINES}


# ============================================================== conversion

def parse(p):
    m = re.match(r"^(.*)@([-\d.]+),([-\d.]+)$", p)
    if not m:
        raise SystemExit(f"point without a coordinate: {p}")
    return m.group(1), (float(m.group(2)), float(m.group(3)))


def own_ways(infra, ways):
    """{relation name: set(way id)}, and "~<name>" for ways whose own name is that."""
    out = defaultdict(set)
    for _rid, (tags, members) in infra.items():
        n = tags.get("name")
        if n:
            out[n] |= {r for ty, r, _ in members if ty == "w"}
    for wid, (tags, _n) in ways.items():
        if tags.get("name"):
            out["~" + tags["name"]].add(wid)
    return out


def convert(log=print, write=True, report=False):
    t0 = time.time()
    import build_model as bm
    import rinf
    ways, rels, stops, cid, cx, cy = bm.load(CC, log)
    coords = bm.Coords(cid, cx, cy)
    track = rinf.Track(ways, coords, log)
    ost = rinf.osm_stations(stops)
    by_key = defaultdict(list)
    for sid, s in ost.items():
        by_key[key(s["name"])].append(sid)
    with open(ROOT / "data" / "proc" / CC / "infra.pkl", "rb") as f:
        infra = pickle.load(f)
    rel_ways = own_ways(infra, ways)
    missing = sorted({o for l in LINES for o in l["own"] if o not in rel_ways})
    if missing:
        log(f"ZA: own track named but not in the extract: {', '.join(missing)}")

    rows, pts_out, names = [], {}, {}
    stat = Counter()
    traced_edges = {}
    for l in LINES:
        placed = []
        for p in l["pts"]:
            label, at = parse(p)
            cs = [c for c in by_key.get(key(label), ())
                  if dist_m(ost[c]["lon"], ost[c]["lat"], *at) <= PLACE_M]
            if cs:
                c = min(cs, key=lambda c: dist_m(ost[c]["lon"], ost[c]["lat"], *at))
                placed.append(("station", label, ost[c]["lon"], ost[c]["lat"], c))
                stat["points at an OSM station"] += 1
            else:
                placed.append(("junction", label, at[0], at[1], None))
                stat["points with no OSM station of the name (junctions)"] += 1
                log(f"    {l['id']}: no OSM station '{label}' within {PLACE_M} m; a junction")
        own = set().union(*(rel_ways.get(o, set()) for o in l["own"])) if l["own"] else None
        total, mine, k = 0.0, 0.0, 0
        edges_all = []
        for a, b in zip(placed[:-1], placed[1:]):
            sa, sb = track.snap(a[2], a[3]), track.snap(b[2], b[3])
            crow = dist_m(a[2], a[3], b[2], b[3]) / 1000
            got = (track.trace(sa, sb, crow * 2 + 5, own)
                   if sa is not None and sb is not None else None)
            if got is None:
                km = crow * 1.2
                stat["sections with no trace (crow-fly x 1.2)"] += 1
                log(f"    {l['id']}: no trace {a[1]} - {b[1]} ({crow:.1f} km crow-fly)")
            else:
                km = got[1]
                mine += got[4]
                edges_all.extend(got[3])
                stat["sections traced"] += 1
            total += km
            k += 1
            ops = []
            for q in (a, b):
                if q[0] == "station":
                    op = f"{CC}:s:{q[4]}"
                    pts_out[op] = {"op": op, "uopid": f"ZA{q[4]}", "type": "10",
                                   "name": ost[q[4]]["name"], "lon": q[2], "lat": q[3]}
                else:
                    jk = key(q[1])[:24]
                    op = f"{CC}:j:{jk}"
                    pts_out[op] = {"op": op, "uopid": f"ZAJ{jk}", "type": "80",
                                   "name": q[1], "lon": q[2], "lat": q[3]}
                ops.append(op)
            rows.append({"sol": f"{CC}:{l['id']}:{k}", "line": l["id"], "a": ops[0],
                         "b": ops[1], "len": f"{km:.3f}", "im": l["im"],
                         "label": f"{a[1]} - {b[1]}"})
        traced_edges[l["id"]] = set(int(e) for e in edges_all)
        names[l["id"]] = {"name": l["name"], "name_en": l["name_en"], "ref": "",
                          "im": l["im"], "suspended": l["suspended"],
                          "traced_km": round(total, 3)}
        log(f"  {l['id']:>26} {l['name'][:40]:<40} {len(placed):3d} points  traced "
            f"{total:7.1f} km" + (f"  on own track {mine / total:.0%}" if own and total else "")
            + ("  (suspended)" if l["suspended"] else ""))
        if report:
            on = []
            # stations passed: OSM stations within 100 m of the trace's edges
            if edges_all:
                import numpy as np
                e = np.unique(np.asarray(edges_all))
                mx = (track.ax[e] + track.bx[e]) / 2
                my = (track.ay[e] + track.by[e]) / 2
                for sid, s in ost.items():
                    x, y = s["lon"] * track.kx, s["lat"] * track.ky
                    if x < mx.min() - 200 or x > mx.max() + 200 or y < my.min() - 200 \
                            or y > my.max() + 200:
                        continue
                    d = np.hypot(mx - x, my - y).min()
                    if d <= 150:
                        on.append(s["name"])
                log(f"      passes: {', '.join(sorted(set(on)))}")
    if report:
        log("  track two lines' traces share (km, by edge):")
        ids = list(traced_edges)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                sh = traced_edges[a] & traced_edges[b]
                km = sum(float(track.elen[e]) for e in sh) / 1000
                if km >= 0.2:
                    log(f"    {km:6.2f}  {a} / {b}")
    log(f"ZA: {len(LINES)} lines, {len(rows)} section rows, {len(pts_out)} points; {dict(stat)}")
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        stamp = {"endpoint": "za_register.py (hand-written line list)",
                 "fetched": date.today().isoformat()}
        (OUT / "sections.json").write_text(json.dumps({**stamp, "rows": rows},
                                                      ensure_ascii=False), "utf-8")
        (OUT / "points.json").write_text(json.dumps({**stamp, "rows": list(pts_out.values())},
                                                    ensure_ascii=False), "utf-8")
        (OUT / "names.json").write_text(json.dumps(names, ensure_ascii=False, indent=0), "utf-8")
        log(f"wrote {OUT} in {time.time() - t0:.0f} s")
    return rows, pts_out, names


def build(path, log):
    """build_model's register hook: convert, rinf.py traces it over OSM track, then each
    line takes its network, operator and colour from LINES."""
    convert(log)
    import rinf
    lines, stations, geoms = rinf.build(path, log)
    by_name = {l["name"]: l for l in LINES}
    for ln in lines:
        info = by_name.get(ln["name"])
        if info is None:
            log(f"ZA: rinf.py line {ln['name']!r} is in no LINES entry")
            continue
        ln["network"] = info["network"]
        ln["operator"] = info["im"]
        ln["colour"] = info["colour"]
        if info["name_en"]:
            ln["name_en"] = info["name_en"]
    built = {ln["name"] for ln in lines}
    for l in LINES:
        if l["name"] not in built:
            log(f"ZA: no line built for {l['name']}")
    merge_twins(lines, stations, geoms, log)
    path_checks(lines, stations, log)
    return lines, stations, geoms


# Two OSM records of one station under two spellings ("Cleveland" and "Clevenland", "Medunsa"
# and "Mudunsa", "Eersterus" and "Eersterust", Metrorail's and Gautrain's "Rhodesfield") both
# become stops on a traced section, with a section of a few metres between them. A section
# shorter than TWIN_KM between two stops of a similar name is one station; the spelling in
# PREFER is kept, else the one more lines use.
TWIN_KM = 0.2
PREFER = {"Cleveland", "Medunsa", "Eersterust", "Ellis Park Station", "Deerness"}


def merge_twins(lines, stations, geoms, log):
    from difflib import SequenceMatcher

    def similar(a, b):
        ka, kb = key(a), key(b)
        return (ka == kb or ka.startswith(kb) or kb.startswith(ka)
                or SequenceMatcher(None, ka, kb).ratio() >= 0.75)
    into = {}

    def root(s):
        while s in into:
            s = into[s]
        return s
    for ln in lines:
        for a, b, km, *_ in ln["sections"]:
            a, b = root(a), root(b)
            if a == b or km >= TWIN_KM:
                continue
            na, nb = stations[a]["name"], stations[b]["name"]
            if not similar(na, nb):
                continue
            keep, drop = a, b
            if (nb in PREFER and na not in PREFER) or (
                    na not in PREFER and len(stations[b]["lines"]) > len(stations[a]["lines"])):
                keep, drop = b, a
            into[drop] = keep
            log(f"ZA: one station: {stations[drop]['name']} ({drop}) into "
                f"{stations[keep]['name']} ({keep}), {km * 1000:.0f} m apart on {ln['name']}")
    if not into:
        return
    for ln in lines:
        g = geoms.get(ln["id"], {})
        ng, secs = {}, []
        hs = ln.get("highspeed_sections") or {}
        nhs = {}
        for sec in ln["sections"]:
            a, b = root(sec[0]), root(sec[1])
            old = f"{sec[0]}|{sec[1]}"
            if a == b:
                continue
            secs.append([a, b, *sec[2:]])
            if old in g:
                ng[f"{a}|{b}"] = g[old]
            if old in hs:
                nhs[f"{a}|{b}"] = hs[old]
        ln["sections"] = secs
        geoms[ln["id"]] = ng
        if "highspeed_sections" in ln:
            ln["highspeed_sections"] = nhs
        disp = []
        for s in ln["display"]:
            s = root(s)
            if not disp or disp[-1] != s:
                disp.append(s)
        ln["display"] = disp
        ln["km"] = round(sum(s[2] for s in secs), 3)
    for drop in into:
        keep = root(drop)
        stations[keep]["lines"] |= stations[drop]["lines"]
        del stations[drop]


# ============================================================== outside numbers, by path
#
# Few South African lines have a published length of their own, so most checks are paths over
# the built register lines between two stations, against a published route length: (label,
# from, to, km, source, the register lines the path may use).
PATH_CHECKS = [
    ("Cape Town - Worcester", "Cape Town", "Worcester", 174.0,
     "Metrorail Western Cape's main line to Worcester, WP", ["Cape Town–De Aar"]),
    ("Cape Town - Malmesbury", "Cape Town", "Malmesbury", 79.4, "WP Malmesbury Line",
     ["Cape Town–De Aar", "Kraaifontein–Malmesbury"]),
    ("Cape Town - Simon's Town", "Cape Town", "Simon's Town", 36.0, "WP Southern Line",
     ["Cape Town–Bellville (via Mutual)", "Salt River–Simon's Town"]),
    ("Cape Town - Heathfield (Cape Flats)", "Cape Town", "Heathfield", 22.2,
     "WP Cape Flats Line 23.8 to Retreat, less Heathfield - Retreat (1.6, built)",
     ["Cape Town–Bellville (via Mutual)", "Maitland–Heathfield"]),
    ("Pretoria - Cape Town", "Pretoria", "Cape Town", 1600.0,
     "the Blue Train's route, WP (rounded)",
     ["Germiston–Pretoria", "Germiston–Kimberley", "Kimberley–De Aar", "Cape Town–De Aar"]),
    ("Pretoria - Polokwane", "Pretoria", "Polokwane", 284.4,
     "Pretoria - Pietersburg's official length in 1902, 176 mi 58 ch (WP, SAR 1953)",
     ["Pretoria–De Wildt", "Pretoria-Noord–Musina"]),
]


def path_checks(lines, stations, log):
    import heapq
    by_name = defaultdict(set)
    for sid, s in stations.items():
        by_name[key(s.get("name"))].add(sid)
    out = []
    for label, a, b, km, note, allowed in PATH_CHECKS:
        adj = defaultdict(list)
        for ln in lines:
            if ln["name"] not in allowed:
                continue
            for sec in ln["sections"]:
                adj[sec[0]].append((sec[1], sec[2]))
                adj[sec[1]].append((sec[0], sec[2]))
        src = [s for s in by_name.get(key(a), ()) if s in adj]
        dst = set(s for s in by_name.get(key(b), ()) if s in adj)
        dist = {s: 0.0 for s in src}
        h = [(0.0, s) for s in src]
        got = None
        while h:
            d, u = heapq.heappop(h)
            if u in dst:
                got = d
                break
            if d > dist.get(u, math.inf):
                continue
            for v, w in adj[u]:
                if d + w < dist.get(v, math.inf):
                    dist[v] = d + w
                    heapq.heappush(h, (d + w, v))
        out.append((label, got, km))
        log(f"path check {label}: {'no path' if got is None else f'{got:.1f} km'} against "
            f"{km:.1f} ({note})" + ("" if got is None else f", ratio {got / km:.3f}"))
    return out


# ============================================================== rinf.py settings

def _names():
    f = OUT / "names.json"
    return json.loads(f.read_text("utf-8")) if f.exists() else {}


class LineName(str):
    """rinf.py writes a numbered line's name as COUNTRY["name"].format(ref=ref)."""

    def __new__(cls, field):
        o = str.__new__(cls, "{ref}")
        o.field = field
        return o

    def format(self, *args, **kw):
        ref = kw.get("ref", args[0] if args else "")
        e = _names().get(ref)
        return e[self.field] if e else ref


def country_conf():
    def id_name(lid, _uop=None):
        e = _names().get(lid.split("#")[0])
        return (e["name"], e.get("name_en") or "") if e else None

    def suspended(_ref, lids):
        ns = _names()
        return any(ns.get(x.split("#")[0], {}).get("suspended") for x in lids)

    ims = {l["im"] for l in LINES}
    return {
        "iso3": "ZAF", "langs": ["en"],
        "ref": lambda lid: None,
        "rule_certain": True,
        "name": LineName("name"), "name_en": LineName("name_en"),
        "id_name": id_name,
        "suspended": suspended,
        "im": {x: x for x in ims},
        # Stops: every OSM station a passenger route stops at that lies on a traced section
        # (Metrorail's and Gautrain's routes list every stop; Shosholoza Meyl's its calls).
        "osm_stops": True,
        # No line numbers: OSM's relations must not lend one.
        "osm_rel": lambda _t: None,
        # The section lengths are this reader's own traces, not a register's chainage: no
        # km_official, so check_model does not report a chainage check that checks nothing.
        "no_chain": True,
    }


def split_pieces(lines, stations, geoms, reg_ways, state, log):
    """build_model's hook, after drop_unridden_sections: rinf.split_pieces (folds a stop's 0 km
    link junctions back into it, and bridges where the country's settings ask)."""
    import rinf
    rinf.split_pieces(lines, stations, geoms, reg_ways, state, log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="convert without writing, with a report")
    ap.add_argument("--convert", action="store_true")
    a = ap.parse_args()
    t = time.time()
    lg = lambda m: print(f"[{time.time() - t:6.1f}s] {m}", flush=True)  # noqa: E731
    if a.dry:
        convert(lg, write=False, report=True)
    elif a.convert:
        convert(lg)
    else:
        print(__doc__)
