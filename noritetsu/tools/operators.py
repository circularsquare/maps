"""OPERATORS: who runs the passenger trains on each register line, and each operator's short
name, colour and logo (Anita, 2026-10-07; research in operator_branding_sources.md).

    python tools/operators.py measure            # the switch, per country, before/after top 10
    python tools/operators.py seed [--search N]  # fill blank cells of colours/operators.csv from Wikidata
    python tools/operators.py logos [--max N]    # download the chosen logos once, slowly; colours from them
    python tools/operators.py build              # dist/data/operators.json and dist/data/<cc>/ops.json

Reads dist/data/<cc>/lines.json and foot.json and data/proc/<cc>/rels.pkl; never writes any
of them. Writes colours/operators.csv (the hand-kept table), dist/data/operators.json,
dist/data/<cc>/ops.json, dist/data/logos/, and its download cache in data/raw/operators/.

THE SWITCH (decision 1). A register line whose `operator` is an infrastructure manager or a
freight railway (role `infra` in the table, or no operator at all) takes as its operators the
companies whose passenger trains run over it: the OSM lines (src "osm", with an operator)
whose footprint (foot.json) lies on its sections. For each operator, the share is the part of
the register line's km covered by at least one of its lines (a union, so ten services over the
same track count once). Kept, in order of share:
  - operators of OSM routes (operating patterns, `service` false) covering >= SHARE_MIN;
  - operators of named trains (`service` true) covering >= SHARE_MIN, only if they are at home
    in the country (Wikidata P17, else the country where their own routes run the most km).
    Without this one international train (Uzbek Railways to Moscow, an ÖBB Nightjet through
    Germany, the Chișinău train over Romanian lines) took whole lines;
  - nothing kept: the best one, if it covers >= SHARE_FLOOR.
  - still nothing: the country's sole passenger operator (COUNTRY_DEFAULT) where it has one,
    else the register operator stays. So an infrastructure manager is left in the Operators
    list only for track no passenger operator could be found for (decision 2).
A register line whose operator runs trains itself (JR, Korail, a metro, China Railway, the
Indian Railways zones, RZD...) is left alone.

THE TABLE colours/operators.csv, one row per (cc, operator string as in the data):
  cc, operator, qid, short, colour, colour_src, logo, logo_licence, group, role, note, km
  qid    Wikidata item; "-" = none, do not search again.
  short  the name shown ("JR West"); blank = the key.
  colour #RRGGBB; colour_src wikidata / logo / hand.
  logo   the Commons file (P154); logo_licence its licence and author.
  group  the operator key this string counts under ("DB" for DB Regio NRW). Several keys
         separated by ";" for a joint operator written as one string ("Renfe / SNCF
         Voyageurs"). "-" = none (stops GROUP_RULES from filling it).
  role   "infra" = an infrastructure manager or freight railway, never a passenger
         operator; "-" = a passenger operator (stops INFRA_RULES from filling it).
  note   how the row was filled; km is refreshed on every run (listed km after the switch).
Every run adds rows for new strings and fills blank cells; a filled cell is never
overwritten, so hand edits stay.

KEYS. The app groups by a key per operator. A row's key is its group if set; else the key of
the biggest row of the same country with the same Wikidata item (東京地下鉄 and 東京メトロ
become one); else the English name the app's OP_EN vote would give; else the string itself.
Keys are shared across countries (DB in Poland is the same row as DB in Germany). A key two
countries reach with different Wikidata items is printed by `build` for a look; a real clash
is settled with a group cell (India's Southern Railway counts under "Indian Railways", so it
no longer shares a row with the British Southern).
"""
import argparse
import colorsys
import csv
import glob
import io
import json
import os
import pickle
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist")
DATA = os.path.join(DIST, "data")
CSV = os.path.join(ROOT, "colours", "operators.csv")
CACHE = os.path.join(ROOT, "data", "raw", "operators")
LOGO_DIR = os.path.join(DATA, "logos")
UA = {"User-Agent": "noritetsu/0.1 (https://github.com/; rail map, operator logos)"}
COLS = ["cc", "operator", "qid", "short", "colour", "colour_src", "logo", "logo_licence",
        "group", "role", "note", "km"]

SHARE_MIN = 0.15      # an operator kept on a register line covers at least this much of it
SHARE_FLOOR = 0.05    # the best one, when none reaches SHARE_MIN, still needs this much
LOGO_H = 28           # px; shown at 14 px (2x)
LOGO_W = 56           # px at most; shown at 28 px, twice the row's 14 px logo height

# ------------------------------------------------------------------ hand rules (decisions)

# Infrastructure managers and freight railways: register lines naming them are switched.
INFRA_RULES = {
    "at": ["ÖBB-Infrastruktur", "ÖBB-Infrastruktur AG"],
    "au": ["Australian Rail Track Corporation", "Genesee & Wyoming Australia", "Westrail",
           "Aurizon"],
    "be": ["Infrabel"], "bg": ["НКЖИ"], "dk": ["Banedanmark"], "fi": ["Väylävirasto"],
    "ca": ["Canadian National", "CPKC", "Hudson Bay Railway",
           "Quebec North Shore and Labrador Railway", "BNSF Railway", "ARMD"],
    "ch": ["Deutsche Eisenbahn-Infrastruktur in der Schweiz", "ÖBB-Infrastruktur AG"],
    "cz": ["Správa železnic", "SART-stavby a rekonstrukce", "Railway Capital",
           "PKP Cargo International"],
    "de": ["DB InfraGO"], "ee": ["Eesti Raudtee", "Edelaraudtee Infrastruktuur"],
    "es": ["Adif", "Adif AV", "LFP Perthus", "adif", "ADIF"], "fr": ["SNCF Réseau", "Eurotunnel"],
    "gb": ["Network Rail", "HS1 Ltd", "Heathrow Airport Holdings"], "gr": ["ΟΣΕ"],
    "hr": ["HŽ Infrastruktura"],
    "it": ["RFI", "Rete Ferroviaria Italiana", "FER", "FERROVIENORD"],
    "lt": ["LTG Infra"], "lv": ["Latvijas dzelzceļš"],
    "me": ["Željeznička infrastruktura Crne Gore"],
    "mk": ["Македонски железници – Инфраструктура"], "nl": ["ProRail"], "no": ["Bane NOR"],
    "nz": ["Dunedin City Council"], "pl": ["PKP Polskie Linie Kolejowe"],
    "pt": ["Infraestruturas de Portugal"], "ro": ["CFR"],
    "rs": ["Инфраструктура железнице Србије"], "se": ["Trafikverket"],
    "si": ["SŽ-Infrastruktura"], "sk": ["Železnice Slovenskej republiky"], "tr": ["TCDD"],
    "us": ["BNSF Railway", "Union Pacific Railroad", "CSX Transportation", "Norfolk Southern",
           "Canadian National", "CPKC", "Florida East Coast Railway",
           "New England Central Railroad", "Buckingham Branch Railroad",
           "New Mexico Department of Transportation", "Vermont Railway",
           "Massachusetts Coastal Railroad", "Pan Am Southern", "Nashville and Eastern Railroad",
           "FWWR", "Virginia Passenger Rail Authority", "Clarendon and Pittsford Railroad",
           "Bay Colony Railroad", "Portland and Western Railroad", "Conrail Shared Assets",
           "DGNO", "Kansas City Terminal Railway", "Terminal Railroad Association of St. Louis",
           "MassDOT", "Minnesota Commercial Railway", "Tacoma Rail", "Belt Railway of Chicago",
           "Central Maine and Quebec", "Austin Western Railroad (CapMetro)"],
    "xk": ["Infrakos"], "za": ["Transnet Freight Rail"],
}

# Where one company runs every passenger train on the infrastructure manager's network, a
# register line no OSM service was found on takes it rather than the manager.
COUNTRY_DEFAULT = {
    "cn": "中国铁路",   # register lines with no operator at all: China Railway runs the national network
    "be": "NMBS/SNCB", "bg": "Български държавни железници", "ee": "AS Eesti Liinirongid",
    "es": "Renfe", "fi": "VR", "fr": "SNCF Voyageurs", "gr": "Hellenic Train", "hr": "HŽPP",
    "lt": "LTG Link", "lv": "Vivi (Pasažieru vilciens)", "me": "Željeznički prevoz Crne Gore",
    "mk": "Македонски Железници", "pt": "Comboios de Portugal", "rs": "Србија Воз",
    "si": "SŽ", "tr": "TCDD Taşımacılık", "xk": "Trainkos",
}

# Subsidiaries and spellings folded into one operator (decision 3). (countries or None for
# any, regex on the string, the key). First match wins.
GROUP_RULES = [
    (None, r"^(DB(?! InfraGO)\b.*|Deutsche Bahn.*|S-Bahn (Berlin|Hamburg)( GmbH)?|"
           r"Usedomer Bäderbahn.*|Erzgebirgsbahn|Südostbayernbahn|Kurhessenbahn|"
           r"Westfrankenbahn|Oberweißbacher Berg- und Schwarzatalbahn)$", "DB"),
    (None, r"^Renfe / SNCF.*$", "Renfe;SNCF"),
    (None, r"^(SNCF(?!T| Réseau)\b.*|TER .*|Transilien.*|OSLO|Ouigo|OUIGO|TGV INOUI|"
           r"Intercités)$", "SNCF"),
    (None, r"^(Renfe|RENFE)\b(?! / ).*$", "Renfe"),
    (None, r"^(ÖBB(?!-Infrastruktur)\b.*|Österreichische Bundesbahnen)$", "ÖBB"),
    (None, r"^(SBB\b.*|Schweizerische Bundesbahnen)$", "SBB"),
    ({"it", "ch", "at", "de", "fr", "si"}, r"^Trenitalia.*$", "Trenitalia"),
    (None, r"^(NS|NS International|Nederlandse Spoorwegen)$", "Nederlandse Spoorwegen"),
    (None, r"^(SNCB|NMBS|SNCB/NMBS|NMBS-SNCB)$", "NMBS/SNCB"),
    (None, r"^(ČD|cz:ČD|České dráhy.*)$", "České dráhy"),
    (None, r"^(ZSSK|Železničná spoločnosť Slovensko.*)$", "Železničná spoločnosť Slovensko"),
    (None, r"^(PKPIC|PKP IC|PKP Intercity.*)$", "PKP Intercity"),
    ({"pl"}, r"^(POLREGIO|Polregio).*$", "Polregio"),
    ({"cz", "sk", "at", "pl", "hr", "rs", "ro", "si", "ua"}, r"^START$", "MÁV"),
    (None, r"^(MÁV|MÁV-Start|MÁV-START|MÁV SZESZA|MÁV-START Zrt\.?|Magyar Államvasutak|"
           r"MÁV Személyszállítási Zrt\.?)$", "MÁV"),
    (None, r"^(DSB\b.*)$", "DSB"),
    ({"se", "no"}, r"^(SJ|SJ AB|SJ Nord|SJ Norge)$", "SJ"),
    ({"no", "se"}, r"^(Vy|Vy Tog|Vy Tåg|Vy Tåg AB|NSB)$", "Vy"),
    ({"ru"}, r"^(ОАО \"РЖД\"|ОАО «РЖД»|РЖД|Российские железные дороги|АО «ФПК»|АО \"ФПК\"|ФПК|"
             r".*(железная дорога|ж\.д\.)( — филиал.*)?)$", "РЖД"),
    ({"ua"}, r"^(Укрзалізниця|Укзалізниця|АТ «Укрзалізниця»|.*[Зз]алізниця)$", "Ukrainian Railways"),
    ({"cn"}, r"^中国铁路.*$", "China Railway"),
    ({"in"}, r"^(Northern|North Western|Southern|Western|East Central|Northeast Frontier|"
             r"South Coast|Central|North Central|North Eastern|South Western|South Central|"
             r"West Central|South Eastern|Eastern|East Coast|South East Central|"
             r"North East Frontier|Metro) Railways?$", "Indian Railways"),
    ({"in"}, r"^(Indian Railways?|IR)$", "Indian Railways"),
    ({"kz"}, r"^(ҚТЖ|КТЖ|Қазақстан темір жолы|Казахстан темир жолы|«Қала маңы тасымалы» АҚ)$",
     "Қазақстан темір жолы"),
    ({"by"}, r"^(БЧ|Беларуская чыгунка|Белорусская железная дорога|.*отделение БЖД.*)$",
     "Беларуская чыгунка"),
    ({"ge"}, r"^(სს \"საქართველოს რკინიგზა\"|საქართველოს რკინიგზა)$", "საქართველოს რკინიგზა"),
    ({"dz"}, r"^(SNTF|Société Nationale des Transports Ferroviaires|"
             r"الشركة الوطنية للنقل بالسكك الحديدية)$", "Société Nationale des Transports Ferroviaires"),
    ({"ma"}, r"^(ONCF.*|Office National des Chemins de Fer|المكتب الوطني للسكك الحديدية)$",
     "Office National des Chemins de Fer"),
    ({"tn"}, r"^(SNCFT|Société Nationale des Chemins de Fer Tunisiens)$",
     "Société Nationale des Chemins de Fer Tunisiens"),
    ({"id"}, r"^(KAI|KAI Commuter|KAI Bandara|Kereta [Aa]pi Indonesia.*|PT Kereta Api Indonesia.*|"
             r"PT KAI.*)$", "Kereta Api Indonesia"),
    ({"au"}, r"^(NSW TrainLink|NSW Trains|Intercity Train)$", "NSW TrainLink"),
    # Stale franchise names in OSM: the company running the trains now.
    ({"gb"}, r"^(Virgin Trains East Coast|LNER)$", "London North Eastern Railway"),
    ({"gb"}, r"^(Northern Rail|Arriva Rail North|Northern Trains)$", "Northern"),
    ({"de"}, r"^(Ostdeutsche Eisenbahngesellschaft|ODEG.*)$", "Ostdeutsche Eisenbahn GmbH"),
    ({"de", "at"}, r"^agilis.*$", "agilis"),
    ({"de"}, r"^vlexx.*$", "vlexx"),
    ({"de"}, r"^(HLB Hessenbahn.*|HLB|Hessische Landesbahn)$", "Hessische Landesbahn GmbH"),
    ({"de"}, r"^(cantus Verkehrsgesellschaft mbH|Cantus)$", "cantus Verkehrsgesellschaft mbH"),
    ({"de"}, r"^(metronom|Metronom)$", "Metronom Eisenbahngesellschaft mbH"),
    ({"gb"}, r"^Trafnidiaeth Cymru$", "Transport for Wales"),
    ({"us"}, r"^Amtrak( California)?$", "Amtrak"),
    (None, r"^(TGV )?Lyria$", "Lyria"),
    ({"ch"}, r"^BLSN$", "BLS"), ({"ch"}, r"^RhB FR VR$", "RhB"), ({"ch"}, r"^MGI$", "MGB"),
    ({"lu"}, r"^(CFL|Société Nationale des Chemins de [Ff]er Luxembourgeois)$", "CFL"),
    ({"hk"}, r"^香港鐵路有限公司.*$", "MTR Corporation"),
    ({"tw"}, r"^(臺灣鐵路管理局|國營臺灣鐵路股份有限公司)$", "Taiwan Railway Corporation"),
    (None, r"^(GYSEV|GySEV|GySĚV|Raaberbahn|GYSEV Zrt\.?)$", "GYSEV"),
    ({"cz", "sk", "at"}, r"^(cz:)?RegioJet.*$", "RegioJet"),
    ({"se", "no"}, r"^SJ Sverige$", "SJ"), ({"no"}, r"^Vy .*$", "Vy"),
    ({"tr"}, r"^TCDD Taşımacılık.*$", "TCDD Taşımacılık"),
    ({"fr"}, r"^RATP/SNCF$", "RATP;SNCF"),
    # Metros split into operating companies or contractors: the brand riders know (Anita,
    # 2026-10-07: "shanghai metro should be one operator ... in terms of how people think of
    # it"). Chinese systems go by their network tag (NETWORK_GROUPS); these by name.
    ({"gb"}, r"^(MTR Elizabeth line|KeolisAmey Docklands Ltd|Arriva Rail London|London Overground.*|"
             r"Tram Operations Ltd|London Underground.*|TfL)$", "Transport for London"),
    ({"us"}, r"^(MTA New York City Transit|New York City Transit|NYC Transit|"
             r"MTA Staten Island Railway)$", "Metropolitan Transportation Authority"),
    ({"us"}, r"^(PRT|Pittsburgh Regional Transit)$", "Pittsburgh Regional Transit"),
    ({"fr"}, r"^RATP Cap .*$", "RATP"),
    ({"fr"}, r"^Régie des Transports (Marseillais|Métropolitains)$", "Régie des Transports Métropolitains"),
    ({"jp"}, r"^(東京地下鉄|東京メトロ|Tokyo Metro)$", "Tokyo Metro"),
    ({"ru"}, r"^(ГУП )?«?Московский метрополитен»?$", "ГУП «Московский метрополитен»"),
    ({"de"}, r"^(SSB AG|Stuttgarter Straßenbahnen AG)$", "Stuttgarter Straßenbahnen AG"),
    ({"de"}, r"^(VAG Verkehrs-Aktiengesellschaft Nürnberg|Verkehrs-Aktiengesellschaft Nürnberg)$",
     "Verkehrs-Aktiengesellschaft Nürnberg"),
    ({"br"}, r"^(Companhia do Metropolitano de São Paulo|ViaMobilidade Linhas 5 e 17|ViaQuatro)$",
     "Companhia do Metropolitano de São Paulo"),
    ({"br"}, r"^(Metrô Rio|MetrôRio)$", "Metrô Rio"),
    ({"ca"}, r"^(British Columbia Rapid Transit Company|InTransitBC)$", "British Columbia Rapid Transit Company"),
    ({"ca"}, r"^(Edmonton Transit Service|TransEd Partners)$", "Edmonton Transit Service"),
    ({"in"}, r"^(BMRCL|Bangalore Metro Rail Corporation Limited)$", "Bangalore Metro Rail Corporation Limited"),
    ({"in"}, r"^(MMRDA|Mumbai Metro Rail Corporation Ltd\.|Mumbai Metro One.*)$", "Mumbai Metro"),
    ({"th"}, r"^(Bangkok Expressway and Metro Public Company Limited|นอร์ทเทิร์นบางกอกโมโนเรล|"
             r"อีสเทิร์นบางกอกโมโนเรล|Northern Bangkok Monorail.*|Eastern Bangkok Monorail.*)$",
     "Bangkok Expressway and Metro Public Company Limited"),
    ({"se"}, r"^(MTR Nordic|Connecting Stockholm|Stockholms spårvägar)$", "SL"),
    ({"pl"}, r"^Koleje Dolnośląskie/GW Train Regio a\.s\.$", "Koleje Dolnośląskie;GW Train Regio"),
    ({"cz"}, r"^GW Train Regio a\.s\./Koleje Dolnośląskie$", "GW Train Regio;Koleje Dolnośląskie"),
]
# A metro run by several companies under one brand: every operator string whose urban rail
# lines (subway, light rail, monorail, tram) mostly carry this network tag counts under the
# brand. NETWORK_KEEP: strings on that network that are a brand of their own (Shanghai's
# maglev, the Songjiang tram, Guangdong's intercity railway).
NETWORK_GROUPS = {"cn": {
    "上海地铁": "Shanghai Metro", "北京地铁": "Beijing Subway", "广州地铁": "Guangzhou Metro",
    "深圳地铁": "Shenzhen Metro", "杭州地铁": "Hangzhou Metro", "天津地铁": "Tianjin Metro",
    "重庆轨道交通": "Chongqing Rail Transit", "西安地铁": "Xi'an Metro", "南昌地铁": "Nanchang Metro",
    "昆明地铁": "Kunming Metro", "佛山地铁": "Foshan Metro", "南京地铁": "Nanjing Metro",
    "福州地铁": "Fuzhou Metro", "大连地铁": "Dalian Metro", "Changchun Rail Transit": "Changchun Rail Transit",
    "洛阳轨道交通": "Luoyang Metro", "成都地铁": "Chengdu Metro", "武汉地铁": "Wuhan Metro"}}
NETWORK_KEEP = {"cn": {"上海磁浮交通发展有限公司", "上海磁悬浮交通发展有限公司", "上海申凯公共交通运营管理有限公司",
                       "广东城际铁路运营有限公司", "上海市域铁路运营有限公司", "大连交通集团",
                       "Dalian Public Transportation Group"}}
URBAN = ("subway", "light_rail", "monorail", "tram")

# Rows added for group keys and defaults that no line names (their Wikidata item, by hand).
EXTRA_ROWS = {
    ("in", "Indian Railways"): "Q819425", ("de", "DB"): "Q9322", ("fr", "SNCF"): "Q13646",
    ("se", "SL"): "", ("in", "Mumbai Metro"): "", ("cn", "China Railway"): "Q1073489",
    ("ua", "Ukrainian Railways"): "Q923337",
    **{("cn", v): "" for v in NETWORK_GROUPS["cn"].values()},
}

LANG = {"jp": "ja", "kr": "ko", "cn": "zh", "tw": "zh", "hk": "zh", "ru": "ru", "ua": "uk",
        "by": "be", "de": "de", "at": "de", "ch": "de", "fr": "fr", "be": "fr", "lu": "fr",
        "it": "it", "es": "es", "pl": "pl", "cz": "cs", "sk": "sk", "hu": "hu", "ro": "ro",
        "nl": "nl", "se": "sv", "no": "nb", "fi": "fi", "dk": "da", "tr": "tr", "ir": "fa",
        "th": "th", "kz": "kk", "eg": "ar", "ma": "fr", "dz": "fr", "tn": "fr", "rs": "sr",
        "bg": "bg", "pt": "pt", "br": "pt", "gr": "el", "vn": "vi", "uz": "uz", "id": "id",
        "hr": "hr", "si": "sl", "lt": "lt", "lv": "lv", "ee": "et", "ge": "ka", "am": "hy",
        "az": "az", "mk": "mk", "me": "sr", "ba": "bs", "al": "sq", "xk": "sq", "md": "ro",
        "kg": "ky", "tj": "tg", "tm": "tk", "mx": "es", "ar": "es", "cl": "es", "my": "ms",
        "xa": "ru"}
ISO = {"gb": "gb", "xk": "xk", "xa": "ge"}   # our cc -> the P17 country's ISO code
LATIN = re.compile(r"^[\u0000-ɏḀ-ỿ\s]+$")


# ------------------------------------------------------------------ data

def regions():
    return list(json.load(open(os.path.join(DIST, "regions.json"), encoding="utf-8"))["regions"])


def parts(s):
    return [p.strip() for p in (s or "").split(";") if p.strip()]


def merge_spans(spans, km):
    """The app's mergeSpans: gaps under 150 m closed."""
    if not spans:
        return []
    tol = min(0.15 / km, 0.1) if km > 0 else 0
    out = []
    for lo, hi in sorted(spans):
        if out and lo <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    if out[0][0] <= tol:
        out[0][0] = 0
    if out[-1][1] >= 1 - tol:
        out[-1][1] = 1
    return out


def span_len(sp):
    return min(1, sum(b - a for a, b in sp))


class Country:
    """One country's lines with what the app works out from them: footprints, owned km,
    listed lines, and passenger coverage of register lines."""

    def __init__(self, cc):
        self.cc = cc
        self.lines = json.load(open(os.path.join(DATA, cc, "lines.json"), encoding="utf-8"))["lines"]
        try:
            f = json.load(open(os.path.join(DATA, cc, "foot.json"), encoding="utf-8"))
        except FileNotFoundError:
            f = {"foot": {}, "scale": 1}
        sc = f.get("scale", 1)
        self.foot = {}
        for k, v in f.get("foot", {}).items():
            prev, ent = 0, []
            for e in v:
                a = e[3] if len(e) == 5 else prev
                b = e[4] if len(e) == 5 else (e[3] if len(e) == 4 else sc)
                prev = b
                ent.append((e[0], e[1] / sc, e[2] / sc, a / sc, b / sc))
            self.foot[int(k)] = ent
        self.sec = {}          # gid -> (line, km)
        self.closed = set()
        for l in self.lines:
            shut = set(l.get("closed") or [])
            for s in l["sections"]:
                self.sec[s[3]] = (l, s[2])
                if f"{s[0]}|{s[1]}" in shut:
                    self.closed.add(s[3])
        self._own()

    def foot_of(self, gid):
        return self.foot.get(gid) or [(gid, 0, 1, 0, 1)]

    def _own(self):
        raw = defaultdict(list)
        for l in self.lines:
            if l.get("service"):
                continue
            for s in l["sections"]:
                if s[3] in self.closed:
                    continue
                for t, fr, to, a, b in self.foot_of(s[3]):
                    te = self.sec.get(t)
                    if not te or te[0]["id"] != l["id"] or t in self.closed or fr == to:
                        continue
                    raw[t].append((min(fr, to), max(fr, to)))
        own = {t: merge_spans(v, self.sec[t][1]) for t, v in raw.items()}
        for l in self.lines:
            l["_own"] = sum(s[2] * span_len(own.get(s[3], [])) for s in l["sections"]
                            if s[3] not in self.closed)
            off = 0.0
            for s in l["sections"]:
                if s[3] in self.closed:
                    continue
                for t, fr, to, a, b in self.foot_of(s[3]):
                    te = self.sec.get(t)
                    if te and te[0].get("src") == "osm":
                        off += s[2] * abs(b - a)
            l["_off"] = off

    @staticmethod
    def listed(l):
        if l.get("src") != "osm":
            return not (l["km"] > 0.05) or l["_own"] > 0.05
        if l.get("service"):
            return False
        return l["_off"] > 0.05 and l["_off"] >= 0.4 * l["km"]

    def coverage(self):
        """Register line id -> {(tier, raw operator): share of its km}, tier R (routes) or N
        (named trains)."""
        cover = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for l in self.lines:
            if l.get("src") != "osm":
                continue
            ops = parts(l.get("operator"))
            if not ops:
                continue
            tier = "N" if l.get("service") else "R"
            for s in l["sections"]:
                for t, fr, to, a, b in self.foot_of(s[3]):
                    te = self.sec.get(t)
                    if not te or te[0].get("src") == "osm" or fr == to:
                        continue
                    for op in ops:
                        cover[te[0]["id"]][(tier, op)][t].append((min(fr, to), max(fr, to)))
        out = {}
        for l in self.lines:
            if l.get("src") == "osm":
                continue
            km = sum(s[2] for s in l["sections"] if s[3] not in self.closed) or 1e-9
            per = {}
            for k, byt in cover.get(l["id"], {}).items():
                per[k] = sum(self.sec[t][1] * span_len(merge_spans(v, self.sec[t][1]))
                             for t, v in byt.items() if t not in self.closed) / km
            out[l["id"]] = per
        return out


# ------------------------------------------------------------------ the table

def load_table():
    rows = {}
    if os.path.exists(CSV):
        with open(CSV, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                rows[(r["cc"], r["operator"])] = {c: r.get(c, "") or "" for c in COLS}
    return rows


def save_table(rows):
    order = sorted(rows.values(), key=lambda r: (-float(r.get("km") or 0), r["cc"], r["operator"]))
    tmp = CSV + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in order:
            w.writerow({c: r.get(c, "") for c in COLS})
    os.replace(tmp, CSV)


def fill(r, col, val, why=None):
    """Set a blank cell; never overwrite a filled one."""
    if r.get(col):
        return False
    if val in (None, ""):
        return False
    r[col] = val
    if why:
        r["note"] = (r["note"] + "; " if r.get("note") else "") + why
    return True


def apply_rules(rows, nets=None):
    """Fill blank role and group cells from INFRA_RULES, GROUP_RULES and NETWORK_GROUPS.
    `nets`: (cc, raw) -> Counter(network tag -> km of its urban rail lines)."""
    for (cc, op), r in rows.items():
        if not r.get("role") and op in INFRA_RULES.get(cc, ()):
            r["role"] = "infra"
        if r.get("group"):
            continue
        for ccs, rx, g in GROUP_RULES:
            if (ccs is None or cc in ccs) and re.match(rx, op):
                if g != op:
                    r["group"] = g
                break
        if r.get("group") or not nets or op in NETWORK_KEEP.get(cc, ()):
            continue
        groups = NETWORK_GROUPS.get(cc, {})
        c = nets.get((cc, op))
        if groups and c:
            n = c.most_common(1)[0][0]
            if n in groups and groups[n] != op:
                r["group"] = groups[n]
                r["note"] = (r["note"] + "; " if r.get("note") else "") + f"group: network {n}"


def urban_networks(countries):
    nets = defaultdict(Counter)
    for cc, C in countries.items():
        if cc not in NETWORK_GROUPS:
            continue
        for l in C.lines:
            if l.get("src") != "osm" or l.get("kind") not in URBAN:
                continue
            n = (l.get("network") or "").strip()
            if n:
                for op in parts(l.get("operator")):
                    nets[(cc, op)][n] += l["km"]
    return nets


def is_infra(rows, cc, op):
    r = rows.get((cc, op))
    return bool(r) and r.get("role") == "infra"


# ------------------------------------------------------------------ OP_EN as the app has it

def op_en_votes(countries):
    """The app's OP_EN with every country loaded (votes over all lines, first-seen wins ties),
    and the same vote per country."""
    glob_v, per_v = {}, defaultdict(dict)
    for cc, C in countries.items():
        for l in C.lines:
            raw = (l.get("operator") or "").strip()
            en_ = (l.get("operator_en") or "")
            if not raw or ";" in raw or ";" in en_:
                continue
            en = en_ or (raw if LATIN.match(raw) else None)
            if not en:
                continue
            for m in (glob_v.setdefault(raw, {}), per_v[cc].setdefault(raw, {})):
                m[en] = m.get(en, 0) + 1
    best = lambda m: max(m.items(), key=lambda x: x[1])[0] if m else None   # first max wins
    return ({r: best(m) for r, m in glob_v.items()},
            {cc: {r: best(m) for r, m in d.items()} for cc, d in per_v.items()})


# ------------------------------------------------------------------ the switch

def home_country(rows, raw_cc_km, items):
    """(raw) -> set of countries it is at home in: its Wikidata item's P17, else where its
    own routes (or, lacking any, its lines) run the most km."""
    # A grouped string is at home where its group's own row is (START on a Czech EuroCity is
    # MÁV's, Hungarian).
    group_iso = {}
    for (c, o), r in rows.items():
        q = r.get("qid") or ""
        if q.startswith("Q") and items.get(q, {}).get("iso") and "weak" not in (r.get("note") or ""):
            group_iso.setdefault(o, set(items[q]["iso"]))

    def home(cc, op):
        r = rows.get((cc, op)) or {}
        g = r.get("group") or ""
        if g and g != "-" and ";" not in g and g in group_iso:
            return group_iso[g]
        q = r.get("qid") or ""
        if q.startswith("Q") and items.get(q, {}).get("iso"):
            return set(items[q]["iso"])
        rk = raw_cc_km.get(op) or {}
        if not rk:
            return {cc}
        m = max(rk.items(), key=lambda x: x[1])[0]
        return {m}
    return home


def switch(C, rows, home):
    """Register line id -> (list of (raw, share)) for switched lines."""
    cc = C.cc
    cov = C.coverage()
    out = {}
    for l in C.lines:
        if l.get("src") == "osm":
            continue
        reg = parts(l.get("operator"))
        if reg and not all(is_infra(rows, cc, o) for o in reg):
            continue                       # the register operator runs trains itself
        per = {k: s for k, s in cov.get(l["id"], {}).items() if not is_infra(rows, cc, k[1])}
        R = {op: s for (t, op), s in per.items() if t == "R"}
        N = {op: s for (t, op), s in per.items() if t == "N"}
        N = {op: s for op, s in N.items() if ISO.get(cc, cc) in home(cc, op)}
        kept = {op: s for op, s in R.items() if s >= SHARE_MIN}
        for op, s in N.items():
            if s >= SHARE_MIN and op not in kept:
                kept[op] = s
        if not kept:
            best = sorted(((s, op) for op, s in {**N, **R}.items()), reverse=True)
            if best and best[0][0] >= SHARE_FLOOR:
                kept = {best[0][1]: best[0][0]}
        if kept:
            out[l["id"]] = sorted(((op, round(min(1.0, s), 3)) for op, s in kept.items()),
                                  key=lambda x: -x[1])
        elif cc in COUNTRY_DEFAULT:
            out[l["id"]] = [(COUNTRY_DEFAULT[cc], 0.0)]
        else:
            out[l["id"]] = [(o, 0.0) for o in reg] if reg else []
    return out


# ------------------------------------------------------------------ keys

def compute_keys(rows, glob_en):
    """(cc, raw) -> list of keys."""
    base = {}
    for (cc, op), r in rows.items():
        base[(cc, op)] = glob_en.get(op) or op
    # Same strong Wikidata item within a country: the biggest row's key.
    byq = defaultdict(list)
    for (cc, op), r in rows.items():
        q = r.get("qid") or ""
        if q.startswith("Q") and "weak" not in (r.get("note") or "") and r.get("group") in ("", "-"):
            byq[(cc, q)].append((float(r.get("km") or 0), op))
    alias = {}
    for (cc, q), lst in byq.items():
        if len(lst) > 1:
            lst.sort(reverse=True)
            for _, op in lst[1:]:
                alias[(cc, op)] = base[(cc, lst[0][1])]
    keys = {}
    for (cc, op), r in rows.items():
        g = r.get("group") or ""
        if g and g != "-":
            keys[(cc, op)] = [k.strip() for k in g.split(";") if k.strip()]
        elif (cc, op) in alias:
            keys[(cc, op)] = [alias[(cc, op)]]
        else:
            keys[(cc, op)] = [base[(cc, op)]]
    # The same key reached, without a group, from two countries whose rows have different
    # Wikidata items: listed for a look, not renamed. Most are one brand in several countries
    # (Transdev, Arriva, Keolis); a real clash is settled by a group cell, as India's Southern
    # Railway is ("Indian Railways").
    who = defaultdict(lambda: defaultdict(set))      # key -> cc -> qids
    for (cc, op), ks in keys.items():
        r = rows[(cc, op)]
        if len(ks) != 1 or (r.get("group") or "-") != "-":
            continue
        q = r.get("qid") or ""
        if q.startswith("Q") and "weak" not in (r.get("note") or ""):
            who[ks[0]][cc].add(q)
    clash = {k: {cc: sorted(q) for cc, q in d.items()} for k, d in who.items()
             if len(d) > 1 and len(set().union(*d.values())) > 1}
    return keys, clash


# ------------------------------------------------------------------ measure

def gather(only=None):
    ccs = only or regions()
    countries = {}
    for cc in ccs:
        if os.path.exists(os.path.join(DATA, cc, "lines.json")):
            countries[cc] = Country(cc)
    return countries


def raw_km(countries):
    """raw -> cc -> km of OSM routes (non-service) carrying it; named trains when no routes."""
    rk, nk = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(float))
    for cc, C in countries.items():
        for l in C.lines:
            if l.get("src") != "osm":
                continue
            for op in parts(l.get("operator")):
                (nk if l.get("service") else rk)[op][cc] += l["km"]
    return {op: dict(rk.get(op) or nk.get(op)) for op in set(rk) | set(nk)}


def load_items():
    p = os.path.join(CACHE, "items.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def run_all(countries, rows):
    """The switch for every country and the km per row after it."""
    blank = lambda cc, op: {c: "" for c in COLS} | {"cc": cc, "operator": op}
    for cc, C in countries.items():
        for l in C.lines:
            for op in parts(l.get("operator")):
                rows.setdefault((cc, op), blank(cc, op))
    for cc, op in COUNTRY_DEFAULT.items():
        if cc in countries:
            rows.setdefault((cc, op), blank(cc, op))
    for (cc, op), q in EXTRA_ROWS.items():
        r = rows.setdefault((cc, op), blank(cc, op))
        if q:
            fill(r, "qid", q, "qid: hand")
    apply_rules(rows, urban_networks(countries))
    items = load_items()
    home = home_country(rows, raw_km(countries), items)
    sw = {}
    for cc, C in countries.items():
        sw[cc] = switch(C, rows, home)
    # rows for every string, km after the switch (listed km a row's lines own)
    km = defaultdict(float)
    for cc, C in countries.items():
        for l in C.lines:
            if not C.listed(l):
                continue
            ops = [o for o, s in sw[cc][l["id"]]] if l["id"] in sw[cc] else parts(l.get("operator"))
            for op in ops:
                km[(cc, op)] += l["_own"]
    for k, r in rows.items():
        if k[0] in countries:
            r["km"] = f"{km.get(k, 0):.0f}"
    return sw


def before_after(countries, rows, sw, keys, glob_en, n=10):
    out = {}
    for cc, C in countries.items():
        b, a = Counter(), Counter()
        for l in C.lines:
            if not C.listed(l):
                continue
            ob = sorted({glob_en.get(p) or p for p in parts(l.get("operator"))}) or ["(no operator)"]
            for k in ob:
                b[k] += l["_own"]
            if l["id"] in sw[cc]:
                raws = [o for o, s in sw[cc][l["id"]]]
            else:
                raws = parts(l.get("operator"))
            ka = sorted({k for o in raws for k in keys.get((cc, o), [glob_en.get(o) or o])}) or ["(no operator)"]
            for k in ka:
                a[k] += l["_own"]
        out[cc] = (b.most_common(n), a.most_common(n))
    return out


# ------------------------------------------------------------------ Wikidata

def http_json(url, tries=6):
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503, 502, 504):
                wait = int(e.headers.get("Retry-After") or 0) or 20 * (i + 1)
                print(f"  {e.code}, waiting {wait} s", flush=True)
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            print("  retry:", e, flush=True)
            time.sleep(15 * (i + 1))
    raise SystemExit("gave up on " + url[:200])


def sparql(q):
    time.sleep(2)
    return http_json("https://query.wikidata.org/sparql?format=json&query=" + urllib.parse.quote(q))["results"]["bindings"]


def qid_of(uri):
    return uri.rsplit("/", 1)[1]


def cache_json(name, default):
    p = os.path.join(CACHE, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else default


def save_cache(name, obj):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, name)
    json.dump(obj, open(p + ".tmp", "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(p + ".tmp", p)


def line_votes(ccs):
    """(cc, raw) -> Counter(line item) from the route relations' `wikidata` tags."""
    out = defaultdict(Counter)
    for cc in ccs:
        p = os.path.join(ROOT, "data", "proc", cc, "rels.pkl")
        if not os.path.exists(p):
            continue
        r = pickle.load(open(p, "rb"))
        for v in r.values():
            tags = v[0] if isinstance(v, tuple) else v
            if tags.get("type") not in ("route", "route_master"):
                continue
            for wd in parts(tags.get("wikidata")):
                if not re.fullmatch(r"Q\d+", wd):
                    continue
                for op in parts(tags.get("operator")):
                    out[(cc, op)][wd] += 1
    return out


def fetch_p137(qs):
    p137 = cache_json("p137.json", {})
    missing = [q for q in sorted(qs) if q not in p137]
    print(f"P137 of {len(missing)} line items", flush=True)
    for i in range(0, len(missing), 200):
        vals = " ".join("wd:" + q for q in missing[i:i + 200])
        got = defaultdict(list)
        for b in sparql(f"SELECT ?l ?op WHERE {{ VALUES ?l {{ {vals} }} ?l wdt:P137 ?op . }}"):
            got[qid_of(b["l"]["value"])].append(qid_of(b["op"]["value"]))
        for q in missing[i:i + 200]:
            p137[q] = got.get(q, [])
        save_cache("p137.json", p137)
    return p137


def fetch_items(qs):
    """label, P1813 short names, P465 colour, P154 logo, P17 country ISO."""
    items = cache_json("items.json", {})
    missing = [q for q in sorted(qs) if q not in items]
    print(f"properties of {len(missing)} items", flush=True)
    for i in range(0, len(missing), 80):
        vals = " ".join("wd:" + q for q in missing[i:i + 80])
        rows = sparql(f"""SELECT ?i ?lab ?short ?col ?logo ?iso ?lab2 WHERE {{ VALUES ?i {{ {vals} }}
          OPTIONAL {{ ?i rdfs:label ?lab FILTER(lang(?lab)='en') }}
          OPTIONAL {{ ?i wdt:P1813 ?short }}
          OPTIONAL {{ ?i wdt:P465 ?col }}
          OPTIONAL {{ ?i wdt:P154 ?logo }}
          OPTIONAL {{ ?i wdt:P17 ?c . ?c wdt:P297 ?iso }} }}""")
        for q in missing[i:i + 80]:
            items[q] = {"label": "", "short": [], "colour": [], "logo": [], "iso": []}
        for b in rows:
            d = items[qid_of(b["i"]["value"])]
            if b.get("lab"):
                d["label"] = b["lab"]["value"]
            if b.get("short"):
                v = b["short"]["value"] + "@" + b["short"].get("xml:lang", "")
                if v not in d["short"]:
                    d["short"].append(v)
            if b.get("col") and b["col"]["value"] not in d["colour"]:
                d["colour"].append(b["col"]["value"])
            if b.get("logo"):
                f = urllib.parse.unquote(b["logo"]["value"].rsplit("/", 1)[1])
                if f not in d["logo"]:
                    d["logo"].append(f)
            if b.get("iso") and b["iso"]["value"].lower() not in d["iso"]:
                d["iso"].append(b["iso"]["value"].lower())
        save_cache("items.json", items)
    return items


def search(text, lang):
    srch = cache_json("search.json", {})
    k = f"{lang}|{text}"
    if k not in srch:
        time.sleep(4)          # faster than this draws 429s with 40 s Retry-After
        url = "https://www.wikidata.org/w/api.php?" + urllib.parse.urlencode(
            {"action": "wbsearchentities", "search": text[:250], "language": lang, "uselang": "en",
             "strictlanguage": "false", "type": "item", "limit": 7, "format": "json"})
        try:
            srch[k] = [x["id"] for x in http_json(url).get("search", [])]
        except urllib.error.HTTPError as e:
            print("  search failed", e.code, text)
            srch[k] = []
        save_cache("search.json", srch)
    return srch[k]


def seed(args):
    countries = gather()
    rows = load_table()
    run_all(countries, rows)
    ccs = list(countries)
    # 1. line items -> P137
    votes = line_votes(ccs)
    p137 = fetch_p137({q for c in votes.values() for q in c})
    cand = {}
    for k, c in votes.items():
        v = Counter()
        for lq, n in c.items():
            for op in p137.get(lq, []):
                v[op] += n
        if v:
            cand[k] = v.most_common(2)
    # 2. search, for the biggest rows still without a strong vote
    ranked = sorted(rows.values(), key=lambda r: -float(r.get("km") or 0))
    want = []
    for r in ranked:
        if r.get("qid"):
            continue
        k = (r["cc"], r["operator"])
        c = cand.get(k)
        if c and c[0][1] >= 3:
            continue
        want.append(r)
    want = [r for r in want if float(r.get("km") or 0) > 0][:args.search]
    hits = {}
    for i, r in enumerate(want):
        h = search(r["operator"], LANG.get(r["cc"], "en"))
        if not h and LANG.get(r["cc"], "en") != "en":
            h = search(r["operator"], "en")
        hits[(r["cc"], r["operator"])] = h
        if i % 50 == 0:
            print(f"  searched {i}/{len(want)}", flush=True)
    allq = {q for c in cand.values() for q, n in c} | {q for h in hits.values() for q in h}
    allq |= {r["qid"] for r in rows.values() if (r.get("qid") or "").startswith("Q")}
    items = fetch_items(allq)
    for (cc, op), r in rows.items():
        if r.get("qid"):
            continue
        c = cand.get((cc, op))
        iso = ISO.get(cc, cc)
        if c and c[0][1] >= 3:
            fill(r, "qid", c[0][0], f"qid: {c[0][1]} line votes")
            continue
        h = [q for q in hits.get((cc, op), []) if iso in items.get(q, {}).get("iso", [])]
        if h:
            fill(r, "qid", h[0], "qid: search")
        elif c:
            fill(r, "qid", c[0][0], f"qid: weak, {c[0][1]} line vote(s)")
    items = fetch_items({r["qid"] for r in rows.values() if (r.get("qid") or "").startswith("Q")})
    for r in rows.values():
        q = r.get("qid") or ""
        it = items.get(q) if q.startswith("Q") else None
        if not it:
            continue
        sh = [s.rsplit("@", 1) for s in it["short"]]
        en = [s for s, l in sh if l in ("en", "en-gb", "en-us")]
        lat = [s for s, l in sh if LATIN.match(s)]
        fill(r, "short", (en or lat or [""])[0])
        if it["colour"]:
            c = it["colour"][0].lstrip("#").upper()
            if re.fullmatch(r"[0-9A-F]{6}", c) and fill(r, "colour", "#" + c):
                r["colour_src"] = r.get("colour_src") or "wikidata"
        if it["logo"]:
            fill(r, "logo", it["logo"][0])
    save_table(rows)
    print("table:", len(rows), "rows;", sum(1 for r in rows.values() if r.get("qid")), "with a qid")


# ------------------------------------------------------------------ logos

def slug(s):
    """An ASCII file name: the Latin part of the string and a short hash, so 東日本旅客鉄道 and
    ÖBB both get a plain URL."""
    import hashlib
    import unicodedata
    a = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    a = re.sub(r"[^a-z0-9]+", "-", a).strip("-")[:40].strip("-")
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:6]
    return f"{a}-{h}" if a else h


def file_info(files):
    """Commons licence, author and a thumbnail url per file (cached)."""
    info = cache_json("files.json", {})
    missing = [f for f in sorted(files) if f not in info or "w" not in info[f]]
    for i in range(0, len(missing), 40):
        chunk = missing[i:i + 40]
        time.sleep(4)
        url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query", "format": "json", "prop": "imageinfo|categories", "redirects": 1,
            "iiprop": "extmetadata|url|mime|size", "iiurlwidth": 330, "cllimit": "max",
            "iiextmetadatafilter": "LicenseShortName|Artist|AttributionRequired|Restrictions|UsageTerms",
            "titles": "|".join("File:" + f for f in chunk)})
        d = http_json(url)
        norm = {n["to"]: n["from"] for n in d["query"].get("normalized", [])}
        for rd in d["query"].get("redirects", []):
            norm[rd["to"]] = norm.get(rd["from"], rd["from"])
        for pg in d["query"]["pages"].values():
            t = norm.get(pg["title"], pg["title"]).split(":", 1)[1]
            ii = (pg.get("imageinfo") or [{}])[0]
            em = ii.get("extmetadata", {})
            cats = [c["title"].split(":", 1)[1] for c in pg.get("categories", [])]
            artist = re.sub(r"<[^>]+>", "", em.get("Artist", {}).get("value", "")).strip()
            info[t] = {"lic": em.get("LicenseShortName", {}).get("value", "missing" if "missing" in pg else ""),
                       "artist": artist[:200], "thumb": ii.get("thumburl", ""), "page": ii.get("descriptionurl", ""),
                       "attr": em.get("AttributionRequired", {}).get("value", ""),
                       "pdtext": any("textlogo" in c.lower().replace(" ", "") for c in cats),
                       "tm": any("trademark" in c.lower() for c in cats),
                       "w": ii.get("width", 0), "h": ii.get("height", 0), "cats": cats}
        for f in chunk:
            if f not in info or "w" not in info[f]:
                info[f] = {"lic": "missing", "thumb": "", "w": 0, "h": 0, "cats": []}
        save_cache("files.json", info)
    return info


FREE = re.compile(r"(?i)public domain|^pd|cc0|cc[ -]by|gfdl|attribution|free|kogl|ogl")


def logo_colour(img):
    """The most-used colour of the logo that is neither white, black nor grey."""
    im = img.convert("RGBA")
    im.thumbnail((200, 200))
    bins = defaultdict(list)
    n = 0
    for r, g, b, a in im.getdata():
        if a < 160:
            continue
        n += 1
        h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
        if s < 0.35 or l < 0.12 or l > 0.88:
            continue
        bins[(r // 16, g // 16, b // 16)].append((r, g, b))
    if not bins or not n:
        return ""
    k, px = max(bins.items(), key=lambda x: len(x[1]))
    if len(px) < 0.03 * n:
        return ""
    r, g, b = (round(sum(p[i] for p in px) / len(px)) for i in range(3))
    return "#%02X%02X%02X" % (r, g, b)


def mostly_white(img):
    im = img.convert("RGBA")
    im.thumbnail((200, 200))
    px = [(r, g, b) for r, g, b, a in im.getdata() if a >= 160]
    if not px:
        return True
    white = sum(1 for r, g, b in px if min(r, g, b) > 225)
    return white > 0.85 * len(px)


def clear_background(img):
    """A logo on an opaque white sheet (a JPG, a flattened PNG): the white around it made
    transparent, flooding in from each corner, so it sits on the panel like the others."""
    from PIL import ImageDraw
    im = img.convert("RGBA")
    w, h = im.size
    for xy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        r, g, b, a = im.getpixel(xy)
        if a > 200 and min(r, g, b) > 230:
            ImageDraw.floodfill(im, xy, (255, 255, 255, 0), thresh=40)
    return im


def logo_targets(n_max):
    """The table, each key's km, and key -> the row whose logo it shows, for the keys that
    get a logo: the top n_max by km and each country's top 8."""
    countries = gather()
    rows = load_table()
    run_all(countries, rows)
    glob_en, _ = op_en_votes(countries)
    keys, _ = compute_keys(rows, glob_en)
    brand = brands(rows, keys)
    km = Counter()
    bycc = defaultdict(Counter)
    for (cc, op), ks in keys.items():
        for k in ks:
            km[k] += float(rows[(cc, op)].get("km") or 0)
            bycc[cc][k] += float(rows[(cc, op)].get("km") or 0)
    want = {k for k, v in km.most_common(n_max) if v > 0}
    for cc, c in bycc.items():
        want |= {k for k, v in c.most_common(8) if v > 0}
    files = {}
    for k in want:
        b = brand.get(k) or {}
        if b.get("logo_row") is not None:
            files[k] = b["logo_row"]
    return rows, km, files


SQUARE = 1.4         # a logo this wide for its height or less counts as near-square
OLD = re.compile(r"(?i)\b(old|former|historic|retro|heritage|1[89]\d\d|white|negative|inverted|"
                 r"invert|mono|livery|train|station|sign|map|uniform|ticket|wagon|car)\b")
GENERIC_CAT = re.compile(r"(?i)^(pd|svg|png|files|images|logos? of|logos$|uploaded|media|"
                         r"trademark|cc-|text logos|.*textlogo|.*with .* logos?|.*\bcolou?r\b)")


def fetch_icons(qs):
    """P8972 (small logo or icon) and P2910 (icon) files per item (cached)."""
    icons = cache_json("icons.json", {})
    missing = [q for q in sorted(qs) if q not in icons]
    for i in range(0, len(missing), 80):
        vals = " ".join("wd:" + q for q in missing[i:i + 80])
        got = defaultdict(list)
        for b in sparql(f"""SELECT ?i ?f WHERE {{ VALUES ?i {{ {vals} }}
              {{ ?i wdt:P8972 ?f }} UNION {{ ?i wdt:P2910 ?f }} }}"""):
            f = urllib.parse.unquote(b["f"]["value"].rsplit("/", 1)[1])
            got[qid_of(b["i"]["value"])].append(f)
        for q in missing[i:i + 80]:
            icons[q] = got.get(q, [])
        save_cache("icons.json", icons)
    return icons


def category_files(cat):
    cats = cache_json("cats.json", {})
    if cat not in cats:
        time.sleep(4)
        url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query", "format": "json", "list": "categorymembers", "cmtype": "file",
            "cmlimit": 100, "cmtitle": "Category:" + cat})
        d = http_json(url)
        cats[cat] = [m["title"].split(":", 1)[1] for m in d.get("query", {}).get("categorymembers", [])]
        save_cache("cats.json", cats)
    return cats[cat]


NAME_STOP = {"rail", "railway", "railways", "railroad", "logo", "logos", "company", "corporation",
             "the", "of", "and", "de", "la", "le", "national", "state", "trains", "train",
             "transport", "transportation", "group", "ltd", "limited", "gmbh", "ag", "sa", "co",
             "svg", "png", "jpg", "with", "from", "operator", "new", "official", "text", "icon"}


def name_tokens(s):
    import unicodedata
    a = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return {t for t in re.findall(r"[a-z0-9]+", a) if len(t) >= 2 and t not in NAME_STOP
            and not t.isdigit()}


def aspect(fi):
    return (fi.get("w") or 0) / (fi.get("h") or 1) if fi.get("h") else 99


def square(args):
    """Swap a wide logo for a squarer version of the same mark (Anita, 2026-10-07: "ideally we
    use a closer to square version, if it exists"): the item's P8972 or P2910 file first,
    then a file in the current logo's own logo category on Commons; free files only; the
    current logo stays when nothing squarer exists. A hand-set logo is left alone."""
    rows, km, files = logo_targets(args.max)
    order = sorted(files, key=lambda k: -km[k])
    info = file_info({r["logo"] for r in files.values()})
    icons = fetch_icons({r["qid"] for r in files.values() if (r.get("qid") or "").startswith("Q")})
    items = load_items()
    info = file_info({f for r in files.values() for f in icons.get(r.get("qid") or "", [])})
    changed = []
    for k in order:
        r = files[k]
        cur = r["logo"]
        if "logo: hand" in (r.get("note") or "") or aspect(info.get(cur, {})) <= SQUARE:
            continue
        ok = lambda f: (FREE.search(info.get(f, {}).get("lic", "")) and info[f].get("thumb")
                        and aspect(info[f]) <= SQUARE and not OLD.search(f))
        pick, how = None, ""
        cands = [f for f in icons.get(r.get("qid") or "", []) if f != cur]
        for f in sorted(cands, key=lambda f: not f.lower().endswith(".svg")):
            if ok(f):
                pick, how = f, "icon"
                break
        if not pick:
            # Only a category named for this operator ("Amtrak logos"), and only files
            # named for it too: colour categories ("Red logos with transparent background")
            # had offered Acid Jazz Records for DB.
            toks = name_tokens(" ".join([k, r.get("short") or "", r["operator"],
                                         (items.get(r.get("qid") or "", {}) or {}).get("label", ""),
                                         os.path.splitext(cur)[0]]))
            has = lambda s: bool(toks & name_tokens(s))
            pool = []
            for c in info.get(cur, {}).get("cats", []):
                if "logo" in c.lower() and not GENERIC_CAT.match(c) and has(c):
                    # JPGs in a logo category are mostly photos of the logo on something
                    pool += [f for f in category_files(c) if f != cur and has(f)
                             and not f.lower().endswith((".jpg", ".jpeg"))]
            if pool:
                info = file_info(set(pool))
                good = [f for f in dict.fromkeys(pool) if ok(f)]
                # an SVG, then one named as a logo or symbol, then the squarest
                good.sort(key=lambda f: (not f.lower().endswith(".svg"),
                                         not re.search(r"(?i)logo|symbol|emblem|icon|mark", f),
                                         abs(aspect(info[f]) - 1)))
                if good:
                    pick, how = good[0], "category"
        if pick:
            changed.append((k, cur, pick, how))
            r["note"] = (r["note"] + "; " if r.get("note") else "") + f"logo: squarer ({how}), was {cur}"
            r["logo"], r["logo_licence"] = pick, ""
            print(f"{km[k]:8.0f} {k[:34]:34} {aspect(info[cur]):4.1f} -> {aspect(info[pick]):4.1f}  {cur} -> {pick}")
    save_table(rows)
    print(len(changed), "logos swapped for squarer ones")


def logos(args):
    from PIL import Image
    rows, km, files = logo_targets(args.max)
    info = file_info({r["logo"] for r in files.values()})
    os.makedirs(LOGO_DIR, exist_ok=True)
    os.makedirs(os.path.join(CACHE, "logos"), exist_ok=True)
    credits = {}
    total = 0
    done = 0
    for k in sorted(files, key=lambda k: -km[k]):
        r = files[k]
        f = r["logo"]
        fi = info.get(f) or {}
        if not fi.get("thumb") or not FREE.search(fi.get("lic", "")):
            print("  skip (licence or no thumb):", k, f, fi.get("lic"))
            continue
        raw = os.path.join(CACHE, "logos", slug(f) + ".png")
        if not os.path.exists(raw):
            time.sleep(2.5)
            data = None
            for i in range(6):
                try:
                    data = urllib.request.urlopen(urllib.request.Request(fi["thumb"], headers=UA), timeout=60).read()
                    break
                except urllib.error.HTTPError as e:
                    if e.code in (429, 503):
                        wait = int(e.headers.get("Retry-After") or 0) or 30 * (i + 1)
                        print(f"  {e.code}, waiting {wait} s", flush=True)
                        time.sleep(wait)
                        continue
                    print("  failed", e.code, f)
                    break
                except Exception as e:
                    print("  failed", e, f)
                    time.sleep(10)
            if not data:
                continue
            open(raw, "wb").write(data)
        try:
            img = Image.open(raw)
            img.load()
        except Exception as e:
            print("  unreadable", f, e)
            continue
        img = clear_background(img)
        if mostly_white(img):
            print("  skip (white, would vanish on the light map):", k, f)
            continue
        col = logo_colour(img)
        if col and fill(r, "colour", col):
            r["colour_src"] = r.get("colour_src") or "logo"
        out = img.convert("RGBA")
        bbox = out.getbbox()
        if bbox:
            out = out.crop(bbox)
        # 28 px high (14 shown, 2x), but never wider than LOGO_W: a wide wordmark comes out
        # shorter instead of stretching the row. The app shows it at half size, aspect kept.
        if out.width / out.height <= LOGO_W / LOGO_H:
            size = (max(1, round(out.width * LOGO_H / out.height)), LOGO_H)
        else:
            size = (LOGO_W, max(1, round(out.height * LOGO_W / out.width)))
        out = out.resize(size, Image.LANCZOS)
        name = slug(r["cc"] + "-" + r["operator"]) + ".png"
        p = os.path.join(LOGO_DIR, name)
        out.info.clear()                   # no ICC profile or EXIF carried over (SNCF's was 0.5 MB)
        out.save(p, optimize=True)
        r["logo_licence"] = r.get("logo_licence") or (fi.get("lic", "") + (f"; {fi['artist']}" if fi.get("artist") else ""))
        credits[name] = {"file": f, "page": fi.get("page", ""), "licence": fi.get("lic", ""),
                         "author": fi.get("artist", ""), "pd_textlogo": fi.get("pdtext", False),
                         "trademark": fi.get("tm", False)}
        total += os.path.getsize(p)
        done += 1
        if total > 30e6:
            print("logos past 30 MB, stopping")
            break
    for p in glob.glob(os.path.join(LOGO_DIR, "*.png")):
        if os.path.basename(p) not in credits:
            os.remove(p)                   # a logo since dropped from the table
    json.dump(credits, open(os.path.join(LOGO_DIR, "credits.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    save_table(rows)
    print(f"{done} logos, {total / 1e6:.2f} MB")


def cache_json_path(p, default):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else default


# ------------------------------------------------------------------ build

def brands(rows, keys):
    """key -> {short, colour, logo_row}: from the row that is the key itself (its own string
    or the biggest row of a Wikidata alias), else, for colour and logo, the biggest member."""
    members = defaultdict(list)
    for (cc, op), ks in keys.items():
        if len(ks) == 1:
            members[ks[0]].append(rows[(cc, op)])
    out = {}
    for k, ms in members.items():
        strong = lambda r: (r.get("qid") or "").startswith("Q") and "weak" not in (r.get("note") or "")
        # The row that is the key: its own string, a hand-set or strongly joined one first.
        ms = sorted(ms, key=lambda r: (r["operator"] != k, "hand" not in (r.get("note") or ""),
                                       not strong(r), -float(r.get("km") or 0)))
        own = [r for r in ms if r["operator"] == k or (r.get("group") or "-") == "-"]
        lead = own[0] if own else None
        b = {"short": ((lead or {}).get("short") or "").strip("-")}
        # A lead row with its own Wikidata item (or "-") speaks for the key alone: Indian
        # Railways must not borrow its Northern Railway zone's logo.
        src = [lead] if lead and lead.get("qid") and "weak" not in (lead.get("note") or "") \
            else ([lead] if lead else []) + [r for r in ms if r is not lead]
        b["colour"] = next((r["colour"] for r in src if r.get("colour") not in ("", "-")), "")
        if lead and lead.get("colour") == "-":
            b["colour"] = ""
        b["logo_row"] = next((r for r in src if r.get("logo")), None)
        if b["logo_row"] is not None and b["logo_row"]["logo"] == "-":
            b["logo_row"] = None
        out[k] = b
    return out


def build(args):
    countries = gather()
    rows = load_table()
    sw = run_all(countries, rows)
    glob_en, _ = op_en_votes(countries)
    keys, rename = compute_keys(rows, glob_en)
    br = brands(rows, keys)
    credits = cache_json_path(os.path.join(LOGO_DIR, "credits.json"), {})
    ops = {}
    for k, b in br.items():
        e = {}
        if b["short"] and b["short"] != k:
            e["short"] = b["short"]
        if b["colour"]:
            e["colour"] = b["colour"]
        lr = b["logo_row"]
        if lr is not None:
            name = slug(lr["cc"] + "-" + lr["operator"]) + ".png"
            if name in credits and os.path.exists(os.path.join(LOGO_DIR, name)):
                e["logo"] = "data/logos/" + name
        if e:
            ops[k] = e
    # The keys the app gives today (OP_EN), where they now fold into another key.
    for (cc, op), ks in keys.items():
        old = glob_en.get(op) or op
        if len(ks) == 1 and ks[0] != old and old not in br:
            e = dict(ops.get(ks[0], {}))
            e["group"] = ks[0]
            ops.setdefault(old, e)
    tmp = os.path.join(DATA, "operators.json.tmp")
    json.dump(dict(sorted(ops.items())), open(tmp, "w", encoding="utf-8"), ensure_ascii=False,
              separators=(",", ":"))
    os.replace(tmp, os.path.join(DATA, "operators.json"))
    for cc, C in countries.items():
        used = {o for l in C.lines for o in parts(l.get("operator"))}
        used |= {o for v in sw[cc].values() for o, s in v}
        kmap = {}
        for o in sorted(used):
            ks = keys.get((cc, o), [glob_en.get(o) or o])
            kmap[o] = ks[0] if len(ks) == 1 else ks
        lines = {}
        for lid, v in sw[cc].items():
            agg = {}
            for o, s in v:
                for k in keys.get((cc, o), [o]):
                    agg[k] = max(agg.get(k, 0), s)
            lines[lid] = [[k, round(s, 2)] for k, s in sorted(agg.items(), key=lambda x: -x[1])]
        p = os.path.join(DATA, cc, "ops.json")
        json.dump({"keys": kmap, "lines": lines}, open(p + ".tmp", "w", encoding="utf-8"),
                  ensure_ascii=False, separators=(",", ":"))
        os.replace(p + ".tmp", p)
    save_table(rows)
    print(f"operators.json: {len(ops)} keys; ops.json for {len(countries)} countries")
    print("keys several countries reach with different Wikidata items (settle with a group cell "
          "if they are different companies):")
    for k, d in sorted(rename.items()):
        print(f"  {k}: {d}")


def measure(args):
    countries = gather(args.cc or None)
    rows = load_table()
    sw = run_all(countries, rows)
    glob_en, _ = op_en_votes(countries)
    keys, rename = compute_keys(rows, glob_en)
    ba = before_after(countries, rows, sw, keys, glob_en)
    res = {}
    for cc, (b, a) in ba.items():
        n_sw = sum(1 for v in sw[cc].values() if v)
        res[cc] = {"before": b, "after": a, "switched": n_sw}
        print(f"== {cc} ({n_sw} register lines switched)")
        for i in range(max(len(b), len(a))):
            x = f"{b[i][0][:38]} {b[i][1]:.0f}" if i < len(b) else ""
            y = f"{a[i][0][:38]} {a[i][1]:.0f}" if i < len(a) else ""
            print(f"   {x:48} | {y}")
    if args.out:
        json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if args.save:
        save_table(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("cc", nargs="*")
    m.add_argument("--out")
    m.add_argument("--save", action="store_true", help="also write the table (new rows, km)")
    s = sub.add_parser("seed")
    s.add_argument("--search", type=int, default=400)
    lg = sub.add_parser("logos")
    lg.add_argument("--max", type=int, default=300)
    sq = sub.add_parser("square")
    sq.add_argument("--max", type=int, default=300)
    sub.add_parser("build")
    a = ap.parse_args()
    {"measure": measure, "seed": seed, "logos": logos, "build": build, "square": square}[a.cmd](a)
