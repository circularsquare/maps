"""The line lists mideast_register.py converts (nafrica_lines.py's format): one entry per
passenger line, its stations in order. Point syntax: nafrica_register.py's docstring. Sources
and reasoning: sa_, ae_, iq_, jo_sources.md. Qatar has no main-line railway, so no list."""
from nafrica_lines import L

# Border points: id -> (lon, lat, [countries]). No passenger train crosses a border here.
BORDERS = {}

# Route relations that are no scheduled passenger service, or that OSM maps before they open;
# mideast_register --clip <cc> drops them (nafrica_register.clip). Each extract is cut from the
# one gcc-states file by bbox, so a relation can turn up in a neighbour's extract too: the
# Gulf's are listed under every Gulf code.
_GULF = {
    # Saudi Arabia
    1273546: "Al Mashaaer Al Mugaddassah Metro (Mecca), southbound: runs only about a week "
             "a year, at Hajj, for permit holders (sa_sources.md)",
    7734267: "Al Mashaaer Al Mugaddassah Metro, northbound: as above",
    7734268: "Al Mashaaer Al Mugaddassah Metro's route_master",
    # Qatar
    10563805: "Lusail Tram Purple Line: under construction, not open (qa_sources.md)",
    # UAE
    7826308: "Dubai Trolley (Westward): the Downtown tourist tram stopped years ago "
             "(ae_sources.md)",
    7826309: "Dubai Trolley (Eastward): as above",
}
NOT_SERVICE = {"sa": dict(_GULF), "ae": dict(_GULF), "qa": dict(_GULF), "iq": {}, "jo": {}}

SAR = "الخطوط الحديدية السعودية"          # Saudi Arabia Railways
ETIHAD = "الاتحاد للقطارات"               # Etihad Rail

# Saudi Arabia: SAR's three passenger lines, stations as SAR's booking site sells them
# (sa_sources.md). `listed_only`: the trains call at these and nowhere else.
SA = [
    L("east", "قطار الشرق: الرياض – الدمام", "East Train: Riyadh – Dammam", SAR,
      ["Riyadh Railway Station", "Al Hufuf Railway Station", "Abqaiq",
       "Dammam Railway Station"], listed_only=True,
      note="the 1981 direct line Riyadh - Hofuf; the old line via Al Kharj and Harad is "
           "freight"),
    L("north", "قطار الشمال: الرياض – القريات", "North Train: Riyadh – Al Qurayyat", SAR,
      ["Riyadh North railway station", "Majmmah", "Al Qassim", "Hail", "Al-Jawf",
       "Al Qurayyat"], listed_only=True),
    L("haramain", "قطار الحرمين السريع", "Haramain High Speed Railway", SAR,
      ["Haramain High Speed Railway - Makkah",
       "Haramain High Speed Railway - Al-Sulimaniyah - Jeddah",
       "Haramain High Speed Railway - King Abdulaziz International Airport",
       "King Abdullah Economic City", "Haramain High Speed Railway - Madinah"],
      listed_only=True),
]

# The UAE: Etihad Rail's passenger trains, Abu Dhabi (Mohamed Bin Zayed City) - Al Dhaid -
# Fujairah, and Abu Dhabi - Dubai (Al Yalayis), which part 10.5 km short of Al Yalayis
# (nafrica_register --fork, 2026-10-08). One line with the Dubai stretch as its branch.
JCT_DUBAI = "~Al Yalayis junction@55.23932,24.95923"
AE = [
    L("etihad", "قطار الاتحاد: أبوظبي – الفجيرة", "Etihad Rail: Abu Dhabi – Fujairah", ETIHAD,
      ["Abu Dhabi Mohamed bin Zayed City Station", JCT_DUBAI, "Al Dhaid Station",
       "Fujairah Station"],
      more=[[JCT_DUBAI, "Dubai Al Yalayis Station"]], listed_only=True),
]

IRR = "الشركة العامة لسكك حديد العراق"     # Iraqi Republic Railways
JHR = "سكة حديد الحجاز الأردنية"            # Jordan Hejaz Railway

# Iraq: IRR's three trains that run daily (iq_sources.md). OSM's track is in pieces a few
# metres apart in ~90 places; `mideast_register --join iq` joins them after --clip.
IQ = [
    # The Baghdad - Basra night train, at the calling points Seat61 gives (no published stop
    # list): every OSM halt it passes would otherwise become a stop.
    L("south", "الخط الجنوبي: بغداد – البصرة", "Southern Line: Baghdad – Basra", IRR,
      ["Central Baghdad Railway Station", "Hilla/Babylon Train Station", "Al Diwaniyah",
       "Samawah Railway Station", "Nasiriya Railway Station", "Maqal Railway Station"],
      listed_only=True),
    # The two commuter trains: their stops are the line's stations, as OSM maps them.
    L("north", "الخط الشمالي: بغداد – سامراء", "Northern Line: Baghdad – Samarra", IRR,
      ["Central Baghdad Railway Station", "Kadhimya Railway Station", "Taji Railway Station",
       "Al Mushahidah", "Balad Railway Station", "Ishaq Railway Station",
       "Samarra Railway Station"]),
    L("west", "الخط الغربي: بغداد – الفلوجة", "Western Line: Baghdad – Fallujah", IRR,
      ["Central Baghdad Railway Station", "Abu Ghraib Train Station",
       "Al-Hamdaniya Train Station", "Garma Railway Station", "Fallujah Railway Station"]),
    # Branches with pilgrim specials only (Arbaeen, a few weeks a year): greyed, so a rider
    # who went can still mark them. Each starts at the main-line station its trains come
    # through; the branch leaves ~1 km beyond it.
    L("karbala", "فرع المسيب – كربلاء", "Musayyib – Karbala branch", IRR,
      ["Musayeb Railway Station", "Karbala Train Station"], suspended=True, listed_only=True),
    L("umqasr", "فرع الشعيبة – أم قصر", "Shuaiba – Umm Qasr branch", IRR,
      ["Shuaiba Railway Station", "Umm Qasr"], suspended=True, listed_only=True),
]

# Jordan: the Hejaz Railway's excursion train out of Amman, Fridays and Saturdays
# (jo_sources.md). OSM has no Al Jeezah station: it is placed at the station's yard on the
# line (--fill adds it there).
JO = [
    L("jeezah", "سكة حديد الحجاز: عمان – الجيزة", "Hejaz Railway: Amman – Al Jeezah", JHR,
      ["Amman Station", "Al Jeezah@35.9625,31.7120"], listed_only=True),
]

LINES = {"sa": SA, "ae": AE, "qa": [], "iq": IQ, "jo": JO}
