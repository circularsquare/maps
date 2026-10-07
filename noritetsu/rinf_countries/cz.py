"""Czechia: Správa železnic's RINF id is "540-00_1501": SŽ's own line number (540), a branch
suffix (-00 the line, -01.. connecting and parallel tracks, -90.. siding leads) and SŽ's
traťový úsek (1501). Neither part is the number riders know. Riders know the timetable
(KJŘ) number, "trať 010 Kolín - Česká Třebová": it heads every table in the printed and online
timetable, is on station departure boards, and is how cs.wikipedia and Wikidata's P1671 number
lines. SŽ's 540 appears only in its network statement. So the ref is the KJŘ number.

There is no id-to-KJŘ rule (SŽ 540 is KJŘ 010, 220 is 190, 280 is 220), so every number comes
from OSM's route=tracks relations, which carry the KJŘ number ("010 – Kolín – Česká
Třebová"). THE SAME `ref` KEY ALSO CARRIES SŽ's NUMBERS on a second set of relations
("220 Nemanice - Plzeň hlavní nádraží" is SŽ 220 = KJŘ 190, while "220 – Benešov u Prahy –
České Budějovice" is KJŘ 220). `cz_rel` tells them apart by name: a KJŘ relation's name
starts with its number, then a dash or a name written with en dashes; SŽ's are "NNN A - B"
with hyphens only. KJŘ numbers run 010-346, so anything higher is SŽ's (or Polish PLK's,
whose relations reach over the border). ČD's duplicates of a few Jeseník lines are dropped
(one is labelled 291 for the Sobotín branch, which is 293), and so are freight-only and
disused relations. REL_FIX corrects five relations that carry another line's number and two
named in German.

Siding ids (SŽ's -90.. leads and the private sidings' 0xx ids) never take a number
(`cz_no_ref`), or every siding junction would become a branch point of its line, and all but
the Zubrnice museum railway are left out of the build (`cz_skip`, rinf.py's `skip_line`).

Names are the timetable's form, "010 Kolín – Česká Třebová", read from the relation's name.
Wikidata is not fetched ("wikidata": None): its labels are "železniční trať A – B" or
"railway line 125", and rinf.py would prefer them to the relation's name.
"""
import re

KJR_MAX = 346


def kjr_ref(ref):
    r = (ref or "").strip()
    m = re.fullmatch(r"(\d{3})([a-z]?)", r)
    if not m or int(m.group(1)) > KJR_MAX:
        return None
    return r


def clean_name(ref, name):
    """"250 - Havlíčkův Brod - Brno - Břeclav - Kúty" -> "250 Havlíčkův Brod – Brno – Břeclav
    – Kúty". Only spaced hyphens are separators; "Praha-Libeň" keeps its hyphen."""
    n = name.strip()
    n = re.sub(rf"^{re.escape(ref)}\s*[–:-]?\s*", "", n)
    n = re.sub(r"^(AŽD|JMHD|JHMD|Souhrnná doprava)\s+", "", n)
    n = re.sub(r"\s*\((?:[–-]\s*)?[^)]*\)", "", n)         # "(provozovaná část)", "(– Ždánice)"
    n = re.sub(r"\s+[–-]\s+", " – ", n)
    n = re.sub(r"(\s–)+\s*$", "", n)
    n = re.sub(r"(–\s*){2,}", "– ", n)
    return f"{ref} {n.strip()}"


# OSM relations whose timetable number is another line's, by (OSM ref, cleaned name). The
# right numbers are Wikidata's P1671 on each line's item (Q576892 Nepomuk - Blatná, whose own
# label says 191; Q1033137 Křižanov - Studenec 252; Q779732 Šakvice - Hustopeče 254). Where no
# source gives the number, the relation keeps its name and loses the number rather than
# joining an unrelated line: Děčín - Schöna and Hrušovany u Brna - Židlochovice were each
# being grouped with a line at the other end of the country.
REL_FIX = {
    ("192", "Nepomuk – Blatná"): ("191", "191 Nepomuk – Blatná"),
    ("257", "Křižanov – Studenec"): ("252", "252 Křižanov – Studenec"),
    ("251", "Šakvice – Hustopeče u Brna"): ("254", "254 Šakvice – Hustopeče u Brna"),
    ("251", "Hrušovany u Brna – Židlochovice"): (None, "Hrušovany u Brna – Židlochovice"),
    ("083", "Děčín – Schöna Gr."): (None, "Děčín – Schöna"),
    # Named in German in OSM; the same places in Czech (Falkenau = Sokolov, Graslitz =
    # Kraslice, Franzensbad = Františkovy Lázně).
    ("145", "Falkenau – Graslitz – Zwotental"): ("145", "145 Sokolov – Kraslice – Zwotental"),
    ("147", "Franzensbad – Bad Brambach"): ("147", "147 Františkovy Lázně – Bad Brambach"),
}


# Connecting curves and short pieces OSM files under a line's number with a name of their own.
# They keep the number but not the name, so the line is named after its main relation.
PIECES = {("090", "Ústí nad Labem jih – Ústí nad Labem západ"),
          ("114", "odb. Vrbka – odb. Bažantnice"), ("131", "Obrnice – odb. České Zlatníky"),
          ("135", "Třebušice – Most nové nádr."), ("161", "Blatno-Rakovník"),
          ("164", "Kadaňský Rohozec – Doupov"),
          ("171", "Praha-Vršovice vjezd. n. – Praha-Radotín"),
          ("270", "spojka Drahotuše – Hranice na Moravě"),
          ("320", "Dětmarovice – Petrovice u Karviné – státní hranice"),
          ("322", "odb. Závada – odb. Koukolná"), ("063", "Dolní Bousov – Kopidlno")}


def cz_rel(tags):
    """(ref, name) of an OSM route=railway/route=tracks relation, or None to ignore it."""
    got = _cz_rel(tags)
    if got and got[0]:
        key = (got[0], got[1][len(got[0]) + 1:])
        if key in PIECES:
            return got[0], None
        return REL_FIX.get(key, got)
    return got


def _cz_rel(tags):
    name = (tags.get("name") or "").strip()
    ref = (tags.get("ref") or "").strip()
    # "line Praha-Libeň - Praha-Bubeneč", "železniční trať Cheb–Schirnding" (ref 179)
    name = re.sub(r"^line\s+", "", name)
    trat = re.match(r"železniční trať\s+", name, re.I)
    if ref and trat:
        name = f"{ref} – {name[trat.end():]}"
    if re.search(r"nákladní|bývalá|opuštěná|zrušená|nikdy", name, re.I):
        return None
    if tags.get("operator") == "ČD":
        return None
    if not ref:
        # "293 ŽD Železnice Desná" (SART's Petrov nad Desnou - Kouty / Sobotín) has its
        # number only in the name.
        m = re.match(r"(\d{3})\s+(?:ŽD\s+)?(.+)", name)
        if m and kjr_ref(m.group(1)):
            return m.group(1), f"{m.group(1)} {m.group(2)}"
        return (None, name) if name else None
    k = kjr_ref(ref)
    if not k or not name.startswith(k):
        return None
    rest = name[len(k):]
    if int(k[:3]) >= 100 and not (re.match(r"\s*[–-]", rest) or "–" in rest
                                  or re.match(r"\s*(AŽD|JMHD|JHMD)\b", rest)):
        return None                              # SŽ's own number, "120 Chomutov - Cheb"
    return k, clean_name(k, name)


# Siding ids never take a line number: SŽ's siding leads (-90..-99, "562-90_1491" from the main
# line to the siding's boundary point "hi ...") and the private sidings' own ids (0xx-yy, and
# ArcelorMittal Ostrava's V34-xx). They lie beside their line's relation and would join its
# group, which turns every siding junction into a branch point that build_model then cuts out
# wherever no OSM route runs. 000-00_2051 is the one real line among the 0xx ids (Vranovice -
# Pohořelice, SŽ's own).
SIDING = re.compile(r"^(\d{3}-9\d_|0\d\d-|V34-)")
NOT_SIDING = {"000-00_2051"}


def cz_no_ref(lid):
    return bool(SIDING.match(lid or "")) and lid not in NOT_SIDING


# And they are left out of the build altogether (`skip_line`), except the one with passenger
# trains: 032-64_1001 Velké Březno - Zubrnice, the Zubrnice museum railway (OSM route T3).
# Built as lines of their own, 56 sidings ending at a matched station survived build_model
# as register lines nobody can ride.
RIDDEN_SIDING = {"032-64_1001"}


def cz_skip(lid):
    return cz_no_ref(lid) and lid not in RIDDEN_SIDING


COUNTRY = {
    # join a line's pieces where RINF leaves a stretch out, over its own OSM relation
    # (rinf.fill_holes; trialled 2026-10-05)
    "fill_holes": True,
    "iso3": "CZE", "wikidata": None, "langs": ["cs"],
    "osm_rel": cz_rel, "no_ref": cz_no_ref, "skip_line": cz_skip,
    # RINF names no infrastructure manager; these are read off the lines each one holds and
    # the lines' cs.wikipedia articles (cz_sources.md). Codes holding only sidings are left
    # unnamed.
    "im": {"0054_IM": "Správa železnic", "3218_IM": "JHMD", "3324_IM": "PDV Railway",
           "3325_IM": "SART-stavby a rekonstrukce", "3559_IM": "AŽD Praha",
           "3642_IM": "Railway Capital", "3145_IM": "PKP Cargo International"},
}
