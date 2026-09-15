"""The "nothing is truly dead" list: every negative record that is missing a part it needs.

WORKFLOW_PLAN.md item 8; the rule is spec §12, *No country is closed for good*. A negative is a
record of what was tried, and the next session can only reopen it if the record says:

    date      when it was checked: YYYY-MM-DD, or a month and year. A date in the section heading
              also counts and is shown in `flags` as `date(h)`.
    searched  WHICH RELEASE was read: a URL or host, a numbered table, form item or variable, a
              named file, the UNSD oracle, a survey round, or a document with its year. An office,
              a series or an instrument on its own does not count.
    result    what came back: a 403, a login, no religion item, national only, N categories...
    reopen    a trigger: REOPEN, reopen when, would reopen, next place to look, not checked,
              unchecked, and similar.
    scope     the record either closes at the questionnaire or data dictionary (spec §12's most
              durable kind), or says what it did NOT check.

`searched` as above and `scope` are the tightened test. `--loose` is the four-part test as first
written, where a named office or instrument counted as searched and there was no `scope`; it rated
every negative that later turned out wrong as missing only its trigger, level with most of the
live list. `--calibrate` shows this.

Where records come from: all of queue.md (the *Closed, with the reason* section and any *Closed
in §11xx* list are negatives by position, everything else by wording) and sources.md's §11
sections. A country registered in countries.py is not a negative any more unless the record is
about a finer or newer source for it: those rows are kind `upgrade`, the rest `country`. Dropped,
and counted: records about drawn countries that are not upgrades; records whose own sentence says
the country was drawn, reopened or is buildable; and every record for a country that some status
line calls buildable or gives a route (Sierra Leone's closed Afrobarometer row is not Sierra Leone
being closed).

    python tools/negatives.py                     each country's most complete record, worst 40
    python tools/negatives.py --records           every record rather than one per country
    python tools/negatives.py --top 0             no limit
    python tools/negatives.py --out PATH          also write both tables as markdown
    python tools/negatives.py --cc gn,zm --all    only these; --all keeps drawn countries' records
    python tools/negatives.py --loose             the untightened four-part test
    python tools/negatives.py --calibrate         both tests on the original Guinea, Zambia,
                                                  Mozambique, Honduras and Argentina negatives

`flags`: `oracle YYYY/N` when UNSD table 28 holds a census religion tabulation (N categories) and
the record is not a questionnaire negative. That was true of Guinea, Zambia and Mozambique when
each was closed: the census asked, so a table existed somewhere. `no-wayback` marks a record that
reports a dead, moved or refusing host and never mentions the Wayback Machine, which is how
Mozambique's tables were reached in the end. `same-shape` marks the shape every wrong negative in
the calibration had: missing both `reopen` and `scope`, and closed on a table or volume not being
there (`not found` / `not published`) rather than on the questionnaire; `--shape` lists only those.
`before §9xx` marks a sources.md record older than that country's latest §9 build section, which
may already answer it and which this scan does not read.

Sorted by parts missing, then population: the record's own `people` cell (`rec`), else the oracle's
latest census total (`unsd`), else Natural Earth's estimate (`ne`).

Matches on text, never on line position (CLAUDE.md), so other sessions' edits do not break it; the
line numbers it prints are only where to look.
"""

import argparse
import json
import os
import re
import sys
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

NE = os.path.join(ROOT, "data", "geo", "ne_10m_admin_0_countries.geojson")

# ---------------------------------------------------------------------------------------------
# The parts
# ---------------------------------------------------------------------------------------------

MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
         r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
DATE = re.compile(r"\b(?:19|20)\d\d-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b"
                  rf"|\b{MONTH}\.?\s+(?:19|20)\d\d\b")

YEAR = r"\b(?:1[89]\d\d|20\d\d)\b"
DOC = (r"(?:census(?:es)?|censos?|recensements?|surveys?|enqu[eê]tes?|encuestas?|reports?|"
       r"rapports?|volumes?|booklets?|livrets?|brochur[ea]s?|yearbooks?|questionnaires?|forms?|"
       r"dictionar(?:y|ies)|rounds?|waves?|releases?|results|r[ée]sultats|resultados|tables?|"
       r"PHC|HPC|RGPH\w*|RGPL|RGP/H|"
       r"MICS\d*|DHS|EDS|ENDESA|LSIS|PESS|SHDS|EPHS|EDHS|CNPV|EICV\w*|ECVMAS|ENEMDU|ENHOGAR|"
       r"monographs?|abstract|atlas|profiles?|bulletin|compendium|publications?|analytical|"
       r"thematic|th[èe]me|chapters?|chapitre|annex)")
_SEARCHED_TIGHT = [
    r"https?://",
    # a host: lowercase labels, a gTLD or a two-letter ccTLD, not a file extension or a call
    r"\b[a-z0-9][a-z0-9-]+(?:\.[a-z0-9-]{2,})*\.(?:gov|gob|gouv|org|com|net|int|edu|info|"
    r"(?!md\b|py\b|js\b|sh\b|gz\b|db\b|pb\b|ts\b|rs\b)[a-z]{2})\b(?![(_])",
    r"\b(?:Table|Tableau|Tabla|Quadro|QUADRO|Cuadro|CUADRO|Graphique|Figure|Form|item|question|"
    r"Q)\s?[A-Z]?\d+(?:[.\-]\d+)*[a-z]?\b",
    r"\bAnne(?:x|xe|xo)\s+(?:\d|[A-Z]\b)",
    r"`[A-Za-z][A-Za-z0-9]*[0-9_][A-Za-z0-9_]*`",               # a variable: `HC1`, `q3c`
    r"(?i)[\w-]+\.(?:pdf|xlsx?|zip|sav|dta|csv|rar|docx?)\b",   # a named file
    r"\boracle\b|\bUNSD\b|[Tt]able 28|Demographic Yearbook",
    rf"(?i){YEAR}[^.;|]{{0,60}}?\b{DOC}\b|\b{DOC}\b[^.;|]{{0,60}}?{YEAR}",
    r"\bRGPH-?\s?\d|\b[IVX]+ RGPH|\bRGPH\d",
    r"\b(?:Afrobarometer|Arab Barometer|LAPOP|AmericasBarometer|ESS|WVS|EVS|LiTS|Central Asia "
    r"Barometer|Caucasus Barometer|Latinobar[oó]metro|ISSP|JGSS|GFS)\s+(?:R\d|[Rr]ounds?\s+\d|"
    r"[Ww]aves?\s+(?:\d|[IVX]+\b)|\d{4}|[IVX]+\b)",
    r"\bR\d(?:-R\d)?\b",                                        # Afrobarometer rounds
    r"\bU\.S\.C\. \d+",                                         # a statute read
]
_SEARCHED_LOOSE = _SEARCHED_TIGHT + [
    r"\b(?:INE|INEC|INEI|INIDE|INS|INSD|INSEED|INSTAT|INSBU|ISTEEBU|IHSI|DANE|INDEC|IBGE|BPS|NBS|"
    r"UBOS|ZamStats|Stats SL|GBoS|ANSD|ANStat|CNSEE|BUCREP|CAPMAS|PBS|BBS|CBS|SSB|SCB|Statbel|"
    r"Destatis|INSEE|ISTAT|ELSTAT|KSH|GUS|PSA|DOSM|NSO|NISR|KNBS|GSS|CSA|LSB|PACI|DOS|CAS|BNS|"
    r"Diyanet|Presidency|USCB|IPUMS|DataFirst|World Bank|UNICEF|SPC|Knoema|[Mm]inistry|"
    r"[Mm]inist[èe]re|[Cc]ensus [Bb]ureau|[Ss]tatistics [Oo]ffice)\b",
    r"\b(?:Afrobarometer|Arab Barometer|LAPOP|AmericasBarometer|ESS|WVS|EVS|LiTS|Central Asia "
    r"Barometer|Caucasus Barometer|Latinobar[oó]metro|ISSP|JGSS|Global Flourishing Study|Pew|"
    r"DHS|MICS|REDATAM|NADA|PX-?Web|Wayback|CDX|ArcGIS|HDX|COD-AB|microdata|census)\b",
]
SEARCHED = {"tight": [re.compile(p) for p in _SEARCHED_TIGHT],
            "loose": [re.compile(p) for p in _SEARCHED_LOOSE]}

RESULT = re.compile(
    r"(?i)\b(?:40[0134]|418|429|50[0-3]|520)s?\b|SERVFAIL|\btime[sd]? out\b|\btimeout|"
    r"refus\w* connection|Cloudflare|captcha|interstitial|\blog-?in\b|sign-?in|\baccount\b|"
    r"registration|licen[cs]ed|embargo|\bgated\b|restricted|port-filtered|squatted|"
    r"(?:does not|no longer) resolve|\bnot ask|\bdo(?:es)? not ask|\basks? (?:none|no\b)|"
    r"\basked none|never asked|\bno religion\b|\bno (?:census |religion )?(?:item|question|"
    r"variable|chapter|table|volume|file|sheet|geography|district|province|region|path)|"
    r"\bnone (?:is|carries|mentions|found)|\bzero\b|\babsent\b|\bempty\b|not found|\bnothing\b|"
    r"\bnational(?:ly)?\b|urban/rural only|\d+ categor|\bshallow|\bundivided|as a chart|"
    r"unpublished|never published|published (?:no|nothing)|not published|\bdeleted|\bdropped|"
    r"\bquota\b|not usable|\bstale\b|\bexcluded\b|\bno route\b|\bdead\b|\bgone\b|\bthin\b|"
    r"nothing passes|\bswings\b|\+0\.\d\d")

TRIGGER = [
    re.compile(r"\bREOPEN\b"),
    re.compile(r"(?i)\breopens?\s+(?:when|if|once|the moment|as soon|after)\b|\bwould reopen\b|"
               r"\bto reopen\b|\b(?:next|first) place to look\b|\bunchecked\b|"
               r"\bnot (?:yet )?(?:checked|chased|opened|tried|searched|probed|confirmed|"
               r"verified|looked for)\b|\bunverified\b|\bunconfirmed\b|\bstays open\b|"
               r"\bstill open\b|\b(?:also|should|to|worth a) re-?(?:probe|test|check)\b|"
               r"\bworth (?:one|a) (?:browser )?(?:session|look|re-?check)\b|\bunexhausted\b|"
               r"\bever revisited\b|\bre-?check in\b|"
               r"\b(?:when|if|once) (?:the |its |a |an )?[\w\s'-]{0,50}?\b(?:publishes|is "
               r"published|are published|opens|releases|is released)\b"),
]

NOT_ASKED = re.compile(
    r"(?i)\bdo(?:es)? not ask|\bdid not ask|\bnot asked|\bnever asked|\basks? (?:none|no "
    r"religion)|\basked none|\bno (?:census )?religion (?:question|item|variable)|deleted the "
    r"religion question|lost its religion question|none is religion|\bno item\b|without the item|"
    r"\bno religion or ethnicity sheet")
QUESTIONNAIRE = re.compile(
    r"(?i)\bquestionnaire|\bforms?\b|\bdictionar|\bcodebook|\bvariable (?:list|picker)|"
    r"\b\d+ (?:person )?variables\b|\bmetadata\b|\bcard\b")
NOT_CHECKED = re.compile(
    r"(?i)\b(?:not|never) (?:yet )?(?:checked|chased|opened|tried|searched|probed|looked (?:for|"
    r"at))\b|\bunchecked\b|\bunverified\b|\bunconfirmed\b|\bnot confirmed\b")

# spec §12's four kinds, most durable first
TYPES = [
    ("not asked", NOT_ASKED),
    ("not published", re.compile(
        r"(?i)published? (?:no|nothing|none)\b|never published|not published|unpublished|"
        r"national(?:ly)? only|nationally or not at all|published nationally|national and "
        r"urban/rural|urban/rural only|nothing (?:below|finer)|stops at the national|"
        r"religion (?:table )?is national|prints? religion nationally")),
    ("not found", re.compile(
        r"(?i)not found|no religion (?:volume|table|chapter|file|sheet|statistics)|zero religion|"
        r"nothing (?:found|official)|none (?:found|mentions)|\babsent\b|\b\d+ categories|"
        r"shallow|undivided|as a chart|no route|not usable|\bquota\b|no (?:province|district|"
        r"region)|no path")),
    ("walled", re.compile(
        r"(?i)\b40[13]\b|\b418\b|\b429\b|SERVFAIL|time[sd]? out|refus\w* connection|Cloudflare|"
        r"captcha|interstitial|\blog-?in\b|sign-?in|\baccount\b|registration|licen[cs]ed|"
        r"embargo|\bgated\b|restricted|port-filtered|blocked on reach|unreachable")),
]


def parts(text, heading_date=None, mode="tight"):
    """-> (missing: list[str], notes: dict). Only the record's own text is tested."""
    missing, notes = [], {}
    if DATE.search(text):
        notes["date"] = "text"
    elif heading_date:
        notes["date"] = "heading"
    else:
        missing.append("date")
    if not any(p.search(text) for p in SEARCHED[mode]):
        missing.append("searched")
    if not RESULT.search(text):
        missing.append("result")
    if not any(p.search(text) for p in TRIGGER):
        missing.append("reopen")
    if mode == "tight":
        durable = NOT_ASKED.search(text) and QUESTIONNAIRE.search(text)
        if not (durable or NOT_CHECKED.search(text)):
            missing.append("scope")
    notes["type"] = "+".join(n for n, rx in TYPES if rx.search(text)) or "?"
    return missing, notes


# ---------------------------------------------------------------------------------------------
# Which records are negatives, and which have been undone since
# ---------------------------------------------------------------------------------------------

NEG = [
    re.compile(
        r"(?i)(?<!not )(?<!not\* )(?<!never )\bclosed\b"
        r"(?! (?:to|on) (?:the |its )?(?:person|national|printed|totals?))|"
        r"\bclose[sd]? (?:at|on) (?:the |its )?(?:questionnaire|forms?|census form|a document|"
        r"what the offices)|\bcloses (?:twice|outright|for good)\b|\bnot open\b|\bno route\b|"
        r"\bblocked\b|\bdead\b(?! end)|\bunobtainable\b|\bnot available\b|\bunavailable\b|"
        r"\bunreachable\b|\bdo not build\b|\bdo(?:es)? not ask\b|\bdid not ask\b|"
        r"\bnever asked\b|\basks? none\b|\basked none\b|\bno (?:census )?religion (?:question|"
        r"item|variable|table|volume|chapter|data|statistics)\b|\bzero religion\b|"
        r"\bnational(?:ly)? only\b|\bnationally or not at all\b|\bpublished nationally\b|"
        r"\bnothing (?:below|finer|to draw)\b|\bnot published\b|\bnever published\b|"
        r"\bpublished nothing\b|\bhas published no\b|\bruled out\b|\bexhausted\b|"
        r"\bnot usable\b|\bdisqualif\w+|\bdo not chase\b|\bdo not re-derive\b|\bwalled\b|"
        r"\bnot drawable\b|\bdeclined\b|\bno (?:official )?source\b|\bno path\b|"
        r"\bnobody should (?:go )?look|\bno (?:province|district|provincial|regional) table|"
        r"\breligion (?:table )?is national\b|\bnational and urban/rural\b|"
        r"\brefus\w* connections\b"),
    re.compile(r"\bOUT\."),
]
REVERSED = re.compile(
    r"\b(?:DRAWN|REDRAWN|BUILT|[Bb]uildable|[Dd]rawable|REOPENED|[Rr]eopened|[Rr]eopens? on|"
    r"[Ss]uperseded|SUPERSEDED|[Ww]ithdrawn|[Ss]urvey (?:build|route)|[Rr]oute exists|[Ww]ired|"
    r"(?:is|are|was|were) (?:now )?(?:drawn|built)|drawn 20\d\d|[Cc]orrected 20\d\d|"
    r"(?:would have been|was|were) wrong|settled|overturned|after all)\b")
OPEN = re.compile(
    r"\b(?:[Bb]uildable|[Dd]rawable|REOPENED|[Rr]eopened|[Rr]eopens? on|[Ss]urvey (?:build|"
    r"route)|[Rr]oute exists|BUILT)\b")
FINER_NEWER = re.compile(
    r"(?i)\bfiner\b|\bnewer\b|\bbelow the (?:nation|province|region|r[ée]gion|department|"
    r"d[ée]partement|district|state|oblast|county|governorate|prefecture)|\b(?:district|commune|"
    r"municipal\w*|municipio|rayon|village|constituenc\w+|ward|parish|upazila|tehsil|county|"
    r"counties|arrondissement|canton|kommune|gemeente\w*|barrio|comarca|zoba|chiefdom|woreda|"
    r"kebele|LGA|prefecture\w*|sub-?(?:district|county|province|prefecture))s?\b|"
    r"\bprovince tables\b|\bby province\b|\bprovincial\b|\bnext census\b|"
    r"\b20(?:2[3-9]) (?:census|round|wave|RGPH|PHC)\b|\bupgrade\b|\bwould improve\b|"
    r"\bcustom (?:cross-)?tabulation|\bmaatwerk\b|\blater (?:round|wave|census)\b")
HOST_TROUBLE = re.compile(r"(?i)refus\w* connection|\bSERVFAIL\b|does not resolve|no longer "
                          r"resolves|\bmoved\b|\bretired\b|server is gone|\b404\b|time[sd]? out")
WAYBACK = re.compile(r"(?i)wayback|\bCDX\b|archive\.org|archived")


def neg_spans(s):
    return [m.span() for p in NEG for m in p.finditer(s)]


CITE = re.compile(r"§\d+[a-z]*(?:'s)?\b[^.;]{0,40}$|\bhad (?:been )?$|\*\"[^\"]*$|“[^”]*$")


def live_negs(s):
    """NEG matches in s that are not a citation of an earlier record: `§11t declined Peru`,
    `what §11ad had disqualified`, a quoted *"no route of any kind"*."""
    s = flat(s)
    return [(a, b) for a, b in neg_spans(s) if not CITE.search(s[max(0, a - 60):a])]


# ---------------------------------------------------------------------------------------------
# Country names
# ---------------------------------------------------------------------------------------------

ALIAS = {
    "ivory coast": "ci", "cote d'ivoire": "ci", "turkiye": "tr", "turkey": "tr", "eswatini": "sz",
    "swaziland": "sz", "cabo verde": "cv", "cape verde": "cv", "sao tome": "st", "the gambia": "gm",
    "gambia": "gm", "dr congo": "cd", "drc": "cd", "democratic republic of the congo": "cd",
    "congo-brazzaville": "cg", "republic of the congo": "cg", "congo": "cg", "burma": "mm",
    "laos": "la", "vietnam": "vn", "viet nam": "vn", "south korea": "kr", "north korea": "kp",
    "russia": "ru", "czechia": "cz", "czech republic": "cz", "north macedonia": "mk",
    "macedonia": "mk", "bosnia": "ba", "micronesia": "fm", "caribbean netherlands": "bq",
    "bonaire": "bq", "bonaire, sint eustatius and saba": "bq", "curacao": "cw",
    "saint martin": "mf", "saint barthelemy": "bl", "st kitts": "kn", "uae": "ae",
    "britain": "uk", "england": "uk", "scotland": "uk", "wales": "uk", "northern ireland": "uk",
    "timor-leste": "tl", "east timor": "tl", "the bahamas": "bs", "falklands": "fk",
    "us virgin islands": "vi", "u.s. virgin islands": "vi", "bvi": "vg", "turks and caicos": "tc",
    "state of palestine": "ps", "palestine": "ps", "holy see": "va", "vatican": "va",
    "china, hong kong sar": "hk", "hong kong": "hk", "china, macao sar": "mo", "macao": "mo",
    "moldova": "md", "brunei": "bn", "syria": "sy", "iran": "ir", "tanzania": "tz",
    "kyrgyzstan": "kg", "tajikistan": "tj", "png": "pg", "kosovo": "xk", "sint maarten": "sx",
    "saint pierre and miquelon": "pm", "saint helena": "sh", "the netherlands": "nl",
    "the philippines": "ph", "united states": "us", "usa": "us",
}
NE_FIELDS = ("NAME", "NAME_LONG", "ADMIN", "NAME_EN", "BRK_NAME", "GEOUNIT", "SUBUNIT",
             "FORMAL_EN", "NAME_SORT")


def norm(s):
    """Lower-case, accents off, whitespace collapsed. Keeps positions close to `flat(s)`'s."""
    s = unicodedata.normalize("NFKD", flat(s).replace("’", "'"))
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def flat(s):
    return re.sub(r"\s+", " ", s)


class Countries:
    def __init__(self):
        self.names, self.pop, self.label = {}, {}, {}
        with open(NE, encoding="utf-8") as fh:
            feats = json.load(fh)["features"]
        for f in feats:
            p = f["properties"]
            iso = str(p.get("ISO_A2_EH") or "")
            if len(iso) != 2 or not iso.isalpha():
                continue
            cc = "uk" if iso == "GB" else iso.lower()
            self.label.setdefault(cc, p.get("NAME") or cc)
            if p.get("POP_EST"):
                self.pop[cc] = max(self.pop.get(cc, 0), int(float(p["POP_EST"])))
            for k in NE_FIELDS:
                if p.get(k):
                    self.names.setdefault(norm(p[k]), cc)
        self.names.update(ALIAS)
        self.codes = set(self.names.values())
        alts = sorted(self.names, key=len, reverse=True)
        self.rx = re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(a) for a in alts)
                             + r")(?![\w]|-bissau)")
        self.code_rx = re.compile(r"`([a-z]{2})`")

    def spans(self, text):
        """[(cc, start, end)] in `flat(text)` coordinates, names then backticked codes."""
        out = [(self.names[m.group(0)], m.start(), m.end()) for m in self.rx.finditer(norm(text))]
        out += [(m.group(1), m.start(), m.end()) for m in self.code_rx.finditer(flat(text))
                if m.group(1) in self.codes]
        return out

    def find(self, text):
        out = []
        for cc, _, _ in self.spans(text):
            if cc not in out:
                out.append(cc)
        return out

    def exact(self, name):
        name = re.sub(r"\s*\(.*?\)\s*", " ", name).strip()
        return self.names.get(norm(name))


# ---------------------------------------------------------------------------------------------
# Markdown into units
# ---------------------------------------------------------------------------------------------

SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
HEAD = re.compile(r"^(#{1,6})\s+(.*)")
BULLET = re.compile(r"^\s{0,3}(?:[-*]|\d+\.)\s+")
SENT = re.compile(r"(?<=[.;!?])\s+(?=[\"*_(`~A-Z§])")


def cells_of(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def units(lines):
    """-> dicts: kind row/bullet/para, line, text, cells, header, h2, h3, closed_list."""
    h2 = h3 = ""
    header, fence, closed_list, cur = None, False, False, None
    out = []

    def flush():
        nonlocal cur
        if cur:
            out.append(cur)
        cur = None

    for i, ln in enumerate(lines):
        if ln.strip().startswith("```"):
            flush()
            fence = not fence
            continue
        if fence:
            continue
        m = HEAD.match(ln)
        if m:
            flush()
            if len(m.group(1)) <= 2:
                h2, h3 = m.group(2), ""
            else:
                h3 = m.group(2)
            header, closed_list = None, False
            continue
        if not ln.strip():
            flush()
            header = None
            continue
        if ln.lstrip().startswith("|"):
            flush()
            if SEP.match(ln):
                continue
            if i + 1 < len(lines) and SEP.match(lines[i + 1]):
                header = [c.lower() for c in cells_of(ln)]
                continue
            out.append(dict(kind="row", line=i + 1, text=ln, cells=cells_of(ln), header=header,
                            h2=h2, h3=h3, closed_list=False))
            continue
        base = dict(line=i + 1, text=ln, cells=None, header=None, h2=h2, h3=h3)
        if BULLET.match(ln):
            flush()
            cur = dict(base, kind="bullet", closed_list=closed_list)
        elif cur and (cur["kind"] == "para" or ln.startswith(" ")):
            cur["text"] += "\n" + ln
        else:
            flush()
            closed_list = bool(re.match(r"\s*\**Closed in ", ln))
            cur = dict(base, kind="para", closed_list=False)
    flush()
    return out


BOUND = re.compile(r"[.;!?][\"”*_)]*\s+(?=[\"“*_(`~A-Z§])")


def sentences(text):
    """Split on sentence ends, including `build.** Germany` where bold closes after the stop."""
    t, out, start = flat(text), [], 0
    for m in BOUND.finditer(t):
        out.append(t[start:m.end()].strip())
        start = m.end()
    out.append(t[start:].strip())
    return [s for s in out if s]


def pieces(text):
    """A list written as `Name (reason); Name (reason)` -> [(names_chunk, piece)], else []."""
    chunks, kinds, depth, buf = [], [], 0, ""
    for ch in text:
        if ch == "(":
            if depth == 0:
                chunks.append(buf)
                kinds.append("out")
                buf = ""
            depth += 1
        buf += ch
        if ch == ")" and depth > 0:
            depth -= 1
            if depth == 0:
                chunks.append(buf)
                kinds.append("in")
                buf = ""
    chunks.append(buf)
    kinds.append("out")
    outside = sum(len(c) for c, k in zip(chunks, kinds) if k == "out")
    if outside > 0.45 * len(text):
        return []
    got, prev = [], ""
    for c, k in zip(chunks, kinds):
        if k == "out":
            prev = c
        else:
            got.append((prev, prev + c))
            prev = ""
    return got


def strip_struck(s):
    t = re.sub(r"~~.*?~~", " ", s)
    return t if re.search(r"[A-Za-z]{2}", plain(t)) else s


def plain(s):
    s = re.sub(r"\*\*|~~|`|\|", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clip(s, n=130):
    s = plain(s)
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------

def is_neg(text):
    return any(p.search(text) for p in NEG)


def heading_date(u):
    for h in (u["h3"], u["h2"]):
        m = DATE.search(h or "")
        if m:
            return m.group(0)
    return None


def about(C, sents, cc, lead=False):
    """The sentences that speak about cc: the ones naming it, and the ones naming no country."""
    out = []
    for i, s in enumerate(sents):
        named = C.find(s)
        if cc in named or not named or (lead and i == 0):
            out.append(s)
    return out


def row_parts(u):
    hdr = u["header"] or []
    idx = [j for j, h in enumerate(hdr) if re.search(r"\b(?:code|cc|country)\b", h)] or [0]
    subj = strip_struck(" ".join(u["cells"][j] for j in idx if j < len(u["cells"])))
    status = [c for j, c in enumerate(u["cells"]) if j not in idx]
    return subj, status, hdr


def records(fname, lines, C, only11):
    """-> (raw records, open countries). A record: cc, file, line, text, mine, quote, hdate, pop."""
    recs, opened = [], set()
    for u in units(lines):
        if only11 and not u["h2"].startswith("11"):
            continue
        in_closed = u["h2"].startswith("Closed, with the reason")
        hd = heading_date(u)
        text = u["text"]
        base = dict(file=fname, line=u["line"], hdate=hd)

        if u["kind"] == "row":
            subj, status, hdr = row_parts(u)
            ccs = C.find(subj)
            if not ccs:
                continue
            sents = [s for c in status for s in sentences(c)]
            for cc in ccs:
                if any(OPEN.search(s) for s in about(C, sents, cc)):
                    opened.add(cc)
            if not is_neg(" | ".join(status)):
                continue
            pop = None
            for j, h in enumerate(hdr):
                if j < len(u["cells"]) and re.search(r"people|population|pop\b", h):
                    pop = parse_people(u["cells"][j])
            for cc in ccs:
                mine = about(C, sents, cc)
                q = next((s for s in mine if is_neg(s)), None)
                if q:
                    recs.append(dict(base, cc=cc, text=text, mine=mine, quote=q, src="row",
                                     pop=pop if len(ccs) == 1 else None))
            continue

        forced = in_closed or u["closed_list"]
        split = [(n, t) for n, t in pieces(text) if C.find(n)]
        if len(split) >= 2:
            lead_neg = forced or is_neg(re.sub(r"\([^()]*(?:\([^()]*\)[^()]*)*\)", "", text))
            for names, t in split:
                if lead_neg or is_neg(t):
                    for cc in C.find(names):
                        recs.append(dict(base, cc=cc, text=t, mine=[t], quote=t, pop=None,
                                         src="piece"))
            continue

        sents = sentences(text)
        lead = re.match(r"^\s*(?:[-*]|\d+\.)?\s*(?:~~)?\*\*(.+?)\*\*", text)
        lead_cc = C.find(lead.group(1)) if lead else []
        if not lead_cc and lead is None and forced:
            head = re.split(r" — |: ", text, maxsplit=1)[0]
            lead_cc = C.find(head) if len(head) < 90 else []
        if lead_cc:
            for cc in lead_cc:
                if any(OPEN.search(s) for s in about(C, sents[:1], cc, lead=True)):
                    opened.add(cc)
            if not (forced or is_neg(text)):
                continue
            q = next((s for s in sents if is_neg(s)), sents[0] if sents else text)
            for cc in lead_cc:
                mine = sents[:1] + [s for s in sents[1:] if cc in C.find(s) or not C.find(s)]
                if not forced and not any(is_neg(s) for s in mine):
                    continue      # the paragraph's negative is about a country it names later
                qc = next((s for s in mine if is_neg(s)), q)
                recs.append(dict(base, cc=cc, text=text, mine=mine, quote=qc, pop=None,
                                 src="lead"))
            continue
        if forced:
            continue      # a closed-section line naming no country (PDH.stat) is not a country
        named_here = C.find(text)
        h3_cc = C.find(u["h3"])
        for s in sents:
            negs = live_negs(s)
            if not negs:
                continue
            near = []
            for cc, a, b in C.spans(s):
                if cc not in near and any(max(a, x) - min(b, y) <= 60 for x, y in negs):
                    near.append(cc)
            if not near and not C.find(s) and len(h3_cc) == 1:
                near = h3_cc
            if not near or len(near) > 6:
                continue
            for cc in near:
                own = [t for t in sents if cc in C.find(t)]
                body = text if len(named_here) <= 2 else " ".join(dict.fromkeys([s] + own))
                recs.append(dict(base, cc=cc, text=body, mine=[s] + own, quote=s, pop=None,
                                 src="sentence"))
    return recs, opened


def parse_people(cell):
    s = plain(cell).replace("~", "")
    m = re.search(r"(\d+(?:\.\d+)?)\s*([MmKk])\b", s)
    if m:
        return int(float(m.group(1)) * (1e6 if m.group(2) in "Mm" else 1e3))
    m = re.search(r"\d{1,3}(?:,\d{3})+", s)
    return int(m.group(0).replace(",", "")) if m else None


def drawn():
    from claim import drawn as _drawn
    return _drawn()


def oracle_index(C):
    """cc -> (latest year, categories, total) from UNSD table 28, when the cache is on disk."""
    try:
        import contextlib
        import io
        from oracle import table, partition, TOTAL
        with contextlib.redirect_stdout(io.StringIO()):
            t = table()
    except (SystemExit, Exception) as e:
        print(f"  (oracle unavailable: {e}; no oracle flag, population from Natural Earth)")
        return {}
    idx = {}
    for name, years in t.items():
        cc = C.exact(name)
        if not cc:
            continue
        y = max(years, key=int)
        cats, total, _ = partition(years[y].get(TOTAL, {}))
        total = total or sum(cats.values())
        if cc not in idx or int(y) > int(idx[cc][0]):
            idx[cc] = (y, len(cats), total)
    return idx


def assess(recs, opened, C, reg, orc, mode, keep_drawn):
    rows = []
    dropped = {"drawn": 0, "reversed": 0, "open": 0}
    for r in recs:
        cc = r["cc"]
        quote = r["quote"]
        if cc in reg:
            # an upgrade negative says the finer or newer thing next to the negative itself
            up = []
            for s in r["mine"]:
                t = flat(s)
                spans = live_negs(t) or ([] if r["src"] == "sentence" else neg_spans(t))
                if REVERSED.search(t) or not spans:
                    continue
                if any(max(m.start(), a) - min(m.end(), b) <= 80
                       for m in FINER_NEWER.finditer(t) for a, b in spans):
                    up.append(s)
            if up:
                kind, quote = "upgrade", up[0]
            elif keep_drawn:
                kind = "drawn"
            else:
                dropped["drawn"] += 1
                continue
        elif cc in opened:
            dropped["open"] += 1
            continue
        elif any(REVERSED.search(s) for s in r["mine"]):
            dropped["reversed"] += 1
            continue
        else:
            kind = "country"
        missing, notes = parts(r["text"], r["hdate"], mode)
        flags = []
        if notes.get("date") == "heading":
            flags.append("date(h)")
        if cc in orc and "not asked" not in notes["type"]:
            flags.append(f"oracle {orc[cc][0]}/{orc[cc][1]}")
        if HOST_TROUBLE.search(r["text"]) and not WAYBACK.search(r["text"]):
            flags.append("no-wayback")
        # the shape Guinea, Zambia, Mozambique, Honduras and Argentina had when they were closed:
        # absence in a release or listing, not the questionnaire, and nothing said to be unchecked
        if ("scope" in missing and "reopen" in missing and "not asked" not in notes["type"]
                and ("not found" in notes["type"] or "not published" in notes["type"])):
            flags.append("same-shape")
        if r["pop"]:
            pop, src = r["pop"], "rec"
        elif cc in orc and orc[cc][2]:
            pop, src = orc[cc][2], "unsd"
        else:
            pop, src = C.pop.get(cc, 0), "ne"
        rows.append(dict(r, kind=kind, missing=missing, type=notes["type"], flags=flags,
                         popn=pop, popsrc=src, name=C.label.get(cc, cc), quote=quote))
    seen, uniq = set(), []
    for x in rows:
        k = (x["cc"], x["file"], x["line"], x["quote"][:60])
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    return uniq, dropped


def scan(C, reg, orc, mode, keep_drawn=False):
    q = open(os.path.join(ROOT, "queue.md"), encoding="utf-8").read().split("\n")
    s = open(os.path.join(ROOT, "sources.md"), encoding="utf-8").read().split("\n")
    rq, oq = records("queue.md", q, C, only11=False)
    rs, os_ = records("sources.md", s, C, only11=True)
    rows, dropped = assess(rq + rs, (oq | os_) - reg, C, reg, orc, mode, keep_drawn)
    # a §11 record older than the country's latest §9 build section may have been answered by it,
    # which this scan cannot read (Uzbekistan's "not published" table was found in §9dp)
    builds = {}
    for i, ln in enumerate(s):
        m = re.match(r"^## (9[a-z]*)\. (.*)", ln)
        if m:
            for cc in C.find(m.group(2).split(" — ")[0]):
                builds[cc] = (i + 1, m.group(1))
    for x in rows:
        b = builds.get(x["cc"])
        if b and x["file"] == "sources.md" and x["line"] < b[0]:
            x["flags"].append(f"before §{b[1]}")
    return rows, dropped


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------

def fmt_pop(n, src):
    if not n:
        return "?"
    s = f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k" if n >= 1e3 else str(n)
    return f"{s} {src}"


def by_country(rows):
    best = {}
    for x in rows:
        k = (x["cc"], x["kind"])
        key = (len(x["missing"]), 1 if x["src"] == "sentence" else 0,
               0 if x["file"] == "queue.md" else 1, -x["line"])
        if k not in best or key < best[k][0]:
            best[k] = (key, x)
    return [dict(x, nrec=sum(1 for y in rows if (y["cc"], y["kind"]) == k))
            for k, (_, x) in best.items()]


def order(rows):
    return sorted(rows, key=lambda x: (-len(x["missing"]), -x["popn"], x["cc"], x["line"]))


def table_lines(rows, md, with_n):
    head = ["#", "cc", "country", "kind", "missing", "type", "pop", "where", "flags", "quote"]
    if with_n:
        head.insert(8, "records")
    out = []
    if md:
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "---|" * len(head))
    for i, x in enumerate(rows, 1):
        miss = ", ".join(x["missing"]) or "none"
        where = f"{x['file']}:{x['line']}"
        if md:
            vals = [str(i), x["cc"], x["name"], x["kind"], miss, x["type"],
                    fmt_pop(x["popn"], x["popsrc"]), where, "; ".join(x["flags"]),
                    clip(x["quote"])]
            if with_n:
                vals.insert(8, str(x["nrec"]))
            out.append("| " + " | ".join(v.replace("|", "/") for v in vals) + " |")
        else:
            fl = "{" + "; ".join(x["flags"]) + "}" if x["flags"] else ""
            n = f" ({x['nrec']} records)" if with_n else ""
            out.append(f"{i:>3} {x['cc']} {x['kind']:<7} {len(x['missing'])} [{miss}] "
                       f"{fmt_pop(x['popn'], x['popsrc'])} {where} {x['type']} {fl}{n}\n"
                       f"      {clip(x['quote'], 150)}")
    return out


def histogram(rows):
    h = {}
    for x in rows:
        h[len(x["missing"])] = h.get(len(x["missing"]), 0) + 1
    return "  ".join(f"{k} missing: {h[k]}" for k in sorted(h))


# ---------------------------------------------------------------------------------------------
# Calibration: the negatives that turned out wrong, as each read when it closed the country
# ---------------------------------------------------------------------------------------------

CALIBRATION = [
    # cc, where, heading date, a snippet still in the file, the text as it read, what reopened it
    ("gn", "sources.md §11w table", "2026-09-07", "full RGPH-3 thematic series, no religion volume",
     "| **Guinea** | 2014 | 10.5M | 6 | full RGPH-3 thematic series, no religion volume |",
     "§9dh: Tableau 5.10 inside the structure volume"),
    ("zm", "sources.md §11p table", "2026-09-06", "Germany-shaped and worse",
     "| **Zambia** | 2022 Census Analytical Report | province, **as a chart** | **5, Christianity "
     "undivided at 98%** | **do not build.** Germany-shaped and worse. |",
     "§9db: a Series B religion volume published after both sweeps"),
    ("mz", "sources.md §11w paragraph", "2026-09-07",
     "the catalogue answers now, and the religion table is national",
     "**Mozambique — the catalogue answers now, and the religion table is national.** §11b and §11d "
     "both recorded `ine.gov.mz` as refusing connections. It still does. **`mozdata.ine.gov.mz` "
     "does not** — the NADA catalogue returns 79 datasets, `Censo 2017, IV` is id 24, and its nine "
     "attached documents download without a licence. The *Brochura dos Resultados Definitivos* is "
     "214 pages and its religion tables are **QUADRO 11 (religion × residence × age × sex)** and "
     "**QUADRO 13 (religion × somatic type)** — national and urban/rural, no district. `LISTA DE "
     "QUADROS.xlsx`, the census's own table list, confirms it: of 60-odd quadros only 3, 6, 7 and 8 "
     "are *\"segundo distrito\"*, and neither religion table is among them. The microdata is "
     "`form_model=licensed`.",
     "§9df: INE's provincial tables on its retired site, in the Wayback Machine"),
    ("mz", "queue.md Closed section, before §11aq's note", None,
     "- **Togo, Uganda, Mozambique, Guinea-Bissau, Namibia**",
     "- **Togo, Uganda, Mozambique, Guinea-Bissau, Namibia** — religion published nationally or "
     "not at all (§11p, §11w).",
     "§9df, and Uganda and Guinea-Bissau the same way"),
    ("hn", "sources.md §11x table", "2026-09-07", "closed: 2,021 media items, zero religion",
     "| **Honduras** | INE | open, honest 404s, WordPress | closed: 2,021 media items, zero "
     "religion; **Censo 2026 is in the field now** |",
     "§11ap/§9dl: INE's ENDESA-MICS 2019 household file"),
    ("hn", "queue.md Closed section", None,
     "- **Panama, Ecuador, Guatemala, Honduras, El Salvador, Costa Rica**",
     "- **Panama, Ecuador, Guatemala, Honduras, El Salvador, Costa Rica** — no religion variable in "
     "any census; Panama across five censuses and 252 tables (§11x). Strongest negatives here. "
     "**But a census negative is not an office negative**: §9cm found that Panama's MICS 2013 and "
     "its April 2022 household survey both ASK the question and that INEC has published neither "
     "answer below the nation. The same should be asked of the other five.",
     "the same route: a household survey, not a census"),
    ("hn", "sources.md §11ad", "2026-09-08", "Honduras is not usable from this file as it stands",
     "1. **Honduras is not usable from this file as it stands.** 22 codes for 18 departments, with "
     "**Choluteca, Copán, Olancho and Valle each appearing under two codes in the same waves**. "
     "Worse, the sample sizes do not match the country: the largest code is labelled `Copán` and "
     "holds **17.9% of the national sample**, while `Francisco Morazán`, the department containing "
     "Tegucigalpa and about a fifth of Honduras, holds **4.7%**. Something is shifted, and joining "
     "on either the code or the label would produce a map that looks fine and is wrong. "
     "**Reconcile against the per-wave releases or LAPOP's country questionnaires before drawing "
     "Honduras.** Everything else in its column above is national and stands.",
     "§11ap: 2012-2018 codes are LAPOP's own department order"),
    ("ar", "sources.md §11ag", "2026-09-08", "Six regions is the CEILING",
     "**CONICET is genuinely new and appears nowhere in this project.** The *Segunda Encuesta "
     "Nacional sobre Creencias y Actitudes Religiosas en la Argentina* (CEIL-CONICET, 2019, the "
     "second edition after 2008) is openly published — an infographic at `conicet.gov.ar` and the "
     "full report in the CONICET institutional repository at `ri.conicet.gov.ar`, no account. "
     "**Six regions is the CEILING and not a starting point, which is now checked rather than "
     "assumed.** The report publishes the regional breakdown on **page 18** and **there is no "
     "province table anywhere in it**, nor is any microdata offered for download. So there is no "
     "path from this source to the 24 provinces, and nobody should go looking for one: **Argentina "
     "is buildable at six units or not at all.**",
     "§9dc: the microdata was deposited, embargoed to 2026-12-31"),
]


def calibrate(C, reg, orc):
    print("CALIBRATION: the negatives that turned out wrong, tested as each read when it closed\n")
    live = {m: scan(C, reg, orc, m)[0] for m in ("loose", "tight")}
    for m in ("loose", "tight"):
        print(f"  live list, {m} test, {len(live[m])} records: {histogram(live[m])}")
    print()
    complete = {"loose": 0, "tight": 0}
    for cc, where, hd, find, text, fix in CALIBRATION:
        flag = f"oracle {orc[cc][0]}/{orc[cc][1]}" if cc in orc else "not in oracle"
        print(f"  {cc}  {where}")
        for m in ("loose", "tight"):
            miss, notes = parts(text, hd, m)
            complete[m] += not miss
            worse = sum(1 for x in live[m] if len(x["missing"]) > len(miss))
            level = sum(1 for x in live[m] if len(x["missing"]) == len(miss))
            print(f"      {m}: {len(miss)} missing {miss}; the live list has {worse} records "
                  f"worse and {level} level with it")
        print(f"      type {notes['type']}; {flag}; reopened by {fix}")
    n = len(CALIBRATION)
    print(f"\n  passed as complete: loose {complete['loose']} of {n}, tight {complete['tight']} "
          f"of {n}")

    print("\n  and does the scan itself find each record? (drawn countries kept for this check)")
    rows = scan(C, reg, orc, "tight", keep_drawn=True)[0]
    for cc, where, hd, find, text, fix in CALIBRATION:
        fname = where.split()[0]
        joined = open(os.path.join(ROOT, fname), encoding="utf-8").read()
        pos = joined.find(find)
        if pos < 0:
            print(f"  {cc}  {where}: the snippet is no longer in {fname}")
            continue
        at = joined.count("\n", 0, pos) + 1
        hit = [x for x in rows if x["cc"] == cc and x["file"] == fname
               and x["line"] <= at <= x["line"] + x["text"].count("\n")]
        if hit:
            x = max(hit, key=lambda y: y["line"])
            print(f"  {cc}  {fname}:{x['line']} found; kind {x['kind']}; as it reads today "
                  f"{len(x['missing'])} missing {x['missing']}")
        else:
            print(f"  {cc}  {fname}:{at} NOT FOUND by the scan")
    return 1 if complete["tight"] else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--records", action="store_true")
    ap.add_argument("--cc")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--loose", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--shape", action="store_true",
                    help="only records flagged same-shape (the wrong negatives' shape)")
    a = ap.parse_args()

    C = Countries()
    reg = drawn()
    orc = oracle_index(C)
    if a.calibrate:
        sys.exit(calibrate(C, reg, orc))

    mode = "loose" if a.loose else "tight"
    rows, dropped = scan(C, reg, orc, mode, a.all)
    if a.cc:
        want = set(a.cc.lower().split(","))
        rows = [x for x in rows if x["cc"] in want]
    hist = histogram(rows)
    # a country with one complete record is not on the list, whatever its thinner records say
    complete = {(x["cc"], x["kind"]) for x in rows if not x["missing"]}
    if a.shape:
        rows = [x for x in rows if "same-shape" in x["flags"]]
    per_cc = order([x for x in by_country(rows)
                    if a.all or (x["missing"] and (x["cc"], x["kind"]) not in complete)])
    rows = [x for x in rows if x["missing"] or a.all]
    per_rec = order(rows)
    n_cc = len({x["cc"] for x in per_cc})
    summary = (f"{len(rows)} records missing a part ({mode} test), {n_cc} countries "
               f"({sum(1 for x in per_cc if x['kind'] == 'country')} country rows, "
               f"{sum(1 for x in per_cc if x['kind'] == 'upgrade')} upgrade rows). "
               f"Dropped: {dropped['drawn']} about drawn countries and not about a finer or newer "
               f"source, {dropped['open']} for countries a status line calls buildable or routed, "
               f"{dropped['reversed']} reversed in their own sentence. Before dropping the "
               f"complete ones: {hist}.")
    print(summary + "\n")
    shown = per_rec if a.records else per_cc
    print("\n".join(table_lines(shown if a.top == 0 else shown[: a.top], md=False,
                                with_n=not a.records)))
    if a.out:
        md = [f"# Negatives missing a part ({mode} test)", "",
              "Written by `python tools/negatives.py`; its docstring defines the parts and the "
              "flags.", "", summary, "",
              "## One row per country and kind, its most complete record", ""]
        md += table_lines(per_cc, md=True, with_n=True)
        md += ["", "## Every record", ""]
        md += table_lines(per_rec, md=True, with_n=False)
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(md) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
