"""Croatia: HŽ Infrastruktura (HŽI, 0078_IM) is the one infrastructure manager in RINF.

RINF HAS NO `era:nationalLine` FOR CROATIA, BUT THE LINE NUMBER IS IN EVERY SECTION'S URI.
HŽI's sections are named after its own line numbers: SectionOfLine_M202__M20214J is the 14th
section of line M202 (Zagreb GK - Rijeka), SectionOfLine_M101__M10103D_M10103L the third of
M101 on its right (D) and left (L) tracks together. Those are the numbers HŽI publishes (M =
international, R = regional, L = local line), the numbers on OSM's route=railway relations
(every one of the 47 ids matches a relation of that ref) and on Wikidata's P1671. So `hr_fix`
(rinf.py's `fix` hook) reads each section's line from its URI, and the id IS the number
(`rule_certain`). Two lines are filed in two parts: M402A/M402B (the two tracks of Sava -
Zagreb Klara round the marshalling yard) and M502_1/M502_2 (HŽI's own M502-1 Zagreb GK -
Velika Gorica and M502-2 Velika Gorica - Sisak - Novska); each pair is one line.

`hr_fix` builds the section list again from sections.json rather than correcting the one
rinf.load_rinf made. load_rinf keys a section on (line id, start point, end point) to choose
between validity versions, and with every Croatian line id empty, two lines' sections between
the same pair of points were one key and one of them was lost: Čakovec - Čakovec Buzovec is on
both L101 and M501, Zagreb Klara - Zagreb Rk PS on both M402 and M403. Croatia's sections have
no validity periods, and every one is listed twice in the fetch (two rows per URI), so a
section here is simply one URI.

NAMES are HŽI's own, from table 1.4 of "Statistika HŽ Infrastrukture za 2025" (the line list
with constructional lengths, ROUTES below), written "M202 Zagreb GK – Rijeka", English "Line
M202 (Zagreb GK – Rijeka)". HŽI writes a border end as "DG" (državna granica); it is left off
where two places remain ("M101 Savski Marof – Zagreb GK"), as si.py leaves off "d. m.". GK is
Glavni kolodvor (main station), ZK Zapadni kolodvor, RK OS/PS the marshalling yard's
departure and arrival groups.

What RINF leaves out of HŽI's 2,617 km: R103 (Knin - Ličko Dugo Polje - border, the Una line,
closed to traffic), L210 Sisak Caprag - Petrinja (closed), L213 beyond Učka to Raša and L102
beyond Harmica to Kumrovec (both closed), L205's closed Čaglin - Našice cement stretch, and
nothing else of note: the 47 ids sum to 2,074 km.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "rinf" / "hr"

# HŽI's line names, table 1.4 of its 2025 statistics (Građevinske dužine pojedinih pruga u
# 2025.), with "DG" (the state border) left off where two places remain.
ROUTES = {
    "M101": "Savski Marof – Zagreb GK",
    "M102": "Zagreb GK – Dugo Selo",
    "M103": "Dugo Selo – Novska",
    "M104": "Novska – Tovarnik",
    "M201": "Botovo – Dugo Selo",
    "M202": "Zagreb GK – Rijeka",
    "M203": "Rijeka – Šapjane",
    "M301": "Beli Manastir – Osijek",
    "M302": "Osijek – Strizivojna-Vrpolje",
    "M303": "Strizivojna-Vrpolje – Slavonski Šamac",
    "M304": "Metković – Ploče",
    "M401": "Sesvete – Sava",
    "M402": "Sava – Zagreb Klara",
    "M403": "Zagreb RK PS – Zagreb Klara",
    "M404": "Zagreb Klara – Delta",
    "M405": "Zagreb ZK – Trešnjevka",
    "M406": "Zagreb Borongaj – Zagreb Resnik",
    "M407": "Sava – Velika Gorica",
    "M408": "Zagreb RK OS – Mićevac",
    "M409": "Zagreb Klara – Zagreb RK PS",
    "M410": "Zagreb RK OS – Zagreb RK PS",
    "M501": "Čakovec – Kotoriba",
    "M502": "Zagreb GK – Sisak – Novska",
    "M601": "Vinkovci – Vukovar",
    "M602": "Škrljevo – Bakar",
    "M603": "Sušak – Rijeka Brajdica",
    "M604": "Oštarije – Knin – Split",
    "M605": "Ogulin – Krpelj",
    "M606": "Knin – Zadar",
    "M607": "Perković – Šibenik",
    "R101": "Buzet – Pula",
    "R102": "Sunja – Volinja",
    "R103": "Ličko Dugo Polje – Knin",
    "R104": "Vukovar-Borovo naselje – Erdut",
    "R105": "Vinkovci – Drenovci",
    "R106": "Zabok – Đurmanec",
    "R201": "Zaprešić – Čakovec",
    "R202": "Varaždin – Dalj",
    "L101": "Čakovec – Mursko Središće",
    "L102": "Savski Marof – Kumrovec",
    "L103": "Karlovac – Kamanje",
    "L201": "Varaždin – Golubovec",
    "L202": "Hum-Lug – Gornja Stubica",
    "L203": "Križevci – Bjelovar – Kloštar",
    "L204": "Banova Jaruga – Pčelić",
    "L205": "Nova Kapela – Našice",
    "L206": "Pleternica – Velika",
    "L207": "Bizovac – Belišće",
    "L208": "Vinkovci – Osijek",
    "L209": "Vinkovci – Županja",
    "L210": "Sisak Caprag – Petrinja",
    "L211": "Ražine – Šibenik Luka",
    "L212": "Rijeka Brajdica – Rijeka",
    "L213": "Lupoglav – Raša",
    "L214": "Gradec – Sveti Ivan Žabno",
}

# "SectionOfLine_M202__M20214J", "SectionOfLine_M101__M10103D_M10103L",
# "SectionOfLine_M402A__M402A01L", "SectionOfLine_M502_1__M502_101J"
SOL_ID = re.compile(r"SectionOfLine_([MRL]\d{3})(?:[A-Z]|_\d)?__")


def line_of(sol):
    m = SOL_ID.search(sol or "")
    return m.group(1) if m else ""


def _tail(v):
    return (v or "").rstrip("/").rsplit("/", 1)[-1]


def hr_fix(secs, points):
    """rinf.py's `fix` hook: every section again from sections.json, one per URI, with the
    line number read from the URI (module docstring). Returns log lines."""
    rows = json.loads((RAW / "sections.json").read_text(encoding="utf-8"))["rows"]
    out, seen, odd = [], set(), []
    for r in rows:
        if r["sol"] in seen:
            continue
        seen.add(r["sol"])
        lid = line_of(r["sol"])
        if not lid:
            odd.append(_tail(r["sol"]))
        try:
            km = float(r.get("len"))
        except (TypeError, ValueError):
            km = None
        out.append({"sol": r["sol"], "line": lid, "base": lid, "a": r["a"], "b": r["b"],
                    "km": km, "im": _tail(r.get("im")), "label": r.get("label", "")})
    msgs = [f"{len(out)} sections rebuilt from {len(rows)} rows, one per URI, the line number "
            f"read from the URI: {len({s['line'] for s in out})} lines (load_rinf had "
            f"{len(secs)} sections)"]
    if odd:
        msgs.append(f"{len(odd)} section URIs carry no line number: {', '.join(odd[:10])}")
    # A line in several unconnected pieces is taken piece by piece, as load_rinf does.
    by_id = defaultdict(list)
    for s in out:
        by_id[s["line"]].append(s)
    for lid, ss in by_id.items():
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for s in ss:
            parent[find(s["a"])] = find(s["b"])
        comps = defaultdict(list)
        for s in ss:
            comps[find(s["a"])].append(s)
        if len(comps) < 2:
            continue
        order = sorted(comps.values(), key=lambda c: min(
            points.get(op, {}).get("uopid") or op for s in c for op in (s["a"], s["b"])))
        for k, c in enumerate(order):
            for s in c:
                s["line"] = f"{lid}#{k + 1}"
        msgs.append(f"{lid} is {len(comps)} unconnected pieces, each taken on its own")
    # R106's last section, R10611J, is filed as Hromec -> Đurmanec (2.724 km), the same pair
    # as the one before it read backwards; it is Hromec -> the Slovenian border towards
    # Rogatec (HŽI's 2025 statistics: Đurmanec - Đurmanec DG 5.922 km = 3.198 + 2.724), and
    # Croatia's RINF has no point there. SŽ-Infrastruktura's RINF has it: "Rogatec d.m.",
    # EU00220, 46.21231 N 15.77691 E (data/raw/rinf/si). It is added under that uopid.
    border = "http://data.europa.eu/949/OperationalPoint_EU00220"
    for s in out:
        if s["sol"].endswith("SectionOfLine_R106__R10611J"):
            hromec = s["b"] if (points.get(s["b"], {}).get("name") == "Hromec") else s["a"]
            points.setdefault(border, {"op": border, "uopid": "EU00220", "name": "Đurmanec DG",
                                       "type": "90", "lon": 15.77691, "lat": 46.21231})
            s["a"], s["b"] = hromec, border
            msgs.append("R106 Hromec - Đurmanec (R10611J) is Hromec - the border, EU00220")
    secs[:] = out
    return msgs


# Lines with no passenger trains: "-" in every section of table 3.7 (passenger train movements
# by line section) of HŽI's 2025 statistics, or "zatvoreno za promet" (closed to traffic), or
# a handful of trains in the year (M602 1, M408 24, M401 30: empty stock and diversions, not a
# service). Most are dropped anyway as unridden junction sections; M606 Knin - Zadar, L103
# Karlovac - Kamanje and L213 Lupoglav - Učka run stop to stop and would be drawn as running.
# They stay on the map, greyed by not_running.py, as Slovakia's do.
SUSPENDED = {"M606", "L103", "L213", "L207", "L211", "L212", "M603", "M602",
             "M401", "M403", "M408", "M409", "M410"}


class HrName(str):
    """rinf.py writes a numbered line's name as COUNTRY["name"].format(ref=ref)."""
    def format(self, *args, **kw):
        ref = kw.get("ref", args[0] if args else "")
        route = ROUTES.get(ref)
        if route:
            return str.format(self, ref=ref, route=route)
        return str.format(self.split(" {route}")[0].split(" ({route})")[0], ref=ref)


COUNTRY = {
    "iso3": "HRV", "wikidata": "Q224", "langs": ["hr"],
    "fix": hr_fix,
    "ref": lambda lid: lid if re.fullmatch(r"[MRL]\d{3}", lid or "") else None,
    "rule_certain": True,
    "suspended": lambda ref, _ids: ref in SUSPENDED,
    "name": HrName("{ref} {route}"), "name_en": HrName("Line {ref} ({route})"),
    "im": {"0078_IM": "HŽ Infrastruktura"},
}
