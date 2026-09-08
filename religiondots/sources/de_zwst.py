"""Germany — Jewish community membership, ZWST Mitgliederstatistik 2025.

Reads (or fetches) data/raw/de_zwst/ and writes data/normalized/de_zwst.csv.

WHY THIS SOURCE EXISTS.  Zensus 2022 reads religion off the Melderegister, which knows
only what church tax requires it to know, so Germany is drawn as three categories and
51.8% of the country — 42,845,220 people — lands in "Sonstige, keine, ohne Angabe"
(sources/de.md §2).  Germany's Muslims, its Orthodox Christians, its free churches and
its Jewish communities are all inside that one grey cell, and the map currently draws
none of them.  This is the one route out of it that adds a COUNTED number rather than a
modelled share.

THE BASIS MATCHES, and that is the whole reason this is the cheap route.  Germany is
`roll` because the Melderegister is an institution's records; the Zentralwohlfahrtsstelle
der Juden in Deutschland publishes an institution's records too.  Nothing here mixes
bases (spec §3.1), no survey is involved, and Germany's `measured` tier survives intact
— unlike the ESS split scouted in sources/de.md §6, which spends it.

And these people are already inside `unrecorded`.  Jewish communities are public-law
religious societies in most Laender, so the 87,934 sit in the 42.8M bucket today.
Drawing them CARVES A COUNTED NUMBER OUT OF IT rather than estimating a share of it, and
whatever consumes this file must subtract before it adds, or Germany gains people.

THE FILE.  No Excel, no CSV, no API — ZWST publishes PDFs and nothing else.  But there
is a real text layer, a complete per-community table, and a published national total to
reconcile against, which is worth more than a spreadsheet with no check in it.  105
communities come out summing to exactly the 87,934 printed on page 5, and thirteen
association subtotals reconcile individually on the way, so the check is thirteen checks
and not one.  105 is also the community count ZWST states for itself, reached here
without being told it.

THREE LAYOUTS, and no reading-order rule survives all of them:

    p12  marker then count   M 217 1 1 7 ...        / " Baden-Baden" / G 469 ...
    p27  count then marker   1.006 M 3 12 26 ...    / " Hamburg" 2.238 G 7 33 ...
    p15  TRANSPOSED — the age bands run down the left edge and the communities run
         across the page as columns

So "the number after G" is the total on some pages and the 0-3 age band on others, and
Hamburg's 2,238 members come out as 7 — a number small enough to pass for a real small
community and never be questioned.

AND THE OBVIOUS CHECK DOES NOT CATCH IT.  Every block publishes male, female and total,
so M + W == G looks like the way to tell the readings apart.  It is not: the table is
male/female/total in EVERY column, so the age bands satisfy it too.  Hamburg's wrong
reading is 3 + 4 == 7 and validates cleanly; a pass over the whole file accepted 120
blocks with zero rejections and was still wrong.  The invariant that DOES discriminate
is the other one — THE TOTAL IS THE SUM OF ITS OWN TWELVE AGE BANDS.  That is false for
the misreading, true for the real column, and true on all three layouts, so it picks the
reading and validates the row in one step.  spec §12: an arithmetic check that the wrong
answer also passes is not a check.

TWO MORE TRAPS:

  * The thousands separator is used in the TOTAL column and omitted in the AGE-BAND
    columns of the same table — Duesseldorf is "6.371" and its 71-80 band is "1232".  A
    `\\d{1,3}(\\.\\d{3})*` pattern silently drops every large community and the national
    total still looks plausible, because the associations that survive still sum to
    something.
  * Community names are a private association's house style, not Gemeinde names, and
    matching them is where the silent error lives.  "Weiden" resolves by exact
    name-match to a village of 84 people in Rheinland-Pfalz rather than to Weiden in der
    Oberpfalz; a community of 178 in a Gemeinde of 84 is the only sign.  Hence SEATS is
    hand-authored and MAX_SHARE_OF_SEAT is asserted.

WHAT THIS CANNOT DO, and it has to be said on the map.  spec §3.6 — a roll counts the
institution's location, not the member's — bites harder here than anywhere else on the
map, because these are REGIONAL CATCHMENTS and not parishes.  Duesseldorf's 6,371 covers
much of the lower Rhine; the Juedische Landesgemeinde Thueringen is one community for an
entire Bundesland.  The seat is what is known and the seat is what is drawn.

AND IT IS A FLOOR, NOT A POPULATION.  87,934 is affiliated membership.  The unaffiliated
and much of the post-2022 Ukrainian arrival are outside it, and estimates of Germany's
Jewish population run half again as high.  That is the same shape of undercount the
register itself has, so it is consistent rather than a new problem — but spec §3.5 says
it gets marked, not filled.

Licence: unstated.  ZWST is a private association and the PDF carries no reuse grant.

Usage:
    python sources/de_zwst.py --fetch    download the PDF (~2MB) if missing
    python sources/de_zwst.py            parse and normalise from data/raw/de_zwst/
"""

import csv
import os
import re
import sys

# Community names are German and the reconciliation prints them; a Windows console is
# cp1252 and dies on an umlaut at the PRINT, which reads like a data error (spec §12).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "de_zwst")
OUT = os.path.join(ROOT, "data", "normalized", "de_zwst.csv")
DE_CSV = os.path.join(ROOT, "data", "normalized", "de.csv")

SOURCE_ID = "de_zwst_2025"
YEAR = 2025
# An association's own membership register, exactly like the Melderegister basis of
# sources/de.py.  NOT self_id: nobody was asked, they joined (spec §3.1).
BASIS = "roll"

CATEGORY = "Jüdische Gemeinde (ZWST-Mitgliedsgemeinde)"

PDF_NAME = "ZWST-Mitgliederstatistik-2025.pdf"
PDF_URL = ("https://zwst.org/sites/default/files/2026-07/"
           "ZWST-Mitgliederstatistik-2025-web.pdf")
MIN_BYTES = 1_500_000

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

# The published national figure, from the trend table on page 5.  Hard-coded so that a
# reissue with different figures fails loudly instead of being normalised in silence.
PUBLISHED_TOTAL = 87_934
N_COMMUNITIES = 105
FIRST_PAGE = 11           # the per-community section starts here (Inhalt, page 4)
N_BANDS = 12              # 0-3, 4-7, 8-11, 12-18, 19-21, 22-30, ... , 71-80, > 80

# The thousands separator is present in the total column and absent in the age bands of
# the same table, so this must NOT be `\d{1,3}(\.\d{3})*`.  See the module docstring.
NUM = re.compile(r"^\d+(\.\d{3})*$")
AGE_BAND = re.compile(r"^(\d+-\d+|>|80)$")
MARKERS = ("M", "W", "G")

# A community is seated in a Gemeinde far smaller than itself only if the join is wrong.
# The real maximum across the 105 is Straubing at 1.7%.
MAX_SHARE_OF_SEAT = 0.05

# The association's own directly-registered members — people in Baden who belong to the
# Landesverband rather than to any of its ten local communities.  They are not seated
# anywhere, so putting all 528 in Karlsruhe would assert a concentration that does not
# exist.  They are spread across the association's communities in proportion to those
# communities instead, which is the closest thing to what every other row here already
# is: a catchment rather than a residence.  0.6% of the layer.
SPREAD_OVER_VERBAND = {"IRG Baden"}

# ---------------------------------------------------------------------------
# The seat of each community, hand-checked against data/normalized/de.csv.
#
# NOT a fuzzy join.  72 of the 105 names match a Gemeinde name exactly and ONE OF THOSE
# 72 IS WRONG — "Weiden" is a village of 84 people in Rheinland-Pfalz as well as a city
# of 42,047 in the Oberpfalz, and the exact match takes the village.  The rest are
# abbreviations ("Mönchengladb."), the association's own compounds ("Herford-Detm.",
# "Rheinpfalz/Speyer", "Kiel u. Region"), or association-level rows whose seat is a fact
# about the body rather than about its name ("IKG München", "Württemberg" -> Stuttgart,
# "Thüringen" -> Erfurt).
#
# Two associations run communities in the same city — Hannover, Göttingen, Hameln,
# Wolfsburg, Kiel and Lübeck each appear twice, orthodox and liberal.  That is not double
# counting; both are real and the seats simply coincide.
# ---------------------------------------------------------------------------
SEATS = {
    # Israelitische Religionsgemeinschaft Baden
    "Baden-Baden":            "082110000000",
    "Emmendingen":            "083165003011",
    "Freiburg":               "083110000000",   # im Breisgau, not Freiburg (Elbe)
    "Heidelberg":             "082210000000",
    "Karlsruhe":              "082120000000",
    "SG Konstanz":            "083355004043",
    "Lörrach":                "083365003050",
    "Mannheim":               "082220000000",
    "Pforzheim":              "082310000000",
    "Rottweil":               "083255003049",
    "IRG Baden":              "082120000000",   # seat only; see SPREAD_OVER_VERBAND
    # Landesverband der Israelitischen Kultusgemeinden in Bayern
    "Amberg":                 "093610000000",   # Oberpfalz, not Amberg bei Buchloe
    "Augsburg":               "097610000000",
    "Bamberg":                "094610000000",
    "Bayreuth":               "094620000000",
    "Erlangen":               "095620000000",
    "Fürth":                  "095630000000",   # Bavaria, not Fürth im Odenwald
    "Hof/Saale":              "094640000000",   # not Hof, a village of 1,308 in RLP
    "Nürnberg":               "095640000000",
    "Regensburg":             "093620000000",
    "Straubing":              "092630000000",
    "Weiden":                 "093630000000",   # i.d.OPf, NOT the 84-person Weiden
    "Würzburg":               "096630000000",
    # Jüdische Gemeinde zu Berlin
    "Berlin":                 "110000000000",
    # Landesverband der jüdischen Gemeinden Land Brandenburg
    "Bernau/LK Barnim":       "120600020020",   # Bernau bei Berlin
    "Cottbus":                "120520000000",
    "Frankfurt/Oder":         "120530000000",
    "Oranienburg/Oberhavel":  "120650256256",
    "Königswusterhausen":     "120610260260",   # Gemeinde is "Königs Wusterhausen"
    "Stadt Potsdam":          "120540000000",
    # independent communities
    "Bremen":                 "040110000000",
    "Frankfurt":              "064120000000",   # am Main; Frankfurt/Oder is above
    "Hamburg":                "020000000000",
    "Köln":                   "053150000000",
    "IKG München":            "091620000000",
    # Landesverband der Jüdischen Gemeinden in Hessen
    "Bad Nauheim":            "064400002002",
    "Darmstadt":              "064110000000",
    "Fulda":                  "066310009009",
    "Gießen":                 "065310005005",
    "Hanau":                  "064350014014",
    "Kassel":                 "066110000000",
    "Limburg":                "065330009009",   # Limburg a.d. Lahn, not Limburgerhof
    "Marburg/Lahn":           "065340014014",
    "Offenbach/M.":           "064130000000",   # am Main, not Offenburg
    "Wiesbaden":              "064140000000",
    # Landesverband der Jüdischen Gemeinden in Mecklenburg-Vorpommern
    "Rostock":                "130030000000",
    "Schwerin":               "130040000000",   # Landeshauptstadt, not the 964-person one
    # Landesverband der Jüdischen Gemeinden von Niedersachsen
    "Bad Nenndorf":           "032575403006",
    "Braunschweig":           "031010000000",
    "Delmenhorst":            "034010000000",
    "Göttingen/Südn.":        "031590016016",
    "Hannover":               "032410001001",
    "Hameln/Pyrmont":         "032520006006",
    "Hildesheim (KG)":        "032540021021",
    "Hildesheim (JG)":        "032540021021",
    "Oldenburg":              "034030000000",   # (Oldenburg), not Oldenburg in Holstein
    "Osnabrück":              "034040000000",
    "JSB Hannover":           "032410001001",
    "Wolfsburg":              "031030000000",
    # Landesverband der Israelitischen Kultusgemeinden von Niedersachsen
    "Bad Pyrmont":            "032520003003",
    "Celle":                  "033510006006",
    "Göttingen":              "031590016016",
    "Hameln":                 "032520006006",
    # Landesverband der Jüdischen Gemeinden von Nordrhein
    "Aachen":                 "053340002002",
    "Bonn":                   "053140000000",
    "Düsseldorf":             "051110000000",
    "Essen":                  "051130000000",
    "Krefeld":                "051140000000",
    "Mönchengladb.":          "051160000000",
    "Duisburg":               "051120000000",
    "Wuppertal":              "051240000000",
    # Landesverband der Jüdischen Gemeinden von Rheinland-Pfalz
    "Bad Kreuznach":          "071330006006",
    "Koblenz":                "071110000000",
    "Rheinpfalz/Speyer":      "073180000000",
    "Trier":                  "072110000000",
    # Synagogengemeinde Saar
    "Saarbrücken":            "100410100100",
    # Landesverband Sachsen der Jüdischen Gemeinden
    "Chemnitz":               "145110000000",
    "Dresden":                "146120000000",
    "Leipzig":                "147130000000",
    # Landesverband Jüdischer Gemeinden Sachsen-Anhalt
    "Dessau":                 "150010000000",   # Gemeinde is "Dessau-Roßlau"
    "Halle/Saale":            "150020000000",
    "Magdeburg":              "150030000000",
    # Jüdische Gemeinschaft Schleswig-Holstein
    "Flensburg":              "010010000000",
    "Kiel u. Region":         "010020000000",
    "Lübeck":                 "010030000000",
    # Landesverband der Jüdischen Gemeinden von Schleswig-Holstein
    "Ahrensburg-St.":         "010620001001",
    "Bad Segeberg":           "010600005005",
    "Elmshorn":               "010560015015",
    "Kiel":                   "010020000000",
    "Pinneberg":              "010560039039",
    # Jüdische Landesgemeinde Thüringen — one community for the whole Bundesland
    "Thüringen":              "160510000000",   # seat Erfurt
    # Landesverband der Jüdischen Gemeinden von Westfalen-Lippe
    "Bielefeld":              "057110000000",
    "Bochum":                 "059110000000",
    "Dortmund":               "059130000000",
    "Gelsenkirchen":          "055130000000",
    "Hagen":                  "059140000000",   # Stadt, not the 476-person Hagen
    "Herford-Detm.":          "057580012012",   # seat Herford
    "Minden":                 "057700024024",   # Stadt, not the 257-person Minden
    "Münster":                "055150000000",   # Stadt, not the 1,372-person Münster
    "Paderborn":              "057740032032",
    "Recklinghausen":         "055620032032",
    # Israelitische Religionsgemeinschaft Württemberg
    "Württemberg":            "081110000000",   # seat Stuttgart
}


def fetch():
    os.makedirs(RAW, exist_ok=True)
    dest = os.path.join(RAW, PDF_NAME)
    if os.path.exists(dest) and os.path.getsize(dest) >= MIN_BYTES:
        print("already have", dest)
        return
    import urllib.request
    print("downloading", PDF_URL)
    urllib.request.urlretrieve(PDF_URL, dest)
    _validate(dest)


def _validate(path):
    """Assert size AND type AND completeness.  HTTP 200 is not a download (spec §12),
    and a PDF can arrive truncated with a Content-Length that matches the damage."""
    size = os.path.getsize(path)
    if size < MIN_BYTES:
        raise SystemExit(f"{path} is {size:,} bytes, expected >= {MIN_BYTES:,}")
    with open(path, "rb") as fh:
        head = fh.read(8)
        fh.seek(max(0, size - 2048))
        tail = fh.read()
    if not head.startswith(b"%PDF"):
        raise SystemExit(f"{path} does not start with %PDF, got {head!r}")
    if b"%%EOF" not in tail:
        # PyMuPDF reports page_count 0 for a truncated file and raises nothing.
        raise SystemExit(f"{path} has no %%EOF trailer — truncated at source")
    print(f"  {size:,} bytes, %PDF header, %%EOF present")


def _to_int(s):
    return int(s.replace(".", ""))


def _read_block(lines, i):
    """The membership count for the M/W/G marker at `i`.

    Taken from whichever side of the marker satisfies `total == sum of the twelve age
    bands`.  That invariant, and NOT `M + W == G`, is what distinguishes the real column
    from the 0-3 age band on the pages that print the count before the marker.  See the
    module docstring.
    """
    for value_at, bands_at in ((i + 1, i + 2), (i - 1, i + 1)):
        if not (0 <= value_at < len(lines)) or not NUM.match(lines[value_at]):
            continue
        bands = lines[bands_at:bands_at + N_BANDS]
        if len(bands) < N_BANDS or not all(NUM.match(b) for b in bands):
            continue
        total = _to_int(lines[value_at])
        if total == sum(_to_int(b) for b in bands):
            return total
    return None


def _name_before(lines, i):
    """Walk back past the numbers and markers to the community's name."""
    for j in range(i - 1, -1, -1):
        p = lines[j]
        if NUM.match(p) or p in MARKERS or AGE_BAND.match(p):
            continue
        return p
    return None


def read():
    """Every M/W/G block in the per-community section, in document order."""
    try:
        import fitz
    except ImportError:
        raise SystemExit("needs PyMuPDF: pip install pymupdf")

    path = os.path.join(RAW, PDF_NAME)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run with --fetch")
    _validate(path)

    doc = fitz.open(path)
    if doc.page_count < FIRST_PAGE:
        raise SystemExit(f"{path} has {doc.page_count} pages, expected a full report")

    blocks = []
    for pno in range(FIRST_PAGE, doc.page_count):
        lines = [ln.strip() for ln in doc[pno].get_text().split("\n") if ln.strip()]
        marks = [i for i, ln in enumerate(lines) if ln in MARKERS]

        for a in range(len(marks) - 2):
            i, j, k = marks[a], marks[a + 1], marks[a + 2]
            if (lines[i], lines[j], lines[k]) != ("M", "W", "G"):
                continue
            m, w, g = (_read_block(lines, x) for x in (i, j, k))
            name = _name_before(lines, k)
            if None in (m, w, g) or name is None:
                continue
            if m + w != g:
                raise SystemExit(f"p{pno + 1} {name}: {m} + {w} != {g}, so the block "
                                 "was read off the wrong column after all")
            blocks.append((pno + 1, name, g))

    doc.close()
    return blocks


def split_subtotals(blocks):
    """Separate communities from the association subtotals that follow them.

    Detected arithmetically rather than by page furniture: an association's subtotal is
    a block equal to the sum of a TRAILING RUN of the communities before it.  It has to
    be a trailing run and not everything pending, because the six independent communities
    (Berlin, Bremen, Frankfurt, Hamburg, Köln, München) belong to no association and sit
    in the sequence between the associations that do.

    This keeps the subtotals out of the sum and reconciles each association on the way,
    so the national check below is thirteen checks and not one.
    """
    associations, pending = [], []
    for rec in blocks:
        run, hit = 0, None
        for start in range(len(pending) - 1, -1, -1):
            run += pending[start][2]
            if run == rec[2] and len(pending) - start > 1:
                hit = start
                break
        if hit is None:
            pending.append(rec)
        else:
            associations.append((rec, pending[hit:]))
            pending = pending[:hit]
    return associations, pending


SONSTIGE = "Sonstige, keine, ohne Angabe"


def gemeinde_names():
    """AGS -> (name, population, Sonstige), from the already-normalised Zensus file.

    `Sonstige` is carried because it is the cell these people are currently inside, and
    whatever consumes this file has to take them OUT of it before adding them back as
    Jews — otherwise Germany gains 87,934 people. See check_room().
    """
    if not os.path.exists(DE_CSV):
        raise SystemExit(f"{DE_CSV} not found — run sources/de.py first")
    pop, sonstige = {}, {}
    with open(DE_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["geo_level"] != "gemeinde":
                continue
            if r["source_category"] == "Einwohnerzahl":
                pop[r["geo_id"]] = (r["geo_name"], int(r["count"]))
            elif r["source_category"] == SONSTIGE:
                sonstige[r["geo_id"]] = int(r["count"])
    return {k: (n, p, sonstige.get(k, 0)) for k, (n, p) in pop.items()}


def build(blocks):
    associations, standalone = split_subtotals(blocks)
    communities = [r for _, members in associations for r in members] + standalone

    print(f"  {len(blocks)} blocks read")
    print(f"  {len(associations)} associations reconciled, "
          f"{len(standalone)} standalone communities")
    for rec, members in associations:
        print(f"    p{rec[0]:>3}  {rec[1]:<22} {rec[2]:>7,}  "
              f"= {len(members):>2} communities")

    national = sum(n for _, _, n in communities)
    ok = True
    if national != PUBLISHED_TOTAL:
        ok = False
    print(f"\n  {'OK ' if national == PUBLISHED_TOTAL else 'BAD'} national total "
          f"{national:,} against the {PUBLISHED_TOTAL:,} published on page 5")
    if len(communities) != N_COMMUNITIES:
        ok = False
    print(f"  {'OK ' if len(communities) == N_COMMUNITIES else 'BAD'} "
          f"{len(communities)} communities, ZWST states {N_COMMUNITIES}")

    missing = sorted({n for _, n, _ in communities} - set(SEATS))
    if missing:
        raise SystemExit("no seat for: " + ", ".join(missing) +
                         "\nAdd it to SEATS by hand — do NOT fuzzy-match (docstring).")

    # spread the association-direct rows over their own association's communities
    spread = {}
    for rec, members in associations:
        direct = [r for r in members if r[1] in SPREAD_OVER_VERBAND]
        if not direct:
            continue
        rest = [r for r in members if r[1] not in SPREAD_OVER_VERBAND]
        base = sum(n for _, _, n in rest)
        for _, name, n in direct:
            print(f"\n  {name}: {n:,} association-direct members spread over "
                  f"{len(rest)} communities in proportion to them")
            # largest-remainder, so the spread sums back to n exactly
            exact = [(r, n * r[2] / base) for r in rest]
            given = {r[1]: int(v) for r, v in exact}
            for r, v in sorted(exact, key=lambda t: -(t[1] % 1))[:n - sum(given.values())]:
                given[r[1]] += 1
            for k, v in given.items():
                spread[k] = spread.get(k, 0) + v

    gem = gemeinde_names()
    seated, rows = {}, []
    for _, name, n in communities:
        if name in SPREAD_OVER_VERBAND:
            continue
        ags = SEATS[name]
        if ags not in gem:
            raise SystemExit(f"{name}: AGS {ags} is not a Gemeinde in de.csv")
        seated[ags] = seated.get(ags, 0) + n + spread.get(name, 0)

    worst, tightest = (0.0, None), (0.0, None)
    for ags, n in sorted(seated.items()):
        gname, pop, sonstige = gem[ags]
        share = n / pop if pop else 1.0
        if share > worst[0]:
            worst = (share, (gname, n, pop))
        headroom = n / sonstige if sonstige else 1.0
        if headroom > tightest[0]:
            tightest = (headroom, (gname, n, sonstige))
        rows.append({
            "geo_id": ags,
            "geo_level": "gemeinde",
            "geo_name": gname,
            "source_category": CATEGORY,
            "count": n,
            "basis": BASIS,
            "year": YEAR,
            "source_id": SOURCE_ID,
            "note": "ZWST Mitgliederstatistik 2025; seat of the community, whose "
                    "catchment is regional (spec §3.6)",
        })

    good = worst[0] <= MAX_SHARE_OF_SEAT
    ok &= good
    gname, n, pop = worst[1]
    print(f"\n  {'OK ' if good else 'BAD'} no community outsizes its seat: worst is "
          f"{gname} at {100 * worst[0]:.2f}% ({n:,} of {pop:,})")
    print(f"      a name matched to the wrong Gemeinde shows up here and nowhere else")

    # the consumption contract: these people are inside `Sonstige` today, so the seat
    # must have enough of it to take them out of before adding them back as Jews
    room = tightest[0] <= 1.0
    ok &= room
    gname, n, sonstige = tightest[1]
    print(f"  {'OK ' if room else 'BAD'} every seat has room in '{SONSTIGE}': "
          f"tightest is {gname}, {n:,} of {sonstige:,} ({100 * tightest[0]:.1f}%)")

    placed = sum(r["count"] for r in rows)
    if placed != PUBLISHED_TOTAL:
        ok = False
    print(f"  {'OK ' if placed == PUBLISHED_TOTAL else 'BAD'} {placed:,} members seated "
          f"in {len(rows)} Gemeinden, against {PUBLISHED_TOTAL:,}")

    if not ok:
        raise SystemExit("reconciliation FAILED")
    return rows


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows = build(read())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
