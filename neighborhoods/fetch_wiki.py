"""
Stage 4: notability. Wikidata sitelinks + Wikipedia pageviews -> data/wiki.json.

WHAT THIS IS FOR. The browser wants every unit in a level (§0b); the quiz wants the
5-30 most recognisable units per city and three difficulty tiers. Nothing in OSM knows
which of New York's 325 neighbourhoods a stranger could name. Wikipedia readership does,
and OSM already carries the join key: a `wikidata` tag on ~39% of units.

WHY ALL LANGUAGES, NOT ENGLISH. Decided 2026-08-29 (spec §7). The audience is "the
people that use Wikipedia", so the measure is global readership. enwiki alone would
badly under-rate Shibuya, whose traffic is mostly `ja`, and Copacabana, mostly `pt` —
which is exactly backwards, since those are two of the most quizzable neighbourhoods on
earth. So: pageviews summed over every language edition that has an article.

TWO PASSES, CHEAP ONE FIRST — the same shape as the survey/geom split in fetch_osm.py,
and for the same reason.

    sitelinks  ~9,800 QIDs at 50 per request        =  ~200 requests   ~2 minutes
    views      one request per article per language = ~86,000 requests  ~7 hours

Both scale with the corpus, which is still growing as cities are surveyed; `--plan`
prints today's real numbers and makes no requests. Both passes are incremental, so
rerunning after a `build.py` costs only the delta.

Sitelink COUNT is itself a fame proxy and comes free with the cheap pass, so a usable
ranking exists before a single pageview is fetched. That matters twice over: if the
sitelink ranking already looks wrong the expensive pass should not be started at all,
and 86,000 requests is not something to launch on an assumption.

Measured spread over 9,826 QIDs on 2026-08-29: 3% have no Wikipedia article at all,
18% have exactly one, half have four or more, 70 have 64+. Mean 9.1. The long tail is
where the cost lives — the top 0.7% of QIDs are ~5% of the requests.

WHY A LEDGER AND NOT A FILE PER RESPONSE. 86,000 small files is a slow directory on
NTFS and a slow rerun. Each pass appends one JSON line per answered question to a
`.jsonl` under cache/wiki/ and flushes immediately, so an interrupt loses at most the
request in flight and a rerun costs zero requests. Nothing is ever rewritten in place.
A truncated last line (kill during write) is skipped on load, costing one refetch.

WHY ONE KEPT-OPEN CONNECTION. Measured on this machine: 5.4 req/s opening a fresh
connection per request, 19.6 req/s reusing one. That is not just our convenience — it is
one TLS handshake asked of Wikimedia instead of 50,000.

PACING, AND WHAT THE SERVER ACTUALLY ALLOWS. Deliberately NOT the duty-cycle pacing
fetch_osm.py uses: that exists because Overpass rations CPU-seconds against a query
whose cost cannot be predicted, and none of it applies to a fixed-cost REST lookup. A
steady rate is the right shape here. The number was measured, not guessed — 10 req/s
drew a 429 roughly every 300 requests, so the default is 5 and the pacer is adaptive:
a 429 costs 20% of the rate permanently-ish, and 200 clean requests win 10% of it back.
That settles just under whatever the endpoint is willing to give on the day, which is
better than any fixed guess. Sustained in practice: ~3-4 req/s.

WHY sitematrix DECIDES WHAT IS A WIKIPEDIA. Wikidata sitelinks include wikiquote,
wikivoyage, commons and friends, and a dbname is not a language code:
`be_x_oldwiki` lives at be-tarask.wikipedia.org and `zh_yuewiki` at zh-yue.wikipedia.org.
Guessing either the family or the host from the dbname gets both wrong. One cached
`action=sitematrix` request gives the authoritative dbname -> (language, host) map.

Usage:
    python fetch_wiki.py --plan                  # zero requests, says what it would do
    python fetch_wiki.py                         # sitelinks pass (default), resumable
    python fetch_wiki.py --pass views            # pageviews pass, resumable, ~7 h
    python fetch_wiki.py --pass views --max-requests 2000  # a measured slice
    python fetch_wiki.py --pass views --limit 2000         # only the 2000 most-linked
    python fetch_wiki.py --pass merge            # rewrite data/wiki.json from cache only
    python fetch_wiki.py --pass pop              # P1082 population, ~200 requests, ~1 min
    python fetch_wiki.py --only Q150188,Q170579  # smoke test

--only and --limit scope what is FETCHED, never what is written: data/wiki.json is
always rebuilt over the whole corpus from whatever the caches hold, so a smoke test
cannot truncate it. `--pass merge` is that rebuild with no fetching at all.
"""

import argparse
import datetime as dt
import http.client
import json
import pathlib
import re
import socket
import ssl
import sys
import time
import urllib.parse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
CACHE = HERE / "cache" / "wiki"
BASE = DATA / "base.json"
OUT = DATA / "wiki.json"

SITEMATRIX = CACHE / "sitematrix.json"
SITELINKS_LEDGER = CACHE / "sitelinks.jsonl"
VIEWS_LEDGER = CACHE / "views.jsonl"
P31_LEDGER = CACHE / "p31.jsonl"
POP_LEDGER = CACHE / "pop.jsonl"

# P31 gets its OWN ledger and its OWN output file rather than joining data/wiki.json.
# The views pass is a ~7 h run that rewrites wiki.json when it finishes, so anything
# sharing that file races with it for hours at a time. p31.json is written once in about
# four minutes and read by pick_levels.py; nothing has to be coordinated.
P31_OUT = DATA / "p31.json"

# Population (P1082) follows p31 exactly, for the same reason and one more: it CANNOT
# share the p31 ledger even though it wants the identical request. That ledger already
# holds every corpus QID, so `--pass p31` resumes to an empty worklist and would fetch
# nothing at all — the pass would appear to succeed and produce no populations.
POP_OUT = DATA / "pop.json"

# Wikimedia's User-Agent policy asks for something descriptive with a way to reach a
# human. Same string as fetch_osm.py, and it stays exactly as written.
UA = "maps-neighborhoods/0.1 (https://anita.garden; anitaxinchen@gmail.com)"

WIKIDATA_HOST = "www.wikidata.org"
PAGEVIEWS_HOST = "wikimedia.org"

# 50 is the API maximum for an anonymous wbgetentities call. Asking for more silently
# truncates rather than erroring, so this must not be raised.
BATCH = 50

MONTHS = 12
TIMEOUT = 60
MAX_RETRIES = 4

# Successes needed before the pacer eases back toward --rps after a 429. See Http.get.
RECOVER_AFTER = 200

QID_RE = re.compile(r"^Q[1-9][0-9]*$")


# --------------------------------------------------------------------------- transport


class NotFound(Exception):
    """404 from the pageviews API. NORMAL: it means this article had no views in the
    window (or did not exist then). It is an answer, not a failure — recorded as zero
    and never asked again."""


class Http:
    """One kept-open HTTPS connection per host, with the retry rules the two APIs need.

    http.client is deliberate: the response body must be read in full before the next
    request or the connection is unusable, which is easy to get right here and is what
    buys the 3.6x. urllib.request cannot reuse a connection at all.
    """

    def __init__(self, rps):
        self.conns = {}
        # `floor` is the fastest we will ever go — the --rps ceiling expressed as a
        # minimum gap between requests. `interval` starts there, grows on a 429, and
        # creeps back down over clean requests, but never below `floor`.
        self.floor = 1.0 / rps if rps else 0.0
        self.interval = self.floor
        self.next_ok = 0.0
        self.rps = rps
        self.ok_streak = 0
        # Per host, because the two APIs are nothing like each other: a 50-QID
        # wbgetentities batch takes ~0.4 s and a pageviews lookup ~0.08 s. One pooled
        # average would project the views pass off the wrong number by 5x.
        self.n = {}
        self.spent = {}
        self.began = {}
        self.reconnects = 0
        self.throttled = 0

    def _conn(self, host):
        c = self.conns.get(host)
        if c is None:
            c = self.conns[host] = http.client.HTTPSConnection(host, timeout=TIMEOUT)
        return c

    def _drop(self, host):
        c = self.conns.pop(host, None)
        if c is not None:
            try:
                c.close()
            except OSError:
                pass

    def get(self, host, path):
        """GET one JSON document. Raises NotFound on 404, RuntimeError once polite
        retries are exhausted."""
        tries = 0
        while True:
            wait = self.next_ok - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            t0 = time.monotonic()
            self.began.setdefault(host, t0)
            try:
                conn = self._conn(host)
                conn.request(
                    "GET", path, headers={"User-Agent": UA, "Accept": "application/json"}
                )
                resp = conn.getresponse()
                raw = resp.read()  # always drain, or the connection cannot be reused
                status = resp.status
            except (http.client.HTTPException, socket.error, ssl.SSLError, OSError) as err:
                # A kept-open connection gets closed by the far end routinely — after
                # some number of requests, or some idle time. That is not an error, it
                # is the cost of reuse, and the FIRST retry is just reopening it, so it
                # must not sleep. Sleeping there cost 2 s roughly every 15 requests in
                # testing: 1,200 requests took 324 s instead of 165 s. Only a second
                # consecutive failure means something is actually wrong.
                self._drop(host)
                self.reconnects += 1
                tries += 1
                if tries > MAX_RETRIES:
                    raise RuntimeError(f"{type(err).__name__}: {err}") from err
                if tries > 1:
                    time.sleep(min(30, 2**tries))
                continue
            finally:
                # Paced from the request's START, not its end. Pacing from the end
                # makes the real rate 1/(latency + interval) — at 80 ms latency and a
                # nominal 10/s that is 5.5/s, half the intended budget for no reason.
                self.next_ok = t0 + self.interval

            self.n[host] = self.n.get(host, 0) + 1
            self.spent[host] = self.spent.get(host, 0.0) + time.monotonic() - t0

            if status == 200:
                # Additive recovery after a multiplicative back-off, so one transient
                # 429 does not halve the rest of a three-hour run. Slow and one-tenth
                # at a time: the aim is to sit just under whatever the server is
                # actually willing to give, not to race back up to --rps.
                self.ok_streak += 1
                if self.ok_streak >= RECOVER_AFTER and self.interval > self.floor:
                    self.interval = max(self.floor, self.interval * 0.9)
                    self.ok_streak = 0
                return json.loads(raw)
            if status == 404:
                raise NotFound(path)
            if status == 429:
                # Honour Retry-After; fall back to a flat minute when it is absent.
                # Then take 20% off the rate — a 429 is the server saying our chosen
                # rate was wrong, and going straight back to it just earns another one.
                # The 200-clean-request recovery above is what climbs back. Never
                # slower than 1/s: below that something else is wrong and crawling
                # through 66,000 requests at 0.5/s helps nobody.
                delay = float(resp.headers.get("Retry-After") or 60)
                self.throttled += 1
                self.ok_streak = 0
                self.interval = min(1.0, (self.interval or 0.2) * 1.25)
                print(
                    f"      429, waiting {delay:.0f}s, backing off to "
                    f"{1 / self.interval:.1f} req/s",
                    flush=True,
                )
                time.sleep(delay + 2)
                continue
            if status in (500, 502, 503, 504) and tries < MAX_RETRIES:
                tries += 1
                delay = min(60, 5 * 2**tries)
                print(f"      {status}, retry {tries}/{MAX_RETRIES} in {delay}s", flush=True)
                time.sleep(delay)
                continue
            raise RuntimeError(f"HTTP {status}: {raw[:200]!r}")

    def close(self):
        for host in list(self.conns):
            self._drop(host)

    def rate(self, host):
        """Requests per second this host would sustain if latency were the only limit.
        The rate actually achieved is min(this, the pacer's ceiling)."""
        spent = self.spent.get(host, 0.0)
        return self.n.get(host, 0) / spent if spent else 0.0

    def wall_rate(self, host):
        """Requests per second actually achieved, pacing sleeps and Retry-After waits
        included. This is the only number worth projecting a long run from."""
        began = self.began.get(host)
        span = time.monotonic() - began if began else 0.0
        return self.n.get(host, 0) / span if span > 0 else 0.0


# ----------------------------------------------------------------------------- ledgers


def read_ledger(path):
    """Every line of a .jsonl cache. A trailing partial line — the one that was being
    written when a run was killed — is dropped rather than crashing the rerun."""
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


class Ledger:
    """Append-and-flush. Never rewritten, so the file on disk is always a valid prefix
    of what this run has actually learned."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("a", encoding="utf-8")

    def add(self, rec):
        self.fh.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


# -------------------------------------------------------------------------- the inputs


def units_by_qid(only=None):
    """qid -> list of (city qid, name, level) for every unit carrying that QID.

    A `wikidata` tag can hold several QIDs separated by ';' (one unit in the current
    corpus does); each is a real item and each is taken. Anything that is not a plain
    Q-number is dropped rather than sent to the API.
    """
    if not BASE.exists():
        sys.exit(f"{BASE} not found — run `python build.py` first")
    data = json.loads(BASE.read_text("utf-8"))
    cities = data["cities"]
    out = {}
    for u in data["units"]:
        raw = u.get("q")
        if not raw:
            continue
        for qid in (p.strip() for p in raw.split(";")):
            if not QID_RE.match(qid):
                continue
            if only and qid not in only:
                continue
            city = cities.get(u["c"], {}).get("name", u["c"])
            out.setdefault(qid, []).append((u["c"], u.get("n"), city, u.get("k")))
    return out


def month_window(end_yyyymm=None):
    """The last MONTHS complete months, as the API's YYYYMMDD pair.

    Complete months only: the current month is partial and would make a unit fetched on
    the 2nd look ten times less famous than the same unit fetched on the 28th. The
    window is stamped into every cached record so that running this again next month
    refetches rather than silently mixing two windows in one sum.
    """
    if end_yyyymm:
        y, m = int(end_yyyymm[:4]), int(end_yyyymm[4:6])
    else:
        today = dt.date.today()
        y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    last = dt.date(y, m, 1) + dt.timedelta(days=32)
    end = dt.date(last.year, last.month, 1) - dt.timedelta(days=1)
    sy, sm = y, m - (MONTHS - 1)
    while sm < 1:
        sm += 12
        sy -= 1
    return dt.date(sy, sm, 1).strftime("%Y%m%d"), end.strftime("%Y%m%d")


def sitematrix(http_, allow_fetch=True):
    """dbname -> (language code, host) for every Wikipedia language edition.

    `smtype=language` is what excludes commons/meta/species and the rest of the
    "special" group; `code == "wiki"` is what excludes wikiquote/wikivoyage/wiktionary
    inside a language. Closed wikis are KEPT — their articles still exist and still get
    read, and a sitelink to one is still a signal.
    """
    if SITEMATRIX.exists():
        raw = json.loads(SITEMATRIX.read_text("utf-8"))
    elif not allow_fetch:
        return None
    else:
        path = (
            "/w/api.php?action=sitematrix&format=json&formatversion=2"
            "&smtype=language&smsiteprop=dbname|url|code"
        )
        raw = http_.get(WIKIDATA_HOST, path)
        CACHE.mkdir(parents=True, exist_ok=True)
        SITEMATRIX.write_text(json.dumps(raw, ensure_ascii=False), "utf-8")

    wikis = {}
    for key, group in raw["sitematrix"].items():
        if not key.isdigit():
            continue
        lang = group["code"]
        for site in group.get("site", []):
            if site.get("code") != "wiki" or site.get("private"):
                continue
            host = urllib.parse.urlsplit(site["url"]).netloc
            wikis[site["dbname"]] = (lang, host)
    return wikis


# ------------------------------------------------------------------- pass 1: sitelinks


def pass_sitelinks(http_, wikis, want, limit):
    have = {r["q"] for r in read_ledger(SITELINKS_LEDGER)}
    todo = [q for q in want if q not in have]
    if limit:
        todo = todo[:limit]
    print(
        f"sitelinks: {len(todo):,} to fetch, {len(want) - len(todo):,} cached or skipped "
        f"({len(todo) // BATCH + bool(len(todo) % BATCH)} requests)"
    )
    if not todo:
        return

    ledger = Ledger(SITELINKS_LEDGER)
    missing = 0
    try:
        for i in range(0, len(todo), BATCH):
            batch = todo[i : i + BATCH]
            path = (
                "/w/api.php?action=wbgetentities&format=json&formatversion=2"
                "&props=sitelinks&ids=" + urllib.parse.quote("|".join(batch))
            )
            payload = http_.get(WIKIDATA_HOST, path)
            entities = payload.get("entities") or {}
            for qid in batch:
                ent = entities.get(qid)
                if ent is None or "missing" in ent:
                    # A deleted or merged-away item. Recorded as zero so it is never
                    # asked about again; the OSM tag is simply stale.
                    missing += 1
                    ledger.add({"q": qid, "s": {}, "gone": 1})
                    continue
                titles = {}
                for dbname, link in (ent.get("sitelinks") or {}).items():
                    hit = wikis.get(dbname)
                    if hit is None:
                        continue  # wikiquote, wikivoyage, commons, ...
                    titles[hit[0]] = link["title"]
                ledger.add({"q": qid, "s": titles})
            done = min(i + BATCH, len(todo))
            print(
                f"  [{done:>5,}/{len(todo):,}] {http_.rate(WIKIDATA_HOST):.1f} req/s",
                flush=True,
            )
    finally:
        ledger.close()
    if missing:
        print(f"  {missing} QIDs no longer exist on Wikidata (stale OSM tags)")


# ------------------------------------------------------------------------- pass: p31


def pass_p31(http_, want, limit):
    """`P31` (instance of) for every unit QID — what KIND of division each one is.

    This is the only signal that survives the admin/place split. Whether a division is
    tagged `boundary=administrative` or `place=borough` is a mapping convention, not a
    statement about the place: Berlin's twelve Bezirke are `place=borough`, Mexico City's
    sixteen alcaldías likewise, Amsterdam's stadsdelen are `place=suburb` — while Tokyo's
    wards are `admin=7` and Paris's arrondissements `admin=9`. Classifying by tag family
    therefore mis-tiers about a third of cities. P31 reads the semantics instead:

        Tokyo    admin=7            -> special ward of Japan
        Berlin   place=borough      -> borough of Berlin
        Amsterdam place=suburb      -> borough of Amsterdam
        Paris    admin=8            -> commune of France
        London   place=suburb       -> area of London          (informal)
        LA       place=neighbourhood-> neighborhood            (informal)
        Seoul    admin=6            -> city of South Korea     (a NEIGHBOURING city)

    It also supplies the user-facing label for a tier — "wards", "arrondissements",
    "boroughs" — which is the thing no rule over `admin_level` numbers can produce.

    **Tiers are mapped from the target QID, never from the English string.** The label is
    for reading; `Q19730508` is what identifies "special ward of Japan" stably across
    languages and renamings. Two records per entity are stored: the unit's P31 targets,
    and one label per distinct target.

    `props=claims` returns every claim, not just P31, because wbgetentities has no way to
    ask for one property. That is ~40 MB over the whole corpus and it buys reuse of the
    batching, pacing and ledger that the sitelinks pass already proved against 429s;
    `wbgetclaims` would fetch exactly P31 but only one entity per request, which is 9,755
    requests instead of 196.
    """
    have = {r["q"] for r in read_ledger(P31_LEDGER) if "q" in r}
    todo = [q for q in want if q not in have]
    if limit:
        todo = todo[:limit]
    print(
        f"p31: {len(todo):,} to fetch, {len(want) - len(todo):,} cached or skipped "
        f"({len(todo) // BATCH + bool(len(todo) % BATCH)} requests)"
    )

    ledger = Ledger(P31_LEDGER)
    try:
        for i in range(0, len(todo), BATCH):
            batch = todo[i : i + BATCH]
            path = (
                "/w/api.php?action=wbgetentities&format=json&formatversion=2"
                "&props=claims&ids=" + urllib.parse.quote("|".join(batch))
            )
            entities = (http_.get(WIKIDATA_HOST, path).get("entities") or {})
            for qid in batch:
                ent = entities.get(qid)
                if ent is None or "missing" in ent:
                    ledger.add({"q": qid, "t": [], "gone": 1})
                    continue
                targets = []
                for claim in (ent.get("claims") or {}).get("P31", []):
                    try:
                        targets.append(
                            claim["mainsnak"]["datavalue"]["value"]["id"]
                        )
                    except (KeyError, TypeError):
                        continue  # novalue/somevalue snaks carry no id
                ledger.add({"q": qid, "t": targets})
            done = min(i + BATCH, len(todo))
            print(
                f"  [{done:>5,}/{len(todo):,}] {http_.rate(WIKIDATA_HOST):.1f} req/s",
                flush=True,
            )
    finally:
        ledger.close()

    # Labels for whatever P31 targets the corpus actually uses. Small — the long tail of
    # neighbourhood types is only a few hundred distinct entities over 56 cities.
    types, labels = load_p31()
    need = sorted({t for v in types.values() for t in v} - set(labels))
    print(f"p31 labels: {len(need):,} distinct types to name "
          f"({len(need) // BATCH + bool(len(need) % BATCH)} requests)")
    if need:
        ledger = Ledger(P31_LEDGER)
        try:
            for i in range(0, len(need), BATCH):
                batch = need[i : i + BATCH]
                path = (
                    "/w/api.php?action=wbgetentities&format=json&formatversion=2"
                    "&props=labels&languages=en&ids="
                    + urllib.parse.quote("|".join(batch))
                )
                entities = (http_.get(WIKIDATA_HOST, path).get("entities") or {})
                for tid in batch:
                    ent = entities.get(tid) or {}
                    lab = ((ent.get("labels") or {}).get("en") or {}).get("value")
                    ledger.add({"lab": tid, "l": lab or tid})
        finally:
            ledger.close()


def load_p31():
    """(qid -> [type qids], type qid -> english label) from the ledger."""
    types, labels = {}, {}
    for rec in read_ledger(P31_LEDGER):
        if "lab" in rec:
            labels[rec["lab"]] = rec["l"]
        elif "q" in rec:
            types[rec["q"]] = rec.get("t") or []
    return types, labels


def write_p31(want):
    types, labels = load_p31()
    out = {
        "types": {q: types[q] for q in want if types.get(q)},
        "labels": labels,
    }
    P31_OUT.write_text(
        json.dumps(out, ensure_ascii=False, separators=(",", ":")), "utf-8"
    )
    return out


# ------------------------------------------------------------------- pass 4: P1082 pop


def best_p1082(claims):
    """(population, year) from an entity's P1082 claims, or (None, None).

    An item routinely carries a dozen census figures, one per year, so picking is the
    whole job. Order: `preferred` rank first — that is the editors' own statement of
    which figure is current — then the latest `P585` point in time. `deprecated` is
    dropped outright; it marks a value known to be wrong, not merely old.

    An undated claim sorts below every dated one rather than being discarded, because on
    small places it is often the only figure there is.
    """
    best = None
    for claim in claims.get("P1082", []):
        rank = claim.get("rank")
        if rank == "deprecated":
            continue
        try:
            # Wikidata quantities are strings with an explicit sign: "+21503".
            amount = int(float(claim["mainsnak"]["datavalue"]["value"]["amount"]))
        except (KeyError, TypeError, ValueError):
            continue  # novalue/somevalue snaks, and the odd malformed quantity
        year = None
        for qual in (claim.get("qualifiers") or {}).get("P585", []):
            try:
                # "+2021-01-01T00:00:00Z" — the leading sign is why this is [1:5].
                year = int(qual["datavalue"]["value"]["time"][1:5])
            except (KeyError, TypeError, ValueError):
                pass
        key = (rank == "preferred", year or 0)
        if best is None or key > best[0]:
            best = (key, amount, year)
    return (best[1], best[2]) if best else (None, None)


def pass_pop(http_, want, limit):
    """P1082 for every corpus QID, 50 at a time.

    Measured against a 600-unit stratified sample of units that have no OSM `population`
    tag: 37% carry a P1082 claim. Per city it is very uneven — Rome, Prague, Istanbul,
    Melbourne, Sydney, Osaka and Tel Aviv all came back at 100%, while Hanoi was 0% —
    but the cost does not depend on the yield.

    THE SAME REQUEST AS `pass_p31`, deliberately not merged with it. Sharing one fetch
    would halve the traffic, but p31.jsonl is already complete for the whole corpus, so
    a combined pass would resume to nothing; and the two products have different
    lifetimes — a tier map is stable, a census figure is superseded. 201 requests is
    about 1.3 minutes and is not worth coupling them over.

    A QID with no P1082 is recorded as `p: null` rather than skipped, so the resume set
    is "asked", not "answered", and a rerun does not re-ask the 63%.
    """
    have = {r["q"] for r in read_ledger(POP_LEDGER) if "q" in r}
    todo = [q for q in want if q not in have]
    if limit:
        todo = todo[:limit]
    print(
        f"pop: {len(todo):,} to fetch, {len(want) - len(todo):,} cached "
        f"({len(todo) // BATCH + bool(len(todo) % BATCH)} requests)"
    )

    ledger = Ledger(POP_LEDGER)
    try:
        for i in range(0, len(todo), BATCH):
            batch = todo[i : i + BATCH]
            path = (
                "/w/api.php?action=wbgetentities&format=json&formatversion=2"
                "&props=claims&ids=" + urllib.parse.quote("|".join(batch))
            )
            entities = (http_.get(WIKIDATA_HOST, path).get("entities") or {})
            for qid in batch:
                ent = entities.get(qid)
                if ent is None or "missing" in ent:
                    ledger.add({"q": qid, "p": None, "gone": 1})
                    continue
                pop, year = best_p1082(ent.get("claims") or {})
                rec = {"q": qid, "p": pop}
                if year is not None:
                    rec["y"] = year
                ledger.add(rec)
            done = min(i + BATCH, len(todo))
            print(
                f"  [{done:>5,}/{len(todo):,}] {http_.rate(WIKIDATA_HOST):.1f} req/s",
                flush=True,
            )
    finally:
        ledger.close()


def load_pop():
    """qid -> (population, year or None) for every QID that has a figure."""
    out = {}
    for rec in read_ledger(POP_LEDGER):
        if rec.get("q") and rec.get("p") is not None:
            out[rec["q"]] = (rec["p"], rec.get("y"))
    return out


def write_pop(want):
    pop = load_pop()
    out = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "property": "P1082",
        # Wikidata is CC0, so this travels for provenance rather than obligation. The
        # figures underneath are national census data and the YEAR is the part the UI
        # must not drop — a 2010 count and a 2024 one do not belong on the same axis.
        "license": "CC0 1.0 (Wikidata)",
        "units": {q: {"p": pop[q][0], "y": pop[q][1]} for q in want if q in pop},
    }
    POP_OUT.write_text(
        json.dumps(out, ensure_ascii=False, separators=(",", ":")), "utf-8"
    )
    return out


# ----------------------------------------------------------------------- pass 2: views


def views_worklist(want, sitelinks, wikis, window):
    """[(lang, title)] still needing a fetch, plus the total the pass would ever make.

    Ordered by descending sitelink count. A partial run is then already enough to rank
    the top of the deck, which is the only part the quiz cares about — and since the
    request total is known exactly, front-loading costs nothing in the projection.
    """
    done = {(r["l"], r["t"]) for r in read_ledger(VIEWS_LEDGER) if r.get("r") == window}
    langs = {lang for lang, _ in wikis.values()}
    todo, total, seen = [], 0, set()
    order = sorted(want, key=lambda q: -len(sitelinks.get(q, {})))
    for qid in order:
        for lang, title in sorted(sitelinks.get(qid, {}).items()):
            if lang not in langs:
                continue
            key = (lang, title)
            if key in seen:
                continue  # two units can share an article; ask once
            seen.add(key)
            total += 1
            if key not in done:
                todo.append(key)
    return todo, total


def pass_views(http_, wikis, want, sitelinks, window, limit, max_requests):
    hosts = {lang: host for lang, host in wikis.values()}
    if limit:
        want = sorted(want, key=lambda q: -len(sitelinks.get(q, {})))[:limit]
    todo, total = views_worklist(want, sitelinks, wikis, window)
    start, end = window.split("-")
    if max_requests:
        todo = todo[:max_requests]

    print(
        f"views: {len(todo):,} to fetch of {total:,} articles "
        f"({total - len(todo):,} cached or skipped), window {start}..{end}"
    )
    if not todo:
        return

    ledger = Ledger(VIEWS_LEDGER)
    t0 = time.monotonic()
    empty = 0
    try:
        for i, (lang, title) in enumerate(todo, 1):
            host = hosts.get(lang)
            if host is None:
                continue
            # Underscores then full percent-encoding, slashes included: an article
            # called "Sant Andreu/Sagrera" is one path segment, not two.
            slug = urllib.parse.quote(title.replace(" ", "_"), safe="")
            path = (
                f"/api/rest_v1/metrics/pageviews/per-article/{host}"
                f"/all-access/all-agents/{slug}/monthly/{start}/{end}"
            )
            try:
                payload = http_.get(PAGEVIEWS_HOST, path)
                months = [int(it.get("views") or 0) for it in payload.get("items", [])]
            except NotFound:
                # Expected and common. Zero, cached, never retried.
                months = []
                empty += 1
            except RuntimeError as err:
                # Not cached: an unanswered question must stay unanswered so the next
                # run asks it again.
                print(f"      FAILED {lang}:{title} — {err}", flush=True)
                continue
            ledger.add(
                {"l": lang, "t": title, "r": window, "v": sum(months), "m": months}
            )
            if i % 250 == 0 or i == len(todo):
                elapsed = time.monotonic() - t0
                print(
                    f"  [{i:>6,}/{len(todo):,}] {i / elapsed:.1f} req/s wall, "
                    f"{http_.rate(PAGEVIEWS_HOST):.1f} req/s in-flight, "
                    f"{empty:,} with no data",
                    flush=True,
                )
    finally:
        ledger.close()
        wall = time.monotonic() - t0
        n = http_.n.get(PAGEVIEWS_HOST, 0)
        pace = 1 / http_.interval if http_.interval else float("inf")
        print(
            f"  {n:,} requests in {wall/60:.1f} min — {n/wall:.1f} req/s wall, "
            f"{empty:,} 404 (no data), {http_.reconnects} reconnects, "
            f"{http_.throttled} 429s, ending at a {pace:.1f} req/s ceiling"
        )


# ------------------------------------------------------------------------- the product


def load_sitelinks(only=None):
    """qid -> {lang: title}. Later lines win, so a --refresh style re-append would."""
    out = {}
    for rec in read_ledger(SITELINKS_LEDGER):
        if only and rec["q"] not in only:
            continue
        out[rec["q"]] = rec.get("s") or {}
    return out


def write_output(want, sitelinks, window):
    views = {}
    for rec in read_ledger(VIEWS_LEDGER):
        if rec.get("r") == window:
            views[(rec["l"], rec["t"])] = rec["v"]

    out = {}
    for qid in sorted(want, key=lambda q: int(q[1:])):
        titles = sitelinks.get(qid)
        if titles is None:
            continue  # sitelinks pass has not reached it yet
        by_lang, fetched = {}, 0
        for lang, title in titles.items():
            v = views.get((lang, title))
            if v is None:
                continue
            fetched += 1
            if v:
                by_lang[lang] = v
        out[qid] = {
            "sitelinks": len(titles),
            "views": sum(by_lang.values()),
            # Only languages with views > 0, sorted biggest first — this is the column
            # anyone will actually read, and the zeroes are noise.
            "byLang": dict(sorted(by_lang.items(), key=lambda kv: -kv[1])),
            "title": dict(sorted(titles.items())),
            # How many of `sitelinks` have view data yet. fetched == sitelinks means the
            # views total is final; anything less means the pass is still mid-run.
            "fetched": fetched,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), "utf-8")
    return out


# --------------------------------------------------------------------------- reporting


def report(want, units, sitelinks, wiki, top=20):
    n = len(want)
    # "not fetched yet" and "fetched, has no article" are different facts and must not
    # share a bar. base.json grows as levels are added, so a stale-by-one-build cache
    # is the normal state here, not an error.
    fetched = [q for q in want if q in sitelinks]
    resolved = [q for q in fetched if sitelinks[q]]
    counts = sorted((len(sitelinks[q]) for q in resolved), reverse=True)
    print()
    print(f"  distinct QIDs in base.json      {n:>7,}")
    print(f"  sitelinks fetched               {len(fetched):>7,}  ({len(fetched)/n:.0%})")
    if not counts:
        return
    d = len(fetched)
    print(f"  with >=1 Wikipedia article      {len(resolved):>7,}  ({len(resolved)/d:.0%}"
          f" of fetched)")
    print(f"  total articles across languages {sum(counts):>7,}")
    print(f"  median / mean sitelinks         {counts[len(counts)//2]:>7,}"
          f" / {sum(counts)/len(counts):.1f}")
    print()
    print("  sitelink count distribution, over the QIDs fetched")
    bands = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 10**6)]
    zero = d - len(resolved)
    print(f"    {'0':>9}  {zero:>6,}  {zero/d:>5.1%}  {'#' * round(40 * zero / d)}")
    for lo, hi in bands:
        c = sum(1 for x in counts if lo <= x <= hi)
        label = f"{lo}" if lo == hi else (f"{lo}+" if hi > 10**5 else f"{lo}-{hi}")
        print(f"    {label:>9}  {c:>6,}  {c/d:>5.1%}  {'#' * round(40 * c / d)}")

    print()
    print(f"  top {top} by sitelink count")
    ranked = sorted(resolved, key=lambda q: (-len(sitelinks[q]), q))[:top]
    for qid in ranked:
        name, city, level = "?", "?", "?"
        if units.get(qid):
            _, name, city, level = units[qid][0]
        v = (wiki.get(qid) or {}).get("views")
        vs = f"{v:>10,}" if v else " " * 10
        # The article title is printed whenever it is not just the OSM name, because a
        # wildly over-famous obscure unit is nearly always a MIS-TAGGED QID rather than
        # a surprise, and seeing the title says so instantly. Real example: a Singapore
        # unit named "Peng Siang" carries a QID whose article is "Common year".
        art = sitelinks[qid].get("en") or next(iter(sitelinks[qid].values()))
        via = f"  -> {art}" if art != name else ""
        print(
            f"    {len(sitelinks[qid]):>4}  {vs}  {qid:<11} {name or '?':<26} "
            f"{city} · {level}{via}"
        )


def project(http_, todo_views, rps):
    if not todo_views:
        return
    # Only the pageviews host says anything about the pageviews pass — a 50-QID
    # wbgetentities batch is five times slower than a pageviews lookup, and projecting
    # off it is how you get 6 hours for a 90-minute job.
    # Prefer the rate this run actually achieved. It already contains the 429 waits and
    # the pacer's own sleeps, which between them cost about 20% — projecting off the
    # ceiling instead reads 4.3 h for a job that takes 5.2 h. Fall back to the ceiling,
    # then to the flag, when there is nothing measured to go on.
    n = http_.n.get(PAGEVIEWS_HOST, 0)
    rate = http_.wall_rate(PAGEVIEWS_HOST) if n > 200 else 0.0
    how = "measured"
    if not rate:
        rate = 1 / http_.interval if http_.interval else rps
        how = "assumed"
    secs = todo_views / rate
    print(
        f"\n  {todo_views:,} view requests remain — at {rate:.1f} req/s ({how}) "
        f"that is {secs/60:.0f} min ({secs/3600:.1f} h)"
    )


# -------------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pass", dest="which", choices=["sitelinks", "views", "merge", "p31", "pop"],
        default="sitelinks",
    )
    ap.add_argument("--only", help="comma-separated QIDs")
    ap.add_argument("--limit", type=int, help="at most N QIDs this run")
    ap.add_argument("--max-requests", type=int, help="views pass: stop after N requests")
    ap.add_argument("--rps", type=float, default=5.0, help="request ceiling (default 5)")
    ap.add_argument("--months-end", help="YYYYMM of the last month in the window")
    ap.add_argument("--plan", action="store_true", help="print the plan, make no requests")
    ap.add_argument(
        "--no-merge", action="store_true",
        help="fetch into the ledger but do not rewrite data/wiki.json — safe to run "
             "alongside a views pass, which rewrites that file when it finishes",
    )
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    units = units_by_qid(only)
    want = sorted(units, key=lambda q: int(q[1:]))
    start, end = month_window(args.months_end)
    window = f"{start}-{end}"

    http_ = Http(args.rps)
    try:
        if args.plan:
            wikis = sitematrix(http_, allow_fetch=False)
            sitelinks = load_sitelinks(only)
            print(f"plan — {len(want):,} distinct QIDs over "
                  f"{sum(len(v) for v in units.values()):,} units")
            print(f"  window          {start}..{end} ({MONTHS} complete months)")
            print(f"  sitematrix      {'cached' if wikis else 'NOT cached (1 request)'}")
            need = len(want) - len([q for q in want if q in sitelinks])
            print(f"  sitelinks pass  {need:,} QIDs -> "
                  f"{need // BATCH + bool(need % BATCH)} requests")
            if wikis and sitelinks:
                todo, total = views_worklist(want, sitelinks, wikis, window)
                print(f"  views pass      {total:,} articles, {len(todo):,} not yet cached")
                report(want, units, sitelinks, {}, args.top)
                project(http_, len(todo), args.rps)
            else:
                print("  views pass      unknown until the sitelinks pass has run")
            return

        # RETURNS EARLY, and that is the point: every other pass ends by rewriting
        # data/wiki.json, which the ~7 h views run also rewrites when it finishes. The
        # p31 pass touches neither that file nor its ledger, so it is safe to run while
        # views is in flight.
        if args.which == "p31":
            pass_p31(http_, want, args.limit)
            out = write_p31(sorted(units_by_qid()))
            kb = P31_OUT.stat().st_size / 1024
            print(f"\nwrote {P31_OUT} — {len(out['types']):,} QIDs typed, "
                  f"{len(out['labels']):,} distinct types, {kb:,.0f} KB")
            return

        # Same early return, same reason, and it also hits a different HOST from the
        # views pass — wbgetentities is www.wikidata.org, pageviews is wikimedia.org —
        # so the two are not competing for one service's rate limit either.
        if args.which == "pop":
            pass_pop(http_, want, args.limit)
            all_units = units_by_qid()
            out = write_pop(sorted(all_units))
            kb = POP_OUT.stat().st_size / 1024
            n_units = sum(len(all_units[q]) for q in out["units"])
            dated = sum(1 for r in out["units"].values() if r["y"])
            print(f"\nwrote {POP_OUT} — {len(out['units']):,} QIDs with a population "
                  f"({len(out['units'])/len(all_units):.0%} of {len(all_units):,}), "
                  f"{dated:,} dated, covering {n_units:,} units, {kb:,.0f} KB")
            return

        wikis = sitematrix(http_)

        if args.which == "sitelinks":
            pass_sitelinks(http_, wikis, want, args.limit)
        elif args.which == "views":
            sitelinks = load_sitelinks(only)
            if not sitelinks:
                sys.exit("no sitelinks cached — run `python fetch_wiki.py` first")
            pass_views(
                http_, wikis, want, sitelinks, window, args.limit, args.max_requests
            )

        # The ledgers are append-only and per-pass, so fetching is always safe to run
        # concurrently; only this merge is not, because `views` rewrites the same file
        # when it finishes — and it runs for ~7 h, which is a long time to be unable to
        # top up sitelinks for units a level-rule change has just admitted. The views run
        # re-reads the sitelinks ledger before its own merge, so nothing is lost by
        # skipping this one.
        if args.no_merge:
            print("--no-merge: ledger updated, data/wiki.json left alone")
            return

        # The product is always the WHOLE corpus. --only and --limit scope what gets
        # FETCHED; if they also scoped the output then a one-QID smoke test would
        # truncate data/wiki.json to one QID, which is a nasty way to lose five hours
        # of views.
        units = units_by_qid()
        want = sorted(units, key=lambda q: int(q[1:]))
        sitelinks = load_sitelinks()
        wiki = write_output(want, sitelinks, window)
        kb = OUT.stat().st_size / 1024
        print(f"\nwrote {OUT} — {len(wiki):,} QIDs, {kb:,.0f} KB")
        report(want, units, sitelinks, wiki, args.top)
        if args.which != "merge":
            todo, _ = views_worklist(want, sitelinks, wikis, window)
            project(http_, len(todo), args.rps)
    finally:
        http_.close()


if __name__ == "__main__":
    main()
