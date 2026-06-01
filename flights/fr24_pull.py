#!/usr/bin/env python3
"""
fr24_pull.py — pull one day's departures, airport by airport, from the FR24 API.

Design (see CLAUDE.md "API usage rules"):
  * Queries DEPARTURES per airport (airports=outbound:<ICAO>) over a time window,
    so every flight is counted exactly once, at the airport it starts from.
  * Paginated + truncation-safe: FR24's Essential plan caps /flight-summary/light
    at ~300 rows per query, so pull_airport calls /count for the true total,
    then pages /light (advancing a time cursor) until it has every record.
  * Resumable: each airport is written to pull/<DAY>/<ICAO>.json the moment it
    finishes. Re-running skips airports whose file exists with ok=true AND
    matches the current window; anything else is re-pulled. Interrupting loses
    only the airport in flight.
  * Credit-metered from FR24's x-fr24-credits-remaining header. Hard-stops at
    CREDIT_CAP, or after STOP_AFTER_FAILS airports fail in a row.
  * Pulls EVERY flight; build_fr24.py drops non-airliners when aggregating.
    (FR24's aircraft= filter caps at 15 type codes — too few to be useful.)
"""
import csv, json, os, re, sys, time, collections
import urllib.request, urllib.parse, urllib.error
from datetime import datetime, timedelta, timezone

# ============================ CONFIG ============================
DAY          = "2026-05-16"      # UTC day to pull (Sat; then 05-17 Sun, 05-18 Mon)
HOUR_FROM    = 0                 # window start hour (inclusive)
HOUR_TO      = 24                # window end hour (exclusive). 24 = whole day.
MAX_AIRPORTS = 0                 # 0 = all (~3260). Busiest-first.
CREDIT_CAP   = 320000            # hard stop: abort if this run's spend exceeds this
STOP_AFTER_FAILS = 5             # hard stop after this many airports fail in a row
PAGE_LIMIT   = 300               # FR24 Essential caps /light at 300 rows/query
RATE_SLEEP   = 2.2               # seconds between calls (Essential = 30 req/min)

BASE       = "https://fr24api.flightradar24.com/api"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
HERE     = os.path.dirname(os.path.abspath(__file__))
DATA     = os.path.join(HERE, "data")
OUT_DIR  = os.path.join(HERE, "pull", DAY)

# ============================ SECRETS ============================
def load_secret(name):
    env = os.environ.get(name)
    if env and env.strip():
        return env.strip()
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        for fn in ("secrets.env", ".env"):
            p = os.path.join(d, fn)
            if os.path.isfile(p):
                for line in open(p, encoding="utf-8"):
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    if k.strip() == name:
                        v = v.strip().strip('"').strip("'")
                        if v:
                            return v
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent

API_KEY = load_secret("FR24_API_KEY")

# ============================ HTTP + CREDITS ============================
calls = 0
credits_remaining = None  # latest x-fr24-credits-remaining header (live balance)
start_remaining   = None  # balance just before this run started
credits_used      = 0     # = start_remaining - credits_remaining (this run's spend)

def _digits(s):
    # join all digit runs, so "665,432" -> 665432 (not 432). Returns None if
    # the value has no digits — caller must treat that as "unknown", NOT as a
    # number (an earlier version fell back to digits in the key name, which
    # silently returned 24 from "fr24" and defeated the credit-headers check).
    m = re.findall(r"\d+", str(s))
    return int("".join(m)) if m else None

def track_credits(hdrs):
    """Track spend from FR24's credit headers. We measure the drop in the
    'remaining' balance — unambiguous however the 'consumed' header is structured."""
    global credits_remaining, start_remaining, credits_used
    rem = con = None
    for k, v in hdrs.items():
        kl = k.lower()
        if "credits-remaining" in kl:
            rem = _digits(v)
        elif "credits-consumed" in kl:
            con = _digits(v)
    if rem is not None:
        credits_remaining = rem
        if start_remaining is None:
            start_remaining = rem + (con or 0)     # balance just before this run
        credits_used = start_remaining - credits_remaining

def window_iso():
    base = datetime.fromisoformat(DAY).replace(tzinfo=timezone.utc)
    iso = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")
    return (iso(base + timedelta(hours=HOUR_FROM)),
            iso(base + timedelta(hours=HOUR_TO)))

def call(endpoint, params):
    """GET with retries. Returns (status, headers, parsed_or_None)."""
    global calls
    url = BASE + endpoint + "?" + urllib.parse.urlencode(
        {k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(url, headers={
        "Accept": "application/json", "Accept-Version": "v1",
        "Authorization": f"Bearer {API_KEY}", "User-Agent": USER_AGENT})
    for attempt in range(4):
        calls += 1
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                body, hdrs, st = r.read().decode("utf-8", "replace"), dict(r.headers), r.status
        except urllib.error.HTTPError as e:
            body, hdrs, st = e.read().decode("utf-8", "replace"), dict(e.headers or {}), e.code
        except Exception as e:
            print(f"    network error ({e}); retry in 5s")
            time.sleep(5); continue
        track_credits(hdrs)
        time.sleep(RATE_SLEEP)
        if st == 429:
            print("    HTTP 429 — backing off 30s"); time.sleep(30); continue
        if st >= 500:
            print(f"    HTTP {st} — retry in 5s"); time.sleep(5); continue
        try:
            return st, hdrs, json.loads(body)
        except Exception:
            return st, hdrs, {"_raw": body[:300]}
    return None, {}, None

def rows(parsed):
    if isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
        return parsed["data"]
    return parsed if isinstance(parsed, list) else []

# ============================ PULL ONE AIRPORT ============================
def pull_airport(icao, w_from, w_to):
    """Departures for one airport. /count gives the true total; /light is paged
    (advancing a first_seen cursor) until every record is collected."""
    base = {"flight_datetime_from": w_from, "flight_datetime_to": w_to,
            "airports": f"outbound:{icao}"}
    st, hd, parsed = call("/flight-summary/count", base)
    count = parsed.get("record_count") if isinstance(parsed, dict) else None
    if not isinstance(count, int):
        return {"airport": icao, "from": w_from, "to": w_to, "ok": False,
                "error": f"count failed (HTTP {st})", "detail": parsed,
                "n_records": 0, "records": []}

    recs, seen = [], set()
    cursor, pages = w_from, 0
    while len(recs) < count and pages < 60:
        pages += 1
        st, hd, parsed = call("/flight-summary/light", dict(
            base, flight_datetime_from=cursor, sort="asc", limit=PAGE_LIMIT))
        if st != 200:
            return {"airport": icao, "from": w_from, "to": w_to, "count": count,
                    "ok": False, "error": f"HTTP {st}", "detail": parsed,
                    "n_records": len(recs), "records": recs}
        page = rows(parsed)
        fresh = [r for r in page if r.get("fr24_id") not in seen]
        recs += fresh
        seen |= {r.get("fr24_id") for r in fresh}
        if len(page) < PAGE_LIMIT:                     # short page -> last page
            break
        # advance the cursor by first_seen: that is the field sort=asc orders by
        # (and that flight_datetime_from filters on). Stop before the cursor
        # could reach the window end, which would make an invalid query.
        nxt = page[-1].get("first_seen")
        # fail loudly if FR24 ever hands back a non-string first_seen (e.g. an
        # epoch int as in their live-positions API) — the string comparison
        # below would otherwise crash with a confusing TypeError mid-pull.
        if nxt is not None and not isinstance(nxt, str):
            sys.exit(f"  !! unexpected first_seen type {type(nxt).__name__}: "
                     f"{nxt!r} — pagination expects an ISO 8601 string. Aborting.")
        if not nxt or nxt >= w_to or (nxt == cursor and not fresh):
            break
        cursor = nxt

    ok = (len(recs) == count)
    out = {"airport": icao, "from": w_from, "to": w_to, "pages": pages,
           "count": count, "n_records": len(recs), "ok": ok, "records": recs}
    if not ok:
        out["error"] = f"got {len(recs)} of {count}"
    return out

# ============================ AIRPORT LIST ============================
def airport_list():
    """ICAO codes with scheduled service, busiest-first (by OpenFlights route count)."""
    iata2icao = {}
    with open(os.path.join(DATA, "airports.dat"), encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            icao, iata = row[5].strip(), row[4].strip()
            # ICAO Doc 7910 location indicators are strictly 4 letters; codes
            # with digits (TC LIDs like CNE3, FAA LIDs, etc.) are not in FR24's
            # database and return HTTP 400 "field format is invalid".
            if len(icao) == 4 and icao.isalpha() and len(iata) == 3 and iata != "\\N":
                iata2icao[iata] = icao
    rc = collections.Counter()
    with open(os.path.join(DATA, "routes.dat"), encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 5:
                continue
            for code in (row[2], row[4]):
                rc[code] += 1
    icao_rc = collections.Counter()
    for iata, n in rc.items():
        ic = iata2icao.get(iata)
        if ic:
            icao_rc[ic] += n
    ordered = [ic for ic, _ in icao_rc.most_common()]
    return ordered[:MAX_AIRPORTS] if MAX_AIRPORTS else ordered

# ============================ MAIN ============================
def main():
    if not API_KEY:
        sys.exit("No FR24_API_KEY — put it in maps/secrets.env")
    os.makedirs(OUT_DIR, exist_ok=True)
    w_from, w_to = window_iso()
    print(f"FR24 pull — day {DAY}, window {w_from} .. {w_to}")
    print(f"output -> {OUT_DIR}\n")

    # --- airport list FIRST (local, free): fail fast if data files are
    # missing or malformed, before any paid call.
    airports = airport_list()

    # --- pre-flight: confirm auth + window + that credit headers work --------
    print("pre-flight: testing one query (ZBAA) ...")
    st, hd, parsed = call("/flight-summary/count", {
        "flight_datetime_from": w_from, "flight_datetime_to": w_to,
        "airports": "outbound:ZBAA"})
    if st != 200:
        sys.exit(f"pre-flight failed (HTTP {st}): {parsed}")
    print(f"  ok — ZBAA window count = {parsed.get('record_count')}")
    if credits_remaining is None:
        sys.exit("credit headers not detected — aborting to avoid uncapped spend.")
    print(f"  credits: this run {credits_used} used, "
          f"{credits_remaining:,} remaining on the account\n")

    # --- resume: which airports are already done ----------------------------
    done = set()
    for fn in os.listdir(OUT_DIR):
        if fn.endswith(".json") and not fn.startswith("_"):
            try:
                d = json.load(open(os.path.join(OUT_DIR, fn), encoding="utf-8"))
                # only count as done if it matches the CURRENT window — so the
                # leftover first-hour files get re-pulled for the full day.
                if (d.get("ok") and d.get("from") == w_from
                        and d.get("to") == w_to):
                    done.add(d["airport"])
            except Exception:
                pass
    todo = [a for a in airports if a not in done]
    print(f"airports: {len(airports)} total, {len(done)} already done, "
          f"{len(todo)} to pull\n")

    # --- pull loop ----------------------------------------------------------
    streak = 0
    for i, icao in enumerate(todo, 1):
        if credits_used >= CREDIT_CAP:
            print(f"\n!! CREDIT_CAP ({CREDIT_CAP:,}) reached — stopping. "
                  f"Re-run to resume.")
            break
        res = pull_airport(icao, w_from, w_to)
        with open(os.path.join(OUT_DIR, f"{icao}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, separators=(",", ":"))
        streak = 0 if res["ok"] else streak + 1
        flag = "ok" if res["ok"] else f"!! {res.get('error')}"
        print(f"  [{i:>4}/{len(todo)}] {icao}  rows={res.get('n_records',0):<4} "
              f"{flag:<22} {credits_used:,} credits used")
        if streak >= STOP_AFTER_FAILS:
            print(f"\n!! {streak} airports failed in a row — stopping to be safe. "
                  f"Check the last few files, fix the cause, then re-run.")
            break

    # --- summary (recomputed from disk so resumed runs stay accurate) -------
    recs_total, flagged, n_files = 0, [], 0
    for fn in os.listdir(OUT_DIR):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        o = json.load(open(os.path.join(OUT_DIR, fn), encoding="utf-8"))
        n_files += 1
        recs_total += o.get("n_records", 0)
        if not o.get("ok"):
            flagged.append(o["airport"])
    summary = {"day": DAY, "window": [w_from, w_to], "airports_pulled": n_files,
               "records": recs_total, "credits_used_this_run": credits_used,
               "credits_remaining": credits_remaining, "calls_this_run": calls,
               "flagged": flagged}
    json.dump(summary, open(os.path.join(OUT_DIR, "_summary.json"), "w"), indent=1)
    print(f"\ndone — {n_files} airports on disk, {recs_total:,} records total, "
          f"{credits_used:,} credits this run, {calls} calls")
    if credits_remaining is not None:
        print(f"       {credits_remaining:,} credits remaining on the account")
    if flagged:
        print(f"!! {len(flagged)} airports flagged: {flagged[:20]}")
        print("   delete those <ICAO>.json files and re-run to re-pull them.")


if __name__ == "__main__":
    main()
