#!/usr/bin/env python3
"""
fr24_probe.py — Flightradar24 API probe.

Run this BEFORE committing to the real pull. Three modes:

  sandbox — FREE. Uses the sandbox key + /sandbox endpoints. The API returns
            STATIC sample data, so this only checks the script's plumbing
            (auth, endpoint paths, JSON parsing) — NOT real coverage. Do this
            first to shake out script bugs without spending a credit.

  smoke   — ~20-30s, a few hundred real credits. Confirms the real API works:
            auth, the airports-filter format, the datetime format, and that
            flight-summary/light returns our fields.

  full    — ~1.5 min, ~6-9k real credits. The real blocker checks: truncation
            (light rows vs /count), China completeness, GA-noise fraction,
            cancelled flights, aircraft filter.

Stdlib only. Tuned for the Essential plan (30 req/min).

USAGE:  python fr24_probe.py sandbox     # free plumbing check (do this first)
        python fr24_probe.py             # smoke test on the real key
        python fr24_probe.py full        # full probe on the real key
        (keys are read from maps/secrets.env — FR24_API_KEY / FR24_SANDBOX_KEY)
"""
import json, os, sys, time, urllib.request, urllib.parse, urllib.error
from datetime import datetime, timedelta, timezone

# ============================ CONFIG ============================
BASE       = "https://fr24api.flightradar24.com/api"
# FR24's API sits behind Cloudflare, which blocks the default Python-urllib
# user-agent (Cloudflare error 1010). A normal browser UA gets through.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
RATE_SLEEP = 2.2     # seconds between calls — safe for Essential's 30 req/min
                     # (bump to 6.5 if you ever run this on an Explorer key)
HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "probe_out")

# set by main() once the mode is known
API_KEY   = None
EP_PREFIX = ""       # "" for the real API, "/sandbox" for the sandbox

def load_secret(name):
    """Read <name> from the environment, else from a gitignored secrets.env /
    .env found by walking up parent dirs from this script. None if not found."""
    env = os.environ.get(name)
    if env and env.strip():
        return env.strip()
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        for fn in ("secrets.env", ".env"):
            path = os.path.join(d, fn)
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        if k.strip() == name:
                            v = v.strip().strip('"').strip("'")
                            if v:
                                return v
        parent = os.path.dirname(d)
        if parent == d:                 # reached the filesystem root
            return None
        d = parent

# Probe a full UTC day from 7 days ago — recent (2 credits/flight) and safely
# inside even Explorer's 30-day history window.
PROBE_DAY = (datetime.now(timezone.utc) - timedelta(days=7)).date()
DAY_FROM  = f"{PROBE_DAY}T00:00:00Z"
DAY_TO    = f"{PROBE_DAY + timedelta(days=1)}T00:00:00Z"
DISC_TO   = f"{PROBE_DAY}T02:00:00Z"        # 2h window for cheap smoke/discovery

# Gap-region airports (ICAO) + (name, rough real departures/day low, high).
# The reality figures are ballpark — the real check is count vs the FR24 website.
TEST_AIRPORTS = {
    "ZBAA": ("Beijing Capital",     450, 600),
    "ZSPD": ("Shanghai Pudong",     500, 700),
    "ZGGG": ("Guangzhou Baiyun",    500, 700),
    "SBGR": ("Sao Paulo Guarulhos", 300, 420),
}

# Common airliner ICAO type codes — used to gauge the GA / non-airliner fraction.
AIRLINER_TYPES = {
    "A319","A320","A321","A318","A19N","A20N","A21N","A332","A333","A338","A339",
    "A342","A343","A345","A346","A359","A35K","A388","A310","B737","B738","B739",
    "B38M","B39M","B752","B753","B762","B763","B764","B772","B77L","B773","B77W",
    "B788","B789","B78X","B744","B748","E170","E175","E190","E195","E290","E295",
    "E75L","E75S","CRJ2","CRJ7","CRJ9","AT72","AT75","AT76","DH8D","BCS1","BCS3",
}

# ============================ HTTP ============================
CALLS = 0

def api(endpoint, params, label):
    """GET a JSON endpoint. Returns (status, headers_dict, parsed_or_None). Saves raw."""
    global CALLS
    CALLS += 1
    url = BASE + EP_PREFIX + endpoint
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "Accept-Version": "v1",
        "Authorization": f"Bearer {API_KEY}",
        "User-Agent": USER_AGENT,
    })
    status, hdrs, body = None, {}, ""
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            status, hdrs, body = r.status, dict(r.headers), r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        status, hdrs = e.code, dict(e.headers or {})
        body = e.read().decode("utf-8", "replace")
    except Exception as e:
        print(f"  ! network error on {endpoint}: {e}")
        time.sleep(RATE_SLEEP)
        return None, {}, None
    with open(os.path.join(OUT_DIR, f"{label}.json"), "w", encoding="utf-8") as f:
        f.write(body)
    try:
        parsed = json.loads(body)
    except Exception:
        parsed = None
    time.sleep(RATE_SLEEP)
    return status, hdrs, parsed

def rows(parsed):
    """flight-summary list responses are nested under a 'data' key."""
    if isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
        return parsed["data"]
    if isinstance(parsed, list):
        return parsed
    return []

def err_text(parsed, status):
    if isinstance(parsed, dict):
        return json.dumps(parsed)[:300]
    return f"HTTP {status}"

def show_record(rec):
    for k in ("fr24_id","flight","callsign","type","orig_icao","dest_icao",
              "dest_icao_actual","datetime_takeoff","datetime_landed","flight_ended"):
        print(f"      {k:18}= {rec.get(k)}")

# ============================ PROBE ============================
def main():
    global API_KEY, EP_PREFIX
    args = [a.lower() for a in sys.argv[1:]]
    sandbox = "sandbox" in args
    mode = "smoke" if sandbox else ("full" if "full" in args else "smoke")

    if sandbox:
        API_KEY, EP_PREFIX = load_secret("FR24_SANDBOX_KEY"), "/sandbox"
        keyname = "FR24_SANDBOX_KEY"
    else:
        API_KEY, EP_PREFIX = load_secret("FR24_API_KEY"), ""
        keyname = "FR24_API_KEY"
    if not API_KEY:
        sys.exit(f"No {keyname} found. Put it in maps/secrets.env (gitignored) "
                 f"as:\n    {keyname}=your-key-here")

    os.makedirs(OUT_DIR, exist_ok=True)
    if sandbox:
        print("="*64)
        print("SANDBOX MODE — responses are STATIC sample data.")
        print("This only checks plumbing (auth, endpoints, parsing), NOT")
        print("real coverage. Counts below are meaningless. It is FREE.")
        print("="*64)
    print(f"FR24 probe — mode: {'SANDBOX' if sandbox else mode.upper()} "
          f"— probing UTC day {PROBE_DAY}")
    print(f"raw responses -> {OUT_DIR}\n")
    findings = {}

    # --- TEST 0: auth ------------------------------------------------------
    print("[0] AUTH — fetching static airport info for ZBAA ...")
    st, hd, pa = api("/static/airports/ZBAA/light", None, "00_auth")
    if st != 200:
        print(f"    FAILED (HTTP {st}): {err_text(pa, st)}")
        print("    Stopping — fix the key/plan before spending more credits.")
        return
    print(f"    OK — {pa}")
    findings["auth"] = "OK"

    # --- TEST 1: discover the airports filter + datetime format ------------
    # Cheap: uses /count over a 2h window, trying candidate param formats.
    print("\n[1] PARAMS — discovering the airports-filter format (2h window) ...")
    candidates = ["outbound:ZBAA", "ZBAA", "departure:ZBAA", "departures:ZBAA"]
    apt_fmt = None
    for cand in candidates:
        st, hd, pa = api("/flight-summary/count",
                         {"flight_datetime_from": DAY_FROM, "flight_datetime_to": DISC_TO,
                          "airports": cand},
                         f"01_param_{cand.replace(':','_')}")
        rc = pa.get("record_count") if isinstance(pa, dict) else None
        print(f"    airports='{cand}' -> HTTP {st}, record_count={rc}")
        if st == 200 and rc is not None:
            apt_fmt = cand.split("ZBAA")[0]   # e.g. 'outbound:' or ''
            break
    if apt_fmt is None:
        print("    FAILED — no airports-filter format worked. Last error above.")
        print("    Check probe_out/01_param_*.json for the API's error message.")
        return
    print(f"    WORKS — airports filter format: '{apt_fmt}<ICAO>'")
    findings["airports_param"] = f"{apt_fmt}<ICAO>"
    findings["datetime_format"] = f"accepted: {DAY_FROM}"
    aptval = lambda icao: f"{apt_fmt}{icao}"

    # credit-reporting headers?
    cred_hdr = [k for k in hd if any(s in k.lower() for s in
                ("credit","quota","ratelimit","usage","limit-"))]
    if cred_hdr:
        print(f"    credit/quota headers: {[f'{k}={hd[k]}' for k in cred_hdr]}")
        findings["credit_header"] = ", ".join(cred_hdr)
    else:
        print(f"    no obvious credit header. all headers: {sorted(hd.keys())}")
        findings["credit_header"] = "none found (see probe_out/ headers)"

    # ====================== SANDBOX / SMOKE: stop here ================
    if sandbox or mode == "smoke":
        tag = "SANDBOX" if sandbox else "SMOKE"
        print(f"\n[{tag}] one small flight-summary/light call (2h window, ZBAA) ...")
        st_c, _, pc = api("/flight-summary/count",
                          {"flight_datetime_from": DAY_FROM, "flight_datetime_to": DISC_TO,
                           "airports": aptval("ZBAA")}, "smoke_count")
        st_l, _, pl = api("/flight-summary/light",
                          {"flight_datetime_from": DAY_FROM, "flight_datetime_to": DISC_TO,
                           "airports": aptval("ZBAA"), "sort": "asc"}, "smoke_light")
        count = pc.get("record_count") if isinstance(pc, dict) else None
        recs = rows(pl)
        print(f"    /count -> {count}   /light -> {len(recs)} records  (HTTP {st_c}/{st_l})")
        if recs:
            print("    sample flight-summary/light record:")
            show_record(recs[0])
        ok = (st_l == 200 and recs and count is not None)
        print("\n" + "="*60)
        print(f"{tag} " + ("PASSED — plumbing works." if ok else "— CHECK ABOVE"))
        print("="*60)
        for k in ("auth","airports_param","datetime_format","credit_header"):
            print(f"  {k:16}: {findings.get(k)}")
        print(f"\n  API calls: {CALLS}")
        if sandbox:
            print("  (sandbox = free; counts above are static sample data)")
            if ok:
                print("  Plumbing OK. Next, on the REAL key: python fr24_probe.py")
        elif ok:
            print("  Next: python fr24_probe.py full")
        return

    # ====================== FULL MODE =================================
    # --- TEST 2: per-airport count vs light (truncation + China) ----------
    print(f"\n[2] TRUNCATION + CHINA — full day {PROBE_DAY}, count vs light ...")
    print(f"    {'airport':<22}{'FR24 count':>11}{'light rows':>11}{'~reality':>12}  verdict")
    pulled = {}
    for icao, (name, lo, hi) in TEST_AIRPORTS.items():
        st1, _, pc = api("/flight-summary/count",
                         {"flight_datetime_from": DAY_FROM, "flight_datetime_to": DAY_TO,
                          "airports": aptval(icao)}, f"02_count_{icao}")
        count = pc.get("record_count") if isinstance(pc, dict) else None
        st2, _, pl = api("/flight-summary/light",
                         {"flight_datetime_from": DAY_FROM, "flight_datetime_to": DAY_TO,
                          "airports": aptval(icao), "sort": "asc"}, f"02_light_{icao}")
        recs = rows(pl)
        pulled[icao] = recs
        nlight = len(recs)
        if count is None:
            verdict = f"count failed (HTTP {st1})"
        elif nlight < count:
            verdict = f"!! TRUNCATED — light {nlight} < count {count}"
        elif count < lo:
            verdict = "!! LOW vs reality — coverage concern"
        else:
            verdict = "looks complete"
        print(f"    {icao} {name:<17}{str(count):>11}{nlight:>11}{f'{lo}-{hi}':>12}  {verdict}")
    findings["truncation"] = "see table above (light should equal count)"
    findings["china"] = "see table above (count should be >= ~reality)"

    # --- TEST 3: field formats + cancelled flights ------------------------
    print("\n[3] FIELDS + CANCELLED — inspecting a sample record ...")
    sample = next((r for v in pulled.values() for r in v), None)
    if sample:
        print("    sample flight-summary/light record:")
        show_record(sample)
        findings["fields"] = "printed above"
    allrecs = [r for v in pulled.values() for r in v]
    n_total = len(allrecs)
    n_notakeoff = sum(1 for r in allrecs if not r.get("datetime_takeoff"))
    if n_total:
        pct = 100 * n_notakeoff / n_total
        print(f"    records with NULL datetime_takeoff (cancelled/never flew): "
              f"{n_notakeoff}/{n_total} ({pct:.1f}%)")
        findings["cancelled"] = (f"{pct:.1f}% null takeoff — "
                                 f"{'filter these out' if n_notakeoff else 'none present'}")

    # --- TEST 4: GA fraction + aircraft filter ----------------------------
    print("\n[4] GA NOISE — aircraft-type mix at the pulled major airports ...")
    if allrecs:
        from collections import Counter
        types = [(r.get("type") or "?") for r in allrecs]
        n_air = sum(1 for t in types if t in AIRLINER_TYPES)
        n_non = n_total - n_air
        print(f"    airliner-typed: {n_air}/{n_total} ({100*n_air/n_total:.1f}%)  "
              f"| non-airliner/unknown: {n_non} ({100*n_non/n_total:.1f}%)")
        non = Counter(t for t in types if t not in AIRLINER_TYPES)
        if non:
            print(f"    top non-airliner types: {dict(non.most_common(8))}")
        findings["ga_fraction"] = f"{100*n_non/n_total:.1f}% non-airliner at major airports"

    print("    testing whether the 'aircraft' query param filters server-side ...")
    st, _, pf = api("/flight-summary/count",
                    {"flight_datetime_from": DAY_FROM, "flight_datetime_to": DAY_TO,
                     "airports": aptval("ZBAA"), "aircraft": "B738,A320,A321"},
                    "04_aircraft_filter")
    rc = pf.get("record_count") if isinstance(pf, dict) else None
    if st == 200 and rc is not None:
        print(f"    aircraft='B738,A320,A321' at ZBAA -> record_count={rc} "
              f"(HTTP 200 — param accepted, can filter to commercial server-side)")
        findings["aircraft_filter"] = "works — can exclude GA in-query"
    else:
        print(f"    aircraft filter -> HTTP {st}: {err_text(pf, st)}")
        findings["aircraft_filter"] = f"unclear (HTTP {st}) — may need to post-filter"

    # --- SUMMARY ----------------------------------------------------------
    print("\n" + "="*64)
    print("FULL PROBE SUMMARY")
    print("="*64)
    for k in ("auth","airports_param","datetime_format","credit_header",
              "truncation","china","cancelled","ga_fraction","aircraft_filter"):
        if k in findings:
            print(f"  {k:16}: {findings[k]}")
    print(f"\n  API calls made: {CALLS}")
    print("  Raw JSON saved in probe_out/ — inspect 02_light_*.json for full records.")
    print("\nGO / NO-GO checklist:")
    print("  GO if   : light rows == count (no truncation), China counts look")
    print("            plausible vs the website, GA fraction is small.")
    print("  RETHINK : light < count with no pagination param -> truncation risk;")
    print("            or China counts come back far below reality.")


if __name__ == "__main__":
    main()
