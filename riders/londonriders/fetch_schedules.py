"""
Fetch real schedules from the TfL Unified API and cache as JSON.

TfL doesn't publish GTFS, so we hit /Line/{id}/Timetable/{stopId} instead.
Each call returns one direction's schedule from a given origin stop, including
real per-trip departure times and inter-station offsets via `stationIntervals`.

Cached files:
  data/api_cache/lines.json                       — list of all rail lines
  data/api_cache/seq_<line>.json                  — Route/Sequence/all (geometry + branches)
  data/api_cache/stops_<line>.json                — per-line stop points
  data/api_cache/tt_<line>__<stopId>.json         — timetable from one origin

Rate limit: anonymous is 50 req/min. We sleep 1.3s between calls so even with
network jitter we stay under the cap.

Roughly 475 timetable calls total (19 lines * ~25 stops). Allow ~10 minutes.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://api.tfl.gov.uk"
RAIL_MODES = "tube,dlr,elizabeth-line,overground"
CACHE_DIR = "data/api_cache"
DELAY_S = 1.3   # ~46 req/min, safely under the 50/min anonymous cap

# Optional: paste your TfL Product subscription key here (or set TFL_APP_KEY env)
# to lift the rate limit to 500/min.
APP_KEY = os.environ.get("TFL_APP_KEY", "")


def get(url, dest):
    """GET url, save JSON to dest. Skip if dest exists. Returns (wrote, err).
    `wrote` is True if a new file was saved, False otherwise.
    `err` is None on success or if the file already existed; otherwise a string.
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return False, None
    if APP_KEY:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}app_key={APP_KEY}"
    tmp = dest + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "londonriders/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code} {url}"
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return False, f"net error {url}: {e}"
    try:
        json.loads(data)
    except json.JSONDecodeError as e:
        return False, f"bad JSON from {url}: {e}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.rename(tmp, dest)
    time.sleep(DELAY_S)
    return True, None


def fetch_lines():
    print("Fetching line list...")
    dest = f"{CACHE_DIR}/lines.json"
    wrote, err = get(f"{API}/Line/Mode/{RAIL_MODES}", dest)
    if err is not None:
        raise SystemExit(f"FATAL: could not fetch line list: {err}")
    if not os.path.exists(dest):
        raise SystemExit(f"FATAL: {dest} missing after fetch")
    with open(dest, encoding="utf-8") as f:
        lines = json.load(f)
    out = [{"id": ln["id"], "name": ln["name"], "mode": ln["modeName"]} for ln in lines]
    print(f"  {len(out)} lines ({'fetched' if wrote else 'cached'})")
    return out


def fetch_per_line_metadata(lines):
    """Pull route sequences and stop points per line (one call each)."""
    for ln in lines:
        lid = ln["id"]
        seq_dest = f"{CACHE_DIR}/seq_{lid}.json"
        sp_dest = f"{CACHE_DIR}/stops_{lid}.json"
        wrote_seq, err_seq = get(f"{API}/Line/{lid}/Route/Sequence/all?serviceTypes=Regular", seq_dest)
        wrote_sp, err_sp = get(f"{API}/Line/{lid}/StopPoints", sp_dest)
        if err_seq:
            print(f"  {lid:20s} seq FAILED: {err_seq}")
        if err_sp:
            print(f"  {lid:20s} stops FAILED: {err_sp}")
        tag = []
        if wrote_seq: tag.append("seq")
        if wrote_sp: tag.append("stops")
        if not err_seq and not err_sp:
            label = ' + '.join(tag) if tag else 'cached'
            print(f"  {lid:20s} {label}")


def line_stops(line_id):
    """Return list of naptan stop IDs for a line, from cached stops_{line}.json."""
    with open(f"{CACHE_DIR}/stops_{line_id}.json", encoding="utf-8") as f:
        sps = json.load(f)
    return [s["naptanId"] for s in sps if s.get("naptanId")]


def fetch_timetables(lines):
    """For each (line, stop) pair, fetch timetable. Skip on cache hit."""
    total = 0
    fetched = 0
    failed = 0
    skipped = 0
    plan = []
    for ln in lines:
        for sid in line_stops(ln["id"]):
            plan.append((ln["id"], sid))
    print(f"Fetching {len(plan)} timetables...")
    t0 = time.time()
    for i, (lid, sid) in enumerate(plan):
        dest = f"{CACHE_DIR}/tt_{lid}__{sid}.json"
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            skipped += 1
            continue
        url = f"{API}/Line/{lid}/Timetable/{sid}"
        wrote, err = get(url, dest)
        total += 1
        if wrote:
            fetched += 1
        else:
            # write a stub so we don't retry forever on a known-bad pair
            with open(dest, "w", encoding="utf-8") as f:
                json.dump({"error": err or "no_data"}, f)
            failed += 1
            if err and "HTTP 404" not in err:
                # 404s are common (terminal-only stops with no outbound trips);
                # surface anything weirder
                print(f"  {lid}/{sid}: {err}")
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1 - skipped) * (len(plan) - i - 1) if (i + 1 - skipped) else 0
            print(f"  [{i+1}/{len(plan)}] fetched={fetched} failed={failed} skipped={skipped} "
                  f"({elapsed:.0f}s, ~{eta:.0f}s remaining)")
    print(f"Done. fetched={fetched} failed={failed} skipped={skipped}")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    lines = fetch_lines()
    print("Per-line metadata...")
    fetch_per_line_metadata(lines)
    fetch_timetables(lines)


if __name__ == "__main__":
    main()
