"""Where a Geofabrik extract downloads from (T-4027).

Geofabrik moves extracts into per-region folders one region at a time (`asia/india/eastern-zone-latest.osm.pbf`
became `asia/india/eastern-zone/eastern-zone-latest.osm.pbf`), and the old URLs' redirects are broken for some.
An extract id stays the old path (`asia/india/eastern-zone`), as cache file names and plan rows use it;
only the URL is resolved: `url(id)` for the .osm.pbf, `url(id, ".poly")` for the boundary.
"""
import datetime
import json
import os
import re
import time

import requests

BASE = "https://download.geofabrik.de/"
INDEX_URL = BASE + "index-v1-nogeom.json"
UA = {"User-Agent": "python-requests (osm extract download)"}
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "work", "geofabrik-index-urls.json")
DATED_DAYS = 5

_memo = {}


def extract_id(url):
    """The extract id of an index URL or a path, the same from either layout:
    .../asia/india/eastern-zone-latest.osm.pbf and .../asia/india/eastern-zone/eastern-zone-latest.osm.pbf
    both give asia/india/eastern-zone."""
    p = url.split("download.geofabrik.de/")[-1]
    p = re.sub(r"(-latest)?\.osm\.pbf$", "", p)
    parts = p.split("/")
    if len(parts) > 1 and parts[-1] == parts[-2]:
        parts.pop()
    return "/".join(parts)


def _index():
    """{extract id: the .osm.pbf URL the index lists}, from a copy at most a day old (stale if the fetch fails)."""
    if "index" in _memo:
        return _memo["index"]
    if os.path.exists(CACHE) and time.time() - os.path.getmtime(CACHE) < 86400:
        out = json.load(open(CACHE))
    else:
        try:
            r = requests.get(INDEX_URL, headers=UA, timeout=120)
            r.raise_for_status()
            out = {extract_id(f["properties"]["urls"]["pbf"]): f["properties"]["urls"]["pbf"]
                   for f in r.json()["features"] if f["properties"].get("urls", {}).get("pbf")}
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            with open(CACHE + ".part", "w") as f:
                json.dump(out, f)
            os.replace(CACHE + ".part", CACHE)
        except (requests.RequestException, ValueError, KeyError):
            out = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    _memo["index"] = out
    return out


def _head(url):
    """(status, final url) of a HEAD request following redirects; (0, url) if the request fails."""
    try:
        r = requests.head(url, headers=UA, timeout=60, allow_redirects=True)
        return r.status_code, r.url
    except requests.RequestException:
        return 0, url


def _dated_fallback(final):
    """A -latest link redirects to today's dated file before it is published: the newest earlier day's file."""
    m = re.match(r"(.*-)(\d{6})(\.osm\.pbf)$", final)
    if not m:
        return None
    d = datetime.datetime.strptime(m.group(2), "%y%m%d").date()
    for back in range(1, DATED_DAYS + 1):
        u = f"{m.group(1)}{(d - datetime.timedelta(days=back)).strftime('%y%m%d')}{m.group(3)}"
        if _head(u)[0] == 200:
            return u
    return None


def url(extract, suffix=".osm.pbf"):
    """A working URL for the extract: the index's, then the old layout, then the folder layout.
    Raises if none answers 200."""
    key = (extract, suffix)
    if key in _memo:
        return _memo[key]
    name = extract.rsplit("/", 1)[-1]
    bases = []
    listed = _index().get(extract)
    if listed:
        bases.append(re.sub(r"-latest\.osm\.pbf$", "", listed))
    bases += [BASE + extract, f"{BASE}{extract}/{name}"]
    tried = []
    for base in dict.fromkeys(bases):
        u = base + ("-latest" if suffix == ".osm.pbf" else "") + suffix
        status, final = _head(u)
        tried.append(f"{u} -> {status}")
        if status == 200:
            _memo[key] = final
            return final
        if status == 404 and suffix == ".osm.pbf" and final != u:
            d = _dated_fallback(final)
            if d:
                _memo[key] = d
                return d
    raise RuntimeError(f"no working Geofabrik URL for {extract}{suffix}: " + "; ".join(tried))
