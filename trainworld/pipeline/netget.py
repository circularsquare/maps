"""A build's network requests, retried in place (T-1979): a reset connection, a timeout, a 429 or a
5xx waits and asks again, so a server's bad minute costs a build seconds rather than the whole row
(Australia's run rebuilt 8 rows when the World Bank reset 9 builds asking at once). Also the
country inputs a run fetches once before its builds (`country_inputs`), so its builds find them
cached.

    python pipeline/netget.py <ISO2>...      fill the country inputs' caches
"""
import json
import os
import sys
import time

import requests

UA = {"User-Agent": "trainworld-pipeline/0.1 (rail game data build; python-requests)"}
WAITS = (5, 20, 60, 120, 240)  # seconds before each further try: about 7.5 minutes in all
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


def request(url, method="get", waits=None, ok=(), **kw):
    """requests.<method>(url, **kw), retried after each wait in `waits` (WAITS) on a connection error, a
    timeout or a status in RETRY_STATUS; statuses in `ok` (a 404 that means "none") come back as
    they are. Any other error status raises at once, the last failure after the last wait."""
    kw.setdefault("headers", UA)
    kw.setdefault("timeout", 120)
    for n, wait in enumerate(tuple(WAITS if waits is None else waits) + (None,)):
        try:
            r = requests.request(method, url, **kw)
            if r.status_code in ok:
                return r
            if r.status_code not in RETRY_STATUS or wait is None:
                r.raise_for_status()
                return r
            why = f"status {r.status_code}"
        except requests.RequestException as x:
            # an error status raised above is final unless it is one to retry
            resp = getattr(x, "response", None)
            if wait is None or (resp is not None and resp.status_code not in RETRY_STATUS):
                raise
            why = f"{type(x).__name__}"
        print(f"  {url[:120]}: {why}; try {n + 2} in {wait} s", flush=True)
        time.sleep(wait)


def settle(tmp, path):
    """tmp in place as path; if another process got there first and Windows will not replace a file
    it has open, theirs stays."""
    try:
        os.replace(tmp, path)
    except OSError:
        if not os.path.exists(path):
            raise
        os.remove(tmp)
    return path


def cached_json(url, path, **kw):
    """`path`, fetched from `url` (a JSON answer) first if it is not on disk; written whole or not
    at all, since builds at once read and fetch the same file."""
    if not os.path.exists(path):
        data = request(url, **kw).json()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        settle(tmp, path)
    return path


def country_inputs(iso2s):
    """Every national figure a planned row's build fetches, cached once for each country: the World
    Bank's employment and age shares, tertiary enrolment and vulnerable employment, and nocar's GDP
    and income groups. Returns {iso2: error} for those that failed after their retries; a build then
    asks again itself."""
    import nocar
    import plan_config
    import tier0
    failed = {}
    try:
        nocar.worldbank()
    except Exception as x:  # noqa: BLE001 (any failure is reported and left to the builds)
        failed["*"] = f"nocar.worldbank: {type(x).__name__} {x}"
    for iso2 in iso2s:
        for name, fn in (("worldbank", tier0.worldbank), ("tertiary_share", tier0.tertiary_share),
                         ("vulnerable_share", plan_config.vulnerable_share)):
            try:
                fn(iso2)
            except Exception as x:  # noqa: BLE001
                failed[iso2] = f"{name}: {type(x).__name__} {x}"[:300]
    return failed


if __name__ == "__main__":
    bad = country_inputs(sys.argv[1:])
    for k, v in bad.items():
        print(f"{k}: {v}")
    sys.exit(1 if bad else 0)
