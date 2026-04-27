"""Fetch Indonesian population data via the BPS WebAPI (stadata).

Writes: helper1m/data/indonesia/population.csv  (columns: code, level, year, pop)

Pipeline stages:
  probe          — list candidate population tables at a given domain so you can
                   verify table IDs and response shapes before a real run.
  --levels 1     — adm1: province populations from central domain.
  --levels 2     — adm2: regency populations from central domain.
  --levels 3     — adm3: kecamatan populations from each regency domain (slow).

Auth: set BPS_API_KEY in environment or in helper1m/.env (gitignored).
Get a key at https://webapi.bps.go.id/developer/.

Design:
  Each BPS table has a "var" ID within a "domain". The same logical table
  (e.g. "Jumlah Penduduk menurut Kecamatan") appears per regency domain with
  a different var ID. We discover var IDs by title match, then cache them in
  data/indonesia/bps_tables.json so reruns skip discovery.

The specific var IDs and response fields are confirmed on first run against
the live API (BPS has rate limits, no unauthenticated access). See probe mode.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

HELPER = Path(__file__).resolve().parents[2]
DATA = HELPER / "data" / "indonesia"
CACHE_TABLES = DATA / "bps_tables.json"
OUT_CSV = DATA / "population.csv"

CENTRAL_DOMAIN = "0000"
# Title substrings used to identify population tables at each admin level.
TITLE_PATTERNS = {
    1: re.compile(r"jumlah\s+penduduk.*menurut\s+provinsi", re.I),
    2: re.compile(r"jumlah\s+penduduk.*menurut\s+kabupaten", re.I),
    3: re.compile(r"jumlah\s+penduduk.*menurut\s+kecamatan", re.I),
}


def get_client():
    load_dotenv(HELPER / ".env")
    key = os.environ.get("BPS_API_KEY")
    if not key:
        sys.exit(
            "Missing BPS_API_KEY.\n"
            "  1. Register at https://webapi.bps.go.id/developer/\n"
            "  2. Create an application, copy the token.\n"
            "  3. Put BPS_API_KEY=... in helper1m/.env  (gitignored)"
        )
    import stadata
    return stadata.Client(key)


def list_tables(client, domain: str) -> list[dict]:
    """List dynamic tables for a domain. Returns list of {var_id, title, ...}."""
    # stadata's list_dynamictable returns a pandas DataFrame per its README;
    # we normalize to list of dicts.
    df = client.list_dynamictable(domain=domain)
    return df.to_dict("records") if hasattr(df, "to_dict") else list(df)


def find_population_table(tables: list[dict], level: int) -> dict | None:
    pat = TITLE_PATTERNS[level]
    for t in tables:
        title = str(t.get("title") or t.get("judul") or "")
        if pat.search(title):
            return t
    return None


def cmd_probe(args):
    """Print candidate population tables for inspection."""
    client = get_client()
    domain = args.domain or CENTRAL_DOMAIN
    print(f"listing dynamic tables at domain={domain} ...")
    tables = list_tables(client, domain)
    print(f"  found {len(tables)} tables")
    if tables:
        print("  fields per entry:", list(tables[0].keys()))
    print()
    for level in (1, 2, 3):
        hit = find_population_table(tables, level)
        label = {1: "province", 2: "regency", 3: "kecamatan"}[level]
        if hit:
            print(f"adm{level} ({label}): {hit}")
        else:
            print(f"adm{level} ({label}): no match")

    # Dump a sample view_dynamictable response so we can pin down the field names.
    hit_any = next((find_population_table(tables, l) for l in (1, 2, 3) if find_population_table(tables, l)), None)
    if hit_any:
        var_id = hit_any.get("var_id") or hit_any.get("id") or hit_any.get("variable_id")
        if var_id is not None:
            print(f"\nsample view_dynamictable(domain={domain}, var={var_id}):")
            try:
                sample = client.view_dynamictable(domain=domain, var=str(var_id))
                print(json.dumps(sample if isinstance(sample, (dict, list)) else list(sample)[:3], default=str, indent=2)[:4000])
            except Exception as e:
                print(f"  error: {type(e).__name__}: {e}")


# ---- real fetchers (filled in after probe reveals response shape) ----


def fetch_central(client, level: int, rows: list[dict]):
    """TODO: fill in after probe confirms the response shape.
    Expected to append rows like {'code': 'ID11', 'level': 1, 'year': 2020, 'pop': ...}.
    """
    raise NotImplementedError("fetch_central — run probe first")


def fetch_kecamatan_all(client, rows: list[dict]):
    """Crawl every regency domain and pull its kecamatan population table."""
    raise NotImplementedError("fetch_kecamatan_all — phase 4")


def cmd_fetch(args):
    client = get_client()
    rows: list[dict] = []
    if "1" in args.levels:
        fetch_central(client, 1, rows)
    if "2" in args.levels:
        fetch_central(client, 2, rows)
    if "3" in args.levels:
        fetch_kecamatan_all(client, rows)

    rows.sort(key=lambda r: (r["level"], r["code"], r["year"]))
    DATA.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["code", "level", "year", "pop"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT_CSV}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_probe = sub.add_parser("probe", help="List candidate population tables")
    p_probe.add_argument("--domain", help=f"BPS domain code (default {CENTRAL_DOMAIN})")
    p_probe.set_defaults(func=cmd_probe)

    p_fetch = sub.add_parser("fetch", help="Fetch populations")
    p_fetch.add_argument("--levels", nargs="+", default=["1", "2"], choices=["1", "2", "3"])
    p_fetch.set_defaults(func=cmd_fetch)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
