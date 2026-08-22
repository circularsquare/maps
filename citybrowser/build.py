"""
Merge base.json + overrides.json -> cities.json.

The only writer of cities.json. See SCHEMA.md for the contract.

The interesting part is staleness. Overrides are field-level patches that record
the base value they replaced (`was`). On every build we compare that against the
CURRENT base value:

  - unchanged  -> the override still applies, use it silently
  - changed    -> the source has moved since the correction was made, so the
                  correction may no longer be right. Keep the override (never
                  silently discard curation) but mark the field stale so the UI
                  can flag it for review.

That is the entire reason patches beat whole-record copies: a copy cannot tell
"I corrected this" from "this happened to be the value at the time".

Usage:
    python build.py            # data/base.json + overrides -> data/cities.json
    python build.py --stats    # also print a curation-progress summary
"""

import argparse
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data"
BASE = DATA / "base.json"
OVERRIDES = DATA / "overrides.json"
OUT = DATA / "cities.json"

# Fields the UI needs but which no source provides — they only ever come from
# curation, so their absence is expected rather than a gap to report.
CURATED_ONLY = {"altNames", "facts", "languages", "religions", "photo"}

# Accepting or rejecting a GHS match is REVIEW, not curation, and must not set
# `_touched`. Working the match queue would otherwise light up the map with
# "curated" rings for thousands of cities nobody has written a single fact
# about — and "faded = not yet curated" is the entire progress signal.
# The edit is still recorded in `_edited`, so provenance is not lost.
# Keep in step with REVIEW_FIELDS in js/data.js — see SCHEMA.md.
REVIEW_FIELDS = {"ghs", "ghsConf", "ghsRole"}


def load(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def merge(base, overrides):
    cities = {}
    stale_total = 0
    created = 0
    deleted = 0

    for key, rec in base.items():
        ov = overrides.get(key, {})
        if ov.get("_deleted"):
            deleted += 1
            continue
        out = dict(rec)
        edited, stale = [], []
        for field, patch in ov.items():
            if field.startswith("_"):
                continue
            out[field] = patch["value"]
            edited.append(field)
            # `was` is what base said when the edit was made. If base has moved,
            # the correction is suspect — surface it rather than trusting it.
            if "was" in patch and patch["was"] != rec.get(field):
                stale.append(field)
        if edited:
            out["_edited"] = sorted(edited)
            if any(f not in REVIEW_FIELDS for f in edited):
                out["_touched"] = True
        if stale:
            out["_stale"] = sorted(stale)
            stale_total += len(stale)
        cities[key] = out

    # Cities that exist in neither source, created by hand in edit mode.
    for key, ov in overrides.items():
        if key in base or "_created" not in ov:
            continue
        if ov.get("_deleted"):
            continue
        out = dict(ov["_created"])
        fields = [f for f in ov if not f.startswith("_")]
        for field in fields:
            out[field] = ov[field]["value"]
        out["_edited"] = sorted(fields) if fields else []
        out["_touched"] = True
        out["_created"] = True
        cities[key] = out
        created += 1

    return cities, {"stale": stale_total, "created": created, "deleted": deleted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true",
                    help="print a curation-progress summary")
    args = ap.parse_args()

    base = load(BASE, {})
    overrides = load(OVERRIDES, {})
    if not base:
        sys.exit(f"no {BASE} — run the fetch stages first")

    cities, counts = merge(base, overrides)
    DATA.mkdir(exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(cities, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    tmp.replace(OUT)

    touched = sum(1 for c in cities.values() if c.get("_touched"))
    print(f"base       {len(base):,}")
    print(f"overrides  {len(overrides):,} cities patched")
    print(f"cities     {len(cities):,} written -> {OUT.name}")
    print(f"touched    {touched:,} ({100*touched/max(len(cities),1):.2f}%)")
    if counts["created"]:
        print(f"created    {counts['created']:,} hand-added")
    if counts["deleted"]:
        print(f"deleted    {counts['deleted']:,} tombstoned")
    if counts["stale"]:
        print(f"STALE      {counts['stale']:,} overridden fields whose source "
              f"value has since changed — review these")

    if args.stats:
        print("\nfield coverage:")
        n = len(cities)
        keys = set()
        for c in cities.values():
            keys.update(k for k in c if not k.startswith("_"))
        for k in sorted(keys):
            have = sum(1 for c in cities.values() if c.get(k) not in (None, "", []))
            tag = "  (curated only)" if k in CURATED_ONLY else ""
            print(f"  {k:12s} {have:7,}/{n:,}  {100*have/max(n,1):5.1f}%{tag}")


if __name__ == "__main__":
    main()
