"""Lithuania — Statistics Lithuania, Surašymas 2021, religion by municipality.

Reads (or fetches) data/raw/lt/ and writes data/normalized/lt.csv.

One SDMX dataflow, **`S3R778_GBS010306_1`** — *Population | Administrative territory | Religion*
— 16 religions plus a total, over 73 nested territories, for 2001, 2011 **and 2021**.
113 KB, no key, no login. 60 municipalities, 2,810,761 people.

THE HOST THAT WAS "403 TO SCRIPTED CLIENTS" IS NOT THE HOST WITH THE DATA. `sources.md` §11
recorded Lithuania as blocked because `osp.stat.gov.lt` returns Cloudflare's challenge page.
It does — that is the human web UI. The statistics live on **`osp-rs.stat.gov.lt`**, a plain
SDMX REST endpoint with no protection at all, and the earlier probe missed it by asking for
`/rest_json/` and `/api/v1/lt/` (both 404) rather than `/rest_xml/dataflow/`. **A statistical
office's UI host and its API host are two machines and only one of them is usually walled.**

`/rest_xml/dataflow/` is 7.4 MB listing **9,521 dataflows** with bilingual names, which is
the searchable index; `/rest_json/dataflow/` 404s, so the catalogue is XML-only while the
data is available as either.

**THE CUBE IS FOUR NESTED LEVELS**, as ever: country, two NUTS2 regions, ten counties and
sixty municipalities, all in one dimension. Summing it whole gives 4x the country exactly.

**AND THE NULLS MEAN TWO OPPOSITE THINGS.** 414 of the 1,020 municipality cells for 2021 are
null, and `OBS_STATUS` separates *konfidencialūs duomenys* (withheld) from *tokio reiškinio
… nebuvo* (there was no such thing — a true zero). Reading either as the other is wrong in a
different direction, and reading both as zero silently deletes people: the withheld cells are
disclosure control on small religions, so **Karaims are suppressed in 59 of the 60
municipalities and Jews in 55**. spec §3.8 exactly, and the most extreme case of it on the
map. What is withheld is reported per category and not filled (§3.5).

Usage:
    python sources/lt.py --fetch    one GET, 113 KB
    python sources/lt.py            normalise from data/raw/lt/
"""

import csv
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "lt")
OUT = os.path.join(ROOT, "data", "normalized", "lt.csv")

SOURCE_ID = "lt_surasymas_2021"
YEAR = 2021
BASIS = "self_id"

COLUMNS = ["geo_id", "geo_level", "geo_name", "source_category", "count",
           "basis", "year", "source_id", "note"]

FLOW = "S3R778_GBS010306_1"
API = f"https://osp-rs.stat.gov.lt/rest_json/data/{FLOW}/"
CUBE = os.path.join(RAW, f"{FLOW}.json")

NATIONAL = 2_810_761          # resident population enumerated, 2021
COUNTRY_CODE = "00"
TOTAL_CAT = "TOT"
DRAWN_YEAR = "2021"

EXPECTED = {"country": 1, "region": 2, "county": 10, "municipality": 60}

# OBS_STATUS values, matched on their Lithuanian text rather than on their index, because
# an index is a fact about this download and the text is a fact about the source.
STATUS_CONFIDENTIAL = "konfidencial"
STATUS_NO_PHENOMENON = "nebuvo"


def fetch():
    import requests

    os.makedirs(RAW, exist_ok=True)
    if os.path.exists(CUBE) and os.path.getsize(CUBE) > 50_000:
        print("already have", CUBE)
        return
    # TLS verifies normally here. The scouting pass reached this host with verification
    # off and nearly left it that way, which is §9h's reflex: test a second client before
    # disabling anything. `requests` and `curl` both verify `osp-rs.stat.gov.lt` fine.
    print("GET", API)
    r = requests.get(API, timeout=300,
                     headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    r.raise_for_status()
    doc = r.json()
    # §5a: HTTP 200 is not a download. Assert it is the cube and not an error envelope.
    for key in ("structure", "dataSets"):
        if key not in doc:
            raise SystemExit(f"not an SDMX-JSON cube, keys {list(doc)}")
    with open(CUBE, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False)
    print(f"  {os.path.getsize(CUBE):,} bytes")


def _load():
    if not os.path.exists(CUBE):
        raise SystemExit(f"missing {CUBE} -- run with --fetch first")
    with open(CUBE, encoding="utf-8") as fh:
        return json.load(fh)


def _level(code):
    """One dimension, four levels, told apart by the SHAPE of the code (Hungary's rule)."""
    if code == COUNTRY_CODE:
        return "country"
    if not code.isdigit():
        return "region"           # LT01, LT02 — the two NUTS2 regions
    return "county" if int(code) <= 10 else "municipality"


def read():
    doc = _load()
    dims = doc["structure"]["dimensions"]["observation"]
    ids = [d["id"] for d in dims]
    axes = [d["values"] for d in dims]
    if len(dims) != 4:
        raise SystemExit(f"expected 4 observation dimensions, got {ids}")

    def axis(want):
        for n, d in enumerate(dims):
            if want in d["id"] or want in (d.get("name") or ""):
                return n
        raise SystemExit(f"no dimension matching {want!r} among {ids}")

    i_geo, i_cat = axis("Savivaldybes"), axis("religija")
    i_time = axis("LAIKOTARPIS")

    # OBS_STATUS lives in the observation value array, at the position its attribute has in
    # structure.attributes.observation. Never assume it is last.
    obs_attrs = doc["structure"]["attributes"]["observation"]
    pos = next((n for n, a in enumerate(obs_attrs) if a["id"] == "OBS_STATUS"), None)
    if pos is None:
        raise SystemExit("no OBS_STATUS attribute -- the null cells cannot be told apart")
    status_vals = [(v.get("name") or "") for v in obs_attrs[pos]["values"]]
    conf = [n for n, s in enumerate(status_vals) if STATUS_CONFIDENTIAL in s.lower()]
    zero = [n for n, s in enumerate(status_vals) if STATUS_NO_PHENOMENON in s.lower()]
    if len(conf) != 1 or len(zero) != 1:
        raise SystemExit(f"cannot identify the two OBS_STATUS meanings in {status_vals} -- "
                         "reading a withheld cell as a zero deletes people silently")
    conf, zero = conf[0], zero[0]

    rows, stats = [], {"n": 0, "confidential": 0, "zero": 0, "other_year": 0}
    for key, val in doc["dataSets"][0]["observations"].items():
        idx = [int(x) for x in key.split(":")]
        if axes[i_time][idx[i_time]]["id"] != DRAWN_YEAR:
            stats["other_year"] += 1
            continue
        g = axes[i_geo][idx[i_geo]]
        c = axes[i_cat][idx[i_cat]]
        n = val[0]
        status = val[pos + 1] if len(val) > pos + 1 else None
        if n is None:
            if status == conf:
                stats["confidential"] += 1
                continue                     # withheld: not zero, and not drawn (§3.5)
            if status == zero:
                stats["zero"] += 1
                n = 0
            else:
                raise SystemExit(f"{g['id']}/{c['id']}: null with OBS_STATUS {status!r}, "
                                 "which is neither of the two known meanings")
        stats["n"] += 1
        level = _level(str(g["id"]))
        note = f"level={level}; code={c['id']}"
        if c["id"] == TOTAL_CAT:
            note += "; universe total, not a religion category"
        if n == 0 and status == zero:
            note += "; published as 'no such phenomenon', i.e. a true zero"
        rows.append({"geo_id": str(g["id"]), "geo_level": level,
                     "geo_name": g["name"], "source_category": c["name"].strip(),
                     "count": int(n), "basis": BASIS, "year": YEAR,
                     "source_id": SOURCE_ID, "note": note})
    return rows, stats, {v["id"]: v["name"].strip() for v in axes[i_cat]}


def check(rows, stats, cats):
    ok = True
    total_label = cats[TOTAL_CAT]

    print(f"  {stats['n']:,} cells kept for {DRAWN_YEAR}; "
          f"{stats['other_year']:,} dropped as 2001/2011; "
          f"{stats['zero']:,} published as a true zero; "
          f"**{stats['confidential']:,} withheld as confidential**")

    levels = {}
    for r in rows:
        levels.setdefault(r["geo_level"], set()).add(r["geo_id"])
    for lv, want in EXPECTED.items():
        got = len(levels.get(lv, ()))
        good = got == want
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {lv:<13} {got:>3} units (expected {want})")

    nat = {r["source_category"]: r["count"] for r in rows if r["geo_level"] == "country"}
    good = nat.get(total_label) == NATIONAL
    ok &= good
    print(f"\n  {'OK ' if good else 'BAD'} national total {nat.get(total_label):,} "
          f"(published {NATIONAL:,})")

    parts = sum(v for k, v in nat.items() if k != total_label)
    good = parts == NATIONAL
    ok &= good
    print(f"  {'OK ' if good else 'BAD'} the 16 categories partition the country exactly "
          f"({parts:,}) — nothing is withheld at national level")

    # Every level's TOTAL must reproduce the country exactly: the total is never suppressed.
    for lv in ("region", "county", "municipality"):
        s = sum(r["count"] for r in rows
                if r["geo_level"] == lv and r["source_category"] == total_label)
        good = s == NATIONAL
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {lv:<13} totals sum to the country ({s:,})")

    # ---- and the gap that is suppression, per category (§3.5, §3.8) ----
    print("\n  municipality sums vs national — the gap is disclosure control, not loss:")
    n_sup = {}
    for r in rows:
        if r["geo_level"] == "municipality":
            n_sup[r["source_category"]] = n_sup.get(r["source_category"], 0) + 1
    tot_gap = 0
    for label in sorted(nat, key=lambda k: -nat[k]):
        if label == total_label:
            continue
        s = sum(r["count"] for r in rows
                if r["geo_level"] == "municipality" and r["source_category"] == label)
        gap = nat[label] - s
        tot_gap += gap
        held = EXPECTED["municipality"] - n_sup.get(label, 0)
        bad = gap < 0
        ok &= not bad
        pct = 100.0 * gap / nat[label] if nat[label] else 0.0
        print(f"    {'BAD' if bad else 'OK '} {label[:42]:<44} {s:>9,}  "
              f"withheld {gap:>6,} ({pct:5.1f}%) in {held:>2} of 60")
    frac = tot_gap / NATIONAL
    good = frac < 0.001
    ok &= good
    print(f"    {'OK ' if good else 'BAD'} total withheld {tot_gap:,} = {frac:.4%} of the "
          "population")

    print(f"\n  {len(rows):,} rows. Categories, national:")
    for label, n in sorted(nat.items(), key=lambda kv: -kv[1]):
        mark = "  <- universe" if label == total_label else ""
        print(f"    {n:>10,}  {100.0 * n / NATIONAL:6.2f}%  {label}{mark}")

    if not ok:
        raise SystemExit("reconciliation FAILED")


def main():
    if "--fetch" in sys.argv:
        fetch()
    rows, stats, cats = read()
    check(rows, stats, cats)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote", OUT, f"({len(rows):,} rows)")


if __name__ == "__main__":
    main()
