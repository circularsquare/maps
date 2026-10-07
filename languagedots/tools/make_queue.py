"""Write queue.csv from coverage/coverage.csv: one row per country, in the order agents take them.

    python tools/make_queue.py            # refuses to overwrite an existing queue.csv
    python tools/make_queue.py --force    # rebuild it (loses status edits; claim.py's are in claims.json)

Order: tier A by population, then tier B by population. Tiers C, D and E are `ruling`: an agent
does not build them until Anita has said how (AGENT_BRIEF.md §3). `rd_geo` says whether
religiondots has geography for the country, the biggest shortcut there is (§4).
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT.parent / "religiondots"
COV = ROOT / "coverage" / "coverage.csv"
OUT = ROOT / "queue.csv"
DRAWN = {"in", "np", "pk"}
TIER_STATUS = {"A": "free", "B": "free", "C": "ruling", "D": "ruling", "E": "ruling"}
FIELDS = ["cc", "iso3", "country", "tier", "pop_m", "status", "rd_geo", "question", "census_year",
          "finest_level", "n_units", "access", "source_url", "note"]


def iso2_of():
    ne = json.loads((RD / "data" / "geo" / "ne_10m_admin_0_countries.geojson").read_text(encoding="utf-8"))
    m = {}
    for f in ne["features"]:
        p = f["properties"]
        for a3 in (p.get("ISO_A3"), p.get("ADM0_A3"), p.get("ISO_A3_EH")):
            a2 = p.get("ISO_A2") if p.get("ISO_A2") not in (None, "-99") else p.get("ISO_A2_EH")
            if a3 and a3 != "-99" and a2 and a2 != "-99":
                m.setdefault(a3, a2.lower())
    # the UK is three censuses and one country, as in religiondots (`uk`)
    m.update({"XKX": "xk", "GBR_EW": "uk", "GBR_SC": "uk", "GBR_NI": "uk",
              "BES": "bq", "ESH": "eh"})
    return m


def main():
    if OUT.exists() and "--force" not in sys.argv:
        raise SystemExit(f"{OUT} exists; --force to rebuild it")
    a2 = iso2_of()
    rows = list(csv.DictReader(open(COV, encoding="utf-8")))
    out, seen = [], {}
    for r in rows:
        cc = a2.get(r["iso3"])
        if not cc:
            print(f"  !! no two-letter code for {r['iso3']} {r['country']}; skipped")
            continue
        if cc in seen:          # the UK's second and third census
            prev = seen[cc]
            prev["pop_m"] = f"{float(prev['pop_m']) + float(r['pop_m']):.2f}"
            prev["note"] = (prev["note"] + f"; also {r['country']}: {r['question']}, "
                            f"{r['census_year']}, {r['finest_level']}").lstrip("; ")
            continue
        status = "drawn" if cc in DRAWN else TIER_STATUS[r["tier"]]
        seen[cc] = row = {
            "cc": cc, "iso3": r["iso3"], "country": r["country"], "tier": r["tier"],
            "pop_m": r["pop_m"], "status": status,
            # religiondots has DRAWN it: its countries/<cc>.py names the placement layer to reuse
            "rd_geo": "yes" if (RD / "countries" / f"{cc}.py").exists() else "",
            "question": r["question"], "census_year": r["census_year"],
            "finest_level": r["finest_level"], "n_units": r["n_units"], "access": r["access"],
            "source_url": r["source_url"], "note": "",
        }
        out.append(row)
    rank = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    out.sort(key=lambda r: (rank[r["tier"]], -float(r["pop_m"] or 0)))
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    n = {s: sum(1 for r in out if r["status"] == s) for s in ("free", "ruling", "drawn")}
    print(f"wrote {OUT}: {len(out)} rows, {n}")


if __name__ == "__main__":
    main()
