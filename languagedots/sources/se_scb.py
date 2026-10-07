"""Sweden: SCB population by country of birth, per kommun, 31 Dec 2025 (open PxWeb API, no key).

    python sources/se_scb.py --fetch     download into data/raw/se/ (two JSON files)
    python sources/se_scb.py             print the checks; sources/se_build.py writes the CSV

Tables (Statistikdatabasen, BE0101E):
  * FolkmRegFlandKCKM  "Folkmängden efter region, födelseland och kön. År 2025": 290 kommuner,
    21 län, Riket x 187 named birth countries + "okänt födelseland" + "övriga födelseländer".
  * FodelselandArKCKM  "Folkmängden efter födelseland, ålder och kön. År 2025": the nation by
    every birth country SCB names (and its groups); a check on Riket.
  * UtlSvBakgFinCKM (BE0101Q) per kommun: foreign-born, Sweden-born with two / one / no
    foreign-born parents.
  * FolkmForUrspCKMv2 (BE0101Q) nationally: Sweden-born by parents' country of birth.
  * FodelselandArK, 2006: the national birth-country stock in the year Parkvall's estimates
    describe (sources/se_build.py derives its splits from it).
SCB's "CKM" tables carry cell-key perturbation: a few people of noise per cell.

CHECKS (printed): the kommuner sum to Riket for every birth country; each kommun's countries
sum to its total; the national table agrees with the regional one's Riket for every country
both name.
"""
import json
import os
import sys

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "se")
API = "https://api.scb.se/OV0104/v1/doris/sv/ssd/BE/BE0101/BE0101E/"
REGIONAL = os.path.join(RAW, "FolkmRegFlandKCKM_2025.json")
NATIONAL = os.path.join(RAW, "FodelselandArKCKM_2025.json")
BACKGROUND = os.path.join(RAW, "UtlSvBakgFinCKM_2025.json")
PARENTS = os.path.join(RAW, "FolkmForUrspCKMv2_2025.json")
STOCK2006 = os.path.join(RAW, "FodelselandArK_2006.json")
UA = {"User-Agent": "Mozilla/5.0"}


def _post(table, query):
    meta = requests.get(API + table, headers=UA, timeout=60).json()
    q = {"query": query, "response": {"format": "json-stat2"}}
    r = requests.post(API + table, json=q, headers=UA, timeout=300)
    r.raise_for_status()
    return meta, r.json()


def fetch():
    os.makedirs(RAW, exist_ok=True)
    allv = lambda code: {"code": code, "selection": {"filter": "all", "values": ["*"]}}
    _, d = _post("FolkmRegFlandKCKM", [allv("Region"), allv("Fodelseregion"),
                                       {"code": "Kon", "selection": {"filter": "item",
                                                                     "values": ["TotSa"]}}])
    json.dump(d, open(REGIONAL, "w", encoding="utf-8"), ensure_ascii=False)
    _, d = _post("FodelselandArKCKM", [allv("Fodelseland"),
                                       {"code": "Alder", "selection": {"filter": "item",
                                                                       "values": ["tot"]}},
                                       {"code": "Kon", "selection": {"filter": "item",
                                                                     "values": ["TotSa"]}}])
    json.dump(d, open(NATIONAL, "w", encoding="utf-8"), ensure_ascii=False)
    q = "https://api.scb.se/OV0104/v1/doris/sv/ssd/BE/BE0101/BE0101Q/"
    meta = requests.get(q + "UtlSvBakgFinCKM", headers=UA, timeout=60).json()
    tot_age = [v for v in meta["variables"] if v["code"] == "Alder"][0]
    tot_age = [c for c, t in zip(tot_age["values"], tot_age["valueTexts"]) if "totalt" in t][0]
    r = requests.post(q + "UtlSvBakgFinCKM", headers=UA, timeout=300, json={
        "query": [allv("Region"), allv("UtlBakgrund"),
                  {"code": "Alder", "selection": {"filter": "item", "values": [tot_age]}},
                  {"code": "Kon", "selection": {"filter": "item", "values": ["TotSa"]}}],
        "response": {"format": "json-stat2"}})
    r.raise_for_status()
    json.dump(r.json(), open(BACKGROUND, "w", encoding="utf-8"), ensure_ascii=False)
    r = requests.post(q + "FolkmForUrspCKMv2", headers=UA, timeout=300, json={
        "query": [allv("Ursprung"), allv("Fodelseland"),
                  {"code": "Kon", "selection": {"filter": "item", "values": ["TotSa"]}},
                  {"code": "Tid", "selection": {"filter": "item", "values": ["2025"]}}],
        "response": {"format": "json-stat2"}})
    r.raise_for_status()
    json.dump(r.json(), open(PARENTS, "w", encoding="utf-8"), ensure_ascii=False)
    # national birth-country stock in 2006, the year Parkvall's estimates describe
    r = requests.post(API + "FodelselandArK", headers=UA, timeout=300, json={
        "query": [allv("Fodelseland"),
                  {"code": "Tid", "selection": {"filter": "item", "values": ["2006"]}}],
        "response": {"format": "json-stat2"}})
    r.raise_for_status()
    json.dump(r.json(), open(STOCK2006, "w", encoding="utf-8"), ensure_ascii=False)
    print("fetched", REGIONAL, NATIONAL, BACKGROUND, PARENTS, STOCK2006)


def background():
    """{kommun: {UtlBakgrund code: n}}: 08 foreign-born, 4 Sweden-born with two foreign-born
    parents, 5 with one, 6 with none (UtlSvBakgFinCKM, 2025)."""
    rows, _ = _cube(BACKGROUND)
    by = {}
    for rec, v in rows:
        by.setdefault(rec["Region"], {})[rec["UtlBakgrund"]] = v
    return {k: v for k, v in by.items() if len(k) == 4}


def parents():
    """{Ursprung code: {country: n}} nationally, 2025 (FolkmForUrspCKMv2)."""
    rows, labels = _cube(PARENTS)
    by = {}
    for rec, v in rows:
        by.setdefault(rec["Ursprung"], {})[rec["Fodelseland"]] = v
    return by


def stock2006():
    rows, _ = _cube(STOCK2006)
    out = {}
    for rec, v in rows:
        out[rec["Fodelseland"]] = out.get(rec["Fodelseland"], 0) + v
    return out


def _cube(path):
    """json-stat2 -> list of (dict of dimension codes, value), plus code -> label per dim."""
    d = json.load(open(path, encoding="utf-8"))
    ids, sizes = d["id"], d["size"]
    cats = {x: sorted(d["dimension"][x]["category"]["index"],
                      key=d["dimension"][x]["category"]["index"].get) for x in ids}
    labels = {x: d["dimension"][x]["category"]["label"] for x in ids}
    out = []
    for i, v in enumerate(d["value"]):
        rec, k = {}, i
        for x, s in zip(reversed(ids), reversed(sizes)):
            rec[x] = cats[x][k % s]
            k //= s
        out.append((rec, v or 0))
    return out, labels


def regional():
    """{kommun code: {birth code: n}}, Riket's row, and labels (region, country)."""
    rows, labels = _cube(REGIONAL)
    by = {}
    for rec, v in rows:
        by.setdefault(rec["Region"], {})[rec["Fodelseregion"]] = v
    kommuner = {k: v for k, v in by.items() if len(k) == 4}
    return kommuner, by["00"], labels["Region"], labels["Fodelseregion"]


def national():
    rows, labels = _cube(NATIONAL)
    return {rec["Fodelseland"]: v for rec, v in rows}, labels["Fodelseland"]


def check():
    kom, riket, rlab, clab = regional()
    assert len(kom) == 290, len(kom)
    bad = []
    for c in riket:
        s = sum(k.get(c, 0) for k in kom.values())
        if s != riket[c]:
            bad.append((c, s, riket[c]))
    print(f"  {len(kom)} kommuner; Riket {riket['TOTfod']:,}; kommuner summing off for "
          f"{len(bad)} birth countries {bad[:5]}")
    worst = max(abs(sum(v for c, v in k.items() if c != 'TOTfod') - k['TOTfod'])
                for k in kom.values())
    print(f"  worst kommun: countries minus total = {worst}")
    nat, nlab = national()
    named = [c for c in riket if c not in ("TOTfod", "OVFOD", "ÖOF")]
    inv = {v: k for k, v in nlab.items()}
    off = [(clab[c], riket[c], nat.get(inv.get(clab[c]))) for c in named
           if nat.get(inv.get(clab[c])) != riket[c]]
    print(f"  national table vs Riket: {len(named) - len(off)} of {len(named)} countries agree; "
          f"differ/missing: {off[:8]}")
    return kom, riket


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    check()
