"""T-509 stage 4: the everyday layer's outing rates, by country.

    python pipeline/everyday_rates.py show                       the table and the worldwide default
    python pipeline/everyday_rates.py <id> [<id> ...] | --all    write each pack's `everyday` block
    python pipeline/everyday_rates.py --file <header.json> <ISO2>  one header (a calibration copy)
    ... [--dry]                                                   print, write nothing

A pack's header block `everyday` = {"country", "rate": {purpose: outings a resident a weekday},
"source"}; the sim reads it into `Params::everyday_rate` (`DemandApi::new`). Only errands, social
and medical come from surveys; school, university and airport keep the sim's own rates (attendance
per enrolled student, an airport's trips), which are not per resident. The destination decay
lengths are one worldwide set in the sim (`everyday::DECAY`, T-637), not by country; a `decay_m`
left in an older block is ignored.

**Counting** (notes/T-509.md section 12). The game's outing is a round trip from home, outward
and return legs both under the activity's purpose. A survey's trips are legs, so the outing rate
for a purpose = its legs / 2, with every leg home counted under the purpose of the activity it
leaves, so a survey's legs are kept whatever its trip chains:
- "to or from" surveys (NHTS's WHYTRP1S, NTS, Sao Paulo's OD with MOTIVO_O on legs home): legs / 2.
- surveys with "return home" as its own purpose (Japan's and Korea's person trip surveys, EMP's
  retour au domicile): trips to the purpose x (1 + R / A) / 2, R the trips home and A the trips to
  anywhere else, which spreads the trips home over the purposes in proportion.
Errands = shopping, personal business and escorting; social = leisure, visits, eating out,
religious; medical = health care. Business and work-related trips are left out (the game has none).
Per resident of all ages: a survey of people 5 and over (or 6+) is scaled by that age share.
"""
import json
import os
import shutil
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from read_pack import PACKS, ROOT  # noqa: E402

PURPOSES = ("errands", "social", "medical")
RES9 = os.path.join(ROOT, "data", "work", "res9")
BACKUP = os.path.join(ROOT, "data", "work", "pre-T509s4")

# outings a resident a weekday by country, from its own travel survey
SURVEYS = {
    "US": {"rate": {"errands": 0.577, "social": 0.424, "medical": 0.044},
           "source": "NHTS 2017 weekday legs by WHYTRP1S/WHYTO (errands shopping, services, errands, escort; social meals, "
                     "recreation, exercise, visits, religious; medical health care) / 2, persons 5+ x 0.939"},
    "BR": {"rate": {"errands": 0.106, "social": 0.057, "medical": 0.044},
           "source": "Sao Paulo OD 2017 microdata, legs by activity end (shopping and personal business; leisure, "
                     "visits and meals; health) / 2 a resident"},
    "JP": {"rate": {"errands": 0.273, "social": 0.122, "medical": 0.037},
           "source": "Tokyo PT 2018 weekday (persons 5+ x 0.962; trips home spread over the purposes, x 0.870): errands "
                     "shopping, pick-up and 63% of other errands; social eating out, social, leisure and 37% of other "
                     "errands (exercise, by the national 2021 split); medical hospital, rehab, day care (press release "
                     "0324press.pdf). The national survey 2021 (big three metros, COVID year) gives 0.291, 0.120, 0.033"},
    "KR": {"rate": {"errands": 0.189, "social": 0.110},
           "source": "KTDB 2016 household travel survey, Seoul, Incheon and Gyeonggi pooled (persons 5+ x 0.965; trips "
                     "home spread, x 0.930): errands shopping, cram school and other less medical at 0.04; social "
                     "leisure, exercise, eating out, visits"},
    # medical sits inside personal business in these three: taken out of errands at 0.044 (US, Brazil)
    "GB": {"rate": {"errands": 0.554, "social": 0.291},
           "source": "NTS 2015-19 weekday trips (NTS0504b; to or from, all ages) / 2: errands shopping, personal "
                     "business, other and education escort; social visits, sport and entertainment, holiday and day "
                     "trips"},
    "FR": {"rate": {"errands": 0.497, "social": 0.375},
           "source": "EMP 2019 weekday trips within 80 km, legs home under the main activity (SDES motifs table 2), "
                     "persons 6+ x 0.934, / 2: errands achats, accompagner, autres motifs personnels; social loisirs"},
    "DE": {"rate": {"errands": 0.518, "social": 0.365},
           "source": "MiD 2017 trips (legs home under the previous purpose, all ages, every day) made weekdays by the "
                     "weekend factors (errands / 1.063, social / 1.171), / 2: errands Einkauf, Erledigung, Begleitung; "
                     "social Freizeit"},
}


def world():
    """The worldwide default: each rate the median over the surveys that give it."""
    return {p: round(statistics.median([s["rate"][p] for s in SURVEYS.values() if p in s["rate"]]), 3)
            for p in PURPOSES}


def block(iso2):
    w = world()
    s = SURVEYS.get(iso2)
    if s is None:
        return {"task": "T-509", "country": iso2, "rate": w,
                "source": f"worldwide default: the median of the {', '.join(SURVEYS)} surveys"}
    return {"task": "T-509", "country": iso2, "rate": {**w, **s["rate"]}, "source": s["source"]}


def country_of(cid, header):
    import nocar
    c = nocar.country_of(cid, header)
    return "GB" if c.startswith("GB") else c


def write_header(path, b):
    with open(path, encoding="utf-8") as f:
        h = json.load(f)
    h["everyday"] = b
    with open(path + ".part", "w") as f:
        json.dump(h, f, indent=2)
    os.replace(path + ".part", path)


def patch(cid, dry=False):
    """The block into the merged pack and its res-9 copy, keeping merge_cells' stamp in step."""
    import merge_cells as mc
    live = os.path.join(PACKS, f"{cid}.json")
    with open(live, encoding="utf-8") as f:
        header = json.load(f)
    b = block(country_of(cid, header))
    r = b["rate"]
    print(f"{cid}: {b['country']}, errands {r['errands']}, social {r['social']}, medical {r['medical']}", flush=True)
    if dry:
        return
    os.makedirs(BACKUP, exist_ok=True)
    stamped = mc.written(cid) == mc.stamp(PACKS, cid)
    for d, tag in ((PACKS, ""), (RES9, ".res9")):
        p = os.path.join(d, f"{cid}.json")
        if os.path.exists(p):
            keep = os.path.join(BACKUP, f"{cid}{tag}.json")
            if not os.path.exists(keep):
                shutil.copy2(p, keep)
            write_header(p, b)
    if stamped:
        with open(os.path.join(RES9, f"{cid}.merged.json"), "w") as f:
            json.dump(mc.stamp(PACKS, cid), f)


def show():
    print(f"{'':6s} " + "  ".join(f"{p:>8s}" for p in PURPOSES))
    for k, r in [(k, s["rate"]) for k, s in SURVEYS.items()] + [("world", world())]:
        print(f"{k:6s} " + "  ".join(f"{r.get(p, float('nan')):8.3f}" for p in PURPOSES))


if __name__ == "__main__":
    args = sys.argv[1:]
    dry = "--dry" in args
    if not args or args[0] == "show":
        show()
    elif args[0] == "--file":
        b = block(args[2])
        print(args[1], b)
        if not dry:
            write_header(args[1], b)
    else:
        import merge_cells as mc
        ids = mc.registered() if "--all" in args else [x for x in args if not x.startswith("--")]
        for cid in ids:
            patch(cid, dry)
