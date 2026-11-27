"""T-4364 Boston: Boston's hand pack (build_city.py CITIES["bos"]) drops Rockingham and Strafford, New Hampshire
(33015, 33017), which join New Hampshire's zone (his, plan name Hillsborough County, shown as Manchester, NH).

    boston_edit.py          dry: what would change
    boston_edit.py --go     edit build_city.py's bos counties and the plan's his row (units, people, bbox; status
                            planned), under the plan lock
    boston_edit.py --undo   put both back (the his row from data/work/pre-T4364us/pre's world_plan.json is restore.py's)

People are Census 2020 (the plan's US units are Census counties, people by GHSL; the next zones replan recounts them).
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, os.path.join(TW, "pipeline"))
import shared_lock  # noqa: E402

BC = os.path.join(TW, "pipeline", "build_city.py")
PLAN = os.path.join(TW, "pipeline", "world_plan.json")
OLD = '            "33015": "Rockingham NH", "33017": "Strafford",\n'
ADD = {"33015": 314176, "33017": 130889}
BOX = (-71.37, 42.70, -70.55, 43.60)  # Rockingham (Isles of Shoals included) and Strafford, lon/lat


def main():
    go, undo = "--go" in sys.argv, "--undo" in sys.argv
    text = open(BC, encoding="utf-8").read()
    i = text.index('    "bos": {')
    j = text.index("    },\n", i)
    block = text[i:j]
    if undo:
        if OLD in block:
            print("build_city.py: bos already has the NH counties")
        else:
            k = block.index('"25025": "Suffolk MA",\n') + len('"25025": "Suffolk MA",\n')
            text = text[:i] + block[:k] + OLD + block[k:] + text[j:]
            open(BC + ".part", "w", encoding="utf-8").write(text)
            os.replace(BC + ".part", BC)
            print("build_city.py: bos counties put back")
        return
    if OLD not in block:
        print("build_city.py: bos already without the NH counties")
    else:
        print("build_city.py: bos drops 33015 Rockingham NH, 33017 Strafford")
        new = text[:i] + block.replace(OLD, "") + text[j:]
        new = new.replace("# Broad Boston-Cambridge-Newton county metro extent, including southern NH.",
                          "# Boston-Cambridge-Newton's Massachusetts counties; Rockingham and Strafford NH are in New\n"
                          "        # Hampshire's zone (T-4364: 22% and 3% of their workers work in these counties).")
        if go:
            open(BC + ".part", "w", encoding="utf-8").write(new)
            os.replace(BC + ".part", BC)
    with shared_lock.held("plan", "T-4364 boston_edit"):
        doc = shared_lock.read_json(PLAN)
        r = next(r for r in doc["metros"] if r.get("id") == "his")
        us = r["units"]["USA"]
        add = [c for c in ADD if c not in us]
        if not add:
            print("his: already holds 33015, 33017")
            return
        if r["status"] not in ("shipped", "planned", "parked"):
            raise SystemExit(f"his: status {r['status']} (a run has it)")
        r["units"]["USA"] = us + add
        r["people"] = r["people"] + sum(ADD[c] for c in add)
        b = r["bbox"]
        r["bbox"] = [min(b[0], BOX[0]), min(b[1], BOX[1]), max(b[2], BOX[2]), max(b[3], BOX[3])]
        r["status"] = "planned"
        r.setdefault("source", {})["hand_edit"] = {"task": "T-4364", "why": "Rockingham and Strafford NH from Boston's hand pack (bos): 22% and 3% of their workers work in Boston's Massachusetts counties (ACS 2016-2020), under the planner's 25% belt line"}
        print(f"his: units + {add}, people {r['people']:,}, bbox {r['bbox']}, status planned")
        if go:
            shared_lock.write_atomic(PLAN, json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
            print("plan written")
        else:
            print("(dry: nothing written)")


if __name__ == "__main__":
    main()
