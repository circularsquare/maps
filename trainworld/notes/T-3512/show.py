"""Print a trace run's edits: timings and the worker log lines after each edit."""
import json
import sys

j = json.load(open(sys.argv[1], encoding="utf-8"))
pat = sys.argv[2] if len(sys.argv) > 2 else ""
print("load final", j.get("final"), "workers", j.get("workers"), "busiest", j.get("busiest"), "error", j.get("error"))
for e in j["edits"]:
    s = e["solve"]
    print("\n==", e["kind"], "settle", e["settle"], e["res"][:90])
    for k in ("version", "arrivedAt", "sentAt", "firstAt", "earlyAt", "localShownAt", "refinedAt", "roundAt", "roundFirstAt", "heldFrom", "wholeSent", "localPlans", "clusters", "periods", "local", "setupMs", "versions"):
        print("  ", k, s.get(k))
    print("   work", json.dumps(s.get("work"))[:400])
    print("   periodMs", json.dumps(s.get("periodMs"))[:300])
    print("   clusterRes", json.dumps(s.get("clusterRes"))[:600])
    f = e["fig"]
    print("   fig rail", f["rail"], "trips", f["trips"], "icRail", f["icRail"], "stH", f["stH"])
    for t, x in e["log"]:
        if pat and pat not in x:
            continue
        print("   %7d %s" % (t, x[:300]))
