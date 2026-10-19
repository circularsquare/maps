"""T-710: the two trips like Anita's screenshot, old and new: where riders leave the train, how many
rides their routes take, and the listed routes."""
import json
import sys

OUT = r"C:\Users\anita\projects\maps\artifacts\trainworld\t710"
for tag in sys.argv[1].split(","):
    j = json.load(open(f"{OUT}/trips_nyc_{tag}.json"))
    for t in j["trips"][:2]:
        print(f"\n{tag} {t['trip']}: rail {t['rail']:.2f}, listed {t['listed']:.2f}")
        br = t["by_rides"]
        print(f"  rides per route (listed): one {br[1]:.3f}, two {br[2]:.3f}, three+ {br[3]:.3f}")
        print("  egress: " + "; ".join(f"{e['station']} {e['share']:.3f} (walk {e['walk']:.1f})" for e in t["egress"]))
        for r in t["routes"]:
            print(f"   {r['share']:.3f} cost {r['cost']:.1f}: {r['legs']}")
