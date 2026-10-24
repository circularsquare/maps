"""T-1140: plan_zones dry runs for a new country through the planner's own code (tree t1150), written
here, never to the plan. Optional what-if zone rule for the country (harness only; units stay the
state's, as for a country with no rule).

    python dry_country.py KR out.json
    python dry_country.py CA out.json min_density=0 big_km2=50000 no_territories=1 contiguous=1
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys

sys.path.insert(0, r"C:\Users\anita\projects\maps\.tw-trees\t1150\trainworld\pipeline")
sys.stdout.reconfigure(encoding="utf-8")
import world_plan as wp  # noqa: E402

iso2, out = sys.argv[1], sys.argv[2]
kw = dict(a.split("=") for a in sys.argv[3:])
a3 = {"KR": "KOR", "CA": "CAN", "MX": "MEX", "TR": "TUR", "VN": "VNM"}[iso2]
if kw:
    rule = {}
    if "min_density" in kw:
        rule["min_density"] = float(kw["min_density"])
    if "big_km2" in kw:
        rule["big_km2"] = float(kw["big_km2"])
    if kw.get("contiguous") == "1":
        rule["contiguous"] = True

    class Rules(dict):
        """ZONE_RULES with the what-if rule, which plan_zones' `iso in ZONE_RULES` does not see (so it
        keeps the state's units, writing no geoBoundaries cache)."""
        def __contains__(self, k):
            return k != a3 and dict.__contains__(self, k)
    wp.ZONE_RULES = Rules(wp.ZONE_RULES, **{a3: rule})
    if kw.get("no_territories") == "1":
        import pickle

        class Here(pickle.Unpickler):
            def find_class(self, mod, name):
                return wp.Unit if name == "Unit" else super().find_class(mod, name)
        with open(wp.WORK / "state.pkl", "rb") as f:
            st = Here(f).load()
        terr = {u.parents[0][0] for u in st["units"] if u.iso == a3 and u.parents
                and u.parents[0][1] in ("Yukon", "Northwest Territories", "Nunavut")}
        del st
        rule["zoned_only"] = {t: [] for t in terr}
        print("territories left out:", terr, flush=True)
    print("what-if rule:", {k: v for k, v in rule.items() if k != "zoned_only"}, flush=True)
wp.plan_zones([a3], out)
os.remove(os.path.splitext(out)[0] + ".extents.geojson")  # 50 MB, and the disk is short
