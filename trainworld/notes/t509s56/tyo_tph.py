"""A copy of Tokyo's calibration network with every line's trains an hour x a factor (T-509 stage 6:
what would bring Tokyo's commute rail share near). python tyo_tph.py 1.25"""
import json
import os
import sys

f = float(sys.argv[1])
src = r"C:\Users\anita\projects\maps\trainworld\data\work\realnet\tyo_real.json"
n = json.load(open(src))
n["demand"]["tph"] = [x * f for x in n["demand"]["tph"]]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"tyo_tph{f:g}.json")
json.dump(n, open(out, "w"))
print(out)
