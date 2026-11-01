"""Rivers, lakes, mountains and areas in China with a Chinese name, from GeoNames -> features.csv

    python asia1m/naming/oldnames/geonames_extract.py <path to CN.zip>

CN.zip is https://download.geonames.org/export/dump/CN.zip (CC BY 4.0). Only feature classes
H (water), T (relief) and L (areas, e.g. leagues, plateaus) are kept, and only rows with at
least one name in Chinese characters among the alternate names. The map position is the
GeoNames point (for a river, one point on it).
"""
import sys, os, io, csv, zipfile, re
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "regionnaming"))
from geo import lonlat_to_px

CJK = re.compile(r"^[一-鿿]+$")
z = zipfile.ZipFile(sys.argv[1])
rows = []
for line in io.TextIOWrapper(z.open("CN.txt"), encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if p[6] not in ("H", "T", "L"): continue
    zh = [a for a in p[3].split(",") if CJK.match(a)]
    if not zh: continue
    x, y = lonlat_to_px(float(p[5]), float(p[4]))
    rows.append(dict(name=p[1], ascii=p[2], zh="|".join(dict.fromkeys(zh)), cls=p[6], code=p[7],
                     x=round(float(x)), y=round(float(y))))
with open(os.path.join(HERE, "features.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, ["name", "ascii", "zh", "cls", "code", "x", "y"]); w.writeheader(); w.writerows(rows)
print(len(rows), "features")
