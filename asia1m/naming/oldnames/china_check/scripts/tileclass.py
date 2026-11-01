import os, json, csv
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
S = os.path.dirname(os.path.abspath(__file__))
RN = r"C:\Users\anita\projects\maps\asia1m\regionnaming"
NAMING = r"C:\Users\anita\projects\maps\asia1m\naming"
regions = json.load(open(os.path.join(NAMING, "regions.json"), encoding="utf-8"))
cn_cols = {r["color"] for r in regions if r["country"] == "China"}
prelim = list(csv.DictReader(open(os.path.join(RN, "regions_prelim.csv"))))
cn_rids = np.array(sorted(int(p["rid"]) for p in prelim if p["color"] in cn_cols))
rid = np.load(os.path.join(RN, "region_id.npy"), mmap_mode="r")
ins = np.array(Image.open(os.path.join(S, "insets.png")).convert("RGBA"))
ext = json.load(open(os.path.join(S, "extent.json")))["ext"]
TW, TH, OV = 500, 400, 12
x0, y0, x1, y1 = ext
cn_codes = set(int(c[1:], 16) for c in cn_cols)
out = {}
r = 0
for y in range(y0, y1, TH):
    c = 0
    for x in range(x0, x1, TW):
        name = "t_r%02d_c%02d" % (r, c)
        Y0, Y1, X0, X1 = max(y - OV, 0), y + TH + OV, max(x - OV, 0), x + TW + OV
        sub = np.asarray(rid[Y0:Y1, X0:X1])
        n_main = int(np.isin(sub, cn_rids).sum())
        a = ins[Y0:Y1, X0:X1]
        code = (a[:, :, 0].astype(np.int64) << 16) | (a[:, :, 1].astype(np.int64) << 8) | a[:, :, 2]
        m = a[:, :, 3] == 255
        n_ins = int(sum(1 for v in np.unique(code[m]) if int(v) in cn_codes))
        out[name] = (n_main, n_ins)
        c += 1
    r += 1
china = [k for k, v in out.items() if v[0] > 0 or v[1] > 0]
other = [k for k, v in out.items() if not (v[0] > 0 or v[1] > 0)]
json.dump(dict(china=china, other=other, counts=out), open(os.path.join(S, "tiles", "classes.json"), "w"), indent=0)
print(len(china), "tiles with China;", len(other), "without:", other)
# contact sheets of the tiles without China, 1x, 2x2 per sheet
for i in range(0, len(other), 4):
    sheet = Image.new("RGB", (1048 + 8, 848 + 8), "black")
    for j, k in enumerate(other[i:i + 4]):
        t = Image.open(os.path.join(S, "tiles", k + ".png")).resize((524, 424), Image.NEAREST)
        sheet.paste(t, ((j % 2) * 532, (j // 2) * 432))
    sheet.save(os.path.join(S, "tiles", "nochina_%02d.png" % (i // 4)))
    print("nochina_%02d" % (i // 4), other[i:i + 4])
