import os, csv, json
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
S = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(S, "tiles"), exist_ok=True)
ext = json.load(open(os.path.join(S, "extent.json")))["ext"]
im = Image.open(os.path.join(S, "composite.png"))
issues = list(csv.DictReader(open(os.path.join(S, "data_issues.csv"), encoding="utf-8")))
TW, TH, OV = 500, 400, 12
x0, y0, x1, y1 = ext
rows = []
lines = []
r = 0
for y in range(y0, y1, TH):
    c = 0
    for x in range(x0, x1, TW):
        name = "t_r%02d_c%02d" % (r, c)
        box = (x - OV, y - OV, x + TW + OV, y + TH + OV)
        t = im.crop(box).resize(((TW + 2 * OV) * 2, (TH + 2 * OV) * 2), Image.NEAREST)
        t.save(os.path.join(S, "tiles", name + ".png"))
        inside = [i for i in issues if box[0] <= int(i["x"]) < box[2] and box[1] <= int(i["y"]) < box[3]]
        lines.append("%s  map x %d-%d y %d-%d  (tile px = (map - %d, map - %d) * 2)" % (name, box[0], box[2], box[1], box[3], box[0], box[1]))
        for i in inside:
            lines.append("    %s/%s %s %s %s @%s,%s %sx%s: %s" % (i["kind"], i["confidence"], i["key"], i["name"], i["zh"], i["x"], i["y"], i["w"], i["h"], i["saw"][:140]))
        c += 1
    r += 1
open(os.path.join(S, "tiles", "index.txt"), "w", encoding="utf-8").write("\n".join(lines))
print(r, "rows", c, "cols")
