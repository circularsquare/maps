"""Make a naming batch: a folder with fact sheets and pictures for one area.

    python asia1m/naming/batch.py <batch name> <text.png> --province "Ningxia Hui"
    python asia1m/naming/batch.py <batch name> <text.png> --province Sichuan --prefectures Chengdu,Deyang

<text.png> is the 'text' layer exported from a copy of asia1m.ase; main.png is read from
../regionnaming (run.py or a manual export puts it there). A region joins the batch when the
named province (and prefecture, if given) holds most of its people.

Writes batches/<name>/:
  facts.json     the batch's regions from regions.json, each with a number n (1, 2, ...)
  labels_*.png   the area as drawn, with the text layer: settled names are read from these
  numbers_*.png  the same area with each batch region's number on it and other regions faded
  thumbs/<n>.png one picture per region for the viewer
and adds the batch to batches/index.json.
"""
import os, sys, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
RN = os.path.join(HERE, "..", "regionnaming")
import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("text_png")
ap.add_argument("--province", default="")
ap.add_argument("--prefectures", default="")
ap.add_argument("--keys", default="", help="semicolon-separated region keys instead of a province")
a = ap.parse_args()
prefs = [p.strip() for p in a.prefectures.split(",") if p.strip()]

regions = json.load(open(f"{HERE}/regions.json", encoding="utf-8"))
import csv
picked = {}; picked_elsewhere = set()
if os.path.exists(f"{HERE}/picks.csv"):
    for p in csv.DictReader(open(f"{HERE}/picks.csv", encoding="utf-8")):
        picked[p["key"]] = p["name"]
        if p["batch"] != a.name: picked_elsewhere.add(p["key"])
for r in regions:
    for nb in r["neighbours"]:
        if nb["key"] in picked: nb["name"] = picked[nb["key"]]
def top(r, level):
    ps = [p for p in r["parents"] if p["level"] == level]
    return ps[0] if ps else None
keys = [k.strip() for k in a.keys.split(";") if k.strip()]   # keys contain commas ("#c@114.5,35.4")
if not keys and not a.province: sys.exit("give --province or --keys")
def member(r):
    if keys: return r["key"] in keys
    # a region belongs to the province and prefecture holding most of its people, however small
    # that share; a 50% floor let regions split between two prefectures fall through every batch
    p = top(r, "province")
    if not p or p["name"] != a.province: return False
    if prefs:
        f = top(r, "prefecture")
        return bool(f) and f["name"] in prefs
    return True
batch = [r for r in regions if member(r) and r["key"] not in picked_elsewhere]
if not batch: sys.exit("no regions match")
batch.sort(key=lambda r: (-r["lat"], r["lon"]))
for n, r in enumerate(batch, 1): r["n"] = n

out = os.path.join(HERE, "batches", a.name)
os.makedirs(os.path.join(out, "thumbs"), exist_ok=True)

main = np.array(Image.open(f"{RN}/main.png").convert("RGBA"))
text = np.array(Image.open(a.text_png).convert("RGBA"))
rid = np.load(f"{RN}/region_id.npy")
color_of = {}
for r in batch:  # map batch regions to their ids in region_id.npy via colour and bbox
    x0, y0, x1, y1 = r["bbox"]
    sub = main[y0:y1 + 1, x0:x1 + 1, :3]
    c = tuple(int(r["color"][i:i + 2], 16) for i in (1, 3, 5))
    m = np.all(sub == c, axis=2)
    ids = rid[y0:y1 + 1, x0:x1 + 1][m]
    color_of[r["n"]] = int(np.bincount(ids[ids > 0]).argmax())
H, W = rid.shape

def composite(y0, y1, x0, x1, fade_ids=None, with_text=True):
    m = main[y0:y1, x0:x1].astype(np.float32)
    rgb = m[:, :, :3] * (m[:, :, 3:] / 255) + 255 * (1 - m[:, :, 3:] / 255)
    if fade_ids is not None:
        keep = np.isin(rid[y0:y1, x0:x1], list(fade_ids))
        rgb[~keep] = rgb[~keep] * 0.35 + 255 * 0.65
    if with_text:
        t = text[y0:y1, x0:x1].astype(np.float32)
        al = t[:, :, 3:] / 255
        rgb = t[:, :, :3] * al + rgb * (1 - al)
    return rgb

font = ImageFont.truetype(os.path.join(HERE, "..", "Unifontexmono-2vrqo.ttf"), 16)

def interior(rid_n, y0, y1, x0, x1):
    m = rid[y0:y1, x0:x1] == rid_n
    d = ndi.distance_transform_edt(m)
    y, x = np.unravel_index(d.argmax(), d.shape)
    return x, y

# area pictures, tiled so each tile is readable at 1-2x
# wide margin so the neighbours' settled names are in the picture too
M = 200
bx0 = max(min(r["bbox"][0] for r in batch) - M, 0); by0 = max(min(r["bbox"][1] for r in batch) - M, 0)
bx1 = min(max(r["bbox"][2] for r in batch) + M, W); by1 = min(max(r["bbox"][3] for r in batch) + M, H)
scale = 2                      # the pixel font is hard to read at 1x
TILE = 1600 // scale
tiles = []
for ty in range(by0, by1, TILE):
    for tx in range(bx0, bx1, TILE):
        y1_, x1_ = min(ty + TILE, by1), min(tx + TILE, bx1)
        inside = [r for r in batch if not (r["bbox"][2] < tx or r["bbox"][0] > x1_ or r["bbox"][3] < ty or r["bbox"][1] > y1_)]
        if not inside: continue
        tag = f"{len(tiles) + 1}"
        lab = Image.fromarray(composite(ty, y1_, tx, x1_).astype(np.uint8))
        lab = lab.resize((lab.width * scale, lab.height * scale), Image.NEAREST)
        lab.save(f"{out}/labels_{tag}.png")
        num = Image.fromarray(composite(ty, y1_, tx, x1_, fade_ids=set(color_of.values()), with_text=False).astype(np.uint8))
        num = num.resize((num.width * scale, num.height * scale), Image.NEAREST)
        dr = ImageDraw.Draw(num)
        for r in inside:
            x, y = interior(color_of[r["n"]], ty, y1_, tx, x1_)
            if not (rid[ty + y, tx + x] == color_of[r["n"]]): continue
            s = str(r["n"]); w, h = dr.textsize(s, font=font)
            cx, cy = x * scale, y * scale
            dr.rectangle([cx - w / 2 - 2, cy - h / 2 - 1, cx + w / 2 + 2, cy + h / 2 + 1], fill=(255, 255, 255), outline=(200, 0, 0))
            dr.text((cx - w / 2, cy - h / 2), s, fill=(200, 0, 0), font=font)
        num.save(f"{out}/numbers_{tag}.png")
        tiles.append(dict(tag=tag, x=tx, y=ty, w=x1_ - tx, h=y1_ - ty, scale=scale, regions=[r["n"] for r in inside]))

# one thumbnail per region: the region at full colour with a red outline, others faded
for r in batch:
    x0, y0, x1, y1 = r["bbox"]
    pad = max(40, int(0.35 * max(x1 - x0, y1 - y0)))
    cx0, cy0 = max(x0 - pad, 0), max(y0 - pad, 0)
    cx1, cy1 = min(x1 + pad, W), min(y1 + pad, H)
    ident = color_of[r["n"]]
    rgb = composite(cy0, cy1, cx0, cx1, fade_ids={ident})
    m = rid[cy0:cy1, cx0:cx1] == ident
    edge = ndi.binary_dilation(m, iterations=2) & ~m
    rgb[edge] = (210, 30, 30)
    im = Image.fromarray(rgb.astype(np.uint8))
    k = 1.0
    s = max(1, min(6, 420 // max(im.width, im.height)))
    if s > 1:
        im = im.resize((im.width * s, im.height * s), Image.NEAREST); k = float(s)
    elif max(im.width, im.height) > 520:
        f = 520 / max(im.width, im.height)
        im = im.resize((int(im.width * f), int(im.height * f)), Image.LANCZOS); k = f
    im.save(f"{out}/thumbs/{r['n']}.png")
    # thumbnail pixel = (map pixel - origin) * k, for overlay.py
    r["thumb"] = dict(x0=cx0, y0=cy0, k=k, w=im.width, h=im.height)

json.dump(dict(name=a.name, province=a.province, prefectures=prefs, tiles=tiles, regions=batch),
          open(f"{out}/facts.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
idx_path = os.path.join(HERE, "batches", "index.json")
idx = json.load(open(idx_path, encoding="utf-8")) if os.path.exists(idx_path) else []
idx = [b for b in idx if b["name"] != a.name] + [dict(name=a.name, province=a.province, prefectures=prefs, count=len(batch))]
json.dump(idx, open(idx_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{a.name}: {len(batch)} regions, {len(tiles)} picture tiles at {scale}x -> {out}")
import subprocess
subprocess.run([sys.executable, os.path.join(HERE, "overlay.py"), a.name], check=True)
