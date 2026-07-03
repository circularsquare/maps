"""Extract regions from the asia1m 'main' layer.
A region = one exact color, with same-color blobs merged if within ~300px.
Outputs region_id.npy (int32 per-pixel region index, 0=none) and regions_prelim.csv.
"""
import sys, json, time
sys.path.insert(0, r"C:/Users/anita/projects/maps/asia1m/regionnaming")
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from geo import px_to_lonlat
Image.MAX_IMAGE_PIXELS = None

SCR = r"C:/Users/anita/projects/maps/asia1m/regionnaming"
MERGE_PX = 300
B = 8                       # coarse pooling factor
R = int(round((MERGE_PX/2)/B))   # dilation radius in coarse cells (~150px each side)

SPECIALS = {(0xd9,0xf6,0xff), (0x53,0x5d,0x8d), (0x64,0xa0,0xd2),  # water, border, coast
            (0xee,0xad,0x75)}  # grid overlay (100px graticule), not a region

t0=time.time()
im = np.array(Image.open(f"{SCR}/main.png").convert('RGBA'))
H, W = im.shape[:2]
r,g,b = im[:,:,0].astype(np.uint32), im[:,:,1].astype(np.uint32), im[:,:,2].astype(np.uint32)
a = im[:,:,3]
cid = (r<<16)|(g<<8)|b
cid[a==0] = 0
for (sr,sg,sb) in SPECIALS:
    cid[cid == ((sr<<16)|(sg<<8)|sb)] = 0
del r,g,b,im
print(f"[{time.time()-t0:.0f}s] loaded {W}x{H}")

# compact color ids; ensure background(0) stays label 0
uniq, inv = np.unique(cid, return_inverse=True)
inv = inv.astype(np.int32).reshape(H, W)
assert uniq[0] == 0, uniq[:3]
K = len(uniq) - 1   # number of distinct region colors
print(f"[{time.time()-t0:.0f}s] {K} region colors")

# disk structure for coarse dilation
yy,xx = np.ogrid[-R:R+1, -R:R+1]
disk = (xx*xx+yy*yy) <= R*R

slices = ndi.find_objects(inv, max_label=K)  # index k-1 -> bbox of color label k

region_id = np.zeros((H, W), np.int32)
regions = []
reuse = []           # colors that split into >1 region
next_rid = 1

def maxpool(mask, B):
    h,w = mask.shape
    ph = (-h) % B; pw = (-w) % B
    if ph or pw:
        mask = np.pad(mask, ((0,ph),(0,pw)))
    H2,W2 = mask.shape
    return mask.reshape(H2//B, B, W2//B, B).max(axis=(1,3))

for k in range(1, K+1):
    sl = slices[k-1]
    if sl is None:
        continue
    ys, xs = sl
    sub = (inv[ys, xs] == k)
    color = int(uniq[k])
    hexc = '#%06x' % color
    # coarse merge
    coarse = maxpool(sub, B)
    dil = ndi.binary_dilation(coarse, structure=disk)
    lbl, n = ndi.label(dil)
    # group id for each ON coarse cell
    grp_at = lbl  # label per coarse cell (only meaningful where dil True, covers coarse True)
    # fine pixel coords of this color (within sub)
    fy, fx = np.where(sub)
    cy, cx = fy//B, fx//B
    g = grp_at[cy, cx]
    groups = np.unique(g)
    y0, x0 = ys.start, xs.start
    rids_here = []
    for gi in groups:
        sel = g == gi
        py = fy[sel] + y0
        px = fx[sel] + x0
        rid = next_rid; next_rid += 1
        region_id[py, px] = rid
        area = int(sel.sum())
        cxm = float(px.mean()); cym = float(py.mean())
        lon, lat = px_to_lonlat(cxm, cym)
        regions.append(dict(rid=rid, color=hexc, area_px=area,
                            cx=round(cxm,1), cy=round(cym,1),
                            lon=round(float(lon),4), lat=round(float(lat),4),
                            bx0=int(px.min()), bx1=int(px.max()),
                            by0=int(py.min()), by1=int(py.max())))
        rids_here.append(rid)
    if len(rids_here) > 1:
        reuse.append((hexc, len(rids_here), [regions[r-1] for r in rids_here]))

print(f"[{time.time()-t0:.0f}s] {len(regions)} regions from {K} colors")
np.save(f"{SCR}/region_id.npy", region_id)

import csv
with open(f"{SCR}/regions_prelim.csv","w",newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(regions[0].keys()))
    w.writeheader(); w.writerows(regions)

# ---- reports ----
areas = np.array([r['area_px'] for r in regions])
print("area percentiles px:", np.percentile(areas,[1,5,25,50,75,95,99]).round().astype(int))
print("tiny regions (<200px):", int((areas<200).sum()), " (<50px):", int((areas<50).sum()))
print(f"colors reused (>1 region, >{MERGE_PX}px apart): {len(reuse)}")
for hexc,n,rs in sorted(reuse,key=lambda x:-x[1])[:15]:
    locs = "; ".join(f"({r['lon']:.2f},{r['lat']:.2f}) {r['area_px']}px" for r in rs)
    print(f"  {hexc} -> {n} regions: {locs}")

# near-duplicate colors
cols = np.array([[ (int(r['color'][1:],16)>>16)&255, (int(r['color'][1:],16)>>8)&255, int(r['color'][1:],16)&255 ] for r in regions])
ucols = np.unique(cols, axis=0)
nd = []
for i in range(len(ucols)):
    d = np.abs(ucols[i+1:].astype(int)-ucols[i].astype(int)).sum(1)
    for j in np.where(d<=3)[0]:
        nd.append((tuple(ucols[i]), tuple(ucols[i+1+j]), int(d[j])))
print(f"near-duplicate color pairs (L1<=3): {len(nd)}")
for c1,c2,d in nd[:15]:
    print(f"  #{c1[0]:02x}{c1[1]:02x}{c1[2]:02x} ~ #{c2[0]:02x}{c2[1]:02x}{c2[2]:02x} d={d}")
print(f"[{time.time()-t0:.0f}s] done")
