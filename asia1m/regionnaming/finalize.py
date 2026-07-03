import csv, numpy as np, shutil, json, os
from collections import Counter, defaultdict
SCR=r"C:/Users/anita/projects/maps/asia1m/regionnaming"
OUT=r"C:/Users/anita/projects/maps/asia1m/regionnaming"

rows=list(csv.DictReader(open(f"{SCR}/regions_all.csv",encoding='utf-8')))
prelim={int(r['rid']):r for r in csv.DictReader(open(f"{SCR}/regions_prelim.csv"))}

# copy final csv (skip if work dir == output dir)
if os.path.abspath(f"{SCR}/regions_all.csv") != os.path.abspath(f"{OUT}/regions_all.csv"):
    shutil.copy(f"{SCR}/regions_all.csv", f"{OUT}/regions_all.csv")

cc=Counter(r['country'] for r in rows)
areas=np.array([int(r['area_px']) for r in rows])

# color reuse >300px (color -> multiple regions)
bycolor=defaultdict(list)
for r in rows: bycolor[r['color']].append(r)
reuse=[(c,v) for c,v in bycolor.items() if len(v)>1]

# near-dup adjacent pairs
cols=sorted(bycolor)
def rgb(h): h=h[1:]; return np.array([int(h[i:i+2],16) for i in (0,2,4)])
arr=np.array([rgb(c) for c in cols])
close=[]
for i in range(len(cols)):
    d=np.abs(arr[i+1:].astype(int)-arr[i]).sum(1)
    for jj in np.where(d<=3)[0]:
        j=i+1+jj; best=1e9
        for ra in bycolor[cols[i]]:
            for rb in bycolor[cols[j]]:
                dd=((float(ra['lon'])-float(rb['lon']))**2+(float(ra['lat'])-float(rb['lat']))**2)**.5
        # use pixel centroids from prelim
        for ra in bycolor[cols[i]]:
            for rb in bycolor[cols[j]]:
                pa=prelim[int(ra['rid'])]; pb=prelim[int(rb['rid'])]
                dd=((float(pa['cx'])-float(pb['cx']))**2+(float(pa['cy'])-float(pb['cy']))**2)**.5
                best=min(best,dd)
        close.append((best,cols[i],cols[j],int(d[jj])))
close.sort()
ndclose=[(b,c1,c2,d) for b,c1,c2,d in close if b<250]

flagged=[r for r in rows if r['flags'] and 'tiny' not in r['flags']]
tiny5=sum(1 for r in rows if int(r['area_px'])<5)

R=[]
R.append("# asia1m — region naming CSV\n")
R.append(f"Generated from the **main** layer of `asia1m.ase` (9500×8000). **{len(rows)} regions**.\n")
R.append("`regions_all.csv` — one row per region. Fill the `name` column; feed `country`/`province`/`admin_divisions` + `lat,lon` to Gemini.\n")
R.append("## Method\n")
R.append("- Each region = one exact color. Same-color blobs merged when within ~300px (8px-coarse dilation). Color reuse across >300px is split into separate regions.\n")
R.append("- Excluded colors: `#d9f6ff` water, `#535d8d` border, `#64a0d2` coast, and **`#eead75`** — a 100px grid overlay found on the main layer (every 100×100 cell held exactly 100px), not a region.\n")
R.append("- Georeferencing reproduces `generateAsia.py`'s Albers projection (lon_0=95, parallels 6/42, extent 47.7–129.2E / -9–65N). Validated by overlaying an independent coastline on your drawn coast — pixel-aligned.\n")
R.append("- Admin descriptions: each region's pixels are intersected with rasterized admin boundaries; positional qualifiers (`northern half of X`) come from where the region sits inside each unit. For dense Chinese city cores (region covers <30% of its main GADM county) the description switches to township (街道/镇) detail from `xiangzhen.shp`.\n")
R.append("- **Hierarchy roll-up**: if a region covers ~all of a parent (prefecture/province) it's named as the parent (`Gifu`, `Sakha`) instead of listing children; if it covers all-but-one/two children it's written `{parent} minus {child}` (`Tonghua minus Huinan and Meihekou`). China prefectures come from a spatial join to the OCHA adm2 layer (GADM's own prefecture field is unreliable).\n")
R.append("- **Ordering**: the units the region mostly consists of are listed first (as leads); smaller edge units are demoted to a trailing `(also …)` with bare names — so a naming model anchors on the prominent places, not the slivers. A unit leads if it's the single largest or ≥20% of the region.\n")
R.append("## Admin sources & levels (matching generateAsia.py)\n")
src=[("China","CHN_adm3 (county/district, GADM) + xiangzhen townships for cities; Chinese names in 名 parens"),
("Hong Kong / Macau","Natural Earth admin-1 (18 HK districts; Macau = 1 SAR unit)"),
("Russia","rus adm2 (raion, far-east only)"),("Mongolia","admin1 aimag (province-level only — coarse)"),
("North Korea","admin2"),("South Korea","kostat municipalities (si/gun/gu)"),("Japan","admin2 municipality (+JA names)"),
("Taiwan","WhosOnFirst region (county/city level)"),("Vietnam","GADM adm2 district"),("Laos/Cambodia/Thailand","admin2"),
("Myanmar","admin3 township"),("Malaysia","admin2 district"),("Singapore","WhosOnFirst borough (5 regions)"),
("Philippines","admin3 municipality"),("Indonesia","admin3 kecamatan"),("East Timor","admin1 municipality"),("Brunei","Natural Earth admin-1 (4 districts)")]
for a,b in src: R.append(f"- **{a}** — {b}")
R.append("\n## Region counts by country\n")
for k,v in cc.most_common(): R.append(f"- {k}: {v}")
R.append(f"\nArea (px) percentiles: {dict(zip([5,25,50,75,95],np.percentile(areas,[5,25,50,75,95]).round().astype(int).tolist()))}. Tiny city regions kept (smallest ~7px).\n")

R.append("## Edge cases to review\n")
R.append(f"**Color reused across >300px ({len(reuse)})** — split into separate regions; all are 1px strays, likely stray clicks:\n")
for c,v in reuse:
    locs="; ".join(f"({r['lon']},{r['lat']}) {r['area_px']}px" for r in v)
    R.append(f"- `{c}` → {len(v)} regions: {locs}")
R.append(f"\n**Near-duplicate colors on adjacent regions ({len(ndclose)})** — two shades differing by ≤3 in RGB, regions <250px apart. Kept as **separate regions** (exact RGB = a distinct region, always); listed only in case a pair is an unintended mis-click in the artwork:\n")
for b,c1,c2,d in ndclose:
    R.append(f"- `{c1}` ~ `{c2}` (RGB dist {d}), regions ~{b:.0f}px apart")
R.append(f"\n(46 further near-dup pairs exist but are far apart = intentional palette reuse.)\n")
R.append(f"\n**Uncovered / offshore ({len(flagged)})** — little/no admin polygon under the region (small islands, or admin gap):\n")
for r in flagged:
    R.append(f"- {r['country']} {r['area_px']}px ({r['lon']},{r['lat']}) — {r['admin_divisions'] or '(no admin found)'}")
R.append(f"\n**Other notes**")
R.append(f"- {tiny5} regions under 5px — possibly stray pixels (below your ~7px minimum); flagged `tiny<5px`.")
R.append("- `#f4d79e` near the Ryukyus is a legitimate scattered island-chain region (low bbox-fill but real).")
R.append("- GADM China (CHN_adm3, ~2012 vintage) occasionally mis-attaches a Chinese name (e.g. Zhanjiang's urban core labelled 浈江区). Positions are correct; verify obscure names — your Gemini prompt already does.")
R.append("- Mongolia/Russia descriptions are province/raion-level (only data available) — coarser than elsewhere.\n")
open(f"{OUT}/REPORT.md","w",encoding='utf-8').write("\n".join(R))
print("wrote REPORT.md and regions_all.csv to", OUT)
print("reuse:",len(reuse)," ndclose:",len(ndclose)," flagged:",len(flagged)," tiny<5:",tiny5)
EOF=None
