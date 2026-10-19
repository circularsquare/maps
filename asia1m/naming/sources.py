"""Gather cheap reading for a batch before any agent starts -> batches/<name>/sources.md

    python asia1m/naming/sources.py <batch name>

China only for now. Per region:
  - historical seats from CHGIS v6 (Harvard/Fudan, time series prefecture and county points,
    222 BC-1911): commanderies, prefectures and abolished counties whose seat lies in the region,
    with years. Licence: free for research, no commercial use or redistribution; used here only
    to find candidate names. Files in data/asia1m/chgis/.
  - Chinese Wikipedia, plain text, for each county/district holding 10%+ of a region's people
    and each prefecture holding half: the lead plus history, name and geography sections, which
    is where 古称/别称 and the local rivers and mountains are.
Article text is cached in batches/<name>/_wiki/ so a rerun costs nothing.
"""
import os, sys, json, re, time, urllib.request, urllib.parse
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
RN = os.path.join(HERE, "..", "regionnaming")
sys.path.insert(0, RN)
import numpy as np
import scipy.ndimage as ndi
import pyogrio
from geo import lonlat_to_px

name = sys.argv[1]
B = os.path.join(HERE, "batches", name)
facts = json.load(open(f"{B}/facts.json", encoding="utf-8"))
regions = facts["regions"]
CACHE = f"{B}/_wiki"; os.makedirs(CACHE, exist_ok=True)
UA = {"User-Agent": "asia1m-naming/0.1 (personal map project)"}

# ---- CHGIS seats -> regions (border pixels go to the nearest region, as in facts.py) ----
rid = np.load(f"{RN}/region_id.npy")
H, W = rid.shape
color_rid = {}
main = None
def rid_of(r):
    x0, y0, x1, y1 = r["bbox"]
    sub = rid[y0:y1 + 1, x0:x1 + 1]
    from PIL import Image
    global main
    if main is None:
        Image.MAX_IMAGE_PIXELS = None
        main = np.array(Image.open(f"{RN}/main.png").convert("RGB"))
    c = tuple(int(r["color"][i:i + 2], 16) for i in (1, 3, 5))
    m = np.all(main[y0:y1 + 1, x0:x1 + 1] == c, axis=2)
    ids = sub[m]
    return int(np.bincount(ids[ids > 0]).argmax())
ids = {r["n"]: rid_of(r) for r in regions}
bx0 = min(r["bbox"][0] for r in regions) - 5; by0 = min(r["bbox"][1] for r in regions) - 5
bx1 = max(r["bbox"][2] for r in regions) + 5; by1 = max(r["bbox"][3] for r in regions) + 5
sub = rid[by0:by1, bx0:bx1]
d, (iy, ix) = ndi.distance_transform_edt(sub == 0, return_indices=True)
sub = np.where((sub == 0) & (d <= 3), sub[iy, ix], sub)
n_of = {v: k for k, v in ids.items()}

CH = os.path.join(HERE, "..", "..", "data", "asia1m", "chgis")
seats = defaultdict(lambda: defaultdict(list))   # n -> name -> [(type, beg, end, level, loc)]
for z, shp, level in [("v6_time_pref_pts_utf_wgs84.zip", "v6_time_pref_pts_utf_wgs84.shp", "pref"),
                      ("v6_time_cnty_pts_utf_wgs84.zip", "v6_time_cnty_pts_utf_wgs84.shp", "cnty")]:
    df = pyogrio.read_dataframe(f"zip://{CH}/{z}!{shp}", read_geometry=False)
    x, y = lonlat_to_px(df["X_COOR"].to_numpy(float), df["Y_COOR"].to_numpy(float))
    x = np.round(x).astype(int) - bx0; y = np.round(y).astype(int) - by0
    ok = (x >= 0) & (x < sub.shape[1]) & (y >= 0) & (y < sub.shape[0])
    for i in np.flatnonzero(ok):
        n = n_of.get(int(sub[y[i], x[i]]))
        if n is None: continue
        row = df.iloc[i]
        seats[n][row["NAME_CH"]].append((row["TYPE_CH"], int(row["BEG_YR"]), int(row["END_YR"]), level, row["PRES_LOC"]))

def yr(v): return f"{-v} BC" if v < 0 else str(v)
def seat_lines(n):
    out = []
    items = []
    for nm, spans in seats[n].items():
        spans.sort(key=lambda t: t[1])
        total, reach = 0, None           # years covered by the union of the spans
        for _, b, e, _, _ in spans:
            if reach is None or b > reach: total += e - b; reach = e
            elif e > reach: total += e - reach; reach = e
        pref = any(s[3] == "pref" for s in spans)
        items.append((not pref, -total, nm, spans))
    items.sort(key=lambda t: t[:3])
    for _, negt, nm, spans in items[:18]:
        kinds = "/".join(dict.fromkeys(t for t, *_ in spans if t)) or "?"
        rng = f"{yr(spans[0][1])}-{yr(max(e for _, _, e, _, _ in spans))}"
        out.append(f"- {nm} ({kinds}, {rng}, {-negt} yrs in all) {spans[0][4] or ''}".rstrip())
    return out

# ---- Chinese Wikipedia ----
def api(params):
    u = "https://zh.wikipedia.org/w/api.php?" + urllib.parse.urlencode({**params, "format": "json"})
    for attempt in range(3):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=30))
        except Exception as e:
            time.sleep(2 + attempt * 3)
    return None

KEEP = re.compile(r"历史|沿革|名称|得名|地名|由来|建置|地理|地形|地貌|河流|水文|水系|山脉|自然")
def wiki(title):
    path = os.path.join(CACHE, re.sub(r'[\\/:*?"<>|]', "_", title) + ".json")
    if os.path.exists(path): return json.load(open(path, encoding="utf-8"))
    r = api(dict(action="query", titles=title, redirects=1, prop="pageprops|extracts",
                 explaintext=1, ppprop="disambiguation", variant="zh-cn"))
    if not r or "query" not in r:
        return None                      # network trouble: don't cache, a rerun tries again
    out = None
    p = next(iter(r["query"]["pages"].values()))
    if "missing" not in p:
        out = dict(title=p["title"], disambig="disambiguation" in p.get("pageprops", {}),
                   text=p.get("extract", ""))
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    time.sleep(0.3)
    return out

def resolve(local, prefecture_local):
    """Article for a unit, stepping past disambiguation pages with the prefecture."""
    if not local: return None
    a = wiki(local)
    if a and not a["disambig"]: return a
    for t in [f"{local} ({prefecture_local})", f"{local} ({(prefecture_local or '').rstrip('市地区盟州')}市)"]:
        if prefecture_local:
            a = wiki(t)
            if a and not a["disambig"]: return a
    return None

def trim(text, cap=2200):
    parts = re.split(r"\n(==+ .*? ==+)\n", text)
    out = [parts[0].strip()[:600]]
    for i in range(1, len(parts) - 1, 2):
        head, body = parts[i], parts[i + 1].strip()
        if KEEP.search(head) and body:
            out.append(f"{head.strip('= ').strip()}: {body[:900]}")
    s = "\n".join(out)
    return s[:cap]

def city_name(c):
    return c if (c or "").endswith(("市", "州", "盟", "地区")) else (c or "") + "市"

articles = {}   # title -> trimmed text
region_units = defaultdict(list)
for r in regions:
    if r["country"] != "China": continue
    wanted = []
    # current counties from townships (their names and prefectures are newer than the county layer)
    cnty = defaultdict(float); cnty_city = {}
    for t in r["towns"]:
        cnty[t["county"]] += t["people_share"]; cnty_city[t["county"]] = t["city"]
    for c, s in sorted(cnty.items(), key=lambda t: -t[1]):
        if s >= 0.10: wanted.append((c, cnty_city[c]))
    for u in r["units"]:
        if u["people_share"] >= 0.10 and u["local"] and all(u["local"] != w[0] for w in wanted):
            wanted.append((u["local"], u["prefecture_local"]))
    for p in r["parents"]:
        if p["level"] == "prefecture" and p["people_share"] >= 0.5 and p["local"]:
            wanted.append((p["local"], None))
    for local, pref in wanted:
        a = resolve(local, city_name(pref) if pref else None)
        if a:
            if a["title"] not in articles: articles[a["title"]] = trim(a["text"])
            if a["title"] not in region_units[r["n"]]: region_units[r["n"]].append(a["title"])
        else:
            region_units[r["n"]].append(f"{local} (no article found)")
    print(f"region {r['n']}: {len(region_units[r['n']])} units, {len(seats[r['n']])} historical seats", flush=True)

lines = [f"# Sources for batch {name}", "",
         "Gathered by sources.py. Historical seats are from CHGIS v6 (seat inside the region; years are",
         "when a unit of that name existed there). Articles are Chinese Wikipedia, trimmed to the lead and",
         "the history, name and geography sections.", ""]
for r in regions:
    lines.append(f"## Region {r['n']}")
    sl = seat_lines(r["n"])
    lines.append("Historical seats:" if sl else "Historical seats: none in CHGIS.")
    lines += sl
    lines.append("Articles: " + "; ".join(region_units[r["n"]]))
    lines.append("")
lines.append("# Articles")
for t, txt in articles.items():
    lines += ["", f"## {t}", f"https://zh.wikipedia.org/wiki/{urllib.parse.quote(t)}", txt]
open(f"{B}/sources.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"wrote {B}/sources.md: {len(articles)} articles")
