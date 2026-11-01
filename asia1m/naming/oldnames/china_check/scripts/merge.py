"""merge the data check (data_issues.csv) with the visual sweep notes below; write the china_check outputs"""
import os, csv, json, math, collections, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
S = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:\Users\anita\projects\maps\asia1m\naming\oldnames\china_check"
NAMING = r"C:\Users\anita\projects\maps\asia1m\naming"
os.makedirs(os.path.join(OUT, "crops"), exist_ok=True)
names = {r["key"]: r for r in csv.DictReader(open(os.path.join(NAMING, "all_names.csv"), encoding="utf-8-sig"))}
R = {r["key"]: r for r in json.load(open(os.path.join(NAMING, "regions.json"), encoding="utf-8"))}
def zh(k): return (names.get(k) or {}).get("zh", "")
nonhan_rows = {r["key"]: r["native"] for r in csv.DictReader(open(os.path.join(NAMING, "nonhan_regions.csv"), encoding="utf-8-sig"))}
def nm(k):
    n = (names.get(k) or {}).get("name")
    return n or (R[k]["units"][0]["name"] if k in R and R[k]["units"] else "")

# ---- visual sweep notes: (kind, key, what I saw, box, confidence, data issues it replaces "kind:key;...")
V = [
 ("missing", "#f9eaf5", "large pale pink region west of Ghulja (Yining) with no label at all on textlocal", (4083, 2280, 396, 217), "high", "missing:#f9eaf5"),
 ("wrong_characters", "#5bd7f4", "label reads 晋城 (Jincheng in Shanxi); the region is Lanzhou's 金城, so 晋 should be 金", (6205, 3228, 23, 11), "high", "wrong_characters:#5bd7f4"),
 ("wrong_colour", "#e8dc43", "Xi'an list entry 'e. 曲江' is drawn in Weiyang's colour (#b1b31f, same as 'd. 未央'); Qujiang's label colour is #aea533 and nothing on the layer has it", (6168, 3005, 38, 14), "high", "wrong_colour:#ebee29"),
 ("wrong_characters", "#6d9bdc", "Shanghai list entry o: the key 'o.' and the inset key o are in Minhang's colour (the key sits on Minhang), but the name next to it is 崇明 in Chongming's colour; the entry probably wants 闵行 in Minhang's colour", (8288, 3549, 38, 14), "high", "missing:#6d9bdc"),
 ("duplicate", "#83bdfa", "崇明 on the main map (7760,3548) and on the Shanghai inset (8183,3416); the third copy is list entry o, reported under Minhang", (7760, 3416, 446, 143), "medium", "duplicate:#83bdfa"),
 ("duplicate", "#84f6cc", "烟台 drawn twice in one colour: once over the city core and once just above it in the sea", (7629, 2905, 33, 24), "high", "duplicate:#84f6cc"),
 ("duplicate", "#9cf8bb", "黄岛 drawn twice, offset by about 10 px (one on land, one in the sea)", (7549, 3102, 48, 17), "high", "duplicate:#9cf8bb"),
 ("illegible", "#7ce9d1", "Qingdao core: vertical 青岛 (in Fushan's colour), vertical 城阳 and 李沧 and keys a-d all overlap the tiny cores and each other; 青岛 is in Fushan's colour while list entry a. 青岛 is in Qingdao's", (7555, 3055, 62, 55), "medium", "wrong_colour:#7ce9d1"),
 ("duplicate", "#94e5d4", "李沧 written on the map (vertical, crossing key c) and again as list entry c. 李沧", (7593, 3061, 58, 32), "high", "duplicate:#94e5d4"),
 ("duplicate", "#88f0bb", "城阳 written on the map (vertical, across water) and again as list entry d. 城阳", (7558, 3066, 93, 39), "medium", "duplicate:#88f0bb"),
 ("duplicate", "#c1f47d", "two labels in Lüshun's colour: 大连 on the peninsula and 旅顺 below it in the sea; the Dalian core has its own 大连 too, so 大连 shows twice side by side", (7588, 2770, 28, 42), "medium", "wrong_colour:#c1f47d"),
 ("wrong_characters", "#9eee6a", "label reads 大連湾 with the Japanese/traditional 連; simplified is 大连湾", (7585, 2741, 35, 11), "medium", "wrong_characters:#9eee6a"),
 ("wrong_characters", "#b4f47d", "label 锦州 is right for Jinzhou, Liaoning; all_names.csv gives this region 金州, which belongs to the Dalian district (#b4f098, labelled 金州 correctly). Fix all_names.csv, not the map", (7541, 2541, 14, 23), "low", "wrong_characters:#b4f47d"),
 ("wrong_characters", "#bfdece", "label reads 興京 with the traditional 興; simplified is 兴京", (7822, 2501, 23, 11), "medium", "wrong_characters:#bfdece"),
 ("wrong_characters", "#8990df", "on the Shanghai inset the Taicang region is labelled 千灯; Qiandeng has its own 千灯 a little south, so 千灯 shows twice and 太仓 is missing", (8096, 3443, 23, 11), "high", "wrong_colour:#8990df"),
 ("wrong_characters", "#7fc5f5", "on the main map the Nanhui region (Pudong) is labelled 奉贤 in Nanhui's colour; the inset has 南汇 and 奉贤 as separate labels", (7792, 3617, 23, 11), "medium", "wrong_colour:#7fc5f5"),
 ("wrong_characters", "#c48cd8", "Mogan region (Yuhang + Deqing) is labelled 杭州 on the main map; the Hangzhou core has its own label/key", (7642, 3693, 11, 23), "medium", "wrong_colour:#c48cd8@7642"),
 ("wrong_characters", "#c48cd8", "Mogan region (Yuhang + Deqing) is labelled 杭州 on the Hangzhou inset too, next to the core's own 杭州", (7940, 3715, 11, 23), "medium", "wrong_colour:#c48cd8@7940"),
 ("duplicate", "#c2afdc", "高邮 written twice side by side: vertical in the big Gaoyou region (#c2afdc, no name in all_names.csv) and horizontal for the 49 px sliver #c6b4e7 that all_names.csv calls Gaoyou", (7560, 3420, 32, 50), "medium", "wrong_colour:#c2afdc"),
 ("wrong_colour", "#e69acd", "丹阳 is drawn in the colour of #e69adf, a 10 px sliver, instead of Danyang's own #e69acd (label colour #ad749a); reads fine on the map", (7578, 3540, 23, 11), "high", "wrong_colour:#e69adf"),
 ("wrong_colour", "#4ee10a", "Guiyang list entry c. 南明 is in #50a91b; the Nanming core (#4ee10a, 43 px) has label colour #3ba908 and no other ink in it, so the core is effectively unlabelled", (6558, 4286, 31, 11), "high", "wrong_colour:#c2edae;missing:#4ee10a"),
 ("wrong_characters", "#c2edae", "region labelled 红枫 (Hongfeng lake); it is mostly Qingzhen and Pingba, so 红枫 fits. all_names.csv gives it 南明, which belongs to the core #4ee10a. Fix all_names.csv", (6494, 4273, 11, 23), "low", "wrong_characters:#c2edae"),
 ("wrong_characters", "#dbd4e9", "Hegang region labelled 兴安; the big 3x 兴安 label sits on Nenjiang to the west, so 兴安 appears twice on neighbouring regions", (8106, 1746, 23, 11), "medium", "wrong_characters:#dbd4e9"),
 ("other", "#d3ddf4", "Turpan is labelled in Uyghur script (تۇرپان) but is not in nonhan_regions.csv; fine if meant, else add it to the list", (4952, 2461, 30, 13), "low", "wrong_characters:#d3ddf4"),
 ("other", "#e6ccf9", "Aksu is labelled in Uyghur script (ئاقسۇ) but is not in nonhan_regions.csv; fine if meant, else add it to the list", (4316, 2629, 27, 12), "low", "wrong_characters:#e6ccf9"),
 ("wrong_characters", "#ceb1fc", "label reads 邵水; all_names.csv has 砂石, which looks like a bad match for 'Shaoshui'. Fix all_names.csv", (6985, 4153, 11, 23), "low", "wrong_characters:#ceb1fc"),
 ("wrong_characters", "#b5dfd0", "label reads 耒水 (the Lei river); all_names.csv has 波水. Probably fix all_names.csv", (7119, 4261, 23, 11), "low", "wrong_characters:#b5dfd0"),
 ("wrong_characters", "#b2e5e5", "label reads 灌河 (the Guan river); all_names.csv has 关河. Probably fix all_names.csv", (7502, 3312, 23, 11), "low", "wrong_characters:#b2e5e5"),
 ("wrong_characters", "#fcb184", "label reads 海坛 (old name of Pingtan); all_names.csv has 平潭", (7730, 4256, 11, 23), "low", "wrong_characters:#fcb184"),
 ("wrong_characters", "#d0a8e9", "label reads 九嶷 (the mountain); all_names.csv has 九疑亭", (6985, 4344, 23, 11), "low", "wrong_characters:#d0a8e9"),
 ("wrong_characters", "#7beab5", "list entry reads 临沂 (the city); all_names.csv has 临邑, a county in Dezhou far to the north. Probably fix all_names.csv", (7399, 3215, 23, 11), "low", "wrong_characters:#7beab5"),
 ("missing", "#ed96ea", "one 新北 label straddles the two Xinbei regions: 新 on #ed96ea, 北 on #e496ed; #ed96ea has no ink of its own colour", (7607, 3530, 22, 29), "low", "missing:#ed96ea"),
 ("illegible", "#7464e0", "Harbin core: keys c and d are dark violet on dark violet and touch the core borders; c is barely visible", (7866, 1972, 30, 18), "medium", ""),
 ("illegible", "#d73125", "Nanjing core: key a is a dark red blob on the red core next to key b; it does not read as a letter", (7513, 3538, 12, 13), "medium", ""),
 ("illegible", "#f12101", "Pearl River Delta inset: keys 1 and 5 sit in neighbouring cores 1 px apart in the same dark red and read as '15' (15 is also a key, at Xintang)", (7380, 4908, 21, 13), "medium", ""),
 ("illegible", "#f0741d", "Xiamen core: keys 9 and 10 overlap each other and the core border", (7589, 4376, 14, 16), "medium", ""),
 ("illegible", "#e24adb", "Hangzhou inset: 杭州 over the dark magenta core is low contrast and a river cuts its 州; 朝晖 above it touches it", (7973, 3738, 32, 30), "medium", ""),
 ("other", "#dee200", "Xi'an inset: the list a-e (西安 小寨 枣园 未央 曲江) has no keys on the inset, so the cores cannot be matched to the list", (6168, 2952, 154, 101), "high", ""),
 ("other", "#ee7709", "Chongqing inset: the list a-f (重庆 江北 南岸 九龙坡 沙坪坝 嘉陵) has no keys on the inset cores", (6081, 3679, 125, 105), "high", ""),
 ("illegible", "#f12101", "Guangzhou list 1-14 (left of the Pearl River inset): rows have no gap, so 黄埔/石井 and 大沥/佛山 touch", (7208, 4834, 52, 262), "low", ""),
 ("illegible", "#db567f", "Shenzhen / Hong Kong list a-v (right of the Pearl River inset): rows have no gap and characters touch row to row", (7660, 4878, 56, 265), "low", ""),
 ("wrong_characters", "#eb3e12", "Guangzhou list entry 2 reads 西關 with the traditional 關; simplified is 西关 (all_names.csv has no zh for Xiguan)", (7222, 4850, 37, 15), "low", ""),
]
# data flags the sweep showed are fine (kind:key[@x])
FALSE = {
 "wrong_characters:#a2b2e7": "齐齐/哈尔 in two lines reads fine",
 "wrong_characters:#b9eedb": "昆嵛 reads fine (another glyph form)",
 "wrong_characters:#e7d59c": "紫金 reads fine (another glyph form), main and inset",
 "wrong_characters:#f19ea6": "藤州 reads fine",
 "wrong_characters:#e4a096": "鉴江 reads fine",
 "wrong_characters:#d3bcf2": "溆浦 reads fine",
 "wrong_characters:#ab8fe8": "海虞 reads fine",
 "wrong_characters:#a2e2af": "本溪 reads fine (溪 from another font)",
 "wrong_characters:#7fe152": "b. 花溪 reads fine",
 "wrong_characters:#e9c493": "安溪 reads fine",
 "wrong_characters:#f3f781": "玉溪 reads fine",
 "wrong_characters:#f0a6a8": "岑溪 reads fine",
 "wrong_characters:#e2b983": "4. 紫帽 reads fine",
 "wrong_characters:#dac900": "the 15x17 piece is a leader arrow from the Zhengzhou core, not a label",
 "duplicate:#dac900": "list 1. 郑州 plus the leader arrow; not a duplicate",
 "duplicate:#eaf0ab": "曹县 drawn staggered, not 曹 twice",
 "wrong_characters:#cdb7ee": "lạng sơn in Vietnamese on a cross-border region that is mostly Vietnamese; all_names.csv zh is just 'Langson'",
 "wrong_characters:#f06969": "武汉 is the Wuhan inset title, drawn in Hankou's colour",
 "duplicate:#f06969": "武汉 is the inset title; Hankou's own label is list entry a. 汉口",
 "wrong_characters:#6bbbf4": "沙依巴克 reads fine",
}
data = list(csv.DictReader(open(os.path.join(S, "data_issues.csv"), encoding="utf-8")))
def dkey(r, with_x=False): return "%s:%s" % (r["kind"], r["key"]) + ("@%s" % r["x"] if with_x else "")
replaced = set()
for v in V:
    for rep in [x for x in v[5].split(";") if x]: replaced.add(rep)
rows, false_seen, border_gaps = [], [], 0
for r in data:
    if "fill colour" in r["saw"]:
        border_gaps += 1; continue
    if dkey(r) in FALSE or dkey(r, True) in FALSE:
        false_seen.append((dkey(r), FALSE.get(dkey(r)) or FALSE.get(dkey(r, True)), r)); continue
    if dkey(r) in replaced or dkey(r, True) in replaced: continue
    note = r["note"]
    if r["kind"] == "wrong_colour" and r["key"] == "#8ae995":
        note = "the sweep shows it on the Haicheng area reading 海城; only the colour is off (the two colours are close)"
    if r["kind"] == "wrong_colour" and r["key"] in ("#97a6da", "#df8680", "#edb0e1", "#77e942", "#95ee6c", "#abee6c", "#f3835a"):
        note = (note + "; " if note else "") + "reads right on the map; only the colour is off"
    if r["kind"] == "missing" and r["confidence"] == "low":
        note = (note + "; " if note else "") + "tiny core or sliver; nothing visible to label in the sweep"
    found, conf = "data", r["confidence"]
    if "another name in this region's area" in r["saw"]:
        found, conf = "both", "low"
        note = "her name for the area, seen in the sweep; all_names.csv could take it"
    elif "a shortening of" in r["saw"]:
        found = "both"
    elif "pair: hers to decide" in note:
        found = "both"
        note = note + "; both copies seen in the sweep"
    rows.append(dict(kind=r["kind"], key=r["key"], name=r["name"], zh=r["zh"] or r["native"], saw=r["saw"],
                     x=r["x"], y=r["y"], w=r["w"], h=r["h"], found=found, confidence=conf, note=note))
present = {dkey(r) for r in data} | {dkey(r, True) for r in data}
for kind, key, saw, box, conf, rep in V:
    hit = any(x in present for x in rep.split(";") if x)
    rows.append(dict(kind=kind, key=key, name=nm(key) if key in R else "", zh=zh(key) or (nonhan_rows.get(key) or ""),
                     saw=saw, x=box[0], y=box[1], w=box[2], h=box[3],
                     found="both" if hit else "visual", confidence=conf, note=""))
ORDER = {"high": 0, "medium": 1, "low": 2}
KORDER = {"missing": 0, "wrong_colour": 1, "wrong_characters": 2, "duplicate": 3, "illegible": 4, "other": 5}
rows.sort(key=lambda r: (ORDER[r["confidence"]], KORDER[r["kind"]], int(r["y"]), int(r["x"])))
for i, r in enumerate(rows, 1): r["id"] = i
with open(os.path.join(OUT, "issues.csv"), "w", encoding="utf-8-sig", newline="") as f:
    wr = csv.DictWriter(f, ["id", "kind", "key", "name", "zh", "saw", "x", "y", "w", "h", "found", "confidence", "note"])
    wr.writeheader(); wr.writerows(rows)

# ---- overlay
COL = {"missing": (255, 0, 0, 255), "duplicate": (255, 140, 0, 255), "wrong_colour": (200, 0, 200, 255),
       "wrong_characters": (0, 90, 255, 255), "illegible": (0, 170, 0, 255), "other": (0, 0, 0, 255)}
W, H = 9500, 8000
ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
d = ImageDraw.Draw(ov)
for r in rows:
    x, y, w, h = int(r["x"]), int(r["y"]), max(int(r["w"]), 1), max(int(r["h"]), 1)
    x0, y0, x1, y1 = x - 4, y - 4, x + w + 3, y + h + 3
    d.rectangle((x0, y0, x1, y1), outline=COL[r["kind"]], width=2)
font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 22)
lx, ly = 30, 7600
d.rectangle((lx - 10, ly - 10, lx + 330, ly + 6 * 32 + 6), fill=(255, 255, 255, 230), outline=(0, 0, 0, 255), width=2)
for i, (k, c) in enumerate(COL.items()):
    d.rectangle((lx, ly + i * 32, lx + 24, ly + i * 32 + 22), outline=c, width=3)
    d.text((lx + 36, ly + i * 32 - 1), "%s (%d)" % (k, sum(1 for r in rows if r["kind"] == k)), fill=(0, 0, 0, 255), font=font)
ov.save(os.path.join(OUT, "issues.png"), optimize=True)

# ---- crops (2x, 40 px of context, the issue box drawn)
comp = Image.open(os.path.join(S, "composite.png")).convert("RGB")
cd = os.path.join(OUT, "crops")
for f_ in os.listdir(cd): os.remove(os.path.join(cd, f_))
used = collections.Counter()
for r in rows:
    x, y, w, h = int(r["x"]), int(r["y"]), max(int(r["w"]), 1), max(int(r["h"]), 1)
    pad = 40
    if w * h > 250000:  # huge box (main + inset pair): crop each end instead of the whole span
        pad = 30
    X0, Y0, X1, Y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H)
    if (X1 - X0) * (Y1 - Y0) > 900 * 900:  # cap: the two ends of a long pair
        X1, Y1 = min(X0 + 900, W), min(Y0 + 900, H)
    c = comp.crop((X0, Y0, X1, Y1)).resize(((X1 - X0) * 2, (Y1 - Y0) * 2), Image.NEAREST)
    dd = ImageDraw.Draw(c)
    dd.rectangle(((x - X0 - 3) * 2, (y - Y0 - 3) * 2, (x + w - X0 + 2) * 2, (y + h - Y0 + 2) * 2), outline=COL[r["kind"]][:3], width=2)
    base = "%s_%s_%s" % (r["kind"], r["key"].lstrip("#").split("@")[0], (r["name"] or "x").replace("'", "").replace("?", "").replace(" ", ""))
    used[base] += 1
    fn = base + ("_%d" % used[base] if used[base] > 1 else "") + ".png"
    r["crop"] = fn
    c.save(os.path.join(cd, fn))
with open(os.path.join(OUT, "issues.csv"), "w", encoding="utf-8-sig", newline="") as f:
    wr = csv.DictWriter(f, ["id", "kind", "key", "name", "zh", "saw", "x", "y", "w", "h", "found", "confidence", "note", "crop"])
    wr.writeheader(); wr.writerows(rows)
json.dump(dict(false=[(a, b) for a, b, _ in false_seen], border_gaps=border_gaps,
               counts=collections.Counter(r["kind"] for r in rows),
               by_conf=collections.Counter((r["kind"], r["confidence"]) for r in rows).most_common(),
               by_found=collections.Counter(r["found"] for r in rows)), open(os.path.join(S, "merge_summary.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=str)
print(len(rows), "issues", collections.Counter(r["kind"] for r in rows), collections.Counter(r["found"] for r in rows))
print("false alarms dropped:", len(false_seen), "border gaps dropped:", border_gaps)
print("unmatched replaces:", [x for x in replaced if not any(dkey(r) == x or dkey(r, True) == x for r in data)])
