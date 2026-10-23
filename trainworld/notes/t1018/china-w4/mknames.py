"""Wave 3 shown names: names_cn.json proposals with hand fixes, as setnames.py input (names_w3.json),
only where the shown name differs from the plan row's name; reports shown names that repeat across
every registered pack and the wave."""
import collections, json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
TW = r"C:\Users\anita\projects\maps\trainworld"
prop = json.load(open(os.path.join(HERE, "names_cn.json"), encoding="utf-8"))
plan = {r["id"]: r for r in json.load(open(os.path.join(TW, "pipeline", "world_plan.json"), encoding="utf-8"))["metros"]}
reg = json.load(open(os.path.join(TW, "app", "src", "game", "registry.json"), encoding="utf-8"))["cities"]
zn = json.load(open(os.path.join(TW, "app", "src", "game", "zone-names.json"), encoding="utf-8"))
wave = set(open(os.path.join(HERE, "wave_ids.txt")).read().strip().split(","))

ONE = {"bnx", "mnx"}  # one-syllable counties keep "County"
FIX = {
    # a prefecture-level city's centre in a county- or district-led zone (prefpacks.py on the built
    # outlines): the zone takes the city's name
    "hig": ("Qinhuangdao", ["Haigang", "Beidaihe", "Shanhaiguan"]),
    "yak": ("Hulunbuir", ["Hailar", "Yakeshi"]),
    "haw": ("Wuhai", ["Shizuishan", "Haibowan"]),   # Wuhai's and Shizuishan's centres are both in haw's outline
    "mig": ("Tieling", ["Mingshan"]),
    "lsh": ("Siping", ["Lishu"]),
    "dnl": ("Liaoyuan", ["Dongliao"]),
    "hnj": ("Tonghua", ["Baishan", "Hunjiang"]),    # Tonghua's and Baishan's centres are both in hnj's outline
    "fuu": ("Songyuan", ["Fuyu"]),
    "jgu": ("Jixi", ["Qitaihe", "Jiguan"]),         # Jixi's and Qitaihe's centres
    "yia": ("Hegang", ["Yilan"]),
    "hna": ("Shuangyashan", ["Huanan"]),
    "baq": ("Yichun", ["Baiquan"]),                 # Heilongjiang's; Jiangxi's Yichun is yih (two real cities)
    "hix": ("Mudanjiang", ["Hailin"]),
    "cea": ("Lhasa", ["Chamdo", "Nyingchi", "Shannan"]),
    "smz": ("Shigatse", ["Nagqu"]),
    "gah": ("Zhangye", ["Jiuquan", "Jiayuguan", "Dunhuang"]),
    "yoa": ("Jinchang", ["Yongchang"]),
    "yzh": ("Baiyin", ["Yuzhong"]),
    "nng": ("Qingyang", ["Ning County"]),
    "hug": ("Dingxi", ["Huining"]),
    "dao": ("Haidong", ["Datong County"]),          # Shanxi's Datong is dat
    "sez": ("Karamay", ["Shihezi", "Kuytun"]),
    "goc": ("Turpan", ["Altay"]),
    "yif": ("Hami", ["Fukang"]),
    # autonomous prefectures', leagues' and prefectures' seats are county-level cities: the zone keeps
    # its lead's own name, the seat an alias (DECISIONS)
    "gol": ("Golmud", ["Delingha", "Yushu"]),
    "yii": ("Yining", []), "caj": ("Changji", []), "krl": ("Korla", []), "yai": ("Yanji", []),
    "xli": ("Xilinhot", []), "usu": ("Usu", ["Bole"]), "jaa": ("Jalaid", ["Ulanhot"]),
    "aks": ("Aksu", ["Hotan"]), "krg": ("Kargilik", ["Artux"]), "mna": ("Manas", ["Tacheng"]),
    "wud": ("Longnan", ["Wudu", "Hezuo"]), "mnh": ("Minhe", ["Tongren"]), "suo": ("Suizhong", ["Xingcheng"]),
}
# earlier waves' rows given an alias from this wave
EXTRA = {"ans": {"aliases": ["Liaoyang"]}}       # Liaoyang's centre is in Anshan's metro outline
NOALIAS = {"nng", "dao"}  # the plan name is given above with its kind
out = {}
for i in sorted(wave):
    r = plan[i]
    name, al = prop.get(i, r["name"]), []
    if i in ONE:
        name = r["name"] + " County"
    if i in FIX:
        name, al = FIX[i]
    if name != r["name"]:
        own = r["name"] not in al and "," not in r["name"] and i not in NOALIAS and r["name"] != name
        al = al + [r["name"]] if own else al
        out[i] = {"name": name, "aliases": al}
    elif al:
        out[i] = {"aliases": al}
for i, v in EXTRA.items():
    out[i] = v  # setnames.py merges the aliases with the entry's own
json.dump(out, open(os.path.join(HERE, "names_w4.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(out), "entries")
shown = collections.defaultdict(list)
for c in reg:
    if c["id"] in wave or c["id"] in out:
        continue
    shown[(zn.get(c["id"]) or {}).get("name") or c["name"]].append(c["id"])
for i in set(wave) | set(out):
    shown[(out.get(i) or {}).get("name") or (zn.get(i) or {}).get("name") or plan[i]["name"]].append(i)
for n, ids in sorted(shown.items()):
    if len(ids) > 1:
        print("repeat", n, ids)
for i, v in sorted(out.items()):
    print(i, plan[i]["name"], "->", v)
