"""T-3324: drop Chinese place-name matches that do not read as the Latin name.

    python tools/place_names_check.py [--pypinyin <dir>] [--dry]      (run after tools/place_names.mjs, with and without --zones)

place_names.mjs matches a registry city or zone to the Wikidata item with the most links among those whose English
label is its name, within 60 km; for small Chinese places that sometimes picks a prefecture or a namesake (Cao -> 曹国,
Jingxi -> 百色). For a place in China or Taiwan the zh-Hans label's pinyin (no tones, spaces or case) must equal one of
its Latin name's parts, once a trailing county, district or city word is dropped from either; a label that fails loses
all its names, the Japanese one too (same item), and its entry becomes `{"q": null}`, which place_names.mjs does not
match again unless told with --redo. Other countries' names cannot be checked this way and stay.
pypinyin (pure Python, `pip install --target <dir> pypinyin`) is read from --pypinyin when it is not installed.
"""
import json
import os
import re
import sys
import time
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "app", "src")
args = sys.argv[1:]
if "--pypinyin" in args:
    sys.path.insert(0, args[args.index("--pypinyin") + 1])
import itertools  # noqa: E402
from pypinyin import pinyin, Style  # noqa: E402

CHECKED = {"CN", "TW"}
# words a Latin or pinyin name may end in without the other having them
TAIL = ["zizhixian", "zizhizhou", "zizhiqu", "diqu", "xian", "shi", "qu", "zhou", "city", "county", "district", "prefecture", "xiang", "zhen", "meng", "qi"]
# Latin names that are not Hanyu pinyin (Uyghur, Mongol, Wade-Giles, old postal) whose Chinese label was read against the
# zone's place by hand, T-3324: Taipei, Taichung, Kaohsiung, Harbin, Urumqi, Hohhot, Xilinhot, Kuqa, Karamay, Korla,
# Aksu, Yarkant, Kargilik, Karakax, Alxa Left, Jungar, Binhai New Area, Chengguan (Lhasa's district), Guoyang (涡阳), Golmud, Kashgar, Altay, Taitung City, Karub (卡若区)
KEEP = {"tai", "tac", "kao", "har", "uru", "hoh", "xln", "kux", "krm", "klx", "aku", "yrk", "kgi", "krx", "alf", "jug", "bin", "cga", "guo", "god", "ksg", "aty", "tig", "kru", "luo"}
FILES = ["i18n/place-names.json", "i18n/place-names-zones.json"]


def fold(s):
    s = unicodedata.normalize("NFKD", s).lower().replace("ü", "u")
    return re.sub(r"[^a-z]", "", "".join(c for c in s if not unicodedata.combining(c)).replace("ü", "u"))


def variants(s):
    """The folded string, and with each trailing word dropped (repeatedly)."""
    out = {s}
    changed = True
    while changed:
        changed = False
        for v in list(out):
            for t in TAIL:
                if v.endswith(t) and len(v) > len(t) + 1 and v[: -len(t)] not in out:
                    out.add(v[: -len(t)])
                    changed = True
    return out


def latin_parts(name):
    parts = [name] + [p for p in re.split(r",| and | - |/|\(|\)|\[|\]", name) if p.strip()]
    return {v for p in parts for v in variants(fold(p))}


def readings(label):
    """Every folded pinyin reading of a label (heteronyms tried: 泊头 is Botou, 都昌 Duchang), at most 256."""
    labels = [label]
    if "自治" in label and "族" in label:  # 威宁彝族回族苗族自治县 is Weining: every prefix before the first 族 is a candidate
        labels += [label[:i] for i in range(2, label.index("族"))]
    out = set()
    for lab in labels:
        rows = pinyin(lab, style=Style.NORMAL, heteronym=True, errors=lambda x: ["#"])
        for combo in itertools.islice(itertools.product(*rows), 256):
            out.add(fold("".join(combo)))
    return out


def reads_as(label, names):
    want = latin_parts_all(names)
    return any(variants(py) & want for py in readings(label))


def latin_parts_all(names):
    return {v for n in names for v in latin_parts(n)}


def main():
    dry = "--dry" in args
    reg = {c["id"]: c for c in json.load(open(os.path.join(SRC, "game", "registry.json"), encoding="utf-8"))["cities"]}
    hand = json.load(open(os.path.join(SRC, "game", "zone-names.json"), encoding="utf-8"))
    for f in FILES:
        path = os.path.join(SRC, f)
        data = json.load(open(path, encoding="utf-8"))
        kept = dropped = unchecked = 0
        gone = []
        for id_, e in data.items():
            c = reg.get(id_)
            if not c or c.get("country") not in CHECKED or not (e.get("zh-Hans") or e.get("zh-TW")):
                unchecked += 1
                continue
            names = [hand[id_]["name"]] if hand.get(id_, {}).get("name") else [c["name"]]  # the name the game shows
            labels = [e[k] for k in ("zh-Hans", "zh-TW") if e.get(k)]
            if id_ in KEEP or any(reads_as(l, names) for l in labels):
                kept += 1
            else:
                dropped += 1
                gone.append((id_, c["name"], e.get("zh-Hans") or e.get("zh-TW")))
                data[id_] = {"q": None}
        print(f"{f}: {kept} kept, {dropped} dropped, {unchecked} not checked (other countries or no zh label)")
        with open(os.path.join(os.environ.get("TEMP", "."), "place_names_dropped.txt"), "a", encoding="utf-8") as g:
            g.write(f"== {f}\n" + "\n".join("\t".join(x) for x in gone) + "\n")
        if not dry:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as g:
                g.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace('},"', '},\n"').replace("{", "{\n", 1)[:-1] + "\n}\n")
            for attempt in range(10):  # Windows refuses the rename while another process (a dev server) has the file open
                try:
                    os.replace(tmp, path)
                    break
                except PermissionError:
                    time.sleep(1)
            else:
                raise SystemExit(f"{f}: could not replace the file; the new one is {tmp}")


if __name__ == "__main__":
    main()
