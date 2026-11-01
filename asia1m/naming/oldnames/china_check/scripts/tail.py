# second half of datacheck.py (exec'd from it, shares its globals)
want_xy = [(k, R[k]["bbox"]) for k in want]
def near_wanted(box, pad=300):
    return [k for k, bb in want_xy if bbox_dist(box, bb) <= pad or any(bbox_dist(box, c) <= pad for c in copies_of(k))]

# inset frames: big pieces of the insets layer
flab, nf = ndi.label(inset_mask)
FRAMES = []
for i, sl in enumerate(ndi.find_objects(flab), 1):
    if sl is None: continue
    hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
    if hh * ww >= 3000 and (flab[sl] == i).mean() >= 0.3: FRAMES.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
del flab
print("inset frames:", FRAMES)
json.dump(dict(ext=ext, frames=FRAMES), open(os.path.join(S, "extent.json"), "w"))

def in_frame(box):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(f[0] <= cx <= f[2] and f[1] <= cy <= f[3] for f in FRAMES)
def above_frame(box):
    return any(f[0] - 10 <= box[0] and box[2] <= f[2] + 10 and 0 <= f[1] - box[3] <= 40 for f in FRAMES)
def beside_frame(box):
    return any((0 <= f[0] - box[2] <= 120 or 0 <= box[0] - f[2] <= 120) and f[1] - 20 <= box[1] <= f[3] for f in FRAMES)

rid2key = {v: k for k, v in key2rid.items()}
def region_under(box):
    x0, y0, x1, y1 = box
    if in_frame(box):
        sub = ins_rgb[y0:y1, x0:x1].reshape(-1, 3); msk = inset_mask[y0:y1, x0:x1].reshape(-1)
        for c, _ in collections.Counter(map(tuple, sub[msk].tolist())).most_common():
            h_ = hexc(c)
            if h_ in by_color: return min(by_color[h_], key=lambda r: bbox_dist(box, r["bbox"]))["key"]
        return None
    for v, _ in collections.Counter(np.asarray(rid[y0:y1, x0:x1]).reshape(-1).tolist()).most_common():
        if v in rid2key: return rid2key[v]
    return None

def off_region(k, box, pad=40):
    X0, Y0 = max(box[0] - pad, 0), max(box[1] - pad, 0); X1, Y1 = min(box[2] + pad, W), min(box[3] + pad, H)
    if k in key2rid and (np.asarray(rid[Y0:Y1, X0:X1]) == key2rid[k]).any(): return False
    col = np.array([int(R[k]["color"][i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
    return not (ins_rgb[Y0:Y1, X0:X1] == col).all(axis=2).any()

def looks_like_n_chars(crop, n):
    h, w = crop.shape
    for s in list(range(10, 17)) + list(range(20, 33)):
        if abs(h - s) <= 2 and abs(w - (n * s + (n - 1))) <= n + 1: return True
        if abs(w - s) <= 2 and abs(h - (n * s + (n - 1))) <= n + 1: return True
    return False

# pieces -> keys / labels
for h_ in her:
    k, exact = colour_owner(h_["colour"], h_["box"])
    h_["owner"], h_["exact"] = k, exact
    hh, ww = h_["crop"].shape
    h_["key"] = hh < 9 or (ww <= 6 and hh <= 10)

def split_list(crop, box):
    """a block of stacked list entries in one colour ('d. 未央' over 'e. 曲江'): cut at blank rows
    into bands, when every band is a list line (a key at its left)"""
    if crop.shape[0] < 20: return [(crop, box)]
    rows = crop.any(axis=1)
    bands, start = [], None
    for y, r in enumerate(list(rows) + [False]):
        if r and start is None: start = y
        if not r and start is not None: bands.append((start, y)); start = None
    if len(bands) < 2: return [(crop, box)]
    out = []
    for y0, y1 in bands:
        b = crop[y0:y1]
        xs = np.flatnonzero(b.any(axis=0))
        if y1 - y0 < 9 or strip_key(b[:, xs.min():xs.max() + 1]) is None: return [(crop, box)]
        out.append((b[:, xs.min():xs.max() + 1], (box[0] + xs.min(), box[1] + y0, box[0] + xs.max() + 1, box[1] + y1)))
    return out

labels = []
n_split = 0
for h_ in her:
    if h_["key"]: continue
    parts = split_list(h_["crop"], h_["box"])
    n_split += len(parts) > 1
    for crop, box in parts:
        labels.append(dict(box=box, colour=h_["colour"], owner=h_["owner"], exact=h_["exact"], crop=crop, pieces=len(parts),
                           px=int(crop.sum())))
print(len(labels), "labels,", sum(h["key"] for h in her), "keys and marks,", n_split, "list blocks split")

from numpy.lib.stride_tricks import sliding_window_view as swv
CHAR_FONTS = [i for i in range(len(F)) if F[i][0] != "unifont"]
def score_chars(crop, zh):
    """each character of zh found on its own anywhere in the crop (any of the 12 px fonts):
    returns (min per-character IoU, share of the label's ink explained, per-char scores)"""
    if not zh or len(zh) > 8: return 0.0, 0.0, []
    P = np.zeros((crop.shape[0] + 24, crop.shape[1] + 24), bool); P[12:-12, 12:-12] = crop
    explained = np.zeros_like(P)
    per = []
    for ch in zh:
        best = (0.0, None)
        for fi in CHAR_FONTS:
            g = gl(fi, ch)
            if g is None: continue
            gh, gw = g.shape
            if P.shape[0] < gh or P.shape[1] < gw: continue
            win = swv(P, (gh, gw))                       # (Y, X, gh, gw)
            inter = np.einsum("yxij,ij->yx", win.astype(np.int16), g.astype(np.int16))
            ink = win.sum(axis=(2, 3))
            iou = inter / (g.sum() + ink - inter + 1e-9)
            y, x = np.unravel_index(np.argmax(iou), iou.shape)
            if iou[y, x] > best[0]: best = (float(iou[y, x]), (y, x, g))
        per.append(round(best[0], 2))
        if best[1]:
            y, x, g = best[1]; explained[y:y + g.shape[0], x:x + g.shape[1]] |= g
    expl = (explained & P).sum() / max(P.sum(), 1)
    return min(per) if per else 0.0, float(expl), per

def read_label(L):
    crop, k, box = L["crop"], L["owner"], L["box"]
    sk = strip_key(crop)
    if sk is not None and sk.shape[0] < 9: sk = None
    L["listkey"] = sk is not None
    def sc(z):
        a, la = score(crop, z)
        if sk is not None:
            b, lb = score(sk, z)
            if b > a: return b, lb + " (key)"
        return a, la
    L["own"], L["own_layout"] = sc(want[k]) if k in want else (0.0, None)
    L["reads"] = L["alt"] = None; L["best"] = (L["own"], k, L["own_layout"])
    if L["own"] >= 0.8: L["reads"] = k; return
    for j in near_wanted(box):
        if j == k: continue
        s, ln = sc(want[j])
        if s > L["best"][0]: L["best"] = (s, j, ln)
    if L["best"][0] >= 0.8: L["reads"] = L["best"][1]; return
    if k:
        for n in altnames(k):
            s, ln = sc(n)
            if s >= 0.8: L["alt"] = n; L["alt_layout"] = ln; return
    # character by character, wherever each sits
    if k in want:
        c0 = sk if sk is not None else crop
        L["chars"] = score_chars(c0, want[k])
        mn, ex, per = L["chars"]
        if mn >= 0.6 and ex >= 0.8: L["reads"] = k; L["own_layout"] = "chars %s, %.2f of ink" % (per, ex); return
        for j in near_wanted(box):
            if j == k or len(want[j]) != len(want[k]) and abs(len(want[j]) - len(want[k])) > 1: continue
            mn2, ex2, per2 = score_chars(c0, want[j])
            if mn2 >= 0.7 and ex2 >= 0.85 and mn2 > mn + 0.15:
                L["reads"] = j; L["best"] = (mn2, j, "chars %s" % per2); return
        for n in altnames(k):
            mn2, ex2, per2 = score_chars(c0, n)
            if mn2 >= 0.7 and ex2 >= 0.85 and mn2 > mn + 0.15: L["alt"] = n; return

for i, L in enumerate(labels):
    read_label(L)
    b = L["box"]
    big = b[3] - b[1] >= 20 and b[2] - b[0] >= 40
    scaled = any(t in (L.get("own_layout") or "") + (L.get("alt_layout") or "") for t in ("2x", "3x"))
    L["role"] = ("title" if big and (above_frame(b) or (in_frame(b) and (scaled or L["alt"])))
                 else "inset" if in_frame(b) else ("list" if (L["listkey"] or beside_frame(b)) else "main"))
    L["under"] = region_under(b)
    if i % 300 == 0: print(i, flush=True)

with open(os.path.join(S, "data_labels.csv"), "w", encoding="utf-8", newline="") as fo:
    wr = csv.writer(fo)
    wr.writerow(["x", "y", "w", "h", "colour", "owner", "owner_name", "owner_zh", "exact_colour", "role", "under",
                 "reads", "reads_zh", "alt", "own_score", "best_score", "best_region", "best_layout", "pieces", "px"])
    for L in sorted(labels, key=lambda L: (L["box"][1], L["box"][0])):
        b = L["box"]; k = L["owner"]
        wr.writerow([b[0], b[1], b[2] - b[0], b[3] - b[1], hexc(L["colour"]), k or "", ename(k) if k else "",
                     zh_of(k) if k else "", L["exact"], L["role"], L["under"] or "", L["reads"] or "",
                     want.get(L["reads"], "") if L["reads"] else "", L["alt"] or "", round(L["own"], 3),
                     round(L["best"][0], 3), L["best"][1] or "", L["best"][2] or "", L["pieces"], L["px"]])

# ---- issues
issues = []
def add(kind, key, saw, box, conf, note=""):
    issues.append(dict(kind=kind, key=key or "", name=ename(key) if key else "", zh=(zh_of(key) if key else ""),
                       native=(nonhan.get(key, {}).get("native", "") if key else ""), saw=saw,
                       x=int(box[0]), y=int(box[1]), w=int(box[2] - box[0]), h=int(box[3] - box[1]),
                       found="data", confidence=conf, note=note))
def bxs(L):
    b = L["box"]; return "%s at %d,%d %dx%d" % (L["role"], b[0], b[1], b[2] - b[0], b[3] - b[1])
def union(Ls):
    return (min(L["box"][0] for L in Ls), min(L["box"][1] for L in Ls), max(L["box"][2] for L in Ls), max(L["box"][3] for L in Ls))
def what(L):
    if L["reads"]: return "reads " + want[L["reads"]]
    if L["alt"]: return "reads " + L["alt"]
    return "unread"

own_labels = collections.defaultdict(list); reads_labels = collections.defaultdict(list)
for L in labels:
    if L["owner"]: own_labels[L["owner"]].append(L)
    if L["reads"]: reads_labels[L["reads"]].append(L)

for k in CN:
    mine = own_labels.get(k, [])
    nh = k in nonhan and nonhan[k].get("native")
    stray = [L for L in mine if L["reads"] and L["reads"] != k]
    good = [L for L in mine if not (L["reads"] and L["reads"] != k)]
    titles = [L for L in good if L["role"] == "title"]
    good = [L for L in good if L["role"] != "title"]
    for L in stray:
        j = L["reads"]
        j_has = [M for M in own_labels.get(j, []) if M["reads"] == j]
        add("wrong_colour", k, "%s %s (%s %s) but is in the colour of %s %s" % (bxs(L), what(L), j, ename(j), k, ename(k)),
            L["box"], "high" if not j_has else "medium",
            ("%s has no label of its own colour" % ename(j)) if not j_has else
            ("%s already has its own label, so this may be %s's label with the wrong characters" % (ename(j), ename(k))))
    if not good:
        other = [L for L in reads_labels.get(k, []) if L["owner"] != k]
        if other: continue   # reported as wrong_colour from the other side (or no-colour pass)
        if any(L["owner"] is None and L["under"] == k and hexc(L["colour"]) not in by_color for L in labels):
            continue         # a label in no region's colour sits on it: reported as wrong_colour
        if stray: continue   # its colour is used, by another name: reported above
        r = R[k]
        add("missing", k, "no label in this region's colour" + (" (main map or inset)" if copies_of(k) else ""),
            r["bbox"], "high" if (zh_of(k) or nh) and r["area_px"] >= 150 else ("medium" if r["area_px"] >= 150 else "low"),
            "area %d px%s" % (r["area_px"], "" if zh_of(k) else "; no zh in all_names.csv"))
        continue
    # unread pieces of one label (a script with gaps wider than 6 px, Arabic or Tibetan) count once
    groups_ = []
    for L in good:
        for g_ in groups_:
            if not L["reads"] and all(not M["reads"] for M in g_) and any(bbox_dist(L["box"], M["box"]) <= 20 for M in g_):
                g_.append(L); break
        else:
            groups_.append([L])
    good = [g_[0] if len(g_) == 1 else dict(g_[0], box=union(g_)) for g_ in groups_]
    if len(good) >= 2:
        roles = sorted(L["role"] for L in good)
        pair = len(good) == 2 and roles in (["inset", "main"], ["inset", "list"], ["list", "main"])
        add("duplicate", k, "%d labels in this colour: %s" % (len(good), "; ".join(bxs(L) + " " + what(L) for L in good)),
            union(good), "medium" if pair else "high",
            ("%s pair: hers to decide" % " + ".join(roles)) if pair else "")
    for L in good:
        if nh: continue
        z = zh_of(k)
        if L["reads"] == k:
            if L["role"] == "main" and off_region(k, L["box"]):
                add("other", k, "right characters, more than 40 px from its region (%s)" % bxs(L), L["box"], "medium")
            continue
        if not z: continue
        if L["alt"]:
            add("wrong_characters", k, "%s reads %s, another name in this region's area; all_names.csv has %s" % (bxs(L), L["alt"], z),
                L["box"], "medium")
            continue
        n_ok = looks_like_n_chars(L["crop"], len(z))
        ch_ = L.get("chars") or (0, 0, [])
        weak = [z[i] for i, s in enumerate(ch_[2]) if s < 0.6]
        if ch_[2] and min(ch_[2]) >= 0.9 and 0.3 <= ch_[1] <= 0.65:
            add("duplicate", k, "%s holds %s twice (every character found, but only %.0f%% of the ink explained by one copy)" %
                (bxs(L), z, 100 * ch_[1]), L["box"], "medium", "two copies of the name drawn touching or overlapping")
            continue
        kept = "".join(z[i] for i, s in enumerate(ch_[2]) if s >= 0.9)
        if ch_[2] and kept and len(kept) < len(z) and ch_[1] >= 0.9:
            add("wrong_characters", k, "%s reads %s, a shortening of %s (all its ink explained)" % (bxs(L), kept, z), L["box"], "low",
                "likely deliberate; all_names.csv could take the short form")
            continue
        add("wrong_characters", k, "%s does not read as %s (whole-label match %.2f; per character %s, %.0f%% of ink explained%s)%s" %
            (bxs(L), z, L["own"], ch_[2], 100 * ch_[1], ("; weakest " + "".join(weak)) if weak else "",
             "; the size fits %d characters" % len(z) if n_ok else "; the size does not fit %d characters" % len(z)),
            L["box"], "medium" if (weak and ch_[1] < 0.8) else "low")

for L in labels:
    if L["owner"] is None:
        u = L["under"]
        if not (L["reads"] in CNs or (u in CNs) or in_frame(L["box"])): continue
        j = L["reads"]
        if not j and hexc(L["colour"]) in by_color:
            add("other", u, "%s: pixels in a region's own fill colour %s on the text layer (not a label; check they are meant, e.g. a border patch)" %
                (bxs(L), hexc(L["colour"])), L["box"], "low")
            continue
        add("wrong_colour", j or u, "%s in colour %s, which is no region's label colour%s" %
            (bxs(L), hexc(L["colour"]), (" (%s; %s %s's label colour is %s)" % (what(L), j, ename(j), hexc(dark(R[j]["color"])))) if j else ""),
            L["box"], "high" if j else "medium", "" if j else "unread; sits on %s %s" % (u or "", ename(u) if u else ""))
    elif L["owner"] in CNs and not L["exact"]:
        add("other", L["owner"], "%s colour %s is 1 off this region's label colour %s" % (bxs(L), hexc(L["colour"]), hexc(dark(R[L["owner"]]["color"]))),
            L["box"], "low")
    elif L["owner"] not in CNs and L["reads"] in CNs:
        j = L["reads"]
        add("wrong_colour", j, "%s %s but is in the colour of %s %s (%s)" % (bxs(L), what(L), L["owner"], ename(L["owner"]), R[L["owner"]]["country"]),
            L["box"], "high")
    elif L["owner"] in CNs and L["reads"] is None and L["owner"] not in want and L["role"] != "title":
        pass

with open(os.path.join(S, "data_issues.csv"), "w", encoding="utf-8", newline="") as fo:
    wr = csv.DictWriter(fo, ["kind", "key", "name", "zh", "native", "saw", "x", "y", "w", "h", "found", "confidence", "note"])
    wr.writeheader(); wr.writerows(issues)
print(collections.Counter((i["kind"], i["confidence"]) for i in issues))
print("China regions with a label in their colour:", sum(1 for k in CN if own_labels.get(k)), "of", len(CN))
