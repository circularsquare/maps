"""
Ascentshed -- inverse watershed / prominence-territory partition.

Instead of water descending to basins, "flow" ascends to peaks. Every cell is
labelled by the peak its steepest-ascent path terminates at (the cell's
prominence parent). Basins are then merged into N territories of comparable
integrated VOLUME -- the relief analog of an equal-population map.

Pipeline
  1. terrain  -> grid z (synthetic, or a real DEM via load_dem)
  2. ascent_basins(z)        -> every cell labelled by the peak it climbs to
  3. merge_tree(z, label)    -> key col, prominence, parent peak, volume per peak
  4a. prune(...)             -> greedy "ultras" map (smallest peak absorbed first)
  4b. balanced_partition(..) -> equal-heft map (the point: cut the tree at saddles
                                into N subtrees of near-equal volume)

Volume of a basin = integral over its cells of (height - key_col) clipped at 0,
the dual of how a depression hierarchy tracks impounded water volume.

Both O(n)-per-cell loops from the prototype are gone: basin labelling is
pointer-doubling, volumes are bincounts. The merge-tree sweep is still a single
elevation-ordered union-find pass (inherently sequential, but O(n a(n))).
"""
import numpy as np

NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ----------------------------------------------------------------------
# 2. D8 steepest-ascent basins (vectorized successor + pointer doubling)
# ----------------------------------------------------------------------
def ascent_basins(z, land=None):
    """Label every cell by the peak its steepest-ascent path reaches.

    Returns (label2d, peak_flat_indices). label2d[i,j] is the flat index of the
    peak cell. With ties broken (jittered terrain) the successor graph is a
    forest of trees whose roots are peaks (self-pointers); resolve roots by
    pointer doubling -- O(n log n), fully vectorized.

    If `land` (boolean mask) is given, cells outside it (sea) are labelled -1 and
    take no part: ascent runs on land only so the coastline is a hard boundary and
    no basin reaches across water. Land cells never ascend into the lower sea, so
    masking afterward leaves land basins intact.
    """
    n, m = z.shape
    N = n * m
    idxgrid = np.arange(N).reshape(n, m)
    best = z.copy()
    nxt2d = idxgrid.copy()
    padv = np.pad(z, 1, constant_values=-np.inf)
    padi = np.pad(idxgrid, 1, constant_values=-1)
    for dy, dx in NB:
        sv = padv[1 + dy:1 + dy + n, 1 + dx:1 + dx + m]
        si = padi[1 + dy:1 + dy + n, 1 + dx:1 + dx + m]
        mask = sv > best
        best = np.where(mask, sv, best)
        nxt2d = np.where(mask, si, nxt2d)
    nxt = nxt2d.ravel()
    # pointer doubling: root = repeated nxt[nxt[...]] until fixed point
    root = nxt.copy()
    while True:
        nr = root[root]
        if np.array_equal(nr, root):
            break
        root = nr
    label = root.reshape(n, m)
    if land is not None:
        label = np.where(land, label, -1)
        peaks = np.unique(label[label >= 0])
    else:
        peaks = np.unique(label)
    return label, peaks


# ----------------------------------------------------------------------
# 3. Merge (join) tree of super-level sets -> key col + parent for every peak
#    Component identity is its FOUNDING (highest) cell = peak index.
# ----------------------------------------------------------------------
def merge_tree(z, label):
    """Descending priority-flood union-find. Returns dicts keyed by peak index:
        elev[p]        peak elevation
        keycol[p]      key-col (saddle) elevation, None for global max
        parent[p]      higher peak across the key col, None for global max
        prominence[p]  elev - keycol
        volume[p]      integral over basin p of (h - keycol) clipped at 0
        cellcount[p], sumh[p]   raw basin stats (for partition weighting)
      and gmax (the global-max peak index).
    """
    n, m = z.shape
    flat = z.ravel()
    N = n * m
    order = np.argsort(-flat, kind='stable')          # descending elevation
    parent_dsu = np.full(N, -1)                        # -1 = not yet added
    top = {}                                           # dsu-root -> founding cell

    def find(x):
        r = x
        while parent_dsu[r] != r:
            r = parent_dsu[r]
        while parent_dsu[x] != r:
            parent_dsu[x], x = r, parent_dsu[x]
        return r

    peaks = set(np.unique(label).tolist())
    keycol = {}
    parent = {}
    for idx in order:
        i, j = divmod(idx, m)
        parent_dsu[idx] = idx
        top[idx] = idx
        for dy, dx in NB:
            a, b = i + dy, j + dx
            if not (0 <= a < n and 0 <= b < m):
                continue
            nb = a * m + b
            if parent_dsu[nb] == -1:
                continue
            rn, rr = find(nb), find(idx)
            if rn == rr:
                continue
            ta, tb = top[rr], top[rn]
            winner, loser = (ta, tb) if flat[ta] >= flat[tb] else (tb, ta)
            if loser not in keycol:                    # seal the lower peak here
                keycol[loser] = flat[idx]
                parent[loser] = winner
            parent_dsu[rn] = rr
            top[find(idx)] = winner
    gmax = max(peaks, key=lambda p: flat[p])
    for p in peaks:
        keycol.setdefault(p, None)
        parent.setdefault(p, None)
    keycol[gmax] = None
    parent[gmax] = None
    elev = {p: float(flat[p]) for p in peaks}
    prominence = {p: (elev[p] - keycol[p]) if keycol[p] is not None else elev[p]
                  for p in peaks}

    # vectorized per-basin volume / stats
    lab = label.ravel()
    peak_list = np.array(sorted(peaks))
    pos = {p: k for k, p in enumerate(peak_list)}
    pidx = np.array([pos[x] for x in lab])             # 0..len(peaks)-1 per cell
    col_of_cell = np.array([keycol[p] if keycol[p] is not None else 0.0
                            for p in peak_list])[pidx]
    contrib = np.clip(flat - col_of_cell, 0, None)
    vol_arr = np.bincount(pidx, weights=contrib, minlength=len(peak_list))
    cnt_arr = np.bincount(pidx, minlength=len(peak_list))
    sumh_arr = np.bincount(pidx, weights=flat, minlength=len(peak_list))
    volume = {int(p): float(vol_arr[k]) for k, p in enumerate(peak_list)}
    cellcount = {int(p): int(cnt_arr[k]) for k, p in enumerate(peak_list)}
    sumh = {int(p): float(sumh_arr[k]) for k, p in enumerate(peak_list)}
    return dict(elev=elev, keycol=keycol, parent=parent, prominence=prominence,
                volume=volume, cellcount=cellcount, sumh=sumh, gmax=gmax,
                peaks=peak_list)


# ----------------------------------------------------------------------
# 3b. FAST divide tree on the basin graph (supersedes merge_tree at scale)
#     The per-cell sweep above is O(cells); this is O(basins). It also carries
#     the region adjacency + saddle elevations the partition needs, and supports
#     per-cell area weighting (cos lat) so heft is true area*height, not pixels.
# ----------------------------------------------------------------------
def rag_and_stats(z, label, cellarea=None, base=0.0):
    """One vectorized pass over the grid. Returns:
        adj      {basin -> set of adjacent basins}
        saddle   {(lo,hi) -> highest-pass elevation between the two basins}
        cellcount{basin -> summed cell area}   (pixels if cellarea is None)
        sumh     {basin -> summed area*height}
        heft     {basin -> summed area*max(height-base, 0)}  (land mass only)
      The pass/col between basins A,B is max over their shared boundary of
      min(z_a, z_b): the highest point you can cross without dropping below it.
    """
    from collections import defaultdict
    if cellarea is None:
        cellarea = np.ones_like(z)
    lab = label.ravel()
    valid = lab >= 0                                  # sea / nodata cells are -1
    az = (z * cellarea).ravel()
    ar = cellarea.ravel()
    ahc = (np.clip(z - base, 0, None) * cellarea).ravel()   # land mass per cell
    # compact basin ids 0..P-1 for vectorized aggregation (land basins only)
    peaks = np.unique(lab[valid])
    pos = np.full(int(lab.max()) + 1, -1)
    pos[peaks] = np.arange(len(peaks))
    cidv = pos[lab[valid]]
    cellcount = {int(p): float(s) for p, s in
                 zip(peaks, np.bincount(cidv, weights=ar[valid], minlength=len(peaks)))}
    sumh = {int(p): float(s) for p, s in
            zip(peaks, np.bincount(cidv, weights=az[valid], minlength=len(peaks)))}
    heft = {int(p): float(s) for p, s in
            zip(peaks, np.bincount(cidv, weights=ahc[valid], minlength=len(peaks)))}

    def pairs(la, lb, za, zb):
        d = (la != lb) & (la >= 0) & (lb >= 0)       # ignore pairs touching sea
        a = la[d]; b = lb[d]
        lo = np.minimum(a, b); hi = np.maximum(a, b)
        val = np.minimum(za[d], zb[d])
        return lo, hi, val

    L = label
    Z = z
    los, his, vals = [], [], []
    for la, lb, za, zb in [
            (L[:, :-1], L[:, 1:], Z[:, :-1], Z[:, 1:]),
            (L[:-1, :], L[1:, :], Z[:-1, :], Z[1:, :])]:
        lo, hi, val = pairs(la.ravel(), lb.ravel(), za.ravel(), zb.ravel())
        los.append(lo); his.append(hi); vals.append(val)
    lo = np.concatenate(los); hi = np.concatenate(his); val = np.concatenate(vals)
    K = int(peaks.max()) + 1
    key = lo.astype(np.int64) * K + hi.astype(np.int64)
    order = np.argsort(key, kind='stable')
    key = key[order]; val = val[order]
    starts = np.concatenate(([0], np.where(np.diff(key) != 0)[0] + 1))
    segmax = np.maximum.reduceat(val, starts)
    ukey = key[starts]
    ulo = (ukey // K).astype(int); uhi = (ukey % K).astype(int)
    adj = defaultdict(set)
    saddle = {}
    for a, b, s in zip(ulo.tolist(), uhi.tolist(), segmax.tolist()):
        adj[a].add(b); adj[b].add(a)
        saddle[(a, b)] = s
    return adj, saddle, cellcount, sumh, heft


def build_tree(z, label, cellarea=None, base=0.0):
    """Divide tree from the basin graph: union-find over RAG edges in descending
    saddle order (Kruskal-style). When an edge first joins two components, the
    lower-peak component is sealed -- its key col is that saddle, its parent is
    the higher peak. ~1000x faster than the per-cell merge_tree at scale and
    carries adjacency + area-weighted stats. Returns the same dict shape, plus
    'adj' and 'saddle'."""
    adj, saddle, cellcount, sumh, heft = rag_and_stats(z, label, cellarea, base)
    flat = z.ravel()
    peaks = sorted(int(p) for p in np.unique(label) if p >= 0)   # exclude sea (-1)
    elev = {p: float(flat[p]) for p in peaks}
    par_dsu = {p: p for p in peaks}
    top = {p: p for p in peaks}              # component -> highest peak

    def find(x):
        while par_dsu[x] != x:
            par_dsu[x] = par_dsu[par_dsu[x]]
            x = par_dsu[x]
        return x

    keycol = {}
    parent = {}
    edges = sorted(saddle.items(), key=lambda kv: kv[1], reverse=True)  # high->low
    for (a, b), s in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        ta, tb = top[ra], top[rb]
        winner, loser = (ta, tb) if elev[ta] >= elev[tb] else (tb, ta)
        if loser not in keycol:
            keycol[loser] = s
            parent[loser] = winner
        par_dsu[rb] = ra
        top[find(a)] = winner
    gmax = max(peaks, key=lambda p: elev[p])
    for p in peaks:
        keycol.setdefault(p, None)
        parent.setdefault(p, None)
    keycol[gmax] = None
    parent[gmax] = None
    prominence = {p: (elev[p] - keycol[p]) if keycol[p] is not None else elev[p]
                  for p in peaks}
    # prominence-volume per basin (relative to own key col), area-weighted
    volume = {p: max(0.0, sumh[p] - (keycol[p] if keycol[p] is not None else base)
                     * cellcount[p]) for p in peaks}
    return dict(elev=elev, keycol=keycol, parent=parent, prominence=prominence,
                volume=volume, cellcount=cellcount, sumh=sumh, heft=heft, gmax=gmax,
                peaks=np.array(peaks), adj=adj, saddle=saddle)


# ----------------------------------------------------------------------
# 3c. Basin simplification -- collapse noise basins before partitioning
#     A real DEM yields one basin per pixel-scale bump (Japan @ 6 M cells ->
#     ~160 k basins, mostly 1-2 cell noise). Collapsing every basin below a
#     prominence threshold into its prominence-parent leaves a few thousand
#     meaningful basins: faster partition AND a cleaner map.
# ----------------------------------------------------------------------
def simplify_basins(label, tree, min_prominence):
    """Relabel each cell to the nearest ancestor basin (up the prominence-parent
    chain) whose prominence >= min_prominence. Vectorized via a lookup table."""
    parent = tree['parent']
    prom = tree['prominence']
    gmax = int(tree['gmax'])
    peaks = [int(p) for p in tree['peaks']]
    keep = {p for p in peaks if prom[p] >= min_prominence}
    keep.add(gmax)
    remap = {}
    for p in peaks:
        q = p
        while q not in keep and parent[q] is not None:
            q = parent[q]
        remap[p] = q
    peak_arr = np.array(peaks)
    remap_arr = np.array([remap[p] for p in peaks], dtype=label.dtype)
    lut = np.full(int(label.max()) + 1, -1, dtype=label.dtype)
    lut[peak_arr] = remap_arr
    out = lut[np.where(label >= 0, label, 0)]
    out[label < 0] = -1                              # keep sea cells as sea
    return out


# ----------------------------------------------------------------------
# 4. Agglomerate to N regions on the basin ADJACENCY graph (the faithful map)
#    Repeatedly take the smallest-volume region and merge it into the neighbour
#    across its highest saddle (shallowest divide = the massif it is a shoulder
#    of). Every merge is between spatially adjacent regions, so every region
#    stays a single connected blob -- unlike tree-ancestor remapping, which
#    strands basins on far-off dominant peaks. This is the inverse watershed,
#    coarsened by significance; region sizes are set by topography (unequal).
# ----------------------------------------------------------------------
def agglomerate(label, tree, N, score='volume', verbose=False):
    """Merge basins bottom-up (smallest `score` first) across their highest
    saddle until N regions remain. Returns (remapped label array, region roots).
    Region identity is its highest peak. Sea/nodata cells (-1) pass through."""
    import heapq
    adj = tree['adj']
    saddle = tree['saddle']
    elev = tree['elev']
    val = tree[score]
    peaks = [int(p) for p in tree['peaks']]

    parent = {p: p for p in peaks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    rvol = {p: float(val[p]) for p in peaks}          # additive significance
    rpeak = {p: elev[p] for p in peaks}               # highest elevation in region
    nbr = {p: {} for p in peaks}
    for (a, b), s in saddle.items():
        nbr[a][b] = s
        nbr[b][a] = s

    heap = [(rvol[p], p) for p in peaks]
    heapq.heapify(heap)
    count = len(peaks)
    while count > N and heap:
        v, r = heapq.heappop(heap)
        if find(r) != r or v != rvol[r] or not nbr[r]:
            continue                                  # stale / isolated
        # target: neighbour across the highest saddle (shallowest divide),
        # breaking ties toward the higher peak (merge uphill into the massif)
        t = max(nbr[r], key=lambda x: (nbr[r][x], rpeak[x]))
        keeper, gone = (r, t) if rpeak[r] >= rpeak[t] else (t, r)
        parent[gone] = keeper
        rvol[keeper] = rvol[r] + rvol[t]
        rpeak[keeper] = max(rpeak[r], rpeak[t])
        # fuse neighbour tables: highest pass to each common neighbour
        del nbr[gone][keeper]
        del nbr[keeper][gone]
        for c, s in nbr[gone].items():
            nbr[c].pop(gone, None)
            if c == keeper:
                continue
            if c in nbr[keeper]:
                best = max(nbr[keeper][c], s)
            else:
                best = s
            nbr[keeper][c] = best
            nbr[c][keeper] = best
        del nbr[gone]
        count -= 1
        heapq.heappush(heap, (rvol[keeper], keeper))
    if verbose:
        print(f"  {count} regions")

    # remap every basin to its region's representative (highest-peak basin id)
    peak_arr = np.array(peaks)
    rep = np.array([find(p) for p in peaks], dtype=np.int64)
    lut = np.full(int(label.max()) + 1, -1, dtype=np.int64)
    lut[peak_arr] = rep
    out = lut[np.where(label >= 0, label, 0)]
    out[label < 0] = -1
    roots = sorted(set(rep.tolist()))
    return out, roots


# ----------------------------------------------------------------------
# 4a. Greedy prune to N territories (the "ultras" map -- kept for comparison)
# ----------------------------------------------------------------------
def prune(label, tree, N, by='volume'):
    """Greedily absorb the least-significant peak into its (living) parent until
    N remain. Returns (remapped label array, surviving peak set)."""
    score = dict(tree['volume'] if by == 'volume' else tree['prominence'])
    par = dict(tree['parent'])
    remap = {p: p for p in tree['elev']}
    alive = set(tree['elev'].keys())
    while len(alive) > N:
        cand = [p for p in alive if par.get(p) is not None]
        if not cand:
            break
        victim = min(cand, key=lambda p: score[p])
        tgt = par[victim]
        while tgt not in alive and par.get(tgt) is not None:
            tgt = par[tgt]
        if tgt == victim or tgt not in alive:
            alive.discard(victim)
            continue
        for p in list(remap):
            if remap[p] == victim:
                remap[p] = tgt
        score[tgt] = score.get(tgt, 0) + score.get(victim, 0)
        for p in par:
            if par.get(p) == victim:
                par[p] = tgt
        alive.discard(victim)
    lab = label.ravel()
    out = np.array([remap.get(x, x) for x in lab]).reshape(label.shape)
    return out, alive


# ----------------------------------------------------------------------
# 4b. Balanced volume-partition -- cut the divide tree into N equal-heft pieces
# ----------------------------------------------------------------------
def _children_map(parent, peaks):
    ch = {int(p): [] for p in peaks}
    for p in peaks:
        pa = parent[int(p)]
        if pa is not None:
            ch[int(pa)].append(int(p))
    return ch


def _assign(cuts, parent, children, gmax):
    """Map every peak to the root of its cut-bounded subtree."""
    territory = {}
    for r in list(cuts) + [gmax]:
        stack, seen = [r], set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            territory[u] = r
            for c in children[u]:
                if c not in cuts:
                    stack.append(c)
    return territory


def _bounded_subtree_w(node, cuts, children, w):
    """Sum of heft over node's subtree, stopping at any cut edge below it."""
    s, stack = 0.0, [node]
    while stack:
        u = stack.pop()
        s += w[u]
        for c in children[u]:
            if c not in cuts:
                stack.append(c)
    return s


def _adjust_parts(cuts, N, w, parent, children, gmax):
    """Add or remove cut edges until there are exactly N parts. Removing a cut
    merges its part into the parent part; adding one bisects the heaviest part."""
    cuts = set(cuts)
    # too many parts: drop the cut whose own part is lightest (cheapest merge)
    while len(cuts) + 1 > N:
        part_w = {r: _bounded_subtree_w(r, cuts, children, w) for r in cuts}
        victim = min(part_w, key=part_w.get)
        cuts.discard(victim)
    # too few parts: bisect the heaviest part by cutting nearest to half its heft
    while len(cuts) + 1 < N:
        roots = list(cuts) + [gmax]
        part_w = {r: _bounded_subtree_w(r, cuts, children, w) for r in roots}
        heavy = max(part_w, key=part_w.get)
        half = part_w[heavy] / 2.0
        # walk the heavy part, pick the node whose bounded subtree is closest half
        nodes, stack = [], [heavy]
        while stack:
            u = stack.pop()
            nodes.append(u)
            for c in children[u]:
                if c not in cuts:
                    stack.append(c)
        cand = [u for u in nodes if u != heavy]
        if not cand:
            break
        pick = min(cand, key=lambda u: abs(
            _bounded_subtree_w(u, cuts, children, w) - half))
        cuts.add(pick)
    return cuts


def balanced_partition(label, tree, N, base=0.0, verbose=False):
    """Partition the divide tree into N connected subtrees of near-equal HEFT.

    Heft is conserved relief mass relative to a COMMON base (sea level / global
    min):  V(subtree) = SH(subtree) - base * CC(subtree), with SH/CC the
    height-sum and cell-count over the subtree. Unlike prominence-volume
    (relative to each peak's own key col), this tiles -- territory hefts sum to
    the total -- so it is equalizable, the relief analog of equal population.
    The tree still constrains every territory to a connected peak-region; only
    the heft measure is common-base.

    Greedy peel: while parts remain, cut the edge whose subtree heft is closest
    to target = (root-component heft) / (remaining parts). Each cut removes one
    subtree from the root component; aggregates are recomputed over attached
    edges only. Cutting a single edge keeps every piece tree-connected.

    Returns (remapped label array, list of territory-root peaks).
    """
    parent = tree['parent']
    peaks = [int(p) for p in tree['peaks']]
    CC0 = {int(p): tree['cellcount'][int(p)] for p in peaks}
    SH0 = {int(p): tree['sumh'][int(p)] for p in peaks}
    w = {p: SH0[p] - base * CC0[p] for p in CC0}      # conserved per-basin heft
    children = _children_map(parent, peaks)
    gmax = int(tree['gmax'])
    total = sum(w.values())

    # post-order (children before parent) over the full tree, rooted at gmax
    order, stack, seen = [], [gmax], set()
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        order.append(u)
        for c in children[u]:
            stack.append(c)
    order.reverse()                                   # now children precede parents

    def cuts_for_target(target):
        """Bottom-up accumulation: close a subtree as its own part once its
        as-yet-unassigned weight reaches target. Returns the cut-edge set."""
        acc = {}
        cuts = set()
        for u in order:
            a = w[u]
            for c in children[u]:
                a += acc[c]                            # cut children contribute 0
            if a >= target and u != gmax:
                cuts.add(u)
                acc[u] = 0.0
            else:
                acc[u] = a
        return cuts

    # binary-search the target so the partition lands on exactly N parts
    lo, hi = total / (len(peaks) + 1), total
    cut = cuts_for_target(total / N)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        c = cuts_for_target(mid)
        parts = len(c) + 1
        if parts > N:                                 # target too small -> raise it
            lo = mid
        elif parts < N:                               # target too big -> lower it
            hi = mid
        else:
            cut = c
            break
        cut = c
    # trim/pad to exactly N parts if the search couldn't land precisely
    cut = _adjust_parts(cut, N, w, parent, children, gmax)
    if verbose:
        print(f"  {len(cut)+1} parts, target={total/N:.0f}")

    roots = list(cut) + [gmax]
    territory = _assign(cut, parent, children, gmax)
    lab = label.ravel()
    out = np.array([territory.get(x, x) for x in lab]).reshape(label.shape)
    return out, roots


# ----------------------------------------------------------------------
# 4c. Balanced regions on the basin ADJACENCY graph (the working approach)
#     The rooted divide tree is star-like at big peaks (the global max can
#     directly parent ~half of all basins), so edge-cutting it can never peel
#     siblings off the root and the root territory dominates. Partitioning the
#     basin adjacency graph instead lets sibling basins merge, which is what an
#     equal-heft tiling actually needs.
# ----------------------------------------------------------------------
def build_rag(label):
    """Region adjacency graph of basins: {peak -> set of adjacent peaks}.
    Two basins are adjacent if any 4-neighbour cells carry different labels."""
    from collections import defaultdict
    adj = defaultdict(set)
    L = label
    def link(a, b):
        d = a != b
        aa, bb = a[d], b[d]
        for x, y in zip(aa.tolist(), bb.tolist()):
            adj[x].add(y)
            adj[y].add(x)
    link(L[:, :-1], L[:, 1:])     # horizontal neighbours
    link(L[:-1, :], L[1:, :])     # vertical neighbours
    return adj


def seed_peaks(label, tree, N, pool_mult=8, allowed=None):
    """Choose N seed peaks that are both prominent and spatially spread, by
    farthest-point sampling from a pool of the most prominent peaks. Pure
    prominence-ranked seeds cluster in high-relief areas and starve each other;
    spreading them gives one territory per major massif without slivers.
    `allowed` restricts the pool (e.g. to land basins)."""
    m = label.shape[1]
    peaks = [int(p) for p in tree['peaks']]
    if allowed is not None:
        peaks = [p for p in peaks if p in allowed]
    prom = tree['prominence']
    pool = sorted(peaks, key=lambda p: prom[p], reverse=True)[:max(N * pool_mult, N)]
    coords = {p: (p // m, p % m) for p in pool}
    seeds = [pool[0]]                                  # global-ish most prominent
    d2 = {p: (coords[p][0] - coords[seeds[0]][0]) ** 2
             + (coords[p][1] - coords[seeds[0]][1]) ** 2 for p in pool}
    while len(seeds) < N and len(seeds) < len(pool):
        nxt = max((p for p in pool if p not in seeds), key=lambda p: d2[p])
        seeds.append(nxt)
        for p in pool:
            dd = (coords[p][0] - coords[nxt][0]) ** 2 + (coords[p][1] - coords[nxt][1]) ** 2
            if dd < d2[p]:
                d2[p] = dd
    return seeds


def _land_components(land, adj):
    """Connected components of the land basins under the adjacency graph."""
    comp = {}
    cid = 0
    for s in land:
        if s in comp:
            continue
        comp[s] = cid
        stack = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v in land and v not in comp:
                    comp[v] = cid
                    stack.append(v)
        cid += 1
    return comp


def seed_by_component(label, tree, N, land, adj):
    """Allocate N seeds across connected land components in proportion to each
    component's heft (largest-remainder), then place each component's seeds by
    prominence-weighted farthest-point sampling. Tiny islands (heft far below the
    per-territory target) get zero seeds and stay background -- this is what kills
    the sliver territories that uniform farthest-point seeding produced."""
    from collections import defaultdict
    heft = tree['heft']
    comp = _land_components(land, adj)
    members = defaultdict(list)
    cheft = defaultdict(float)
    for p in land:
        members[comp[p]].append(p)
        cheft[comp[p]] += heft[p]
    H = sum(cheft.values())
    if H <= 0:
        return seed_peaks(label, tree, min(N, len(land)), allowed=land)
    target = H / N
    raw = {c: cheft[c] / target for c in cheft}      # sums to N
    alloc = {c: int(raw[c]) for c in raw}
    rem = N - sum(alloc.values())
    for c in sorted(raw, key=lambda c: raw[c] - alloc[c], reverse=True)[:max(rem, 0)]:
        alloc[c] += 1
    seeds = []
    for c, nc in alloc.items():
        nc = min(nc, len(members[c]))
        if nc <= 0:
            continue
        seeds += seed_peaks(label, tree, nc, allowed=set(members[c]))
    return seeds


def balanced_regions(label, tree, N, base=0.0, seeds=None, min_frac=0.35):
    """Partition the LAND basins into N connected, near-equal-heft territories by
    balanced multi-source growth on the basin adjacency graph.

    Only basins with positive land heft (Σ area·max(h-base,0) > 0) take part; the
    sea and fully-submarine basins are background. Seeds default to N prominent,
    spatially-spread land peaks (one territory per major massif). Growth always
    extends the currently-lightest region by absorbing its lightest unassigned
    neighbour land basin, driving regions toward equal heft while keeping each
    connected. Returns (label array with -1 for sea/unassigned, list of seeds).
    """
    import heapq
    peaks = [int(p) for p in tree['peaks']]
    heft = {p: tree['heft'][p] for p in peaks}
    land = {p for p in peaks if heft[p] > 0}        # basins that contain land
    adj = tree.get('adj') or build_rag(label)

    if seeds is None:
        seeds = seed_by_component(label, tree, min(N, len(land)), land, adj)
    seeds = [int(s) for s in seeds]

    from collections import defaultdict
    owner = {}
    region_heft = {}
    frontier = {}                  # region -> set of UNASSIGNED adjacent land basins
    infront = defaultdict(set)     # basin -> regions whose frontier holds it
    unassigned = set(land) - set(seeds)
    heap = []                      # (region_heft, tiebreak, seed)
    tie = 0
    for s in seeds:
        owner[s] = s
        region_heft[s] = heft[s]
        frontier[s] = set()
        for nb in adj[s]:
            if nb in unassigned:
                frontier[s].add(nb)
                infront[nb].add(s)
        heapq.heappush(heap, (region_heft[s], tie, s))
        tie += 1

    while heap and unassigned:
        h, _, s = heapq.heappop(heap)
        if h != region_heft[s]:
            continue               # stale heap entry
        fr = frontier[s]
        if not fr:
            continue               # region sealed in; drop it
        b = min(fr, key=lambda x: heft[x])   # frontier holds only unassigned
        owner[b] = s
        unassigned.discard(b)
        region_heft[s] += heft[b]
        for r in infront[b]:       # b is now assigned: pull it from all frontiers
            frontier[r].discard(b)
        infront[b].clear()
        for nb in adj[b]:          # grow the frontier with b's unassigned nbrs
            if nb in unassigned:
                frontier[s].add(nb)
                infront[nb].add(s)
        heapq.heappush(heap, (region_heft[s], tie, s))
        tie += 1

    # land basins unreachable from any seed (islets behind assigned basins):
    # attach to lightest adjacent assigned region; iterate to a fixpoint.
    changed = True
    while changed and unassigned:
        changed = False
        for b in list(unassigned):
            nbrs = [owner[x] for x in adj[b] if x in owner]
            if nbrs:
                r = min(nbrs, key=lambda r: region_heft[r])
                owner[b] = r
                region_heft[r] += heft[b]
                unassigned.discard(b)
                changed = True

    # dissolve sliver territories: a seed boxed in (e.g. a tiny island whose only
    # neighbours are sea/other seeds) stays far below target. Merge any territory
    # under min_frac*target into its lightest adjacent territory. Reduces the part
    # count below N -- the realised number of territories is returned via the map.
    members = defaultdict(set)
    for b, r in owner.items():
        members[r].add(b)
    radj = defaultdict(set)
    for b, r in owner.items():
        for nb in adj[b]:
            r2 = owner.get(nb)
            if r2 is not None and r2 != r:
                radj[r].add(r2)
    target = sum(region_heft[r] for r in members) / max(N, 1)
    while len(members) > 1:
        r = min(members, key=lambda x: region_heft[x])
        if region_heft[r] >= min_frac * target:
            break
        nbr_regions = [x for x in radj[r] if x in members]
        if not nbr_regions:                        # isolated sliver -> background
            for b in members[r]:
                owner[b] = -1
            del members[r]; del region_heft[r]
            continue
        t = min(nbr_regions, key=lambda x: region_heft[x])
        for b in members[r]:
            owner[b] = t
        members[t] |= members[r]
        region_heft[t] += region_heft[r]
        for x in radj[r]:
            radj[x].discard(r)
            if x != t:
                radj[x].add(t)
                radj[t].add(x)
        del members[r]; del region_heft[r]

    roots = [r for r in members]
    peak_arr = np.array(peaks)
    own_arr = np.array([owner.get(p, -1) for p in peaks], dtype=np.int64)
    lut = np.full(int(label.max()) + 1, -1, dtype=np.int64)
    lut[peak_arr] = own_arr
    out = lut[label]
    return out, roots


# ----------------------------------------------------------------------
# territory volume measurement (exact, from grid) -- the equal-heft test
# ----------------------------------------------------------------------
def territory_volumes(label_terr, z, tree, base=None, cellarea=None):
    """Exact heft of each territory, measured from the grid (area-weighted).

    base=None  -> prominence-volume: sum over cells of (h - territory_keycol)
                  clipped at 0 (relative to each territory's own cut saddle).
                  Ranks territories; does NOT conserve across territories.
    base=value -> conserved heft: sum over cells of (h - base) clipped at 0,
                  a common floor. Tiles (hefts sum to total); this is what the
                  balanced partition equalizes.
    cellarea     per-cell area weight (cos lat for geographic grids); None = 1.
    """
    flat = z.ravel()
    L = label_terr.ravel()
    w = np.ones_like(flat) if cellarea is None else cellarea.ravel()
    keycol = tree['keycol']
    out = {}
    for t in np.unique(L):
        if int(t) < 0:                 # -1 = sea / unassigned background
            continue
        sel = L == t
        if base is None:
            col = keycol.get(int(t))
            col = 0.0 if col is None else col
        else:
            col = base
        out[int(t)] = float((np.clip(flat[sel] - col, 0, None) * w[sel]).sum())
    return out


def cv(values):
    a = np.asarray(list(values), float)
    return float(a.std() / a.mean()) if a.mean() else float('nan')
