// T-4262 stage 2: where a network stroke set's segments sit in its GPU buffers. The buffer order is the paint order
// (no depth test: a later line is drawn over an earlier one, T-190), so the segments stay in their order, with room
// spread between them; an edit's new segments go into the room where they belong, and only they and the slots they
// free are written. Where the room there is too small, the nearest stretch with room enough is laid out again, and
// only when the whole set has too little is it laid out afresh in larger buffers.

/** a slot no segment holds: the stroke shader draws nothing for it (a bit past every `F_` flag, network.ts) */
export const F_HOLE = 32768;

/** where a set's segments are: per block its first segment and the slot it is in, both increasing; block k holds
 * the segments up to block k + 1's first. `count` segments, `end` the slots drawn (to the last block's end),
 * `cap` the slots the buffers have room for. Every slot below `cap` no block holds has `F_HOLE` set. */
export interface SlotLayout {
  src: Int32Array;
  gpu: Int32Array;
  count: number;
  end: number;
  cap: number;
}

/** a plan's writes: spans of slots (start, end) written whole, each slot of a span no item holds as a hole; items
 * (slot, first segment, count), in order, each inside a span; and slots to mark as holes only (start, count) */
export interface SlotPlan { layout: SlotLayout; spans: number[]; items: number[]; holes: number[] }

/** slots for `n` segments: a quarter more */
export const roomFor = (n: number) => n + Math.max(Math.floor(n * slotShare.room), slotShare.min);
/** The room's shares: `room` of the set's segments, `tail` of it after the last (the rest spread after every
 * `block` segments); a stretch laid out again keeps `fit` of its segments free, and reaches at most `reach` times
 * the segments to put (or blocks) before the set is laid out afresh instead (notes/T-4262.md: a mixed session of
 * edits wrote about 6 slots per segment it made). `?slotRoom=tiny` (checks): a few slots, so a small game's edits run
 * every path a big set needs. */
export const slotShare = { room: 1 / 4, min: 256, tail: 0.5, fit: 1 / 32, reach: 64, block: 128 };
export function tinySlots() { Object.assign(slotShare, { room: 0, min: 4, block: 8, reach: 4 }); }
if (typeof location !== "undefined" && new URLSearchParams(location.search).get("slotRoom") === "tiny") tinySlots();

const blockCount = (L: SlotLayout, k: number) => (k + 1 < L.src.length ? L.src[k + 1] : L.count) - L.src[k];

/** `n` segments laid out afresh in `cap` slots: half the room after the last (a new line or blueprint is added at
 * the end), the rest spread between blocks */
export function wholeLayout(n: number, cap = roomFor(n)): SlotLayout {
  const B = slotShare.block, blocks = n ? Math.ceil(n / B) : 0, slack = Math.floor((cap - n) * (1 - slotShare.tail));
  const src = new Int32Array(blocks), gpu = new Int32Array(blocks);
  for (let k = 0; k < blocks; k++) {
    src[k] = k * B;
    gpu[k] = k * B + Math.floor((slack * k) / blocks);
  }
  return { src, gpu, count: n, end: blocks ? gpu[blocks - 1] + n - src[blocks - 1] : 0, cap };
}

/** the block holding segment `i` */
function blockOf(L: SlotLayout, i: number): number {
  let lo = 0, hi = L.src.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (L.src[mid] <= i) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/** the slot segment `i` is in */
export const slotOf = (L: SlotLayout, i: number) => {
  const k = blockOf(L, i);
  return L.gpu[k] + i - L.src[k];
};

/** each stretch of segments `a` to `a + c` that sits in consecutive slots: fn(slot, first segment, count) */
export function forPieces(L: SlotLayout, a: number, c: number, fn: (slot: number, first: number, count: number) => void) {
  if (c <= 0) return;
  for (let k = blockOf(L, a), i = a, e = a + c; i < e; k++) {
    const to = Math.min(e, L.src[k] + blockCount(L, k));
    fn(L.gpu[k] + i - L.src[k], i, to - i);
    i = to;
  }
}

/** segment ranges (start, count pairs, in order) as the slot ranges they are drawn from, holes between included */
export function slotRanges(L: SlotLayout, ranges: readonly number[]): number[] {
  const out: number[] = [];
  for (let r = 0; r + 1 < ranges.length; r += 2) {
    if (ranges[r + 1] <= 0) continue;
    const a = slotOf(L, ranges[r]);
    out.push(a, slotOf(L, ranges[r] + ranges[r + 1] - 1) + 1 - a);
  }
  return out;
}

/** the blocks laid out whole: items (slot, first, count) for every block */
export function wholeItems(L: SlotLayout): number[] {
  const out: number[] = [];
  for (let k = 0; k < L.src.length; k++) out.push(L.gpu[k], L.src[k], blockCount(L, k));
  return out;
}

/**
 * Lay out `n` segments made from the last set's (laid out as `old`) by `runs` (start, count, old start or -1, as
 * `NetPatch`): the runs kept stay in their slots (those out of order, the fewest segments, are written again), new
 * segments go into the room between them, and where that is too small the nearest stretch with room enough is laid
 * out again, its room spread evenly. Null: the buffers have too little room for the set (lay it out afresh).
 */
export function planPatch(old: SlotLayout, runs: Int32Array, n: number): SlotPlan | null {
  // items in the new order: (slot or -1 for segments to write, first segment, count)
  const iG: number[] = [], iS: number[] = [], iC: number[] = [];
  const push = (g: number, s: number, c: number) => {
    const k = iG.length - 1;
    if (k >= 0 && iS[k] + iC[k] === s && (g < 0 ? iG[k] < 0 : iG[k] >= 0 && iG[k] + iC[k] === g)) iC[k] += c;
    else iG.push(g), iS.push(s), iC.push(c);
  };
  for (let r = 0; r + 2 < runs.length; r += 3) {
    const s = runs[r], c = runs[r + 1], o = runs[r + 2];
    if (c <= 0) continue;
    if (o < 0) push(-1, s, c);
    else forPieces(old, o, c, (g, x, m) => push(g, s + x - o, m));
  }
  keepInOrder(iG, iC);
  // the kept items, and between each two (and before the first, after the last) the segments to write
  const kept: number[] = [];
  for (let i = 0; i < iG.length; i++) if (iG[i] >= 0) kept.push(i);
  const m = kept.length;
  const lo = (j: number) => (j === 0 ? 0 : iG[kept[j - 1]] + iC[kept[j - 1]]);
  const hi = (j: number) => (j === m ? old.cap : iG[kept[j]]);
  // per gap j (before kept[j]): the segments to write in it; prefix sums of those and of the kept items
  const want = new Float64Array(m + 2), keptSum = new Float64Array(m + 1);
  for (let i = 0, j = 0; i < iG.length; i++) {
    if (iG[i] >= 0) { keptSum[j + 1] = keptSum[j] + iC[i]; j++; } else want[j + 1] += iC[i];
  }
  for (let j = 0; j <= m; j++) want[j + 1] += want[j];
  const content = (a: number, b: number) => want[b + 1] - want[a] + keptSum[b] - keptSum[a];
  // the items of gaps a to b (and the kept between) laid out from slot `from` over `span` slots
  const place = new Int32Array(iG.length).fill(-1);
  const firstItem = (j: number) => (j === 0 ? 0 : kept[j - 1] + 1), lastItem = (j: number) => (j === m ? iG.length : kept[j]);
  const spans: number[] = [];
  // the last stretch laid out again: a later one reaching into it takes it whole
  let prevA = -1, prevB = -1;
  for (let j = 0; j <= m; j++) {
    if (want[j + 1] === want[j]) continue;
    if (want[j + 1] - want[j] <= hi(j) - lo(j)) {
      // (room enough: written one after another from the gap's start)
      let at = lo(j);
      for (let i = firstItem(j); i < lastItem(j); i++) (place[i] = at), (at += iC[i]);
      spans.push(lo(j), at);
      continue;
    }
    // the nearest stretch of gaps a..b with room for its segments and a sixteenth more, the cheaper side first
    let a = j, b = j;
    const fits = () => hi(b) - lo(a) >= content(a, b) * (1 + slotShare.fit);
    // (a stretch past `reach` times the segments to put, the room there being used up: the set laid out afresh)
    const most = Math.max(slotShare.reach * (want[j + 1] - want[j]), slotShare.reach * slotShare.block);
    while (!fits()) {
      if ((a === 0 && b === m) || content(a, b) > most) return null;
      const left = a > 0 ? iC[kept[a - 1]] : Infinity, right = b < m ? iC[kept[b]] : Infinity;
      if (left <= right) a--;
      else b++;
      if (a <= prevB) a = Math.min(a, prevA);
    }
    // (a window reaching past the last kept item takes room an eighth of its segments, not all left in the buffers)
    const total = content(a, b), from = lo(a), span = b === m ? Math.min(hi(b) - from, Math.ceil(total * 9 / 8) + 64) : hi(b) - from;
    // half the room after the segments that did not fit (more are often added where some just were: track pieces
    // built one after another, lines added at the end), the rest spread over the stretch
    const free = span - total, hot = free >> 1, spread = free - hot, after = lastItem(j);
    let done = 0;
    for (let i = firstItem(a); i < lastItem(b); i++) {
      place[i] = from + done + Math.floor((spread * done) / total) + (i >= after ? hot : 0);
      done += iC[i];
    }
    // spans already taken in the window's gaps are its own now
    while (spans.length && spans[spans.length - 2] >= from) spans.length -= 2;
    spans.push(from, from + span);
    prevA = a;
    prevB = b;
    j = b;
  }
  // the new layout: every item where it was kept or placed, runs of consecutive slots one block
  const src: number[] = [], gpu: number[] = [], items: number[] = [];
  for (let i = 0; i < iG.length; i++) {
    const g = place[i] >= 0 ? place[i] : iG[i];
    if (place[i] >= 0) items.push(g, iS[i], iC[i]);
    const k = src.length - 1;
    if (k >= 0 && gpu[k] + (iS[i] - src[k]) === g) continue;
    src.push(iS[i]), gpu.push(g);
  }
  let end = 0;
  for (let i = 0; i < iG.length; i++) end = Math.max(end, (place[i] >= 0 ? place[i] : iG[i]) + iC[i]);
  const layout: SlotLayout = { src: Int32Array.from(src), gpu: Int32Array.from(gpu), count: n, end, cap: old.cap };
  // slots that held segments and now hold none, outside the spans written: holes
  const now: number[] = [];
  for (let k = 0; k < layout.src.length; k++) now.push(layout.gpu[k], blockCount(layout, k));
  for (let s = 0; s < spans.length; s += 2) now.push(spans[s], spans[s + 1] - spans[s]);
  const was: number[] = [];
  for (let k = 0; k < old.src.length; k++) was.push(old.gpu[k], blockCount(old, k));
  return { layout, spans, items, holes: subtract(was, now) };
}

/** ranges (start, count pairs) of `a` not in any of `b`'s */
function subtract(a: number[], b: number[]): number[] {
  const sort = (x: number[]) => {
    const p: [number, number][] = [];
    for (let i = 0; i + 1 < x.length; i += 2) if (x[i + 1] > 0) p.push([x[i], x[i] + x[i + 1]]);
    return p.sort((u, v) => u[0] - v[0]);
  };
  const A = sort(a), B = sort(b), out: number[] = [];
  let k = 0;
  for (const [s0, e] of A) {
    let s = s0;
    while (k < B.length && B[k][1] <= s) k++;
    for (let q = k; q < B.length && B[q][0] < e && s < e; q++) {
      if (B[q][0] > s) out.push(s, B[q][0] - s);
      s = Math.max(s, B[q][1]);
    }
    if (s < e) out.push(s, e - s);
  }
  return out;
}

/** kept items (slot >= 0) in order of slot: those of the heaviest increasing run kept, the rest marked to write (-1) */
function keepInOrder(iG: number[], iC: number[]) {
  let last = -1, sorted = true;
  for (let i = 0; i < iG.length && sorted; i++) if (iG[i] >= 0) (sorted = iG[i] > last), (last = iG[i]);
  if (sorted) return;
  const idx: number[] = [];
  for (let i = 0; i < iG.length; i++) if (iG[i] >= 0) idx.push(i);
  // the heaviest subsequence increasing in slot (a Fenwick tree of the best weight ending below each slot rank)
  const ranks = [...new Set(idx.map((i) => iG[i]))].sort((x, y) => x - y), rank = new Map(ranks.map((g, r) => [g, r + 1]));
  const tree = new Float64Array(ranks.length + 1), at = new Int32Array(ranks.length + 1).fill(-1);
  const best = new Float64Array(idx.length), prev = new Int32Array(idx.length).fill(-1);
  idx.forEach((i, q) => {
    let w = 0, p = -1;
    for (let r = rank.get(iG[i])! - 1; r > 0; r -= r & -r) if (tree[r] > w) (w = tree[r]), (p = at[r]);
    best[q] = w + iC[i];
    prev[q] = p;
    for (let r = rank.get(iG[i])!; r <= ranks.length; r += r & -r) if (best[q] > tree[r]) (tree[r] = best[q]), (at[r] = q);
  });
  let q = 0;
  for (let k = 1; k < idx.length; k++) if (best[k] > best[q]) q = k;
  const keep = new Set<number>();
  for (; q >= 0; q = prev[q]) keep.add(idx[q]);
  for (const i of idx) if (!keep.has(i)) iG[i] = -1;
}
