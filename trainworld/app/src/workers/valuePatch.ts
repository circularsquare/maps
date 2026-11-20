// T-3803: a value posted again and again as its changes from the last one, so the clock worker does not clone the
// money record whole with every state (about 2 MB on a 2,647-station save, nearly all of it the regions' ledgers,
// 25-65 ms a post there). Plain objects go key by key, arrays entry by entry after dropping the entries their start
// lost (a capped ledger), anything else whole. Applying a patch to the value it was made against gives a value equal
// to the new one, keys in the same order; unchanged parts are the old value's own.

export type ValuePatch =
  /** the new value whole */
  | { v: unknown }
  /** an object: its changed keys' patches, and its keys in order when they are not the old value's */
  | { o: Record<string, ValuePatch>; keys?: string[] }
  /** an array: its length, how many entries the old one's start lost, and the changed entries by new index */
  | { a: number; drop: number; at: Record<number, ValuePatch> };

const plain = (x: unknown): x is Record<string, unknown> => typeof x === "object" && x !== null && Object.getPrototypeOf(x) === Object.prototype;

/** Deep equality of the values a patch walks (arrays and plain objects by content, keys in order). */
export function sameValueDeep(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (Array.isArray(a)) {
    if (!Array.isArray(b) || a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!sameValueDeep(a[i], b[i])) return false;
    return true;
  }
  if (!plain(a) || !plain(b)) return false;
  const ka = Object.keys(a), kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  for (let i = 0; i < ka.length; i++) if (ka[i] !== kb[i] || !sameValueDeep(a[ka[i]], b[ka[i]])) return false;
  return true;
}

/** How `next` differs from `prev`, null when it does not. */
export function diffValue(prev: unknown, next: unknown): ValuePatch | null {
  if (Object.is(prev, next)) return null;
  if (Array.isArray(prev) && Array.isArray(next)) {
    let drop = 0;
    if (next.length && prev.length > 1 && !sameValueDeep(prev[0], next[0]))
      for (let k = 1; k <= 8 && k < prev.length; k++) if (sameValueDeep(prev[k], next[0])) { drop = k; break; }
    const at: Record<number, ValuePatch> = {};
    let changed = drop > 0 || prev.length !== next.length;
    for (let i = 0; i < next.length; i++) {
      const j = i + drop;
      const p = j < prev.length ? diffValue(prev[j], next[i]) : { v: next[i] };
      if (p) { at[i] = p; changed = true; }
    }
    return changed ? { a: next.length, drop, at } : null;
  }
  if (plain(prev) && plain(next)) {
    const kp = Object.keys(prev), kn = Object.keys(next);
    let sameKeys = kp.length === kn.length;
    for (let i = 0; sameKeys && i < kn.length; i++) sameKeys = kp[i] === kn[i];
    const o: Record<string, ValuePatch> = {};
    let changed = !sameKeys;
    for (const k of kn) {
      const p = Object.prototype.hasOwnProperty.call(prev, k) ? diffValue(prev[k], next[k]) : { v: next[k] };
      if (p) { o[k] = p; changed = true; }
    }
    if (!changed) return null;
    return sameKeys ? { o } : { o, keys: kn };
  }
  return { v: next };
}

/** `prev` with `p` applied: a new value along the changed paths, `prev`'s own parts elsewhere. */
export function applyPatch<T>(prev: T, p: ValuePatch): T {
  if ("v" in p) return p.v as T;
  if ("a" in p) {
    const src = prev as unknown as unknown[], out: unknown[] = new Array(p.a);
    for (let i = 0; i < p.a; i++) {
      const q = p.at[i];
      out[i] = q ? applyPatch(src[i + p.drop], q) : src[i + p.drop];
    }
    return out as unknown as T;
  }
  const src = prev as unknown as Record<string, unknown>, out: Record<string, unknown> = {};
  for (const k of p.keys ?? Object.keys(src)) {
    const q = p.o[k];
    out[k] = q ? applyPatch(src[k], q) : src[k];
  }
  return out as unknown as T;
}
