// A bubble selection's everyday far end, added up over the demand workers (T-285; used by
// demandClient.ts `flows`). Each worker answers `EverydayApi.selection_sparse_day` for its periods:
// per purpose kind 30 timing values, m, then m far cells (ascending), their rail, walking and
// driving legs. Only the far cells the selection's samples reach are sent, where every worker used
// to send three values for each of the city's cells per purpose (14.6 MB a worker in New York) and
// the main thread added them all up. The sums are the same f32 sums, in the same order, as that
// dense layout gave. No imports, so a check script can run it under Node (notes/t285-merge-check.mjs).

/** Far cells (ascending) with their legs by mode. */
export interface FarCells {
  cells: Uint32Array;
  farRail: Float32Array;
  farWalk: Float32Array;
  farDrive: Float32Array;
}

export interface EverydayMerge {
  /** each kind's 30 timing values (outward then return, mode-major, five periods), summed */
  timing: Float32Array[];
  /** per shown purpose (`keyOf`): its far cells and its legs by mode (from the timing) */
  shown: Map<string, { far: FarCells; total: [number, number, number] }>;
  /** every kind's far cells and the commutes' (`commute`) together */
  far: FarCells;
  /** the commutes' legs by mode (`start`) with every kind's added, kind by kind */
  legs: [number, number, number];
}

/**
 * Add up the workers' packed answers (`packed`, in the order they are summed) for `kinds` purpose
 * kinds. `keyOf(kind)` names the shown purpose a kind counts in (school and university share one),
 * or null for none; `commute` is the commutes' far end, added last, and `start` their legs by mode.
 * Null if a reply is malformed.
 */
export function mergeEveryday(packed: Float32Array[], kinds: number, keyOf: (kind: number) => string | null, commute: FarCells, start: [number, number, number]): EverydayMerge | null {
  // where each reply's kinds start
  const at: number[][] = [];
  let top = -1;
  for (const v of packed) {
    const o: number[] = [];
    let p = 0;
    for (let kind = 0; kind < kinds; kind++) {
      if (p + 31 > v.length) return null;
      o.push(p);
      const m = v[p + 30];
      if (m > 0) top = Math.max(top, v[p + 30 + m]);
      p += 31 + 4 * m;
    }
    if (p !== v.length) return null;
    at.push(o);
  }
  for (const c of commute.cells) top = Math.max(top, c);
  // every far cell any purpose or the commutes reach, ascending: a slot each
  const slot = new Int32Array(top + 1).fill(-1);
  packed.forEach((v, r) => at[r].forEach((p) => { const m = v[p + 30]; for (let i = 0; i < m; i++) slot[v[p + 31 + i]] = 0; }));
  for (const c of commute.cells) slot[c] = 0;
  const union: number[] = [];
  for (let c = 0; c <= top; c++) if (slot[c] === 0) { slot[c] = union.length; union.push(c); }
  const U = union.length;
  const far = [new Float32Array(U), new Float32Array(U), new Float32Array(U)];
  const shownAcc = new Map<string, { far: Float32Array[]; total: [number, number, number] }>();
  const timing: Float32Array[] = [];
  const legs: [number, number, number] = [start[0], start[1], start[2]];
  // one kind summed over the workers at a time; `touched` its slots
  const kindLegs = [new Float32Array(U), new Float32Array(U), new Float32Array(U)];
  const seen = new Uint8Array(U);
  const touched: number[] = [];
  for (let kind = 0; kind < kinds; kind++) {
    for (const s of touched) { kindLegs[0][s] = kindLegs[1][s] = kindLegs[2][s] = 0; seen[s] = 0; }
    touched.length = 0;
    const t = new Float32Array(30);
    packed.forEach((v, r) => {
      const p = at[r][kind], m = v[p + 30];
      for (let k = 0; k < 30; k++) t[k] += v[p + k];
      for (let i = 0; i < m; i++) {
        const s = slot[v[p + 31 + i]];
        if (!seen[s]) { seen[s] = 1; touched.push(s); }
        kindLegs[0][s] += v[p + 31 + m + i]; kindLegs[1][s] += v[p + 31 + 2 * m + i]; kindLegs[2][s] += v[p + 31 + 3 * m + i];
      }
    });
    timing.push(t);
    for (const s of touched) for (let m = 0; m < 3; m++) far[m][s] += kindLegs[m][s];
    const key = keyOf(kind);
    if (key !== null) {
      let acc = shownAcc.get(key);
      if (!acc) shownAcc.set(key, (acc = { far: [new Float32Array(U), new Float32Array(U), new Float32Array(U)], total: [0, 0, 0] }));
      for (const s of touched) for (let m = 0; m < 3; m++) acc.far[m][s] += kindLegs[m][s];
      for (let m = 0; m < 3; m++) for (let q = 0; q < 5; q++) acc.total[m] += t[m * 5 + q] + t[15 + m * 5 + q];
    }
    for (let m = 0; m < 3; m++) {
      let v = 0;
      for (let q = 0; q < 5; q++) v += t[m * 5 + q] + t[15 + m * 5 + q];
      legs[m] += v;
    }
  }
  const keep = (f: Float32Array[]): FarCells => {
    const own: number[] = [];
    for (let s = 0; s < U; s++) if (f[0][s] + f[1][s] + f[2][s] > 0) own.push(s);
    return { cells: Uint32Array.from(own, (s) => union[s]), farRail: Float32Array.from(own, (s) => f[0][s]), farWalk: Float32Array.from(own, (s) => f[1][s]), farDrive: Float32Array.from(own, (s) => f[2][s]) };
  };
  const shown = new Map<string, { far: FarCells; total: [number, number, number] }>();
  for (const [key, { far: f, total }] of shownAcc) shown.set(key, { far: keep(f), total });
  for (let i = 0; i < commute.cells.length; i++) {
    const s = slot[commute.cells[i]];
    far[0][s] += commute.farRail[i]; far[1][s] += commute.farWalk[i]; far[2][s] += commute.farDrive[i];
  }
  return { timing, shown, far: keep(far), legs };
}
