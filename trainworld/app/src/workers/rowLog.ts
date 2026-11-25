// T-4291: what each read of the state's rows changed (workers/stateRows.ts), numbered, so a reader of the rows (the post,
// the drawn network) takes the union of the reads since the one it last took instead of comparing whole lists. A reader
// that has fallen off the ring, or has a whole read (a load, `?patch=walk`, `reset`) among those it missed, is told
// nothing is known and compares whole.

/** edges, nodes, lines */
export type RowIds = [Set<number>, Set<number>, Set<number>];

interface Entry {
  all: boolean;
  ids: [number[], number[], number[]];
  /** per kind: some id came or went, or the list's order moved */
  order: [boolean, boolean, boolean];
}

export class RowLog {
  private ring: Entry[] = [];
  /** the number of `ring[0]` */
  private first = 1;
  /** reads kept: a reader further behind compares whole (a follow-up dropped this many times running) */
  static readonly SIZE = 64;

  /** The number the next read gets: a reader that has taken every read so far holds it. */
  get next(): number {
    return this.first + this.ring.length;
  }

  /** A read: per kind the ids whose row in the list is another object or is gone, or `all` for anything. */
  add(all: boolean, ids?: readonly Iterable<number>[], order?: readonly boolean[]) {
    this.ring.push({ all, ids: [[...(ids?.[0] ?? [])], [...(ids?.[1] ?? [])], [...(ids?.[2] ?? [])]], order: [!!order?.[0], !!order?.[1], !!order?.[2]] });
    if (this.ring.length > RowLog.SIZE) { this.ring.shift(); this.first++; }
  }

  /** The ids the reads from `from` on changed, by kind, and whether each kind's ids came, went or moved in order;
   * null when that is not known (a whole read among them, or `from` off the ring). */
  since(from: number): { ids: RowIds; order: [boolean, boolean, boolean] } | null {
    if (from < this.first || from > this.next) return null;
    const ids: RowIds = [new Set(), new Set(), new Set()], order: [boolean, boolean, boolean] = [false, false, false];
    for (let k = from - this.first; k < this.ring.length; k++) {
      const e = this.ring[k];
      if (e.all) return null;
      for (let j = 0; j < 3; j++) {
        for (const id of e.ids[j]) ids[j].add(id);
        if (e.order[j]) order[j] = true;
      }
    }
    return { ids, order };
  }
}
