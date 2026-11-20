// Shapes by grid cell for the waiting room and car park search (facilityPlace.ts): which shapes' boxes may meet a
// query box. Callers apply their exact tests to what it returns (T-3659).

export class ShapeGrid<T> {
  private cells = new Map<number, { shape: T; seen: number }[]>();
  /** shapes too large (or not finite) for cells, returned by every query */
  private wide: T[] = [];
  private all: T[] = [];
  private serial = 0;

  constructor(private cell: number) {}

  add(shape: T, x0: number, y0: number, x1: number, y1: number) {
    this.all.push(shape);
    const l = Math.floor(x0 / this.cell), r = Math.floor(x1 / this.cell), t = Math.floor(y0 / this.cell), b = Math.floor(y1 / this.cell);
    if (!Number.isFinite(r - l + b - t) || (r - l + 1) * (b - t + 1) > 256) {
      this.wide.push(shape);
      return;
    }
    const entry = { shape, seen: 0 };
    for (let x = l; x <= r; x++) for (let y = t; y <= b; y++) {
      const key = x * 1_000_003 + y, list = this.cells.get(key);
      if (list) list.push(entry); else this.cells.set(key, [entry]);
    }
  }

  near(x: number, y: number, radius: number): T[] {
    return this.box(x - radius, y - radius, x + radius, y + radius);
  }

  /** each shape whose box may meet this one, once */
  box(x0: number, y0: number, x1: number, y1: number): T[] {
    const l = Math.floor(x0 / this.cell), r = Math.floor(x1 / this.cell), t = Math.floor(y0 / this.cell), b = Math.floor(y1 / this.cell);
    if (!Number.isFinite(r - l + b - t) || (r - l + 1) * (b - t + 1) > 256) return this.all.slice();
    const found = this.wide.slice(), stamp = ++this.serial;
    for (let i = l; i <= r; i++)
      for (let j = t; j <= b; j++) {
        const list = this.cells.get(i * 1_000_003 + j);
        if (list) for (const e of list) if (e.seen !== stamp) {
          e.seen = stamp;
          found.push(e.shape);
        }
      }
    return found;
  }
}
