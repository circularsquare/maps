// T-2045: the zone in view solved alone and to its last round while a game load's whole solve runs,
// in a pool of its own (docs/demand-runtime.md "The focus pool", notes/T-2045.md): n solving workers
// (the main pool's count, at most FOCUS_WORKERS) dealt the periods as a pool of n is. A question for the
// main pool's aux goes to its first solving worker, answered between its populations (T-2495: an aux
// worker of its own, `?perfOff=focusAux`, was about 0.1 GB more during a load). It holds only
// the zones it solves, at most MAX_ZONES, and is ended when the load's whole solve lands, or once the
// main pool's own estimates make it needless (demandClient `retireFocus`).

import type { CityId } from "../game/cities";
import type { FromDemand, ToDemand } from "./demandProtocol";

/** zones the pool holds at once: each is a fine city in every one of its workers */
export const FOCUS_MAX_ZONES = 3;
/** solving workers at most (`?focusWorkers=n` for another), fewer when the main pool is smaller; T-2495:
 * one, since each worker is about 0.1 GB during a load and a second made the zone's first figures no sooner */
export const FOCUS_WORKERS = 1;

export interface FocusHooks {
  /** anything the main client handles for this pool: results, sums, query replies */
  message(i: number, msg: FromDemand): void;
  /** `id` is open in every worker of the pool */
  opened(id: CityId): void;
  /** a worker failed: the pool is ended and the load goes on without it */
  failed(why: string): void;
}

type Pack = { id: CityId; header: string; bin: ArrayBuffer; level?: number };

export class FocusPool {
  readonly workers: Worker[] = [];
  /** open in every worker, oldest first */
  readonly open: CityId[] = [];
  /** sent to the workers, not yet open in all */
  readonly opening = new Set<CityId>();
  private acks = new Map<CityId, number>();
  private ended = false;
  /** each worker's WebAssembly memory, bytes, as it last reported it (?debug=1's `memory()`) */
  readonly mem: number[] = [];

  /** the worker a question for an aux worker goes to: the pool's own aux, else its first solving worker */
  readonly auxAt: number;

  constructor(readonly n: number, periodsOf: number[][], private hooks: FocusHooks, view: ToDemand, withAux: boolean) {
    this.auxAt = withAux ? n : 0;
    for (let i = 0; i < n + (withAux ? 1 : 0); i++) {
      const w = new Worker(new URL("./demand.worker.ts", import.meta.url), { type: "module" });
      w.onmessage = (e: MessageEvent<FromDemand>) => this.receive(i, e.data);
      w.onerror = (e) => this.fail(`focus worker ${i}: ${e.message}`); // i18n-ignore
      const aux = i === n;
      w.postMessage({ kind: "init", packs: [], periods: aux ? [] : periodsOf[i], aux, warmEveryday: false, ...(aux ? { groups: periodsOf.slice(0, n) } : { markets: periodsOf[i], share: false }) } satisfies ToDemand);
      w.postMessage(view);
      this.workers.push(w);
    }
  }

  get alive() { return !this.ended; }
  holds(id: CityId) { return this.open.includes(id) || this.opening.has(id); }

  /** A city pack to every worker (the last takes the original buffer), opened ahead of any other. */
  post(pack: Pack) {
    if (this.ended || this.holds(pack.id)) return;
    this.opening.add(pack.id);
    this.acks.set(pack.id, 0);
    this.workers.forEach((w, i) => {
      const bin = i === this.workers.length - 1 ? pack.bin : pack.bin.slice(0);
      w.postMessage({ kind: "open", packs: [{ ...pack, bin }], first: true } satisfies ToDemand, [bin]);
    });
    // the oldest zone beyond the cap goes (its figures stay shown; questions about it go to the main pool)
    while (this.open.length + this.opening.size > FOCUS_MAX_ZONES && this.open.length) this.close(this.open[0]);
  }

  close(id: CityId) {
    const k = this.open.indexOf(id);
    if (k >= 0) this.open.splice(k, 1);
    this.opening.delete(id);
    this.acks.delete(id);
    if (!this.ended) this.postAll({ kind: "close", ids: [id] });
  }

  /** A solve to every worker, each solving worker told its part of the origins (T-1543). */
  solve(msg: Extract<ToDemand, { kind: "solve" }>) {
    this.workers.forEach((w, i) => w.postMessage(this.n > 1 && i < this.n ? { ...msg, part: [i, this.n] } : msg));
  }

  /** the next round may start (the solve is sent with the barrier) */
  go(version: number, round: number) {
    for (const w of this.workers.slice(0, this.n)) w.postMessage({ kind: "go", version, round, group: 0 } satisfies ToDemand);
  }

  postAll(msg: ToDemand) {
    if (!this.ended) for (const w of this.workers) w.postMessage(msg);
  }

  end() {
    if (this.ended) return;
    this.ended = true;
    for (const w of this.workers) w.terminate();
  }

  private fail(why: string) {
    if (this.ended) return;
    this.end();
    this.hooks.failed(why);
  }

  private receive(i: number, msg: FromDemand) {
    if (this.ended) return;
    const mem = (msg as { memBytes?: number }).memBytes;
    if (mem) this.mem[i] = mem;
    if (msg.kind === "failed" || msg.kind === "dead") return this.fail(`focus worker ${i}: ${msg.error}`); // i18n-ignore
    if (msg.kind === "opened") {
      for (const r of msg.rejected ?? []) if (!r.market) return this.fail(`focus worker ${i}: ${r.id} did not open`); // i18n-ignore
      for (const id of Object.keys(msg.cityInfo) as CityId[]) {
        if (!this.opening.has(id)) continue;
        const a = (this.acks.get(id) ?? 0) + 1;
        this.acks.set(id, a);
        if (a < this.workers.length) continue;
        this.opening.delete(id);
        this.open.push(id);
        this.hooks.opened(id);
      }
      return;
    }
    if (msg.kind === "ready" || msg.kind === "closed" || msg.kind === "packOpened" || msg.kind === "plan" || msg.kind === "loads") return;
    this.hooks.message(i, msg);
  }
}
