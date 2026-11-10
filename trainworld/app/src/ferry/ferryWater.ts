// The ferry tool's questions to its water worker (T-2192): where a click by the water puts a port, what is
// under the pointer, and a line's route in the F1 -> F2 format (notes/T-2177.md). The tiles come from the
// basemap's own tile server, the template read from the style as the bus tool reads it.

import type { Map as MlMap } from "maplibre-gl";
import { roadTileTemplate } from "../map/busRoads";
import type { LegStats, LineRoute, LinePoint, LonLat, PortSnap } from "./route";
import { noteStaleBuild } from "../game/staleBuild";
import type { FerryIn, FerryOut } from "../workers/ferryWater.worker";

export type { LineRoute, LinePoint, LonLat, PortSnap, Refusal } from "./route";
export { KIND_PORT, KIND_ROUTE, KIND_WAYPOINT } from "./route";

type Waiting = (m: FerryOut) => void;

export class FerryWater {
  private worker: Worker | null = null;
  private next = 1;
  private waiting = new Map<number, Waiting>();

  constructor(private map: MlMap) {}

  private start(): Worker {
    if (this.worker) return this.worker;
    const w = new Worker(new URL("../workers/ferryWater.worker.ts", import.meta.url), { type: "module" });
    w.addEventListener("error", (e) => { if (!e.message) noteStaleBuild(); }); // no message: its script did not load
    w.onmessage = (e: MessageEvent<FerryOut>) => {
      const f = this.waiting.get(e.data.id);
      this.waiting.delete(e.data.id);
      f?.(e.data);
    };
    void roadTileTemplate(this.map).then((url) => {
      if (url) w.postMessage({ kind: "template", url } satisfies FerryIn);
    });
    return (this.worker = w);
  }

  private ask(m: FerryIn & { id: number }): Promise<FerryOut> {
    const w = this.start();
    return new Promise((res) => {
      this.waiting.set(m.id, res);
      w.postMessage(m);
    });
  }

  /** A port for a click on land by the water: its landing and berth, or why not. */
  async port(at: LonLat): Promise<PortSnap | null> {
    const m = await this.ask({ kind: "port", id: this.next++, at });
    return m.kind === "port" ? m.snap : null;
  }

  /** What is under a point: water, or land a port could go on. Null while the tiles fail. */
  async probe(at: LonLat): Promise<{ water: boolean; shore: boolean } | null> {
    const m = await this.ask({ kind: "probe", id: this.next++, at });
    return m.kind === "probe" ? m.probe : null;
  }

  /** A line's route through its ports' berths and waypoints in order. A newer request with the same `key`
   * (a drag) drops an older one still waiting: it answers null. */
  async line(points: LinePoint[], key?: string): Promise<{ route: LineRoute; stats: (LegStats | null)[]; ms: number } | null> {
    const m = await this.ask({ kind: "line", id: this.next++, points, key });
    return m.kind === "line" ? m : null;
  }

  dispose() {
    this.worker?.terminate();
    this.worker = null;
    for (const f of this.waiting.values()) f({ kind: "dropped", id: 0 });
    this.waiting.clear();
  }
}
