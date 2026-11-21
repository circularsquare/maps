// T-4004: the search box's towns and neighbourhoods, asked of a worker (workers/placeSearch.worker.ts),
// which loads the near zones' place lists and the world name index's shards and posts back the best
// few. The main thread only sends the query and reads at most MAX_HITS rows back.

import { signal } from "@preact/signals";
import { CITIES, zoneBox } from "./cities";
import { DATA_BASE } from "./dataBase";
import { normalise } from "./nameMatch";
import type { Place } from "./placeList";

export interface ZoneInfo { id: string; box: [number, number, number, number]; name: string; center: [number, number] }
export type PlaceSearchIn =
  | { type: "init"; base: string; zones: ZoneInfo[] }
  /** q "": no search, only the view's lists loaded ahead */
  | { type: "query"; seq: number; q: string; lng: number; lat: number };
/** `near`: within the worker's NEAR_M of the view, which ranks it first among equal matches */
export interface PlaceFound { place: Place; score: number; near: boolean }

/** The latest answer: `q` normalised; `done` once nothing more is coming for it. */
export const placeResults = signal<{ q: string; hits: PlaceFound[]; done: boolean }>({ q: "", hits: [], done: true });

let worker: Worker | null = null, broken = false, seq = 0, sentZones = -1, asked = "";

/** Ask for the places matching `query` near (lng, lat); an empty query loads the view's lists ahead. */
export function findPlaces(query: string, lng: number, lat: number) {
  const q = normalise(query);
  asked = q;
  if (!broken && !worker) {
    try {
      worker = new Worker(new URL("../workers/placeSearch.worker.ts", import.meta.url), { type: "module" });
      worker.onmessage = (e: MessageEvent<{ seq: number; hits: PlaceFound[]; done: boolean }>) => {
        if (e.data.seq === seq) placeResults.value = { q: asked, hits: e.data.hits, done: e.data.done };
      };
      worker.onerror = () => {
        broken = true;
        worker?.terminate();
        worker = null;
        placeResults.value = { q: asked, hits: [], done: true };
      };
    } catch {
      broken = true;
    }
  }
  if (!worker) {
    placeResults.value = { q, hits: [], done: true };
    return;
  }
  if (sentZones !== CITIES.length) {
    sentZones = CITIES.length;
    const zones = CITIES.map((c): ZoneInfo => ({ id: c.id, box: zoneBox(c), name: c.name, center: [c.center[0], c.center[1]] }));
    worker.postMessage({ type: "init", base: DATA_BASE, zones } satisfies PlaceSearchIn);
  }
  seq++;
  if (!q) placeResults.value = { q, hits: [], done: true };
  worker.postMessage({ type: "query", seq, q, lng, lat } satisfies PlaceSearchIn);
}
