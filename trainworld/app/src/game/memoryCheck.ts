// T-1197: before a game loads, the memory it will take against what the device can spare. Demand's
// solving workers each hold every zone the game reaches, so the estimate is per worker from the zones'
// local trips, a stand-in for their packs' size known before anything downloads. A game is refused
// when a worker would pass what one browser worker can hold, or, on a device that reports less than
// 8 GB, when the whole game would take most of it. Thresholds: DECISIONS.md 2026-10-24 (T-1197).
// T-1223: zones not looked at are held coarse (workers/demandLod.ts), so a game is judged at the least
// it can take: its biggest zone fine and the rest coarse.

import { cityById, isCity } from "./cities";
import { decodeSave, readStored, type SaveHeader } from "../workers/saveFile";
import { poolSize } from "../workers/demandClient";
import { COARSE_SHARE, GB_PER_MTRIPS } from "../workers/demandLod";

/** GB a solving worker takes outside its WebAssembly memory (its JS heap, T-1205); the browser's 4 GB
 * is on the WebAssembly memory alone, which is GB_PER_MTRIPS per million local trips a day held fine */
const WORKER_BASE_GB = 0.7;
/** GB the page takes besides the solving workers (map, network, the clock worker, the aux worker),
 * and per zone (New York's real save 4.1-4.4 GB in all, all of China 11.5) */
const PAGE_GB = 1.5, PAGE_GB_PER_ZONE = 0.015;
/** a browser worker's memory cannot pass 4 GB; past this a solve stops partway */
const WORKER_CAP_GB = 3.5;
/** of a device reporting under 8 GB (`navigator.deviceMemory`, which never reports more than 8), the
 * share a game may take */
const DEVICE_SHARE = 0.75;

/** `wasmGB`: a solving worker's WebAssembly memory; `workerGB` with its JS heap */
export interface MemoryNeed { zones: number; wasmGB: number; workerGB: number; workers: number; totalGB: number }

/** The memory a game reaching `zones` will take, its biggest zone fine and the others coarse. */
export function memoryNeed(zones: readonly string[]): MemoryNeed {
  let trips = 0, most = 0;
  for (const id of zones) if (isCity(id)) { const t = cityById(id).localTrips ?? 0; trips += t; most = Math.max(most, t); }
  const wasmGB = GB_PER_MTRIPS * (most + COARSE_SHARE * (trips - most)) / 1e6;
  const workerGB = WORKER_BASE_GB + wasmGB;
  const workers = poolSize();
  return { zones: zones.length, wasmGB, workerGB, workers, totalGB: PAGE_GB + PAGE_GB_PER_ZONE * zones.length + workers * workerGB };
}

const gb = (n: number) => (n < 10 ? n.toFixed(1) : n.toFixed(0));
/** Why a game reaching `zones` will not fit here, in a sentence; null if it will. */
export function memoryRefusal(zones: readonly string[]): string | null {
  const need = memoryNeed(zones);
  if (need.wasmGB > WORKER_CAP_GB) return `This game covers ${need.zones} zones. Each demand calculation needs about ${gb(need.wasmGB)} GB and browsers allow 4.`;
  const device = (navigator as { deviceMemory?: number }).deviceMemory;
  if (device !== undefined && device < 8 && need.totalGB > DEVICE_SHARE * device) return `This game needs about ${gb(need.totalGB)} GB of memory and this device has about ${gb(device)} GB.`;
  return null;
}

/** The zones a saved game reaches: those its save lists as opened, else (a save from before T-729) the
 * cities its slot's stations are in. */
export function zonesOf(header: SaveHeader | null, cities?: string[]): string[] {
  return header?.opened ?? cities ?? [];
}

/** T-1197: why the save in `bytes` (a file) or slot `key` will not fit here; null if it will or cannot be read. */
export async function saveRefusal(src: { bytes: Uint8Array } | { key: string }): Promise<string | null> {
  try {
    const stored = "key" in src ? await readStored(src.key) : null;
    const bytes = "bytes" in src ? src.bytes : stored?.bytes;
    if (!bytes) return null;
    const { header } = await decodeSave(bytes);
    return memoryRefusal(zonesOf(header, stored?.cities));
  } catch { return null; }
}
