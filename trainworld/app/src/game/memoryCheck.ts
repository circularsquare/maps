// T-1197: before a game loads, the memory it will take against what the device can spare. Demand's
// solving workers each hold every zone the game reaches (notes/T-1195.md: New York's 5 zones 0.8 GB a
// worker, all of China's 169 about 2.5 GB), so the estimate is per worker from the zones' local trips,
// a stand-in for their packs' size known before anything downloads. A game is refused when a worker
// would pass what one browser worker can hold, or, on a device that reports less than 8 GB, when the
// whole game would take most of it. Thresholds: DECISIONS.md 2026-10-24 (T-1197).

import { cityById, isCity } from "./cities";
import { decodeSave, readStored, type SaveHeader } from "../workers/saveFile";
import { poolSize } from "../workers/demandClient";

/** GB a solving worker holds with no zone, and per million local trips a day of the zones it holds */
const WORKER_BASE_GB = 0.7, WORKER_GB_PER_MTRIPS = 0.0012;
/** GB the page takes besides the solving workers (map, network, the clock worker, the aux worker),
 * and per zone (New York's real save 4.1-4.4 GB in all, all of China 11.5) */
const PAGE_GB = 1.5, PAGE_GB_PER_ZONE = 0.015;
/** a browser worker's memory cannot pass 4 GB; past this a solve stops partway */
const WORKER_CAP_GB = 3.5;
/** of a device reporting under 8 GB (`navigator.deviceMemory`, which never reports more than 8), the
 * share a game may take */
const DEVICE_SHARE = 0.75;

export interface MemoryNeed { zones: number; workerGB: number; workers: number; totalGB: number }

/** The memory a game reaching `zones` will take. */
export function memoryNeed(zones: readonly string[]): MemoryNeed {
  let trips = 0;
  for (const id of zones) if (isCity(id)) trips += cityById(id).localTrips ?? 0;
  const workerGB = WORKER_BASE_GB + WORKER_GB_PER_MTRIPS * trips / 1e6;
  const workers = poolSize();
  return { zones: zones.length, workerGB, workers, totalGB: PAGE_GB + PAGE_GB_PER_ZONE * zones.length + workers * workerGB };
}

const gb = (n: number) => (n < 10 ? n.toFixed(1) : n.toFixed(0));
/** Why a game reaching `zones` will not fit here, in a sentence; null if it will. */
export function memoryRefusal(zones: readonly string[]): string | null {
  const need = memoryNeed(zones);
  if (need.workerGB > WORKER_CAP_GB) return `This game reaches ${need.zones} zones and needs about ${gb(need.workerGB)} GB for each demand calculation. A browser gives each one at most 4 GB.`;
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
