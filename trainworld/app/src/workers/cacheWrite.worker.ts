// T-789: the demand cache's IndexedDB write (workers/saveFile.ts `writeDemandCache`), off the main
// thread: its structured clone of a whole result (the market's corridors are tens of MB) took
// longer than a frame. The result's arrays arrive transferred.

import { writeDemandCache } from "./saveFile";

export interface CacheWrite { id: number; slot: string; hash: string; value: unknown }

self.onmessage = async (e: MessageEvent<CacheWrite>) => {
  const { id, slot, hash, value } = e.data;
  const ok = await writeDemandCache(slot, hash, value);
  postMessage({ id, ok });
};
