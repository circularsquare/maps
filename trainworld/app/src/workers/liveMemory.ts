// T-3757: the running game's memory from its demand workers' WASM, fitted on headless Chrome's private bytes of all
// the tab's processes (T-3329's saves, two 53-64 city saves and 3 solving workers; notes/T-3757.md). The largest solving
// worker and the aux worker stand for everything that grows with the network (the clock worker, the map, each demand
// worker's JS); each further solving worker adds only itself. WASM memory only: the allocator's live bytes are inside it.

/** bytes with nothing open */
const LIVE_BASE = 1.17 * 2 ** 30;
/** bytes per byte of WASM of the largest solving worker and the aux worker */
const LIVE_PER_FIRST = 2.69;
/** bytes per byte of WASM of each further solving worker */
const LIVE_PER_EXTRA = 1.56;

/** Bytes the game takes with solving workers of `solving` bytes of WASM each and an aux worker of `aux`. */
export function liveBytes(solving: readonly number[], aux: number): number {
  const first = Math.max(0, ...solving), rest = solving.reduce((a, m) => a + m, 0) - first;
  return LIVE_BASE + LIVE_PER_FIRST * (first + aux) + LIVE_PER_EXTRA * rest;
}

/** Bytes outside the demand workers' WASM: what the pool's limit counts as the rest of the page where the browser
 * cannot measure it. */
export const outsideWasmBytes = (first: number, aux: number): number => LIVE_BASE + (LIVE_PER_FIRST - 1) * (first + aux);
