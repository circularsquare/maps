// T-3315: the simulation is built with WASM SIMD, which Safari has from 16.4 (and Chrome 91, Firefox 89). Older
// browsers fail to compile it with their own message from deep in the worker, so main.ts asks first (as for
// WebGL2, `map/webgl2.ts`) and the loading screen says what to do (game/loading.ts `loadNoSimd`).

/** Whether this browser validates a tiny module that splats a byte into a vector and counts its bits (the probe
 * wasm-feature-detect uses). */
export function hasWasmSimd(): boolean {
  try {
    return typeof WebAssembly === "object" && WebAssembly.validate(new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
    ]));
  } catch {
    return false;
  }
}
