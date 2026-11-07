// A worker's WASM start, tried again when its download fails (loadfail, 2026-11-06): one dropped or
// cut-short fetch of the .wasm (a busy local server, a rebuild replacing files, a flaky connection)
// otherwise ended the whole load at "Failed to load". wasm-bindgen's init keeps nothing from a failed
// try, so calling it again starts clean. A memory refusal is not retried: it would only fail again.

const WAITS_MS = [1_000, 4_000];

export async function initWasm<T>(init: () => Promise<T>, what: string): Promise<T> {
  for (let k = 0; ; k++) {
    try {
      return await init();
    } catch (err) {
      if (k >= WAITS_MS.length || /memory/i.test(String(err))) throw err;
      console.warn(`${what}: WASM did not start (${err}); trying again`); // i18n-ignore
      await new Promise((r) => setTimeout(r, WAITS_MS[k]));
    }
  }
}
