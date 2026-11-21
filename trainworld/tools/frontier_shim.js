// T-3887: the browser globals the game's modules touch as they load, for running a few of them in node
// (tools/frontier_fares.ts imports it first). Nothing here is read for the fares.
import.meta.glob = () => ({});
globalThis.location ??= { hash: "", search: "", href: "http://localhost/", pathname: "/", origin: "http://localhost" };
const store = new Map();
globalThis.localStorage ??= { getItem: (k) => store.get(k) ?? null, setItem: (k, v) => store.set(k, String(v)), removeItem: (k) => store.delete(k), clear: () => store.clear(), key: () => null, length: 0 };
globalThis.sessionStorage ??= globalThis.localStorage;
globalThis.window ??= globalThis;
globalThis.addEventListener ??= () => {};
globalThis.matchMedia ??= () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
globalThis.document ??= { addEventListener() {}, documentElement: { style: {}, dataset: {} }, createElement: () => ({ style: {}, getContext: () => null }), querySelector: () => null, body: { style: {} }, visibilityState: "visible" };
