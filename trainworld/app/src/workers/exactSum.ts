/**
 * T-3373 B stage 2 (`?exactSum=1`, notes/T-3373-B-plan.md "Exact sums"): sums across populations that come out
 * the same in any order. Each term is rounded to a whole number of steps (`LOAD_SCALE` or `FARE_SCALE` steps a
 * rider or a dollar) and held in an f64, where such numbers add exactly while a total stays under 2^53 steps
 * (2^33 riders, 2^29 dollars); the total is rounded to f32 once, where the kernel or a result takes it. So a
 * network's figures do not depend on which worker added which population, or in what order. No imports, so a
 * check loads it in Node (`checks/t3373-sums.mjs`).
 */

/** T-3373 B stage 3, off by default: `?split=1` gives each zone and the market one solving worker (`Holding`,
 * demandPlan.ts), `?split=all` holds everything in every worker through the same code (today's pool) */
export const SPLIT: "off" | "all" | "plan" = (() => {
  if (typeof location === "undefined") return "off";
  const v = new URLSearchParams(location.search).get("split");
  return v === "1" ? "plan" : v === "all" ? "all" : "off";
})();

/** What a solving worker's `init` carries, off by default until stage 4: `?exactSum=1` sums this way (and
 * `?split=1`, whose figures must not depend on which worker holds what); dev `?sumOrder=reverse` takes every such
 * sum's populations in reverse order */
export const SUM_FLAGS: { exactSum?: true; reverseSums?: true } = (() => {
  if (typeof location === "undefined") return {};
  const q = new URLSearchParams(location.search), dev = !!(import.meta as { env?: { DEV?: boolean } }).env?.DEV;
  return { ...(q.get("exactSum") === "1" || SPLIT === "plan" ? { exactSum: true as const } : {}), ...(dev && q.get("sumOrder") === "reverse" ? { reverseSums: true as const } : {}) };
})();

/** steps a rider: a load's term is off by at most 2^-21 of a rider */
export const LOAD_SCALE = 2 ** 20;
/** steps a dollar, for the fare extras (`fare_extra`, `base_extra`) */
export const FARE_SCALE = 2 ** 24;
/** steps a unit of `load_of_crush`, a share of crush capacity (totals of a few at most, so room for fine steps) */
export const CRUSH_SCALE = 2 ** 32;

/** `x` rounded to a whole number of steps (exact: the scales are powers of two) */
export const stepOf = (x: number, scale = LOAD_SCALE): number => Math.round(x * scale) / scale;

/** a sum's f64 total rounded to f32, once */
export const toF32 = (a: Float64Array): Float32Array => Float32Array.from(a);

/** `a` += `b` element by element; both hold whole steps, so the order of adding never matters */
export function addInto(a: Float64Array, b: ArrayLike<number>): Float64Array {
  for (let k = 0; k < a.length; k++) a[k] += b[k] ?? 0;
  return a;
}
