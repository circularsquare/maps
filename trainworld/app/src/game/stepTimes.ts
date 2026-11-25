// DEV (T-4284): ms of named steps of the page's work, summed until read. `twSteps()` in a DEV page returns each step's
// ms and calls and starts again; a step inside another counts in both.

const sums = new Map<string, [number, number]>();
const on = !!import.meta.env?.DEV;

/** `f`, its ms added to step `name` (DEV) */
export function timed<T>(name: string, f: () => T): T {
  if (!on) return f();
  const t0 = performance.now();
  try {
    return f();
  } finally {
    const s = sums.get(name) ?? sums.set(name, [0, 0]).get(name)!;
    s[0] += performance.now() - t0;
    s[1]++;
  }
}

/** the ms since `t0` added to step `name` (DEV); the time to pass as the next `t0` */
export function stepSince(name: string, t0: number): number {
  if (!on) return 0;
  const now = performance.now(), s = sums.get(name) ?? sums.set(name, [0, 0]).get(name)!;
  s[0] += now - t0;
  s[1]++;
  return now;
}

if (on && typeof window !== "undefined")
  (window as unknown as { twSteps: () => Record<string, { ms: number; n: number }> }).twSteps = () => {
    const out = Object.fromEntries([...sums].map(([k, [ms, n]]) => [k, { ms: Math.round(ms * 100) / 100, n }]));
    sums.clear();
    return out;
  };
