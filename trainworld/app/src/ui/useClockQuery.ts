// Asking the clock worker an inspector question and keeping the last answer in a panel.

import { useEffect, useState } from "preact/hooks";
import { queryClock } from "../workers/clockClient";

export type ClockQuery = Parameters<typeof queryClock>[0];

/** The worker's answer to `req`, parsed, asked again when `deps` change; null until it comes (and
 * with a null `req`). A newer ask drops an older answer still on its way. The last answer stays
 * while the next is awaited, unless `reset`: then the panel is null in between. */
export function useClockQuery<T>(req: ClockQuery | null, deps: unknown[], parse: (d: Float64Array) => T, reset = false): T | null {
  const [value, set] = useState<T | null>(null);
  useEffect(() => {
    if (!req) return;
    let on = true;
    if (reset) set(null);
    void queryClock(req).then((d) => { if (on && d) set(parse(d)); });
    return () => { on = false; };
  }, deps);
  return value;
}

/** The latest answer of `ask`, asked again when `deps` change; null until one comes. A null answer
 * (a worker busy with something newer) keeps the last one, and `ask` null asks nothing. */
export function useLatestAnswer<T>(ask: (() => Promise<T | null>) | null, deps: unknown[]): T | null {
  const [value, set] = useState<T | null>(null);
  useEffect(() => {
    if (!ask) return;
    let on = true;
    void ask().then((r) => { if (on && r) set(r); });
    return () => { on = false; };
  }, deps);
  return value;
}
