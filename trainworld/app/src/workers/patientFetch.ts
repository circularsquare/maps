// T-3313: `fetch` that gives up when the connection goes quiet. A browser can hold a request for a very long
// time on a connection that died without an error (a laptop waking on another network, a phone switching
// from Wi-Fi), and whatever waits on it waits with it: a demand pack held every solve behind it, an account
// call every sync after it. The limit is on silence, not on the whole download, so a big pack on a slow line
// still loads: the clock restarts at the response's headers and at every chunk of its body. A request given
// up rejects as any network failure does ("TimeoutError"), and its callers' retries take over.

/** No headers, or no byte of the body, for this long, ms: the request is given up */
export const STALL_MS = 45_000;
/** For the account's calls: a save going up (up to 4 MB) is sent before any header comes back, so a slow
 * upload must not count as silence */
export const CLOUD_STALL_MS = 120_000;

export async function patientFetch(input: string, init: RequestInit = {}, stallMs = STALL_MS): Promise<Response> {
  const ctl = new AbortController();
  const outer = init.signal;
  if (outer) {
    if (outer.aborted) ctl.abort(outer.reason);
    else outer.addEventListener("abort", () => ctl.abort(outer.reason), { once: true });
  }
  let timer: ReturnType<typeof setTimeout> | undefined;
  const stop = () => clearTimeout(timer);
  const poke = () => {
    stop();
    // i18n-ignore: a developer diagnostic
    timer = setTimeout(() => ctl.abort(new DOMException(`No data for ${stallMs / 1000} s`, "TimeoutError")), stallMs);
  };
  poke();
  let r: Response;
  try {
    r = await fetch(input, { ...init, signal: ctl.signal });
  } catch (err) {
    stop();
    throw err;
  }
  if (!r.body || r.status === 204 || r.status === 205 || r.status === 304) { stop(); return r; }
  poke();
  const reader = r.body.getReader();
  const body = new ReadableStream<Uint8Array>({
    async pull(c) {
      try {
        const { done, value } = await reader.read();
        if (done) { stop(); c.close(); return; }
        poke();
        c.enqueue(value);
      } catch (err) {
        stop();
        c.error(err);
      }
    },
    cancel(reason) { stop(); return reader.cancel(reason); },
  });
  const out = new Response(body, { status: r.status, statusText: r.statusText, headers: r.headers });
  // the callers name the file by it in their errors
  Object.defineProperty(out, "url", { value: r.url });
  return out;
}
