// T-2499: bytes [start, end) of a file, fetched with a Range request. A server or the browser's cache
// may answer it with the range (206), the whole file (200), or a 200 carrying the range alone (Chrome,
// while the same pack downloads whole for demand): the last was read as the whole file, so a zone's
// street speeds came out empty and its buses ran at the top speed until a reload.

// T-1635: not stored in the browser cache either: Chrome could not answer the whole file that follows from a
// partly stored one, and the retry logged a console error (net::ERR_CACHE_OPERATION_NOT_SUPPORTED).

// T-3196: given up after `timeoutMs`, so a stalled connection never holds the clock worker's queue for ever.

// T-2486: `start` and `end` are always the raw layout's, as the pack header's `arrays` give them. For a packed
// `.bin` (its header has `bin_gz`, workers/packZip.ts) the members covering them are fetched and inflated.

import { binGzOf, inflateSpan, isGzip, membersFor, type BinGz } from "./packZip";

/** each pack's member index, from its header (fetched as the caller just did, so from the browser cache) */
const indexes = new Map<string, Promise<BinGz | null>>();
function indexOf(url: string, timeoutMs: number): Promise<BinGz | null> {
  if (!url.endsWith(".bin")) return Promise.resolve(null);
  let p = indexes.get(url);
  if (!p) {
    p = (async () => {
      const res = await fetch(url.slice(0, -4) + ".json", { signal: AbortSignal.timeout(timeoutMs) });
      if (!res.ok) throw new Error(`${res.status} ${url.slice(0, -4)}.json`);
      const z = binGzOf(await res.json());
      // a packed .bin under a header without the index (another upload's) reads as garbage and stalls the ground:
      // refused, as a failed fetch
      if (!z && isGzip(new Uint8Array(await fetchBytes(url, 0, 2, timeoutMs)))) throw new Error(`${url} is packed but its header has no bin_gz`);
      return z;
    })();
    indexes.set(url, p);
    p.catch(() => indexes.get(url) === p && indexes.delete(url));
  }
  return p;
}

export async function fetchRange(url: string, start: number, end: number, timeoutMs = 60_000): Promise<ArrayBuffer> {
  const z = await indexOf(url, timeoutMs);
  if (!z) return fetchBytes(url, start, end, timeoutMs);
  const m = membersFor(z, start, end);
  const buf = await fetchBytes(url, m.zStart, m.zEnd, timeoutMs);
  return inflateSpan(z, new Uint8Array(buf), start, end);
}

async function fetchBytes(url: string, start: number, end: number, timeoutMs: number): Promise<ArrayBuffer> {
  const res = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` }, cache: "no-store", signal: AbortSignal.timeout(timeoutMs) });
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  const buf = await res.arrayBuffer();
  const want = end - start;
  if (res.status === 206) {
    const from = /bytes (\d+)-/.exec(res.headers.get("content-range") ?? "");
    if (from && Number(from[1]) !== start) throw new Error(`range from ${from[1]}, not ${start}: ${url}`);
    if (buf.byteLength < want) throw new Error(`short range ${buf.byteLength} of ${want}: ${url}`);
    return buf.byteLength === want ? buf : buf.slice(0, want);
  }
  if (buf.byteLength >= end) return buf.slice(start, end);
  if (buf.byteLength === want) return buf;
  throw new Error(`${buf.byteLength} bytes for a range of ${want} from ${start}: ${url}`);
}
