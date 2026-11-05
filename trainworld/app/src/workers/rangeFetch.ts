// T-2499: bytes [start, end) of a file, fetched with a Range request. A server or the browser's cache
// may answer it with the range (206), the whole file (200), or a 200 carrying the range alone (Chrome,
// while the same pack downloads whole for demand): the last was read as the whole file, so a zone's
// street speeds came out empty and its buses ran at the top speed until a reload.

export async function fetchRange(url: string, start: number, end: number): Promise<ArrayBuffer> {
  const res = await fetch(url, { headers: { Range: `bytes=${start}-${end - 1}` } });
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
