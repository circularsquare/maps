import { Deflate, gzip, Inflate } from "pako";

/** T-3395: the error a stream that does not unpack to its end throws; any other (a failed allocation,
 * the output cap) says nothing about the bytes. */
export const GZIP_DAMAGED = "Incomplete or damaged gzip stream."; // i18n-ignore: internal codec error

/** Worker-only strict gzip, including the end marker and checksum. Safari 16's native streams
 * can finish without emitting all their bytes (WebKit 254021), so saves use the bundled codec. */
export function decompressGzip(bytes: Uint8Array, maxBytes = Infinity): Uint8Array<ArrayBuffer> {
  const stream = new Inflate({ windowBits: 31 });
  const parts: Uint8Array[] = [];
  let size = 0;
  stream.onData = (part) => {
    size += part.length;
    if (size > maxBytes) throw new Error("Gzip output is too large."); // i18n-ignore: internal codec error
    parts.push(part);
  };
  if (!stream.push(bytes, true) || stream.err || !stream.ended) throw new Error(GZIP_DAMAGED);
  const out = new Uint8Array(size);
  let at = 0;
  for (const part of parts) { out.set(part, at); at += part.length; }
  return out;
}

/** Verify the bytes before a save can replace the previous slot or rotate its backups. */
export function compressGzip(raw: Uint8Array): Uint8Array<ArrayBuffer> {
  const bytes = gzip(raw);
  const checked = decompressGzip(bytes, raw.length);
  if (checked.length !== raw.length || checked.some((v, i) => v !== raw[i])) throw new Error("Gzip round-trip failed."); // i18n-ignore: internal codec error
  return bytes;
}

/** T-3869: the pieces' bytes are packed and read back 64 KB at a time, so a near-full page need not find room for
 * the whole file, nor for the bytes it packs, in one buffer (a browser short of memory refuses big buffers first). */
export const GZ_PART = 64 << 10;

/** T-3869: what `pieces()` gives, one stream gzipped in parts of at most GZ_PART bytes, the bytes `gzip` makes of
 * them joined. Read back against `pieces()` called again, which must give the same bytes; `room(n)` is asked before
 * each part is kept (checks make it refuse). */
export function compressGzipParts(pieces: () => Iterable<Uint8Array>, room: (n: number) => void = () => {}): Uint8Array<ArrayBuffer>[] {
  const out: Uint8Array<ArrayBuffer>[] = [];
  const d = new Deflate({ gzip: true, chunkSize: GZ_PART });
  d.onData = (part) => { room(part.length); out.push(part); };
  for (const p of pieces()) if (!d.push(p, false)) break;
  if (!d.push(new Uint8Array(0), true) || d.err) throw new Error(`Gzip failed: ${d.msg}`); // i18n-ignore: internal codec error
  readBack(out, pieces);
  return out;
}

const ROUND_TRIP = "Gzip round-trip failed."; // i18n-ignore: internal codec error
function readBack(parts: Uint8Array[], pieces: () => Iterable<Uint8Array>) {
  const want = pieces()[Symbol.iterator]();
  let cur: Uint8Array = new Uint8Array(0), at = 0;
  const stream = new Inflate({ windowBits: 31, chunkSize: GZ_PART });
  stream.onData = (got) => {
    for (let k = 0; k < got.length;) {
      if (at === cur.length) {
        const next = want.next();
        if (next.done) throw new Error(ROUND_TRIP);
        cur = next.value; at = 0;
        continue;
      }
      const n = Math.min(cur.length - at, got.length - k);
      for (let j = 0; j < n; j++) if (got[k + j] !== cur[at + j]) throw new Error(ROUND_TRIP);
      k += n; at += n;
    }
  };
  parts.forEach((p, i) => stream.push(p, i === parts.length - 1));
  if (stream.err || !stream.ended || at !== cur.length) throw new Error(ROUND_TRIP);
  for (let next = want.next(); !next.done; next = want.next()) if (next.value.length) throw new Error(ROUND_TRIP);
}
