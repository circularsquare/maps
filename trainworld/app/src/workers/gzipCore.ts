import { gzip, Inflate } from "pako";

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
  if (!stream.push(bytes, true) || stream.err || !stream.ended) throw new Error("Incomplete or damaged gzip stream."); // i18n-ignore: internal codec error
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
