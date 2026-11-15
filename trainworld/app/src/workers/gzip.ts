import * as codec from "./gzipCore";

type Operation = "compressGzip" | "decompressGzip";
let worker: Worker | null = null, seq = 0;
const waiting = new Map<number, { resolve: (bytes: Uint8Array<ArrayBuffer>) => void; reject: (error: Error) => void }>();

/** The clock already runs in a worker; file pickers and Community use a separate codec worker
 * so a large save never compresses or inflates on the UI thread. */
function call(op: Operation, bytes: Uint8Array, maxBytes = Infinity): Promise<Uint8Array<ArrayBuffer>> {
  if (typeof document === "undefined") return Promise.resolve(op === "compressGzip" ? codec.compressGzip(bytes) : codec.decompressGzip(bytes, maxBytes));
  if (!worker) {
    worker = new Worker(new URL("./gzip.worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = ({ data }: MessageEvent<{ seq: number; bytes?: Uint8Array<ArrayBuffer>; error?: string }>) => {
      const pending = waiting.get(data.seq);
      waiting.delete(data.seq);
      if (data.bytes) pending?.resolve(data.bytes);
      else pending?.reject(new Error(data.error));
    };
    worker.onerror = () => {
      worker?.terminate(); worker = null;
      for (const pending of waiting.values()) pending.reject(new Error("Gzip worker failed.")); // i18n-ignore: internal codec error
      waiting.clear();
    };
  }
  const id = ++seq, copy = bytes.slice();
  return new Promise((resolve, reject) => {
    waiting.set(id, { resolve, reject });
    try { worker!.postMessage({ seq: id, op, bytes: copy, maxBytes }, [copy.buffer]); }
    catch (error) { waiting.delete(id); reject(error); }
  });
}

export const compressGzip = (raw: Uint8Array): Promise<Uint8Array<ArrayBuffer>> => call("compressGzip", raw);
export const decompressGzip = (bytes: Uint8Array, maxBytes = Infinity): Promise<Uint8Array<ArrayBuffer>> => call("decompressGzip", bytes, maxBytes);
