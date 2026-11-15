import * as codec from "./gzipCore";

onmessage = ({ data }: MessageEvent<{ seq: number; op: "compressGzip" | "decompressGzip"; bytes: Uint8Array; maxBytes: number }>) => {
  try {
    const bytes = data.op === "compressGzip" ? codec.compressGzip(data.bytes) : codec.decompressGzip(data.bytes, data.maxBytes);
    postMessage({ seq: data.seq, bytes }, { transfer: [bytes.buffer] });
  } catch (error) {
    postMessage({ seq: data.seq, error: String(error) });
  }
};
