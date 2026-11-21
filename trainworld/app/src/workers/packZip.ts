// T-2486: published `.bin` packs are one gzip member per array (tools/packzip.py), and their header says where
// each member sits: `bin_gz: { size, parts: [raw offset, offset in the file, length][] }`. The header's `arrays`
// keep the raw layout, so a whole read is inflated back to it and a Range read is mapped onto the members
// covering it. A header without `bin_gz` is a raw pack, read as it is.

export interface BinGz { size: number; parts: [number, number, number][] }

/** gzip's magic bytes: `market-drives.bin` published gzipped (a raw one starts "TWD1") */
export const isGzip = (b: Uint8Array) => b.length >= 2 && b[0] === 0x1f && b[1] === 0x8b;

export async function gunzip(b: Uint8Array): Promise<Uint8Array> {
  return new Uint8Array(await new Response(new Blob([b as BlobPart]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
}

/** The member index in a pack header (its JSON, parsed or as bytes), or null for a raw pack */
export function binGzOf(head: unknown): BinGz | null {
  const h = head instanceof Uint8Array ? JSON.parse(new TextDecoder().decode(head)) : head;
  const z = (h as { bin_gz?: BinGz } | null)?.bin_gz;
  return z && typeof z.size === "number" && Array.isArray(z.parts) && z.parts.length ? z : null;
}

const rawEnd = (z: BinGz, k: number) => (k + 1 < z.parts.length ? z.parts[k + 1][0] : z.size);

/** Members `from`..`to` (inclusive) of `buf`, whose byte 0 is the file's byte `base`, inflated into `out` at
 * their raw offsets less `outBase` */
async function inflateInto(z: BinGz, buf: Uint8Array, base: number, from: number, to: number, out: Uint8Array, outBase: number) {
  const jobs: Promise<void>[] = [];
  for (let k = from; k <= to; k++) {
    const [s, zo, zl] = z.parts[k], want = rawEnd(z, k) - s;
    jobs.push(gunzip(buf.subarray(zo - base, zo - base + zl)).then((d) => {
      if (d.length !== want) throw new Error(`pack member ${k} is ${d.length} bytes, not ${want}`);
      out.set(d, s - outBase);
    }));
  }
  await Promise.all(jobs);
}

/** A whole packed `.bin` back to its raw bytes */
export async function unpackBin(bin: Uint8Array, z: BinGz): Promise<Uint8Array> {
  const last = z.parts[z.parts.length - 1];
  if (bin.length < last[1] + last[2]) throw new Error(`packed .bin is ${bin.length} bytes, its index wants ${last[1] + last[2]}`);
  const out = new Uint8Array(z.size);
  await inflateInto(z, bin, 0, 0, z.parts.length - 1, out, 0);
  return out;
}

/** For raw bytes [start, end): the members covering them, and the file bytes holding those members */
export function membersFor(z: BinGz, start: number, end: number): { from: number; to: number; zStart: number; zEnd: number } {
  let from = 0, to = 0;
  for (let k = 0; k < z.parts.length; k++) {
    if (z.parts[k][0] <= start) from = k;
    if (z.parts[k][0] < end) to = k;
  }
  return { from, to, zStart: z.parts[from][1], zEnd: z.parts[to][1] + z.parts[to][2] };
}

/** Raw bytes [start, end) from `buf`, the file's bytes [zStart, zEnd) as `membersFor` named them */
export async function inflateSpan(z: BinGz, buf: Uint8Array, start: number, end: number): Promise<ArrayBuffer> {
  const m = membersFor(z, start, end), lo = z.parts[m.from][0], hi = rawEnd(z, m.to);
  const out = new Uint8Array(hi - lo);
  await inflateInto(z, buf, m.zStart, m.from, m.to, out, lo);
  return out.buffer.slice(start - lo, end - lo);
}
