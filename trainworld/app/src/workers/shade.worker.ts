// T-1026: the realm shade's cut (map/boundaryGeometry.ts, ShadeCut) off the main thread. The coarse
// outlines come once, as the page's packed copy (T-2802); each cut names the zones shown and gets the triangles back.
import { gridRing, ShadeCut, type Ring } from "../map/boundaryGeometry";
import { decodeRing, packedRings, type PackedOutlines } from "../map/coarseOutlines";

// T-2794, T-3493: the page starts a worker for each cut, ended with its answer, so nothing of the world's outlines
// stays in memory between cuts; a zone's rings are decoded only when a cut names it.
export type ShadeIn = { kind: "outlines"; packed: PackedOutlines } | { kind: "cut"; gen: number; ids: string[] };
export type ShadeOut = { gen: number; tri: Float64Array };

let packed: { q: number; rings: Map<string, Int32Array[]> } | null = null;
const rings = new Map<string, Ring[]>();
const shade = new ShadeCut();

function ringsOf(id: string): Ring[] {
  let r = rings.get(id);
  if (!r) rings.set(id, (r = (packed?.rings.get(id) ?? []).flatMap((f) => gridRing(decodeRing(f, packed!.q)) ?? [])));
  return r;
}

self.onmessage = (e: MessageEvent<ShadeIn>) => {
  const m = e.data;
  if (m.kind === "outlines") return void (packed = { q: m.packed.q, rings: packedRings(m.packed) });
  const tri = shade.cut(m.ids.flatMap(ringsOf));
  (self as unknown as Worker).postMessage({ gen: m.gen, tri } satisfies ShadeOut, [tri.buffer]);
};
