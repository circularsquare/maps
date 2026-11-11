// T-1026: the realm shade's cut (map/boundaryGeometry.ts, ShadeCut) off the main thread. The coarse
// outlines come once, as the page's packed copy (T-2802); each cut names the zones shown and gets the triangles back.
import { gridRing, ShadeCut, type Ring } from "../map/boundaryGeometry";
import { decodeRing, packedRings, type PackedOutlines } from "../map/coarseOutlines";

// T-2794: on a small device the page starts a worker for each cut, which is ended with its answer, so nothing
// of the world's outlines stays in memory between cuts.
export type ShadeIn = { kind: "outlines"; packed: PackedOutlines } | { kind: "cut"; gen: number; ids: string[] };
export type ShadeOut = { gen: number; tri: Float64Array };

const rings = new Map<string, Ring[]>();
const shade = new ShadeCut();

function setOutlines(p: PackedOutlines) {
  for (const [id, rs] of packedRings(p)) rings.set(id, rs.flatMap((f) => gridRing(decodeRing(f, p.q)) ?? []));
}

self.onmessage = (e: MessageEvent<ShadeIn>) => {
  const m = e.data;
  if (m.kind === "outlines") return setOutlines(m.packed);
  const tri = shade.cut(m.ids.flatMap((id) => rings.get(id) ?? []));
  (self as unknown as Worker).postMessage({ gen: m.gen, tri } satisfies ShadeOut, [tri.buffer]);
};
