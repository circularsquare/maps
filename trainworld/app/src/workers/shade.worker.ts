// T-1026: the realm shade's cut (map/boundaryGeometry.ts, ShadeCut) off the main thread. The coarse
// outlines' file comes once as text; each cut names the zones shown and gets the triangles back.
import { gridRing, ShadeCut, type Ring } from "../map/boundaryGeometry";
import { decodeRing, type OutlineFile } from "../map/coarseOutlines";

export type ShadeIn = { kind: "outlines"; text: string } | { kind: "cut"; gen: number; ids: string[] };
export type ShadeOut = { gen: number; tri: Float64Array };

const rings = new Map<string, Ring[]>();
const shade = new ShadeCut();

self.onmessage = (e: MessageEvent<ShadeIn>) => {
  const m = e.data;
  if (m.kind === "outlines") {
    const { q, zones } = JSON.parse(m.text) as OutlineFile;
    for (const id in zones) rings.set(id, zones[id].flatMap((f) => gridRing(decodeRing(f, q)) ?? []));
    return;
  }
  const tri = shade.cut(m.ids.flatMap((id) => rings.get(id) ?? []));
  (self as unknown as Worker).postMessage({ gen: m.gen, tri } satisfies ShadeOut, [tri.buffer]);
};
