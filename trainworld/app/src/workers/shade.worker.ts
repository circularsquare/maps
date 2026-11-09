// T-1026: the realm shade's cut (map/boundaryGeometry.ts, ShadeCut) off the main thread. The coarse
// outlines' file comes once as text; each cut names the zones shown and gets the triangles back.
import { gridRing, ShadeCut, type Ring } from "../map/boundaryGeometry";
import { decodeRing, type OutlineFile } from "../map/coarseOutlines";

// T-2794: on a small device the page starts a worker for each cut, which fetches the file itself and is
// ended with its answer, so nothing of the world's outlines stays in memory between cuts.
export type ShadeIn = { kind: "outlines"; text: string } | { kind: "fetch"; url: string } | { kind: "cut"; gen: number; ids: string[] };
export type ShadeOut = { gen: number; tri: Float64Array };

const rings = new Map<string, Ring[]>();
const shade = new ShadeCut();
let ready: Promise<void> = Promise.resolve();

function setOutlines(text: string) {
  const { q, zones } = JSON.parse(text) as OutlineFile;
  for (const id in zones) rings.set(id, zones[id].flatMap((f) => gridRing(decodeRing(f, q)) ?? []));
}

self.onmessage = async (e: MessageEvent<ShadeIn>) => {
  const m = e.data;
  if (m.kind === "outlines") return setOutlines(m.text);
  if (m.kind === "fetch") {
    // a failed fetch is an uncaught error, the page's onerror, which cuts on the main thread instead
    ready = fetch(m.url).then((r) => r.text()).then(setOutlines);
    void ready.catch((err) => { setTimeout(() => { throw err; }); });
    return;
  }
  try { await ready; } catch { return; }
  const tri = shade.cut(m.ids.flatMap((id) => rings.get(id) ?? []));
  (self as unknown as Worker).postMessage({ gen: m.gen, tri } satisfies ShadeOut, [tri.buffer]);
};
