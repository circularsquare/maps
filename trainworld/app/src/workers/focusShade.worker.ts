// T-1145: the outline of a focused zone with its sea (boundaryGeometry.unionOutline) off the main
// thread: a zone's own boundary can hold tens of thousands of vertices.
import { unionOutline } from "../map/boundaryGeometry";

/** `rings`: boundaryGeometry.footprintRings */
export type FocusShadeIn = { id: string; rings: Float64Array[] };
export type FocusShadeOut = { id: string; outline: Float64Array[] };

self.onmessage = (e: MessageEvent<FocusShadeIn>) => {
  const outline = unionOutline(e.data.rings);
  (self as unknown as Worker).postMessage({ id: e.data.id, outline } satisfies FocusShadeOut, outline.map((r) => r.buffer));
};
