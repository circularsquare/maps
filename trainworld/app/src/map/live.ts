// The live map objects, for UI code that needs them (picking, a train's position, the tools).
// Set by main.ts.

import type { Map as MlMap } from "maplibre-gl";
import { behindHorizon } from "./gl";
import type { EdgeEditor } from "./edgeEdit";
import type { NodeDrag } from "./nodeDrag";
import type { Overlay } from "./overlay";
import type { Tools } from "./tools";

/** A place is on the far side of the globe (T-033), so DOM marks placed by `map.project` hide. */
export function hiddenOnGlobe(lng: number, lat: number): boolean {
  return behindHorizon(live.overlay?.cam, lng, lat);
}

export const live:{ map: MlMap | null; overlay: Overlay | null; tools: Tools | null; edgeEditor: EdgeEditor | null; nodeDrag: NodeDrag | null } = {
  map: null,
  overlay: null,
  tools: null,
  edgeEditor: null,
  nodeDrag: null,
};
