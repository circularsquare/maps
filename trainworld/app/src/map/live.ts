// The live map objects, for UI code that needs them (picking, a train's position, the tools).
// Set by main.ts.

import type { Map as MlMap } from "maplibre-gl";
import type { EdgeEditor } from "./edgeEdit";
import type { NodeDrag } from "./nodeDrag";
import type { Overlay } from "./overlay";
import type { Tools } from "./tools";

export const live: { map: MlMap | null; overlay: Overlay | null; tools: Tools | null; edgeEditor: EdgeEditor | null; nodeDrag: NodeDrag | null } = {
  map: null,
  overlay: null,
  tools: null,
  edgeEditor: null,
  nodeDrag: null,
};
