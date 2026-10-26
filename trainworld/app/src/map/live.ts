// The live map objects, for UI code that needs them (picking, a train's position, the tools).
// Set by main.ts.

import type { Map as MlMap } from "maplibre-gl";
import { behindHorizon } from "./gl";
import type { EdgeEditor } from "./edgeEdit";
import type { NodeDrag } from "./nodeDrag";
import type { Overlay } from "./overlay";
import type { Tools } from "./tools";

/** A place is on the far side of the globe (T-033), so DOM marks placed by `map.project` hide. Asked of
 * the map's own camera as it is now, the one `map.project` uses (T-1480: the overlay's camera is a frame
 * old and, as the shaders do, hides nothing below 20% globe, while `map.project` stays a pure globe
 * until the blend ends: the antipode, Brazil from Wuhan, landed mid-screen from zoom 11.8 to 12). */
export function hiddenOnGlobe(lng: number, lat: number): boolean {
  // (MapLibre's own markers ask the same of the camera's transform, which it does not export)
  type Occlusion = { isLocationOccluded?(l: { lng: number; lat: number }): boolean };
  const m = live.map as unknown as { _camera?: { transform?: Occlusion }; transform?: Occlusion } | null;
  const t = m?._camera?.transform ?? m?.transform;
  if (typeof t?.isLocationOccluded === "function") return t.isLocationOccluded({ lng, lat });
  return behindHorizon(live.overlay?.cam, lng, lat);
}

export const live:{ map: MlMap | null; overlay: Overlay | null; tools: Tools | null; edgeEditor: EdgeEditor | null; nodeDrag: NodeDrag | null } = {
  map: null,
  overlay: null,
  tools: null,
  edgeEditor: null,
  nodeDrag: null,
};
