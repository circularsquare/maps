import type { Map as MlMap } from "maplibre-gl";

/** Whether style calls (`setPaintProperty`, `setLayoutProperty`) are safe. MapLibre has no style while
 * its WebGL context is lost and a style that is still loading refuses them, so callers skip and
 * reapply their state on `style.load` (T-957). Not `isStyleLoaded()`, which also waits on every tile. */
export const styleUp = (map: MlMap): boolean => !!(map.style as unknown as { _loaded?: boolean } | undefined)?._loaded;
