// T-4275: OpenFreeMap's vector tiles: the TileJSON at /planet and every tile under it, which the basemap, crossing
// prices, bus routes and ferry water all read. A tracker list can block that path by mistake (EasyPrivacy's
// `||tiles.openfreemap.org/planet`, in uBlock Origin, Brave and NordVPN, left no roads or water), so the
// TileJSON is read here, from the first route that answers it, and its tile addresses are put on that route.
// Unblocked players stay on the first route, the one OpenFreeMap's cache holds.

export const OPENFREEMAP = "https://tiles.openfreemap.org/";
/** The same files by other addresses, in the order tried. "//planet" is served as "/planet" but is outside a rule
 * written for "/planet"; a proxy on a host of ours would go after it (notes/T-4275.md). */
const ROUTES = [OPENFREEMAP, OPENFREEMAP + "/"];
/** A vector source's TileJSON address under this protocol is answered by `openMapTiles` (basemap.ts adds it). */
export const TILEJSON_PROTOCOL = "twtiles";
const PREFIX = `${TILEJSON_PROTOCOL}://`;

/** an OpenFreeMap TileJSON address as the style should name it; any other address unchanged */
export const routedTileJson = (url: string) => (url.startsWith(OPENFREEMAP) ? PREFIX + url.slice(OPENFREEMAP.length) : url);
export const isRoutedTileJson = (url: string) => url.startsWith(PREFIX);

export interface TileJson { tiles: string[]; [field: string]: unknown }

const asked = new Map<string, Promise<TileJson>>();

/** The TileJSON a `routedTileJson` address names, from the first route that gives it, with its tiles on that route.
 * A blocker may refuse the request or answer 204 with no body (NordVPN), so anything without a tile address counts
 * as blocked. */
export function openMapTiles(url: string): Promise<TileJson> {
  const path = url.slice(PREFIX.length);
  let p = asked.get(path);
  if (!p) {
    p = (async () => {
      let why: unknown = null;
      for (const route of ROUTES) {
        try {
          const res = await fetch(route + path);
          const json = res.ok ? (JSON.parse((await res.text()) || "null") as Partial<TileJson> | null) : null;
          if (Array.isArray(json?.tiles) && json.tiles.length) {
            return { ...json, tiles: json.tiles.map((t) => (t.startsWith(OPENFREEMAP) ? route + t.slice(OPENFREEMAP.length) : t)) };
          }
          why = new Error(`${route}${path} answered ${res.status} with no tiles`);
        } catch (e) {
          why = e;
        }
      }
      asked.delete(path); // asked again next time (offline now, back later)
      throw why;
    })();
    asked.set(path, p);
  }
  return p;
}
