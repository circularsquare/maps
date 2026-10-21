// Exact metropolitan footprints, including islands and excluded enclaves, with GeoJSON winding.
import type { Feature, FeatureCollection, MultiLineString, MultiPolygon, Polygon, Position } from "geojson";

function ring(points: Position[], ccw: boolean): Position[] {
  const area = points.slice(1).reduce((sum, p, i) => sum + points[i][0] * p[1] - p[0] * points[i][1], 0);
  return (area > 0) === ccw ? points : [...points].reverse();
}

export function cityGeometry(boundary: FeatureCollection<Polygon | MultiPolygon>): MultiPolygon {
  return {
    type: "MultiPolygon",
    coordinates: boundary.features.flatMap(({ geometry }) =>
      geometry.type === "Polygon" ? [geometry.coordinates] : geometry.coordinates
    ).map(p => p.map((r, i) => ring(r, i === 0))),
  };
}

/** A vertex as one number, on a grid of about a metre: neighbouring zones' shared borders repeat
 * the same vertices (T-799). */
const vertex = (p: Position) => Math.round((p[0] + 180) * 1e5) * 33554432 + Math.round((p[1] + 90) * 1e5);

/** The zones' outlines with each shared border drawn once (T-799): a border two zones both hold
 * belongs to the first zone added, so it is one line at one weight, like an outer edge. A border the
 * focused zone holds is left out entirely, since the focus layers draw it. */
export class OutlineEdges {
  /** each segment's holders: the first zone to hold it, or every zone, in the order they were added */
  private holders = new Map<string, string | string[]>();
  private zones = new Map<string, { ring: Position[]; keys: string[] }[]>();

  get size() { return this.zones.size; }

  add(id: string, city: MultiPolygon) {
    if (this.zones.has(id)) return;
    const rings = city.coordinates.flat().map((ring) => {
      const keys: string[] = [];
      for (let i = 1; i < ring.length; i++) {
        const a = vertex(ring[i - 1]), b = vertex(ring[i]);
        const key = a < b ? `${a}:${b}` : `${b}:${a}`;
        keys.push(key);
        const held = this.holders.get(key);
        if (held === undefined) this.holders.set(key, id);
        else if (Array.isArray(held)) held.push(id);
        else this.holders.set(key, [held, id]);
      }
      return { ring, keys };
    });
    this.zones.set(id, rings);
  }

  /** One feature per zone with the lines it draws (property `id`), none for `focused`. */
  lines(focused: string): FeatureCollection<MultiLineString> {
    const features: Feature<MultiLineString>[] = [];
    for (const [id, rings] of this.zones) {
      if (id === focused) continue;
      const coordinates: Position[][] = [];
      for (const { ring, keys } of rings) {
        const drawn = keys.map((key) => {
          const held = this.holders.get(key)!;
          return typeof held === "string" ? held === id : held[0] === id && !held.includes(focused);
        });
        const start = drawn.indexOf(false);
        if (start < 0) { coordinates.push(ring); continue; }
        // begin just after a gap, so a run that crosses the ring's seam stays one line
        const n = keys.length;
        let line: Position[] = [];
        for (let k = 1; k <= n; k++) {
          const i = (start + k) % n;
          if (drawn[i]) {
            if (!line.length) line.push(ring[i]);
            line.push(ring[i + 1]);
          } else if (line.length) {
            coordinates.push(line);
            line = [];
          }
        }
        if (line.length) coordinates.push(line);
      }
      if (coordinates.length) features.push({ type: "Feature", properties: { id }, geometry: { type: "MultiLineString", coordinates } });
    }
    return { type: "FeatureCollection", features };
  }
}

/** A world-sized polygon with the city's outer rings cut out, plus its excluded enclaves. */
export function outsideCity(city: MultiPolygon): MultiPolygon {
  return {
    type: "MultiPolygon",
    coordinates: [
      [
        [[-180, -85.051129], [180, -85.051129], [180, 85.051129], [-180, 85.051129], [-180, -85.051129]],
        ...city.coordinates.map(p => ring(p[0], false)),
      ],
      ...city.coordinates.flatMap(p => p.slice(1).map(r => [ring(r, true)])),
    ],
  };
}
