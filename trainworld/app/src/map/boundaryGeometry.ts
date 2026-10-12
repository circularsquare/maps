// Exact metropolitan footprints, including islands and excluded enclaves, with GeoJSON winding.
import type { FeatureCollection, MultiPolygon, Polygon, Position } from "geojson";

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
