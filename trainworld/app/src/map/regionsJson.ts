// regions.json (about 1.5 MB, the outlines of noritetsu's regions), fetched once for the page and the
// real-route worker, which gets the outlines in a message (T-2575).
import { noritetsuBases } from "../game/rings";
import { PUBLISHED } from "../game/dataBase";
import { patientFetch } from "../workers/patientFetch";

export interface RegionsJson {
  /** the base that served it, which serves the regions' pmtiles too */
  base: string;
  regions: Record<string, { parts: [number, number][][] }>;
}

let loading: Promise<RegionsJson | null> | null = null;

export function regionsJson(): Promise<RegionsJson | null> {
  return (loading ??= (async () => {
    for (const base of noritetsuBases(document.baseURI, PUBLISHED)) {
      try {
        const r = await patientFetch(base + "regions.json"); // T-3314: a held request gives up, and the next base is tried
        const j = r.ok ? await r.json() : null;
        if (j?.regions) return { base, regions: j.regions };
      } catch {
        // not there, or not JSON: try the next
      }
    }
    return null;
  })());
}
