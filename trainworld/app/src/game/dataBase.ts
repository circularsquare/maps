// Where the game's data lives (T-924): the city packs, market packs, outlines, real saves and
// real-network graphs that a build copies into `packs/`.
//
// Locally (8800, the dev server) they sit beside the page in `packs/`. Published, the page is on
// the website and the data on R2 under a versioned prefix, which tools/deploy.py writes into the
// published index.html as `<meta name="tw-data" content="https://data.anita.garden/trainworld/<version>/">`.
// Never decided by hostname: the website served locally reads R2 like the published one. The meta
// rather than a probe for a local-only file, so a published page makes no failing request, and its
// data version is fixed by the same file that names its scripts (DECISIONS.md, T-924).

const doc = (globalThis as { document?: Document }).document;
const meta = doc?.querySelector?.<HTMLMetaElement>('meta[name="tw-data"]')?.content;

/** Absolute URL of the folder holding the build's `packs/` files, ending in a slash. Node checks stub a document
 * with no baseURI. */
export const DATA_BASE = new URL(meta || "packs/", doc?.baseURI || "http://localhost/").href;

/** T-2901: the terrain blocks' folder: `terrain-world/` beside the packs, published the fixed prefix on R2 that
 * tools/deploy.py names in the `tw-terrain` meta (one upload shared by every release). */
const terrainMeta = doc?.querySelector?.<HTMLMetaElement>('meta[name="tw-terrain"]')?.content;
export const TERRAIN_BASE = new URL(terrainMeta || "terrain-world/", DATA_BASE).href;

/** Whether this is the published page (its index.html has the tw-data meta): data from other maps, such as
 * noritetsu's real rail, is then on R2 rather than beside the page (T-941). */
export const PUBLISHED = !!meta;

/** A file in the data folder: `dataUrl("nyc.bin")`, `dataUrl("realgraph/")`. */
export const dataUrl = (path: string) => new URL(path, DATA_BASE).href;
