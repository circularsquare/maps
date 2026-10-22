// Where the game's data lives (T-924): the city packs, market packs, outlines, real saves and
// real-network graphs that a build copies into `packs/`.
//
// Locally (8800, the dev server) they sit beside the page in `packs/`. Published, the page is on
// the website and the data on R2 under a versioned prefix, which tools/deploy.py writes into the
// published index.html as `<meta name="tw-data" content="https://data.anita.garden/trainworld/<version>/">`.
// Never decided by hostname: the website served locally reads R2 like the published one. The meta
// rather than a probe for a local-only file, so a published page makes no failing request, and its
// data version is fixed by the same file that names its scripts (DECISIONS.md, T-924).

function resolve(): string {
  const doc = (globalThis as { document?: Document }).document;
  // Node checks stub a document with no baseURI
  const page = doc?.baseURI || "http://localhost/";
  const meta = doc?.querySelector?.<HTMLMetaElement>('meta[name="tw-data"]')?.content;
  return new URL(meta || "packs/", page).href;
}

/** Absolute URL of the folder holding the build's `packs/` files, ending in a slash. */
export const DATA_BASE = resolve();

/** A file in the data folder: `dataUrl("nyc.bin")`, `dataUrl("realgraph/")`. */
export const dataUrl = (path: string) => new URL(path, DATA_BASE).href;
