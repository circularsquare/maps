/**
 * trainsworld public copy: the game's data (city packs, terrain, outlines, real networks) is 6 GB and is not in git.
 * With no local `data/packs` folder, the dev server answers every `/packs/...` and `/noritetsu/...` request itself
 * by fetching the published copy from data.anita.garden (a server-to-server fetch, so the bucket's CORS never
 * applies). Only the files the game asks for are fetched, so a first session downloads tens of MB.
 *   TW_DATA_VERSION=<folder>   which published data version (default: data-version.txt beside this repo's trainworld/)
 *   TW_DATA=off                never fetch; use local data/ only
 * With a local `data/packs` (or TW_PACKS) this does nothing.
 */
function dataFallback(): Plugin {
  const ORIGIN = "https://data.anita.garden";
  const versionFile = resolve(here, "../data-version.txt");
  const version = process.env.TW_DATA_VERSION || (existsSync(versionFile) ? readFileSync(versionFile, "utf8").trim() : "");
  const on = process.env.TW_DATA !== "off" && !!version && !existsSync(PACKS);
  // fetch has already undone a gzip Content-Encoding, so that header and the length are not passed on
  const KEEP = ["content-type", "content-range", "accept-ranges", "cache-control", "etag"];
  return {
    name: "trainsworld-data-fallback",
    configureServer(server) {
      if (!on) return;
      console.log(`[data] no local data/packs: fetching the game's data from ${ORIGIN}/trainworld/${version}/ as the page asks`);
      server.middlewares.use(async (req: any, res: any, next: () => void) => {
        const url: string = req.url ?? "";
        let target: string;
        if (url.startsWith("/packs/terrain-world/")) target = `${ORIGIN}/trainworld/terrain-world/v1/${url.slice("/packs/terrain-world/".length)}`;
        else if (url.startsWith("/packs/")) target = `${ORIGIN}/trainworld/${version}/${url.slice("/packs/".length)}`;
        else if (url.startsWith("/noritetsu/")) target = `${ORIGIN}${url}`;
        else return next();
        try {
          const headers: Record<string, string> = {};
          if (req.headers.range) headers.range = req.headers.range;
          const r = await fetch(target, { headers });
          res.statusCode = r.status;
          for (const k of KEEP) { const v = r.headers.get(k); if (v) res.setHeader(k, v); }
          const body = Buffer.from(await r.arrayBuffer());
          res.setHeader("content-length", body.length);
          res.end(body);
        } catch {
          res.statusCode = 502;
          res.end();
        }
      });
    },
  };
}

