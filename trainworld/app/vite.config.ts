import { copyFileSync, createReadStream, existsSync, mkdirSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
// TW_PACKS: a frozen copy instead (pipeline/freeze.py build, T-928)
const PACKS = process.env.TW_PACKS ? resolve(process.env.TW_PACKS) : resolve(here, "../data/packs");
/** The packs the game opens: the registry's cities (T-125), and its regional market. Pipeline output, gitignored. */
const REGISTRY = JSON.parse(readFileSync(resolve(here, "src/game/registry.json"), "utf8")) as { cities: { id: string }[]; market: { pack: string } };
const CITIES = REGISTRY.cities.map((c) => c.id);

/**
 * City packs at `packs/<city>.json`, `.bin`, `.boundary.geojson`, `.market.json` (T-377) and `.places.json` (the OSM place
 * list, pipeline/places.py, fetched only when search or station naming needs it) beside the page: served from
 * data/packs/ by the dev server and copied into dist/packs/ by a build. The clock worker reads the
 * water mask from it with a Range request, the demand workers the cells. `packs/outlines.json` (T-800) is the one file
 * shared by every zone.
 */
function packs(): Plugin {
  return {
    name: "trainworld-packs",
    configureServer(server) {
      server.middlewares.use("/packs", (req, res, next) => {
        const file = join(PACKS, (req.url ?? "").split("?")[0].replace(/^\/+/, ""));
        if (!file.startsWith(PACKS) || !existsSync(file)) return next();
        const size = statSync(file).size;
        const range = /bytes=(\d+)-(\d*)/.exec(req.headers.range ?? "");
        res.setHeader("Accept-Ranges", "bytes");
        res.setHeader("Content-Type", file.endsWith(".geojson") ? "application/geo+json" : file.endsWith(".json") ? "application/json" : "application/octet-stream");
        if (range) {
          const a = Number(range[1]), b = range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
          res.statusCode = 206;
          res.setHeader("Content-Range", `bytes ${a}-${b}/${size}`);
          res.setHeader("Content-Length", b - a + 1);
          createReadStream(file, { start: a, end: b }).pipe(res);
        } else {
          res.setHeader("Content-Length", size);
          createReadStream(file).pipe(res);
        }
      });
    },
    writeBundle(opts) {
      const out = join(opts.dir ?? resolve(here, "dist"), "packs");
      mkdirSync(out, { recursive: true });
      for (const c of CITIES)
        for (const ext of [".json", ".bin", ".boundary.geojson", ".places.json", ".market.json"]) {
          const src = join(PACKS, c + ext);
          if (existsSync(src)) copyFileSync(src, join(out, c + ext));
          else if (ext === ".places.json") this.warn(`place list missing: ${src} (build it with pipeline/places.py ${c})`);
          else if (ext === ".market.json") this.warn(`market pack missing: ${src} (build it with pipeline/market_zones.py ${c})`);
          else this.warn(`city pack missing: ${src} (build it with pipeline/build_city.py ${c})`);
        }
      const intercity = join(PACKS, REGISTRY.market.pack + ".json");
      if (existsSync(intercity)) copyFileSync(intercity, join(out, REGISTRY.market.pack + ".json"));
      else this.warn("intercity market missing: build pipeline/intercity.py");
      // T-800: every zone's coarse outline (pipeline/zone_outlines.py)
      const outlines = join(PACKS, "outlines.json");
      if (existsSync(outlines)) copyFileSync(outlines, join(out, "outlines.json"));
      else this.warn("coarse zone outlines missing: run pipeline/zone_outlines.py");
      // T-731: the real networks a sandbox can start from (pipeline/realnet.py ship)
      const real = join(PACKS, "real");
      if (existsSync(real)) {
        mkdirSync(join(out, "real"), { recursive: true });
        for (const f of readdirSync(real)) if (f.endsWith(".save")) copyFileSync(join(real, f), join(out, "real", f));
      } else this.warn("real networks missing: run pipeline/realnet.py ship");
      // T-727: the real network's routing graphs the track tool snaps along (pipeline/realgraph.py)
      const graphs = join(PACKS, "realgraph");
      if (existsSync(graphs)) {
        mkdirSync(join(out, "realgraph"), { recursive: true });
        for (const f of readdirSync(graphs)) if (f.endsWith(".rgr") || f === "index.json") copyFileSync(join(graphs, f), join(out, "realgraph", f));
      } else this.warn("real-network graphs missing: run pipeline/realgraph.py <regions>");
    },
  };
}

export default defineConfig({
  // Served under /trainworld/ by maps/serve.py, so every URL in the build must be relative.
  base: "./",
  worker: { format: "es" },
  // T-328: the app build, part of the demand cache's key: new at each build and dev-server start
  define: { __TW_BUILD__: JSON.stringify(Date.now().toString(36)) },
  build: { outDir: "dist", target: "es2022" },
  // T-682: the real-network layer reads noritetsu's tiles beside the game (map/realNetwork.ts); 8800 serves them
  // at /noritetsu/ and sends no CORS headers, so the dev server passes the path through
  server: { proxy: { "/noritetsu": "http://localhost:8800" } },
  plugins: [packs()],
});
