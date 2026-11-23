import { copyFileSync, createReadStream, existsSync, linkSync, mkdirSync, readdirSync, readFileSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { execSync } from "node:child_process";
import { createHash } from "node:crypto";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
// TW_PACKS: a frozen copy instead (pipeline/freeze.py build, T-928)
const PACKS = process.env.TW_PACKS ? resolve(process.env.TW_PACKS) : resolve(here, "../data/packs");
/** The packs the game opens: the registry's cities (T-125), and its regional market. Pipeline output, gitignored. */
const REGISTRY = JSON.parse(readFileSync(resolve(here, "src/game/registry.json"), "utf8")) as { cities: { id: string }[]; market: { pack: string } };
const CITIES = REGISTRY.cities.map((c) => c.id);
/** T-2901: the terrain classes of the world's land, one file per degree block (pipeline/world_terrain.py), at
 * `packs/terrain-world/`; TW_TERRAIN_WORLD: another copy. A build holds them as hard links (0.7 GB, 19k files), and
 * tools/deploy.py publishes them on their own, not with each release. */
const TERRAIN_WORLD = process.env.TW_TERRAIN_WORLD ? resolve(process.env.TW_TERRAIN_WORLD) : resolve(here, "../data/terrain-world-v2");
/** T-1285: the world's coarse water (pipeline/world_water.py), served at `packs/world.water`; TW_WORLD_WATER: another copy. */
const WORLD_WATER = process.env.TW_WORLD_WATER ? resolve(process.env.TW_WORLD_WATER) : resolve(here, "../data/terrain/world.water");
/** T-279: the market's routed drives (pipeline/market_drives.py), served at `packs/market-drives.bin`; TW_MARKET_DRIVES: another copy. */
const MARKET_DRIVES = process.env.TW_MARKET_DRIVES ? resolve(process.env.TW_MARKET_DRIVES) : join(PACKS, "market-drives.bin");

/** TW_LINK_PACKS=1 (snapbuild.cmd): a build holds the packs as hard links to the data files instead of copies, 6 GB in
 * 5 s instead of 35. Safe because every pipeline writer replaces a pack (a .part renamed over it), never rewrites it in
 * place (pipeline/freeze.py relies on the same rule); a copy where a link is refused (another volume). Never over an
 * existing file: it could be a link to the source, and copying onto that would truncate the data file. */
const LINK_PACKS = process.env.TW_LINK_PACKS === "1";
function put(src: string, dst: string): void {
  if (!LINK_PACKS) return copyFileSync(src, dst);
  try {
    if (existsSync(dst)) unlinkSync(dst);
    linkSync(src, dst);
  } catch (e) {
    if (existsSync(dst)) throw e;
    copyFileSync(src, dst);
  }
}

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
      server.middlewares.use("/packs/market-drives.bin", (_req, res) => {
        // missing, a 404: every drive in the market is then the rule's
        if (!existsSync(MARKET_DRIVES)) {
          res.statusCode = 404;
          res.end();
          return;
        }
        res.setHeader("Content-Type", "application/octet-stream");
        res.setHeader("Content-Length", statSync(MARKET_DRIVES).size);
        createReadStream(MARKET_DRIVES).pipe(res);
      });
      server.middlewares.use("/packs/terrain-world", (req, res) => {
        const file = join(TERRAIN_WORLD, (req.url ?? "").split("?")[0].replace(/^\/+/, ""));
        // never a 200 with the page for a file that is not there: the clock worker reads a failure as a failure
        if (!file.startsWith(TERRAIN_WORLD) || !existsSync(file) || !statSync(file).isFile()) {
          res.statusCode = 404;
          res.end();
          return;
        }
        res.setHeader("Content-Type", "application/octet-stream");
        res.setHeader("Content-Length", statSync(file).size);
        createReadStream(file).pipe(res);
      });
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
      server.middlewares.use("/packs/world.water", (_req, res) => {
        // missing, a 404: the clock worker then leaves the sea outside the zones as land
        if (!existsSync(WORLD_WATER)) {
          res.statusCode = 404;
          res.end();
          return;
        }
        res.setHeader("Content-Type", "application/octet-stream");
        res.setHeader("Content-Length", statSync(WORLD_WATER).size);
        createReadStream(WORLD_WATER).pipe(res);
      });
    },
    writeBundle(opts) {
      const out = join(opts.dir ?? resolve(here, "dist"), "packs");
      mkdirSync(out, { recursive: true });
      for (const c of CITIES)
        for (const ext of [".json", ".bin", ".boundary.geojson", ".places.json", ".market.json"]) {
          const src = join(PACKS, c + ext);
          if (existsSync(src)) put(src, join(out, c + ext));
          else if (ext === ".places.json") this.warn(`place list missing: ${src} (build it with pipeline/places.py ${c})`);
          else if (ext === ".market.json") this.warn(`market pack missing: ${src} (build it with pipeline/market_zones.py ${c})`);
          else this.warn(`city pack missing: ${src} (build it with pipeline/build_city.py ${c})`);
        }
      if (existsSync(join(TERRAIN_WORLD, "index.bin"))) {
        const world = join(out, "terrain-world");
        mkdirSync(world, { recursive: true });
        for (const f of readdirSync(TERRAIN_WORLD)) {
          const dst = join(world, f);
          if (existsSync(dst)) continue;
          try { linkSync(join(TERRAIN_WORLD, f), dst); } catch { copyFileSync(join(TERRAIN_WORLD, f), dst); }
        }
      } else this.warn("terrain blocks missing: run pipeline/world_terrain.py build");
      if (existsSync(WORLD_WATER)) put(WORLD_WATER, join(out, "world.water"));
      else this.warn("world water mask missing: run pipeline/world_water.py build");
      if (existsSync(MARKET_DRIVES)) put(MARKET_DRIVES, join(out, "market-drives.bin"));
      else this.warn("market drives missing: run pipeline/market_drives.py build");
      // T-800: every zone's coarse outline (pipeline/zone_outlines.py); without it every zone shows (T-1006)
      const outlines = join(PACKS, "outlines.json");
      if (existsSync(outlines)) put(outlines, join(out, "outlines.json"));
      else this.warn("coarse zone outlines missing: run pipeline/zone_outlines.py");
      // T-731: the real networks a sandbox can start from (pipeline/realnet.py ship)
      const real = join(PACKS, "real");
      if (existsSync(real)) {
        mkdirSync(join(out, "real"), { recursive: true });
        for (const f of readdirSync(real)) if (f.endsWith(".save")) put(join(real, f), join(out, "real", f));
      } else this.warn("real networks missing: run pipeline/realnet.py ship");
      // T-727: the real network's routing graphs the track tool snaps along (pipeline/realgraph.py)
      const graphs = join(PACKS, "realgraph");
      if (existsSync(graphs)) {
        mkdirSync(join(out, "realgraph"), { recursive: true });
        for (const f of readdirSync(graphs)) if (f.endsWith(".rgr") || f === "index.json") put(join(graphs, f), join(out, "realgraph", f));
      } else this.warn("real-network graphs missing: run pipeline/realgraph.py <regions>");
      // T-4004: the world name index the search box finds far towns with (pipeline/place_index.py); without it only near ones
      const index = join(PACKS, "place-index");
      if (existsSync(join(index, "index.json"))) {
        mkdirSync(join(out, "place-index"), { recursive: true });
        put(join(index, "index.json"), join(out, "place-index", "index.json"));
        for (const tier of readdirSync(index, { withFileTypes: true })) {
          if (!tier.isDirectory()) continue;
          mkdirSync(join(out, "place-index", tier.name), { recursive: true });
          for (const f of readdirSync(join(index, tier.name))) put(join(index, tier.name, f), join(out, "place-index", tier.name, f));
        }
      } else this.warn("place name index missing: run pipeline/place_index.py");
    },
  };
}

/** The start menu's version: the commit count, so 8800 and the published page can be told apart at a glance
 * (the orchestrator names it in its reports); "dev" on the dev server, whose tree may not match a commit. */
function version(): string {
  if (process.argv.includes("build") === false) return "dev";
  try {
    return execSync("git rev-list --count HEAD", { cwd: dirname(fileURLToPath(import.meta.url)) }).toString().trim();
  } catch {
    return "dev";
  }
}

// T-328: the app build, part of the demand cache's key: new at each build and dev-server start. T-3707: also the stamp
// a score is sent with; tools/deploy.py reads it from dist/build.json into the site's list of published builds.
const BUILD = Date.now().toString(36);
const VERSION = version();
/** T-3707: a build from the public repo (tools/opensource adds this file there) marks its games modded. */
const UNOFFICIAL = existsSync(resolve(here, "unofficial-build.txt"));

function buildStamp(): Plugin {
  return {
    name: "tw-build-stamp",
    apply: "build",
    generateBundle() {
      this.emitFile({ type: "asset", fileName: "build.json", source: JSON.stringify({ build: BUILD, version: VERSION }) + "\n" });
    },
  };
}

/** T-3926: the demand cache's key in a build (`__TW_DEMAND__`, workers/demandCache.ts): a SHA-256 of the source of
 * every module the demand side runs and of every file those modules load by URL (the demand WASM, workers, data),
 * so a release that changes none of them keeps every cached result and one that changes any of them keeps none.
 * The modules come from the bundler's own graph: from `DEMAND_ROOTS`, every import, static or dynamic, and every
 * `new URL(..., import.meta.url)` (a worker's whole graph, from its own build). The key is written over a placeholder
 * once the chunk holding it renders, when every worker's graph is known. The dev server keeps a key per start.
 * Left out, by name only (anything not named is in): `DEMAND_SKIP`, the translations, and the clock worker with the
 * track WASM, whose output reaches a solve only as the network, whose demand-read form the key holds already
 * (`canonical`). */
const DEMAND_ROOTS = ["src/workers/demandClient.ts"].map((f) => resolve(here, f).replace(/\\/g, "/"));
const DEMAND_SKIP = ["src/i18n/messages/", "src/workers/clock.worker.ts"].map((f) => resolve(here, f).replace(/\\/g, "/") + (f.endsWith("/") ? "/" : ""));
const skipped = (f: string) => DEMAND_SKIP.some((s) => (s.endsWith("/") ? f.startsWith(s) : f === s));
const DEMAND_SLOT = "tw-demand-key-" + "0".repeat(50);
const URL_REF = /new URL\(\s*["'`]([^"'`]+)["'`]\s*,\s*import\.meta\.url\s*\)/g;
/** each worker build's entry -> every module in that build */
const workerGraphs = new Map<string, string[]>();
const fileOf = (id: string) => id.split("?")[0].replace(/\\/g, "/");

function demandWorkerGraph(): Plugin {
  return {
    name: "tw-demand-key-worker",
    apply: "build",
    buildEnd() {
      const ids = [...this.getModuleIds()];
      for (const id of ids) if (this.getModuleInfo(id)?.isEntry) workerGraphs.set(fileOf(id), ids);
    },
  };
}

function demandKey(): Plugin {
  return {
    name: "tw-demand-key",
    config: (_c, { command }) => ({ define: { __TW_DEMAND__: JSON.stringify(command === "build" ? DEMAND_SLOT : "dev-" + BUILD) } }),
    renderChunk(code) {
      if (!code.includes(DEMAND_SLOT)) return null;
      const files = new Set<string>();
      const queue = [...DEMAND_ROOTS];
      const take = (id: string) => {
        if (id.startsWith("\0")) return;
        const f = fileOf(id);
        if (!files.has(f) && !skipped(f)) { files.add(f); queue.push(id); }
      };
      // a graph that may be missing part of the demand code gets the build stamp: a cold load, never a stale one
      let short = DEMAND_ROOTS.some((r) => !this.getModuleInfo(r)) ? "a root is not in the graph" : "";
      for (const r of DEMAND_ROOTS) files.add(r);
      while (queue.length) {
        const id = queue.pop()!;
        const info = this.getModuleInfo(id);
        for (const d of [...(info?.importedIds ?? []), ...(info?.dynamicallyImportedIds ?? [])]) take(d);
        const f = fileOf(id);
        if (!/\.(ts|tsx|js|mjs)$/.test(f) || !existsSync(f)) continue;
        for (const m of readFileSync(f, "utf8").matchAll(URL_REF)) {
          const ref = resolve(dirname(f), m[1]).replace(/\\/g, "/");
          if (skipped(ref)) continue;
          const graph = workerGraphs.get(ref);
          if (graph) for (const w of graph) take(w);
          else if (/\.(ts|tsx|js|mjs)$/.test(ref)) short ||= `no worker build's graph for ${m[1]} (${f})`;
          else if (!files.has(ref)) {
            if (!existsSync(ref)) short ||= `${f} loads ${m[1]}, which is not there`;
            else { files.add(ref); queue.push(ref); }
          }
        }
      }
      for (const need of ["src/workers/demand.worker.ts", "src/wasm/demand/sim_bg.wasm"])
        if (!files.has(resolve(here, need).replace(/\\/g, "/"))) short ||= `${need} is not in the graph`;
      if (short) {
        this.warn(`demand cache key: ${short}; this build's key is its build stamp`);
        return { code: code.split(DEMAND_SLOT).join(("build-" + BUILD).padEnd(DEMAND_SLOT.length, "-")), map: null };
      }
      const parts = [...files].map((f) => {
        // the same file in another checkout: its path from app/ (or from node_modules/), text without CRs
        const name = f.includes("/node_modules/") ? f.slice(f.lastIndexOf("/node_modules/") + 1) : relative(here, f).replace(/\\/g, "/");
        const bytes = readFileSync(f);
        const body = /\.(wasm|bin|png|jpg|webp)$/.test(f) ? bytes : Buffer.from(bytes.toString("utf8").replace(/\r\n/g, "\n"));
        return name + " " + createHash("sha256").update(body).digest("hex");
      }).sort();
      if (process.env.TW_DEMAND_KEY_FILES) writeFileSync(process.env.TW_DEMAND_KEY_FILES, parts.join("\n") + "\n");
      const key = createHash("sha256").update(parts.join("\n")).digest("hex");
      return { code: code.split(DEMAND_SLOT).join(key), map: null };
    },
  };
}

export default defineConfig({
  // Served under /trainworld/ by maps/serve.py, so every URL in the build must be relative.
  base: "./",
  worker: { format: "es", plugins: () => [demandWorkerGraph()] },
  // TW_CHANNEL=track: the track branch's build on 8800 (T-4126, game/trackBuild.ts, tools/ops/snaptrack.py)
  define: { __TW_BUILD__: JSON.stringify(BUILD), __TW_VERSION__: JSON.stringify(VERSION), __TW_UNOFFICIAL__: JSON.stringify(UNOFFICIAL), __TW_TRACK__: JSON.stringify(process.env.TW_CHANNEL === "track") },
  build: { outDir: "dist", target: "es2022" },
  // T-682: the real-network layer reads noritetsu's tiles beside the game (map/realNetwork.ts); 8800 serves them
  // at /noritetsu/ and sends no CORS headers, so the dev server passes the path through
  server: { proxy: { "/noritetsu": "http://localhost:8800" } },
  plugins: [packs(), buildStamp(), demandKey()],
});
