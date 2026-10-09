import { copyFileSync, createReadStream, existsSync, mkdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
const PACKS = resolve(here, "../data/packs");
/** City packs the app loads (pipeline output in data/packs/, gitignored). */
const CITIES = ["nyc"];

/**
 * City packs at `packs/<city>.json` and `packs/<city>.bin` beside the page (T-025): served from
 * data/packs/ by the dev server and copied into dist/packs/ by a build. The clock worker reads the
 * water mask from it with a Range request, the demand workers the cells.
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
        res.setHeader("Content-Type", file.endsWith(".json") ? "application/json" : "application/octet-stream");
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
        for (const ext of [".json", ".bin"]) {
          const src = join(PACKS, c + ext);
          if (existsSync(src)) copyFileSync(src, join(out, c + ext));
          else this.warn(`city pack missing: ${src} (build it with pipeline/build_city.py ${c})`);
        }
    },
  };
}

export default defineConfig({
  // Served under /trainworld/ by maps/serve.py, so every URL in the build must be relative.
  base: "./",
  worker: { format: "es" },
  build: { outDir: "dist", target: "es2022" },
  plugins: [packs()],
});
