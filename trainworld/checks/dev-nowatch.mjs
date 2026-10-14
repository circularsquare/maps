// The app's Vite dev server on its own port with file watching and HMR off, so other agents' edits
// do not reload the page mid-check. Modules are read once, when first asked for: restart it after
// editing the app.
//   node trainworld/checks/dev-nowatch.mjs <port>      (pick a port 6100-6199 nobody else uses)
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
const root = fileURLToPath(new URL('../app', import.meta.url)).replace(/\\/g, '/');
const port = Number(process.argv[2] ?? process.env.PORT);
if (!port) { console.error('usage: node checks/dev-nowatch.mjs <port>'); process.exit(2); }
const require = createRequire(root + '/package.json');
const { createServer } = await import(pathToFileURL(require.resolve('vite')).href);
const server = await createServer({ root, configFile: root + '/vite.config.ts', server: { port, strictPort: true, watch: null, hmr: false } });
await server.listen();
server.printUrls();
