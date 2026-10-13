// Static server with HTTP Range and Cache-Control: no-store (like serve.py), for startup timing of
// a scratch production build (notes/startup-timing.mjs). node notes/static-serve.mjs <dir> <port>
import { createServer } from 'node:http';
import { createReadStream, existsSync, statSync } from 'node:fs';
import { join, resolve, extname } from 'node:path';
const root = resolve(process.argv[2]);
const port = Number(process.argv[3] ?? 6101);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.geojson': 'application/geo+json', '.wasm': 'application/wasm', '.png': 'image/png', '.svg': 'image/svg+xml', '.pbf': 'application/x-protobuf', '.woff2': 'font/woff2', '.ttf': 'font/ttf' };
createServer((req, res) => {
  let p = decodeURIComponent((req.url ?? '/').split('?')[0]);
  if (p.endsWith('/')) p += 'index.html';
  const file = join(root, p);
  if (!file.startsWith(root) || !existsSync(file) || statSync(file).isDirectory()) { res.statusCode = 404; return res.end(); }
  const size = statSync(file).size;
  res.setHeader('Accept-Ranges', 'bytes');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('Content-Type', types[extname(file)] ?? 'application/octet-stream');
  const range = /bytes=(\d+)-(\d*)/.exec(req.headers.range ?? '');
  if (range) {
    const a = Number(range[1]), b = range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
    res.statusCode = 206;
    res.setHeader('Content-Range', `bytes ${a}-${b}/${size}`);
    res.setHeader('Content-Length', b - a + 1);
    createReadStream(file, { start: a, end: b }).pipe(res);
  } else {
    res.setHeader('Content-Length', size);
    createReadStream(file).pipe(res);
  }
}).listen(port, '127.0.0.1', () => console.log(`serving ${root} on ${port}`));
