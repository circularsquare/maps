// Entry of the search worker (a module worker; page asks `new Worker('worker.js', { type: 'module' })`).
// The worker owns the grids, runs every search and paints every overlay into an RGBA buffer;
// the page only shows them. Modules are in js/; data URLs are relative to this file.
//
// L10: the coarse world grid (data/grid.bin.gz), solved whole on each click.
// L3, L1: tiles (data/tiles/), solved for a window round the view, seeded at the window
// border from the next coarser solved level. See docs/resolution.md.
import { init, click, setView } from './js/session.js';

self.onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'init') init(msg);
    else if (msg.type === 'run') click(msg.lng, msg.lat);
    else if (msg.type === 'view') setView(msg);
};
