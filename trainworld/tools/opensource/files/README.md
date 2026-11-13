# trainsworld

A railway building game on a map of the whole world. You lay track and stations, set up lines, and
the game works out who rides your trains and who takes the car. It runs in the browser.

The game is at https://anita.garden/trainsworld/. This repository is its source.

Most of the code was written by Claude (Anthropic's AI model), working with Anita, who designs the game, plays it
and reviews the changes. The docs and this README were written by Claude too.

The page is TypeScript with MapLibre and Preact. The simulation and the demand model are Rust compiled to
WebAssembly. The city data is built by Python scripts, and the game fetches the built data when it runs.

Everything is in the `trainworld/` folder (the game was called trainworld while it was built), and the docs write
paths from this repo's root, so `node trainworld/checks/smoke.mjs` means what it says.

## Run it

You need Node 22 or newer, Rust (stable) with the `wasm32-unknown-unknown` target, and wasm-pack 0.15.

```
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
cd trainworld/app
npm install
npm run wasm
npm run dev
```

Open the address Vite prints (http://localhost:5173/). `npm run wasm` takes a few minutes the first time. Run it
again when anything in `trainworld/sim/` changes. `npm run check` type-checks the page.

## Data

The game needs city packs, terrain, outlines and real rail networks, about 6 GB in all. They are not in this repo.
With no `trainworld/data/packs` folder, the dev server fetches the files the page asks for from
https://data.anita.garden/trainworld/. A first session downloads tens of MB. The published version it reads is
named in `trainworld/data-version.txt`. To pin another one, run `TW_DATA_VERSION=<folder> npm run dev`. To stay
offline, run `TW_DATA=off npm run dev` and fill `trainworld/data/` yourself.

The basemap tiles come from OpenFreeMap and need no key.

You can build packs yourself with the scripts in `trainworld/pipeline/`, but a world run needs 30 GB of raw
downloads (OpenStreetMap extracts, GHSL, national censuses) and hours of compute. Treat the pipeline as a
reference for how the numbers are made unless you are adding a place. `trainworld/docs/adding-a-country.md` and
`trainworld/docs/world-run.md` explain the process. Parts of it name private tooling (an AWS account, a cores
ledger) that is not in this repo.

The game's account, leaderboard, community and feedback features talk to a server that is not part of this repo.
They stay off on localhost unless you run your own server and add `?acct=local`. The game plays fully without them.

The four building sounds are not included (see `NOTICE.md`). The game plays without them.

## Where things are

- `trainworld/SPEC.md` says what the game is and how it works.
- `trainworld/docs/` has the rules in detail, one file per area. `docs/code-map.md` says where each module lives.
- `trainworld/app/` is the page, `trainworld/sim/` the Rust crate, `trainworld/pipeline/` the data scripts.
- `trainworld/checks/` has the browser and Node checks. `checks/smoke.mjs` runs the main ones against a dev server
  that has data. Set `TW_CORES=3` to limit the cores a check's Chrome uses. Some older checks use paths from the
  author's machine.
- `trainworld/wiki/` has pages that explain the mechanics to players.

## Licence

The code is under the GNU Affero General Public License 3.0 (`LICENSE`). If you run a modified copy as a website,
you have to offer its source to the people using it.

The data is under the ODbL, and the fonts, flags and libraries keep their own licences. `NOTICE.md` lists each, with
the credits the data needs. The name "trainsworld" is not covered by the code licence.

Settings has links to the author's Ko-fi page and Discord, and the page has privacy and terms files that name
anita.garden. If you host a fork, replace them.

## Contributing

See `CONTRIBUTING.md`. Issues and pull requests are welcome.
