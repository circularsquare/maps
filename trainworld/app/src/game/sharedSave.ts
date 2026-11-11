// T-3115: a save shared to Community so others can open a copy (notes/T-2887.md, "Open a copy"; shonei-server's
// docs/trainworld-api.md). `forShare` makes what goes up; `copyFromShare` checks a stranger's file before the loader
// sees any of it and makes the bytes that open as the viewer's own copy. The server checks only the outer shape.
//
// What a save holds that others then see: the network (stations and their names, lines and their names, groups), the
// money and its history, the zones opened, the achievements, the line sound notes. What it never held: the game's
// name (the slot's, outside the bytes), account or device ids, playtime, settings (localStorage). The Getting
// started card's progress goes; nothing else in it is private.
//
// Against a hostile file: sizes are capped before and while unpacking (a gzip bomb stops at the cap); the header is
// parsed by JSON.parse alone, walked with depth, size and key limits (no `__proto__`, `constructor` or `prototype`
// keys, every number finite, strings short), and each field must have its type or the file is refused; unknown
// fields are dropped; the copy is re-encoded from the checked header, so the loader reads only what passed. The
// network bytes go to the track model as any loaded file's do: its reader refuses what it cannot read, a panic is
// caught and the model replaced (clock.worker.ts `afterLoadPanic`), and the copy loads into a new slot, so nothing
// of the viewer's is touched whatever happens.

import { encodeSave, isRetiredTrackFormat, isThisWorld, SAVE_FORMAT, SAVE_HEADER_FIELDS, SAVE_MAGIC, type SaveHeader } from "../workers/saveFile";

/** shonei-server's MaxTwCommunitySave and MaxTwCommunitySaveUnpacked */
export const MAX_SHARED_SAVE = 2 << 20;
const MAX_RAW = 32 << 20;
const MAX_HEADER = 8 << 20, MAX_DEPTH = 8, MAX_NODES = 4_000_000, MAX_STRING = 1000, MAX_KEYS = 100_000;
const BAD_KEYS = new Set(["__proto__", "constructor", "prototype"]);

/** Why a copy did not open: "version" (another world, a retired or newer format), "bad" (not a save, damaged, too big). */
export class CopyRefused extends Error {
  constructor(readonly why: "version" | "bad") { super(why); }
}

/** `gz` unpacked, or null when it is not gzip or unpacks past `cap` bytes */
async function gunzipCapped(gz: Uint8Array, cap: number): Promise<Uint8Array | null> {
  const reader = new Blob([gz as BlobPart]).stream().pipeThrough(new DecompressionStream("gzip")).getReader();
  const parts: Uint8Array[] = [];
  let n = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      n += value.length;
      if (n > cap) { void reader.cancel().catch(() => {}); return null; }
      parts.push(value);
    }
  } catch { return null; }
  const out = new Uint8Array(n);
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}

const plain = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const num = (v: unknown) => typeof v === "number" && Number.isFinite(v);
const str = (v: unknown) => typeof v === "string";
const arr = (f: (v: unknown) => boolean) => (v: unknown) => Array.isArray(v) && v.every(f);
const rec = (f: (v: unknown) => boolean) => (v: unknown) => plain(v) && Object.values(v).every(f);
const entries = arr((e) => plain(e) && Object.values(e).every((x) => x === undefined || num(x)));
const opt = (f: (v: unknown) => boolean) => (v: unknown) => v === undefined || f(v);

/** Each header field's shape; every field of `SaveHeader` must be here (the type says so). */
const FIELDS: Record<keyof SaveHeader, (v: unknown) => boolean> = {
  game: (v) => v === "anitabuilder", format: (v) => num(v) && Number.isInteger(v), city: str, region: opt(str), cities: opt(arr(str)),
  clock: (v) => num(v) && (v as number) >= 0, cash: num, loan: opt(num), loanPay: opt(num), loanDays: opt(num),
  fleet: opt(num), fleetFast: opt(num), fleetBus: opt(num), fleetFerry: opt(num), fleetTram: opt(num),
  fares: (v) => plain(v) && num(v.base) && num(v.perKm) && opt(arr((p) => Array.isArray(p) && p.length === 2 && p.every(num)))(v.nodes)
    && opt(rec(num))(v.scales) && opt(str)(v.baseline) && opt(str)(v.ref) && opt((b) => b === "usd")(v.basis)
    && ["income", "hsr", "ferry", "bus", "transfer"].every((k) => opt(num)(v[k])),
  ledger: entries, hours: opt(entries),
  regions: opt(rec((r) => plain(r) && opt(entries)(r.ledger) && opt(entries)(r.hours))),
  stars: opt(rec(num)), goals: opt(rec(rec(num))),
  notes: opt(rec((n) => Array.isArray(n) && n.length === 2 && num(n[0]) && str(n[1]))),
  mode: opt((m) => m === "normal" || m === "sandbox"), start: opt(str), opened: opt(arr(str)),
  demandFine: opt(arr(str)), demandBuilt: opt(arr(arr(str))), newCars: opt(num), level: opt(num),
  guide: opt(plain), achievements: opt(rec((a) => plain(a) && num(a.at) && opt(str)(a.of) && opt(num)(a.v) && opt(arr(str))(a.in))),
  home: opt(str), copy: opt(str),
};

/** Whether a parsed value stays inside the limits: depth, node and key counts, string lengths, finite numbers,
 * no key that could reach a prototype. */
function withinLimits(root: unknown): boolean {
  let nodes = 0;
  const walk = (v: unknown, depth: number): boolean => {
    if (++nodes > MAX_NODES) return false;
    if (typeof v === "number") return Number.isFinite(v);
    if (typeof v === "string") return v.length <= MAX_STRING;
    if (v === null || typeof v === "boolean") return true;
    if (typeof v !== "object" || depth >= MAX_DEPTH) return false;
    if (Array.isArray(v)) return v.every((x) => walk(x, depth + 1));
    const keys = Object.keys(v);
    if (keys.length > MAX_KEYS) return false;
    return keys.every((k) => !BAD_KEYS.has(k) && k.length <= MAX_STRING && walk((v as Record<string, unknown>)[k], depth + 1));
  };
  return walk(root, 0);
}

/** A save's header and network out of its file, checked; throws CopyRefused. */
async function readChecked(gz: Uint8Array): Promise<{ header: SaveHeader; track: Uint8Array }> {
  if (gz.length > MAX_SHARED_SAVE) throw new CopyRefused("bad");
  const raw = await gunzipCapped(gz, MAX_RAW);
  if (!raw || raw.length < 8 || new TextDecoder().decode(raw.subarray(0, 4)) !== SAVE_MAGIC) throw new CopyRefused("bad");
  const n = new DataView(raw.buffer, raw.byteOffset).getUint32(4, true);
  if (n > MAX_HEADER || 8 + n + 4 > raw.length) throw new CopyRefused("bad");
  let parsed: unknown;
  try { parsed = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(raw.subarray(8, 8 + n))); } catch { throw new CopyRefused("bad"); }
  if (!plain(parsed) || !withinLimits(parsed)) throw new CopyRefused("bad");
  if (parsed.game !== "anitabuilder") throw new CopyRefused("bad");
  if (!num(parsed.format) || (parsed.format as number) > SAVE_FORMAT) throw new CopyRefused("version");
  const header: Record<string, unknown> = {};
  for (const k of Object.keys(parsed)) if (SAVE_HEADER_FIELDS.has(k)) header[k] = parsed[k];
  for (const [k, ok] of Object.entries(FIELDS)) if (!ok(header[k])) throw new CopyRefused("bad");
  const track = raw.slice(8 + n);
  const magic = new TextDecoder().decode(track.subarray(0, 4));
  if (isRetiredTrackFormat(track)) throw new CopyRefused("version");
  if (!/^TWT[4-6]$/.test(magic)) throw new CopyRefused(/^TWT\d$/.test(magic) ? "version" : "bad");
  const h = header as unknown as SaveHeader;
  if (!isThisWorld(h)) throw new CopyRefused("version");
  return { header: h, track };
}

/** The playing game's save as it goes up with a share, its private parts out; null if it is not a save this build can
 * read back or is over the server's limit. */
export async function forShare(bytes: Uint8Array): Promise<Uint8Array | null> {
  try {
    const { header, track } = await readChecked(bytes);
    const { guide: _g, copy: _c, ...rest } = header;
    const out = await encodeSave(rest, track);
    return out.length <= MAX_SHARED_SAVE ? out : null;
  } catch { return null; }
}

/** A shared save, as downloaded, checked and made the viewer's copy of share `id`: tagged with it (so its figures
 * never rank), no Getting started card. Throws CopyRefused. */
export async function copyFromShare(gz: Uint8Array, id: string): Promise<Uint8Array> {
  const { header, track } = await readChecked(gz);
  const { guide: _g, ...rest } = header;
  return encodeSave({ ...rest, copy: id }, track);
}
