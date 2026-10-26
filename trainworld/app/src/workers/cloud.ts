// T-988: games on the account (notes/T-988.md; the routes in shonei-server's docs/trainworld-api.md).
// The browser stays the working copy; a signed-in player's slots are also kept on the account, so
// they follow the player to other devices. Each slot's tie is its `SyncRecord` (`sync:<slot>`).
//
// Plain functions of a `CloudContext`, so both threads use them: the clock worker uploads after each
// write and claims the game it opens (clock.worker.ts); the main thread settles every slot on the
// start menu before the worker starts, and downloads or deletes the account's other games.

import { deleteSlot, leaseSlot, listSlots, readCloudDeletes, readStored, readSync, writeCloudDeletes, writeSlot, writeSync, type GameMode, type SaveSlot, type StoredSave, type SyncRecord, type ThumbPanel } from "./saveFile";

/** What a page needs to reach the account: the server, the session and this browser's device id.
 * `on`: the server keeps games (`/noritetsu/config`'s `trainworldGames`); records are still tied
 * while it is not, and go up once it is. `ids`: the leaderboards' game ids by slot from before T-988
 * (`trainworld.score-games.v1`), so one id keeps naming a game's figures and its save. */
export interface CloudContext { base: string; token: string; acct: string; device: string; on: boolean; ids?: Record<string, string> }

/** The card's facts sent with each save, so another device draws its card without opening it. */
export interface GameFacts {
  at: number; day: number; city?: string; mode?: GameMode; cities?: string[]; lines?: number; stations?: number; cash?: number;
  panels?: ThumbPanel[]; shade?: "light" | "dark";
}
export interface AccountGame {
  id: string; name: string; rev: number; writer: string; writerAt: number; savedAt: number;
  facts?: Partial<GameFacts>; saveBytes: number; pictureRev?: number;
}
export interface AccountList { games: AccountGame[]; tombstones: { id: string; at: number }[]; used: number; limit: number; maxGames: number }

/** Wall ms between two uploads of an autosaved game (the leaderboards' gap). */
export const GAP_MS = 120_000;
/** The server's limits: a save, a picture, a name (bytes). Past 4.6 MB it closes the connection
 * after its 413, which a browser may report as a network error, so a bigger save is never sent. */
const MAX_SAVE = 4 << 20, MAX_PICTURE = 512 << 10, MAX_NAME = 200, MAX_CITIES = 64;
/** A card whose game another device saved this recently says so. */
export const ELSEWHERE_MS = 10 * 60_000;

export const WHY_OFFLINE = "Offline";
export const WHY_UNREACHABLE = "The server could not be reached";
export const WHY_FULL = "Your account is full. Delete a game to make room.";
export const WHY_TOO_BIG = "This game is too big to keep on your account.";

/** Called on a 401: the session is dead (the main thread drops the token). */
let tokenGone: () => void = () => {};
export function onTokenGone(f: () => void) { tokenGone = f; }

export function newGameId(): string {
  const a = "abcdefghijkmnpqrstuvwxyz23456789", b = crypto.getRandomValues(new Uint8Array(12));
  return Array.from(b, (v) => a[v % a.length]).join("");
}

/** A slot tied to this account and synced (not kept here only). */
export const tiedTo = (rec: SyncRecord | null | undefined, ctx: CloudContext): rec is SyncRecord => !!rec && rec.acct === ctx.acct && !rec.local;
/** Changed here since its last sync. */
export const changedHere = (slot: { at: number }, rec: SyncRecord) => slot.at !== rec.syncedAt;

// ---------------------------------------------------------------- HTTP

interface Answer { status: number; json?: any; bytes?: Uint8Array; headers?: Headers }

async function call(ctx: CloudContext, method: string, path: string, body?: BodyInit | object, want: "json" | "bytes" = "json"): Promise<Answer> {
  const headers: Record<string, string> = { Authorization: "Bearer " + ctx.token };
  let payload: BodyInit | undefined;
  if (body instanceof FormData) payload = body;
  else if (body !== undefined) { headers["Content-Type"] = "application/json"; payload = JSON.stringify(body); }
  let r: Response;
  try {
    r = await fetch(ctx.base + path, { method, headers, body: payload, cache: "no-store" });
  } catch {
    return { status: 0 };
  }
  if (r.status === 401) tokenGone();
  const out: Answer = { status: r.status, headers: r.headers };
  try {
    if (r.ok && want === "bytes") out.bytes = new Uint8Array(await r.arrayBuffer());
    else if (r.status !== 204 && r.status !== 304) out.json = await r.json();
  } catch { /* no body */ }
  return out;
}

const failure = (status: number) => status === 0 ? (typeof navigator !== "undefined" && navigator.onLine === false ? WHY_OFFLINE : WHY_UNREACHABLE)
  : status === 507 ? WHY_FULL : status === 413 ? WHY_TOO_BIG : WHY_UNREACHABLE;

/** Why the last list failed, for "Not on your account yet". */
let listFailure = WHY_UNREACHABLE;
export const lastListFailure = () => listFailure;

export async function listGames(ctx: CloudContext): Promise<AccountList | null> {
  if (!ctx.on) return null;
  const a = await call(ctx, "GET", "trainworld/games");
  if (a.status === 200 && a.json) return a.json as AccountList;
  listFailure = failure(a.status);
  return null;
}

/** A game's picture as a data URL (the card's `thumb`); undefined without one. */
export async function gamePicture(ctx: CloudContext, id: string): Promise<string | undefined> {
  const a = await call(ctx, "GET", `trainworld/games/${id}/picture`, undefined, "bytes");
  if (a.status !== 200 || !a.bytes) return undefined;
  let bin = "";
  for (let k = 0; k < a.bytes.length; k += 0x8000) bin += String.fromCharCode(...a.bytes.subarray(k, k + 0x8000));
  return "data:image/png;base64," + btoa(bin);
}

export interface Fetched { stored: StoredSave; id: string; rev: number; savedAt: number }
/** An account game as a slot's record, from its bytes, facts and picture; null if it cannot be read. */
export async function fetchGame(ctx: CloudContext, g: AccountGame): Promise<Fetched | null> {
  const a = await call(ctx, "GET", `trainworld/games/${g.id}`, undefined, "bytes");
  if (a.status !== 200 || !a.bytes) return null;
  const rev = Number(a.headers?.get("X-Rev")) || g.rev, savedAt = Number(a.headers?.get("X-Saved-At")) || g.savedAt;
  const f = g.facts ?? {};
  const thumb = g.pictureRev ? await gamePicture(ctx, g.id) : undefined;
  const stored: StoredSave = {
    bytes: a.bytes, at: typeof f.at === "number" ? f.at : savedAt * 1000, day: f.day ?? 0, city: f.city, name: g.name,
    thumb, panels: thumb ? f.panels : undefined, shade: f.shade, cities: f.cities ?? [], lines: f.lines ?? 0, stations: f.stations ?? 0, cash: f.cash ?? 0, mode: f.mode,
  };
  return { stored, id: g.id, rev, savedAt };
}

/** A downloaded game written into slot `key` (whose lock the caller holds), its record synced to it. */
export async function writeFetched(ctx: CloudContext, key: string, got: Fetched) {
  await writeSlot(key, got.stored);
  await writeSync(key, { id: got.id, acct: ctx.acct, rev: got.rev, syncedAt: got.stored.at, okAt: got.savedAt * 1000 });
}

/** The account's copy of `g` written into slot `key` (whose lock the caller holds); null if it could
 * not be downloaded. */
export async function takeInto(ctx: CloudContext, key: string, g: AccountGame): Promise<StoredSave | null> {
  const got = await fetchGame(ctx, g);
  if (!got) return null;
  await writeFetched(ctx, key, got);
  return got.stored;
}

/** An account game not in this browser, downloaded into a new slot; its key, or null. */
export async function downloadGame(ctx: CloudContext, g: AccountGame): Promise<string | null> {
  const key = "slot-" + crypto.randomUUID();
  return await takeInto(ctx, key, g) ? key : null;
}

/** Delete a game from the account (its backups and figures go too); true once it is gone. */
export async function deleteGame(ctx: CloudContext, id: string): Promise<boolean> {
  const a = await call(ctx, "DELETE", `trainworld/games/${id}`);
  return a.status === 204 || a.status === 404;
}

/** A slot deleted in this browser: its account game goes too, now or at the next chance. Call before
 * the slot (and its record) is deleted. */
export async function forgetSlot(key: string) {
  const rec = await readSync(key);
  if (!rec?.acct || rec.local) return;
  const list = await readCloudDeletes();
  if (!list.some((d) => d.id === rec.id)) await writeCloudDeletes([...list, { id: rec.id, acct: rec.acct }]);
}
/** Send the deletions this browser owes the account. */
export async function sendDeletes(ctx: CloudContext) {
  if (!ctx.on) return;
  for (const d of await readCloudDeletes()) {
    if (d.acct !== ctx.acct || !await deleteGame(ctx, d.id)) continue;
    await writeCloudDeletes((await readCloudDeletes()).filter((x) => x.id !== d.id));
  }
}

// ---------------------------------------------------------------- uploads

const clip = (name: string) => {
  let s = name;
  while (new TextEncoder().encode(s).length > MAX_NAME) s = s.slice(0, -1);
  return s;
};
function pictureOf(thumb?: string): Blob | null {
  const m = thumb && /^data:image\/png;base64,(.*)$/.exec(thumb);
  if (!m) return null;
  const bin = atob(m[1]), b = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) b[i] = bin.charCodeAt(i);
  return b.length <= MAX_PICTURE ? new Blob([b], { type: "image/png" }) : null;
}
export const factsOf = (s: StoredSave): GameFacts => ({
  at: s.at, day: s.day, city: s.city, mode: s.mode, cities: s.cities?.slice(0, MAX_CITIES), lines: s.lines, stations: s.stations, cash: s.cash,
  panels: s.panels, shade: s.shade,
});

/** "ok"; "elsewhere": another device has the game open; "failed": not sent (the record says why);
 * "skip": nothing to send (the slot went, or is not tied). */
export type Upload = "ok" | "elsewhere" | "failed" | "skip";

/** Upload slot `key` as stored now. A "moved" (this device's own write whose answer was lost) is sent
 * again on the account's rev; a game deleted on the account is kept as a new account game. */
export async function upload(ctx: CloudContext, key: string, force = false): Promise<Upload> {
  if (!ctx.on) return "skip";
  for (let attempt = 0; attempt < 3; attempt++) {
    const [stored, rec] = await Promise.all([readStored(key), readSync(key)]);
    if (!stored || !tiedTo(rec, ctx)) return "skip";
    if (stored.bytes.length > MAX_SAVE) {
      await writeSync(key, { ...rec, error: WHY_TOO_BIG });
      return "failed";
    }
    const fd = new FormData();
    fd.append("meta", JSON.stringify({ name: clip(stored.name || "Saved game"), facts: factsOf(stored), baseRev: rec.rev, device: ctx.device, force }));
    fd.append("save", new Blob([stored.bytes as BlobPart], { type: "application/octet-stream" }), "game.save");
    const pic = pictureOf(stored.thumb);
    if (pic) fd.append("picture", pic, "game.png");
    const a = await call(ctx, "PUT", `trainworld/games/${rec.id}`, fd);
    const now = await readSync(key); // a rename or delete may have come meanwhile
    if (!now) return "skip";
    if (a.status === 200 && a.json) {
      // a slot written again while this went up stays changed here, and goes next time
      await writeSync(key, { ...now, rev: a.json.rev, syncedAt: stored.at, upAt: Date.now(), okAt: a.json.savedAt * 1000, gaveWay: undefined, error: undefined });
      return "ok";
    }
    const code = a.json?.code;
    if (a.status === 409 && code === "elsewhere") return "elsewhere";
    if (a.status === 409 && code === "moved" && typeof a.json?.rev === "number") { await writeSync(key, { ...now, rev: a.json.rev }); continue; }
    if (a.status === 410) { await writeSync(key, { ...now, id: newGameId(), rev: 0, syncedAt: 0 }); continue; }
    await writeSync(key, { ...now, error: failure(a.status) });
    return "failed";
  }
  return "failed";
}

/** Claim the game this device is opening. "ok"; "moved": the account's copy is newer than this
 * slot's (download it, or fork); "gone": the account does not have it (or deleted it); "failed". */
export async function claim(ctx: CloudContext, rec: SyncRecord): Promise<"ok" | "moved" | "gone" | "failed"> {
  const a = await call(ctx, "POST", `trainworld/games/${rec.id}/claim`, { device: ctx.device, baseRev: rec.rev });
  if (a.status === 200) return "ok";
  if (a.status === 409) return "moved";
  if (a.status === 404 || a.status === 410) return "gone";
  return "failed";
}

// ---------------------------------------------------------------- settling

/** A fork's name: the slot's and its own save time, as a restored backup is named. */
export const forkName = (name: string, at: number) => `${name} at ${new Date(at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}`;

/** This browser's version of slot `key` copied into a new slot as a new account game (a fork), and
 * uploaded; the new key. */
export async function forkOut(ctx: CloudContext, key: string, stored: StoredSave): Promise<string> {
  const to = "slot-" + crypto.randomUUID();
  const lease = await leaseSlot(to);
  try {
    await writeSlot(to, { ...stored, name: forkName(stored.name || "Saved game", stored.at) });
    await writeSync(to, { id: newGameId(), acct: ctx.acct, rev: 0, syncedAt: 0 });
  } finally { lease?.release(); }
  await upload(ctx, to);
  return to;
}

/** A record for a slot first written while signed in: tied to the account. */
export async function tieNew(ctx: CloudContext, key: string): Promise<SyncRecord> {
  const rec: SyncRecord = { id: ctx.ids?.[key] || newGameId(), acct: ctx.acct, rev: 0, syncedAt: 0 };
  await writeSync(key, rec);
  return rec;
}

/** Settle slot `key`, whose lock the caller holds, against the account's list: upload what changed
 * here, take a newer account copy into an unchanged slot, delete one deleted on the account, and fork
 * one changed on both sides. True if anything was written. */
export async function settleSlot(ctx: CloudContext, key: string, list: AccountList): Promise<boolean> {
  const [stored, rec] = await Promise.all([readStored(key), readSync(key)]);
  if (!stored || !tiedTo(rec, ctx)) return false;
  const g = list.games.find((x) => x.id === rec.id);
  const changed = changedHere(stored, rec) && !rec.gaveWay;
  if (!g && list.tombstones.some((t) => t.id === rec.id)) {
    if (!changed) { await deleteSlot(key); return true; }
    await writeSync(key, { ...rec, id: newGameId(), rev: 0, syncedAt: 0 });
    return await upload(ctx, key) === "ok";
  }
  if (!g) return changed || rec.rev > 0 ? await upload(ctx, key) === "ok" : false;
  if (g.rev === rec.rev) {
    if (!changed) return false;
    if (g.writer === ctx.device) return await upload(ctx, key) === "ok";
    // another device opened it from this rev: its game goes on there, this one's changes are kept apart
  } else if (!changed) {
    return !!await takeInto(ctx, key, g);
  }
  await forkOut(ctx, key, stored);
  return !!await takeInto(ctx, key, g);
}

/** On load: every slot this page can lock (but `skip`, the one it plays) settled against the
 * account, and the deletions owed sent. The account's list afterwards, or null when unreachable. */
export async function settle(ctx: CloudContext, skip = "", stop = () => false): Promise<AccountList | null> {
  if (!ctx.on) return null;
  await sendDeletes(ctx);
  const list = await listGames(ctx);
  if (!list) return null;
  let wrote = false;
  for (const s of await listSlots().catch(() => [] as SaveSlot[])) {
    if (stop()) break;
    if (s.key === skip || !tiedTo(s.cloud, ctx)) continue;
    const lease = await leaseSlot(s.key);
    if (!lease) continue; // open in another tab, whose worker settles it
    try { wrote = await settleSlot(ctx, s.key, list) || wrote; } catch { /* left for next time */ } finally { lease.release(); }
  }
  return wrote ? await listGames(ctx) ?? list : list;
}

/** The slots the sign-in question is about: tied to no account and not kept here only. */
export const untied = (slots: SaveSlot[]) => slots.filter((s) => !s.cloud?.acct && !s.cloud?.local);

/** The sign-in question's answer for `slots`: added to the account (they go up at the next settle),
 * or kept in this browser only. */
export async function answerMerge(ctx: CloudContext, slots: SaveSlot[], add: boolean) {
  for (const s of slots) {
    const id = s.cloud?.id || ctx.ids?.[s.key] || newGameId();
    await writeSync(s.key, add ? { id, acct: ctx.acct, rev: 0, syncedAt: 0 } : { id, rev: 0, syncedAt: 0, local: true });
  }
}
