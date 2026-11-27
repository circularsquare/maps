// T-972, T-973: the optional anita.garden account and the leaderboards (notes/T-972.md). Signing in
// is never needed to play. The account is noritetsu's (shonei-server's `/noritetsu/` sign-in routes,
// its token kept under noritetsu's storage key, so one sign-in serves both pages on anita.garden);
// the leaderboards are shonei-server's `/trainworld/` routes (its docs/trainworld-api.md).
//
// Which server: market.anita.garden on anita.garden. On localhost none unless `?acct=local` (a
// shonei-server run on 127.0.0.1:8083) or `?acct=live` was given once (remembered; `?acct=off`
// forgets it), so a dev page or a check makes no request that could fail. Everything waits for
// `/noritetsu/config` to say `trainworld`: an older server, or none, keeps every icon hidden.
//
// While signed in, each normal game's points (the forecast weekday's, the Cities tab's) and its weekday profit (T-3879) are sent at
// most every two wall minutes when they change and only from a whole final solve (T-3639), joined or
// not, so a board is current the moment its player joins. A game owns a random id per browser slot
// (`GAMES_KEY`), or its account game's id (T-988); a sandbox sends only the fact, once, which the server keeps for good. A slot deleted here
// takes its figures off.
//
// T-988: the games themselves go on the account too (workers/cloud.ts, game/accountGames.ts); this
// file gives the clock worker the account (`cloudContext`) on every change.

import { effect, signal } from "@preact/signals";
import { incomeByDay, moneyState, moneyWorld } from "./money";
import { dayProfit, dayProfitByRegion } from "./profitEstimate";
import { pointsBeforeCover, pointsPerDay, scoreSettled } from "./points";
import { startPlaytime } from "./playtime";
import { startCrashCount } from "./crashCount";
import { startGameTime } from "./gameTime";
import { startAchievementCounts } from "./achievementCounts";
import { onAccountGone, postAccount, saveSlotsState } from "../workers/clockClient";
import { listGames, newGameId, onTokenGone, type CloudContext } from "../workers/cloud";
import { listSlots, writeSync } from "../workers/saveFile";
import { t } from "../i18n";
import { TRACK_BUILD } from "./trackBuild";
import { patientFetch, CLOUD_STALL_MS } from "../workers/patientFetch";

export interface AccountView {
  id: string;
  logins: string[];
  username?: string;
  email: string;
  pendingEmail: string;
}
interface Session { token: string; account: AccountView }
export interface AccountConfig { googleClientId: string; dev: boolean; password: boolean; signIn: boolean; trainworld?: boolean; trainworldGames?: boolean; trainworldFeedback?: boolean; trainworldFeedbackThreads?: boolean; /** T-3326: how many screenshots a new feedback takes; absent or 0 on a server that drops them */ trainworldFeedbackImages?: number; trainworldPing?: boolean; /** T-4409: POST /trainworld/crash */ trainworldCrash?: boolean; trainworldAchievements?: boolean; trainworldCommunity?: boolean; trainworldCommunityPreview?: boolean; trainworldCommunitySearch?: boolean; trainworldCommunityPages?: boolean; /** T-3652: the list takes view=newest */ trainworldCommunityNewest?: boolean; trainworldFlags?: boolean; /** T-3807: PUT /trainworld/name */ trainworldRename?: boolean; /** T-3879: the `profit` board */ trainworldProfit?: boolean; /** T-3279: the flag list version the server accepts (game/flags.json `set`) */ trainworldFlagSet?: number }
export interface BoardSetting { on: boolean; name: string; /** set when a moderator hid the display name; absent on a server that does not say */ nameHidden?: boolean; /** T-3269: the flag beside the name, game/flags.ts */ flag?: string }
export interface BoardRow { rank: number; name: string; value: number; flag?: string }
export interface Board { board: string; count: number; rows: BoardRow[]; me?: BoardRow }

/** The most cities a game sends (shonei-server's MaxTwCities): its best ones. */
const MAX_CITIES = 64;
/** Wall ms between two sends of a game's figures. */
const SEND_GAP_MS = 120_000;
/** T-988: wall ms between two asks for the config while the server cannot be reached */
const CONFIG_RETRY_MS = 120_000;
const API_KEY = "trainworld.acct-api";
const GAMES_KEY = "trainworld.score-games.v1";
/** T-988: this browser profile's random device id, nothing about the machine */
const DEVICE_KEY = "trainworld.device.v1";
/** T-988: whether the server last said it keeps games */
const GAMES_ON_KEY = "trainworld.games-on.v1";

const LOCAL = /^(localhost|127\.0\.0\.1)$/.test(location.hostname);

function apiChoice(): "live" | "local" | null {
  if (TRACK_BUILD) return null; // T-4126: the track build has no account
  const q = new URLSearchParams(location.search).get("acct");
  try {
    if (q === "off") localStorage.removeItem(API_KEY);
    else if (q === "live" || q === "local") localStorage.setItem(API_KEY, q);
  } catch { /* storage blocked */ }
  if (/(^|\.)anita\.garden$/.test(location.hostname)) return "live";
  if (!LOCAL) return null;
  try {
    const v = localStorage.getItem(API_KEY);
    return v === "live" || v === "local" ? v : null;
  } catch { return null; }
}

const CHOICE = apiChoice();
const BASE = CHOICE === "live" ? "https://market.anita.garden/" : CHOICE === "local" ? "http://127.0.0.1:8083/" : "";
// noritetsu's key where noritetsu talks to the same server (the live site, or local with local), so
// a sign-in is shared; a localhost page on the live server keeps its own, so it never signs a
// local noritetsu out with a token its server refuses.
const AUTH_KEY = (CHOICE === "live") === !LOCAL ? "noritetsu.auth.v1" : "trainworld.auth.live.v1";

const lsGet = <T>(k: string): T | null => { try { return JSON.parse(localStorage.getItem(k) ?? "null") as T; } catch { return null; } };
const lsSet = (k: string, v: unknown) => {
  try { if (v == null) localStorage.removeItem(k); else localStorage.setItem(k, JSON.stringify(v)); } catch { /* this page only */ }
};

/** "off" (no server for this page), "checking", "on", or "unavailable" (no answer, or a server
 * without the leaderboards); the icons show only when "on". */
export const accountState = signal<"off" | "checking" | "on" | "unavailable">(BASE ? "checking" : "off");
export const accountConfig = signal<AccountConfig | null>(null);
const readSession = (): Session | null => { const s = lsGet<Session>(AUTH_KEY); return s?.token ? s : null; };
export const session = signal<Session | null>(readSession());
/** The leaderboard setting, from `/trainworld/me`; null until it answers. */
export const boardSetting = signal<BoardSetting | null>(null);
export const accountOpen = signal(false);
export const boardsOpen = signal(false);

export class ApiError extends Error { constructor(msg: string, readonly status: number, readonly code = "", /** the rest of an error answer, e.g. `played` seconds on a "too-new" */ readonly extra: Record<string, unknown> | null = null) { super(msg); } }

export async function api<T>(method: string, path: string, body?: unknown, opts: { auth?: boolean; keep401?: boolean; headers?: Record<string, string>; form?: FormData } = {}): Promise<T> {
  const headers: Record<string, string> = { ...opts.headers };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const s = session.peek();
  if (opts.auth !== false && s) headers.Authorization = "Bearer " + s.token; // i18n-ignore
  let r: Response;
  try {
    r = await patientFetch(BASE + path, { method, headers, body: opts.form ?? (body === undefined ? undefined : JSON.stringify(body)) }, CLOUD_STALL_MS);
  } catch {
    throw new ApiError(t("account.unreachable"), 0);
  }
  let j: any = null;
  if (r.status !== 204) { try { j = await r.json(); } catch { /* no body */ } }
  if (!r.ok) {
    // a dead token means signed out; a wrong current password (keep401) does not
    if (r.status === 401 && opts.auth !== false && !opts.keep401 && s) tokenGone();
    const msg = typeof j?.error === "string" ? j.error : t("account.answered", { status: r.status });
    throw new ApiError(msg.charAt(0).toUpperCase() + msg.slice(1).replace(/([^.])$/, "$1."), r.status, typeof j?.code === "string" ? j.code : "", j && typeof j === "object" ? j : null);
  }
  return j as T;
}

/** T-3115: a file from the server as bytes, without the session; an ApiError for a failed answer, "too-big" past
 * `max` bytes (read no further than that). */
export async function fetchBytes(path: string, max: number): Promise<Uint8Array> {
  let r: Response;
  try { r = await patientFetch(BASE + path); } catch { throw new ApiError(t("account.unreachable"), 0); }
  if (!r.ok || !r.body) throw new ApiError(t("account.answered", { status: r.status }), r.status);
  const reader = r.body.getReader(), parts: Uint8Array[] = [];
  let n = 0;
  for (;;) {
    let got: ReadableStreamReadResult<Uint8Array>;
    try { got = await reader.read(); } catch { throw new ApiError(t("account.unreachable"), 0); }
    if (got.done) break;
    n += got.value.length;
    if (n > max) { void reader.cancel().catch(() => {}); throw new ApiError("too big", 413, "too-big"); } // i18n-ignore: a code, not shown
    parts.push(got.value);
  }
  const out = new Uint8Array(n);
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}

/** T-1673: the feedback form's send, to the account server (which mails it on; no address is in
 * the page). Signed in, it carries the session so the mail names the account (T-1719). Throws an ApiError whose message says what to do. */
export async function sendFeedback(f: { name: string; message: string; context: Record<string, string>; kind?: string; images?: { type: string; data: string }[] }) {
  if (!BASE) throw new ApiError(t("account.noServer"), 0);
  // a server that does not say it takes feedback answers 404 without CORS headers: do not call it
  const cfg = accountConfig.peek();
  if (cfg && !cfg.trainworldFeedback) throw new ApiError(t("account.feedbackNotYet"), 404);
  // T-2853: only a server that says it keeps threads gets the tag and the browser key (an older one would
  // refuse the key header in its CORS preflight)
  const threads = cfg?.trainworldFeedbackThreads === true;
  if (threads && !session.peek() && !readFeedbackKey()) makeFeedbackKey();
  await api("POST", "trainworld/feedback", { kind: threads && f.kind ? f.kind : "feedback", message: f.message, name: f.name, context: f.context, ...(threads && f.images?.length ? { images: f.images } : {}) },
    { headers: threads ? keyHeaders() : undefined });
}

// ---------------------------------------------------------------- feedback threads (T-2853)

/** The random key a signed-out browser's feedback belongs to (32 bytes, base64url). Made the first time
 * feedback is sent signed out; the server keeps only its hash. Never in a URL; not the device id. */
const FB_KEY = "trainworld.feedback-key.v1";
/** How many of Anita's replies to each thread were already announced, so a reload says nothing twice */
const FB_CHECK_MS = 30 * 60_000;

export interface FeedbackRow { id: string; kind: string; created: number; updated: number; first: string; messages: number; unread: number }
export interface FeedbackMessage { at: number; from: "player" | "anita"; text: string }
export interface FeedbackThread { id: string; kind: string; created: number; updated: number; unread: number; messages: FeedbackMessage[] }

/** Threads with a reply not yet read: the dot on Submit feedback and on the Settings tab. */
export const feedbackUnread = signal(0);

/** Whether the server keeps feedback threads; off, nothing here is called. */
export const feedbackThreadsOn = () => !!BASE && accountConfig.value?.trainworldFeedbackThreads === true;

export function readFeedbackKey(): string {
  try { const k = localStorage.getItem(FB_KEY); return k && /^[A-Za-z0-9_-]{32,128}$/.test(k) ? k : ""; } catch { return ""; }
}
function makeFeedbackKey(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  const k = btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  try { localStorage.setItem(FB_KEY, k); } catch { return ""; } // blocked storage: sending works, no list shows
  return readFeedbackKey();
}
const keyHeaders = (): Record<string, string> => { const k = readFeedbackKey(); return k ? { "X-Feedback-Key": k } : {}; };
/** T-3326: how many screenshots the form may attach: what the server says, else none. */
export const feedbackImagesMax = () => (feedbackThreadsOn() ? Math.max(0, Math.min(4, accountConfig.value?.trainworldFeedbackImages ?? 0)) : 0);
/** Whether this browser can have earlier feedback to list: a session or a key. */
export const canListFeedback = () => feedbackThreadsOn() && !!(session.value || readFeedbackKey());

const fbTrail = (id: string) => `trainworld/feedback/thread/${encodeURIComponent(id)}`;
export async function listFeedback(): Promise<FeedbackRow[]> {
  const r = await api<{ threads?: FeedbackRow[] }>("GET", "trainworld/feedback/mine", undefined, { headers: keyHeaders() });
  const rows = r.threads ?? [];
  feedbackUnread.value = rows.filter((x) => x.unread > 0).length;
  return rows;
}
export const getFeedbackThread = (id: string) => api<FeedbackThread>("GET", fbTrail(id), undefined, { headers: keyHeaders() });
export const replyFeedback = (id: string, message: string) => api<FeedbackThread>("POST", fbTrail(id) + "/reply", { message }, { headers: keyHeaders() });
export const deleteFeedback = (id: string) => api<void>("DELETE", fbTrail(id), undefined, { headers: keyHeaders() });
export async function markFeedbackRead(id: string) {
  await api<void>("POST", fbTrail(id) + "/read", undefined, { headers: keyHeaders() });
}

let claiming = false;
/** Signed in with a key: the key's threads move to the account and the key goes (on a 200 only). */
async function claimFeedbackKey() {
  const k = readFeedbackKey();
  if (!k || !session.peek() || claiming) return;
  claiming = true;
  try {
    await api("POST", "trainworld/feedback/claim", {}, { headers: { "X-Feedback-Key": k } });
    try { localStorage.removeItem(FB_KEY); } catch { /* blocked */ }
  } catch { /* the next check tries again */ } finally { claiming = false; }
}

/** The cheap check, from memory on the server: claims a key if there is one and sets the dot. */
export async function checkFeedback() {
  if (!feedbackThreadsOn()) return;
  await claimFeedbackKey();
  if (!session.peek() && !readFeedbackKey()) { feedbackUnread.value = 0; return; }
  let unread: { id: string; count: number }[];
  try { unread = (await api<{ unread?: { id: string; count: number }[] }>("GET", "trainworld/feedback/unread", undefined, { headers: keyHeaders() })).unread ?? []; }
  catch { return; } // offline or rate limited: the dot stays as it was
  feedbackUnread.value = unread.length;
}

let watching = false;
function startFeedbackWatch() {
  if (watching || !feedbackThreadsOn()) return;
  watching = true;
  effect(() => { void session.value; void checkFeedback(); }); // on load, and when a sign-in or sign-out changes whose threads these are
  setInterval(() => void checkFeedback(), FB_CHECK_MS);
}

export function storeSession(j: { token: string; account: AccountView }) {
  session.value = { token: j.token, account: j.account };
  lsSet(AUTH_KEY, session.value);
  void loadBoardSetting();
}
function tokenGone() {
  session.value = null;
  boardSetting.value = null;
  lsSet(AUTH_KEY, null);
}
export function signOut() { tokenGone(); }

/** Sign in, register or log in: the answer is a session. */
export async function signIn(path: "signin" | "login" | "register", body: unknown) {
  storeSession(await api<{ token: string; account: AccountView }>("POST", "noritetsu/" + path + (path === "register" ? SITE : ""), body, { auth: false }));
}

/** Names this game in the account server's mails (T-3104); the accounts are shared with noritetsu. */
export const SITE = "?site=trainsworld";

/** Deleting asks again: the password, or a fresh Google credential for an account with none (shonei-server's checkReauth). */
export async function deleteAccount(proof: { password: string } | { credential: string } = { password: "" }) {
  const acct = session.peek()?.account.id;
  await api("DELETE", "noritetsu/account", proof);
  tokenGone();
  lsSet(GAMES_KEY, null);
  // T-988: the games stay in this browser, tied to no account
  for (const s of await listSlots().catch(() => [])) if (acct && s.cloud?.acct === acct) await writeSync(s.key, null);
}

// ---------------------------------------------------------------- games on the account (T-988)

const DEVICE = (() => {
  let d = lsGet<string>(DEVICE_KEY);
  if (!d) { d = newGameId() + newGameId(); lsSet(DEVICE_KEY, d); }
  return d;
})();

/** What the account's games need, while signed in (workers/cloud.ts); `on` once the server says it
 * keeps games. */
export function cloudContext(): CloudContext | null {
  const s = session.peek();
  if (!BASE || !s) return null;
  const ids = Object.fromEntries(Object.entries(games()).filter(([slot, e]) => !slot.startsWith(EARLIER) && !isStale(slot, e)).map(([slot, e]) => [slot, e.id]));
  // a page opened while the server cannot be reached keeps the account's games as last known, so the
  // game plays with "Not on your account yet" and goes up once the server answers again
  const cfg = accountConfig.peek(), st = accountState.peek();
  const on = cfg ? st === "on" && !!cfg.trainworldGames : st === "unavailable" && lsGet<boolean>(GAMES_ON_KEY) === true;
  return { base: BASE, token: s.token, acct: s.account.id, device: DEVICE, on, ids };
}
/** The device id, for "Open on another device". */
export const deviceId = () => DEVICE;

export async function loadBoardSetting() {
  if (!session.peek() || accountState.peek() !== "on") return;
  try { boardSetting.value = (await api<{ board: BoardSetting }>("GET", "trainworld/me")).board; }
  catch { /* shown as not joined; a 401 has signed out */ }
}

export async function setBoard(on: boolean, name = "") {
  try { boardSetting.value = (await api<{ board: BoardSetting }>("PUT", "trainworld/optin", { on, name })).board; }
  catch (e) { throw renameRefused(e) ?? e; }
  if (on) await sendScores(true);
}

/** T-3807: whether the server takes `PUT /trainworld/name`; off, the account panel has no Change name. */
export const renameOn = () => accountState.value === "on" && accountConfig.value?.trainworldRename === true;

/** T-3807: the server's two-a-day limit on name changes, in words; null for any other failure. Join, Share and
 * Change name all meet it. */
export function renameRefused(e: unknown): Error | null {
  const err = e as ApiError;
  if (err?.code !== "rename-limit") return null;
  const sec = typeof err.extra?.retryAfter === "number" ? err.extra.retryAfter : 0;
  return new Error(t("accountPanel.renameLimit", { count: Math.max(1, Math.ceil(sec / 3600)) }));
}

/** T-3807: a new display name, without joining or leaving the boards. Throws an Error to show. */
export async function setDisplayName(name: string) {
  try { boardSetting.value = (await api<{ board: BoardSetting }>("PUT", "trainworld/name", { name })).board; }
  catch (e) {
    const err = e as ApiError;
    if (err.code === "bad-name" || err.code === "need-name") throw new Error(t("community.errBadName"));
    throw renameRefused(e) ?? e;
  }
  boards.value = { ...boards.peek(), at: 0 }; // the boards fetch again when next shown, with the new name
}

// ---------------------------------------------------------------- the boards

export const boards = signal<{ list: Record<string, number>; data: Board | null; key: string; at: number; error: string }>({ list: {}, data: null, key: "points", at: 0, error: "" });
let fetching = false;
/** a board asked for while another was fetching (T-3879: Profit clicked as the panel opens), fetched next */
let wanted: string | null = null;
export async function fetchBoards(key = boards.peek().key) {
  if (fetching) { wanted = key; return; }
  fetching = true;
  wanted = null;
  boards.value = { ...boards.peek(), key };
  try {
    const [list, data] = await Promise.all([
      api<{ boards: Record<string, number> }>("GET", "trainworld/boards", undefined, { auth: false }),
      api<Board>("GET", `trainworld/board?board=${encodeURIComponent(key)}`),
    ]);
    boards.value = { list: list.boards ?? {}, data, key, at: Date.now(), error: "" };
  } catch (e) {
    boards.value = { ...boards.peek(), at: Date.now(), error: (e as Error).message };
  } finally {
    fetching = false;
    if (wanted !== null && wanted !== key) void fetchBoards(wanted);
  }
}

// ---------------------------------------------------------------- scores

/** Each browser slot's game id, the account it last sent to, what it sent, and whether its
 * sandbox notice went. */
interface GameEntry { id: string; acct?: string; sent?: string; at?: number; sandbox?: boolean; shared?: boolean; origin?: string }
const games = (): Record<string, GameEntry> => lsGet<Record<string, GameEntry>>(GAMES_KEY) ?? {};

/** T-3720: the slot's game, by its origin (versions.ts). A slot key is reused (`autosave`, once its game was deleted), and
 * an entry left from the game that was there must never become the new game's id or latch. An entry under the slot's own
 * account id is its game whatever the origin: a pull of that game (writeFetched) can bring another device's origin. */
const slotAt = (slot: string) => saveSlotsState.peek().slots.find((x) => x.key === slot);
const originOf = (slot: string) => slotAt(slot)?.origin;
const isStale = (slot: string, e: GameEntry | undefined) => {
  const x = slotAt(slot);
  return !!(e?.origin && x?.origin && e.origin !== x.origin && e.id !== x.cloud?.id);
};
/** The slot here that holds game `id` as its account game, if any: that game is live, and its figures and share stay. */
const holderOf = (id: string) => saveSlotsState.peek().slots.find((x) => x.cloud?.id === id);
/** An earlier game's entry put aside from its slot (`tidyEntries`), kept until its figures can be taken off. */
const EARLIER = "earlier:";
/** The slot's own entry, or none when it is an earlier game's. */
const entryOf = (slot: string): GameEntry | undefined => { const e = games()[slot]; return isStale(slot, e) ? undefined : e; };
const stamped = (slot: string, e: GameEntry): GameEntry => ({ ...e, origin: originOf(slot) ?? e.origin });

/** The playing game's slot, when it is a stored one this tab writes (not a `?new=1` page). */
function playingSlot(): string | null {
  const s = saveSlotsState.peek();
  return s.writable && s.active && s.slots.some((x) => x.key === s.active) ? s.active : null;
}

/** T-2887: the stored game being played, read as a signal; null for a page with none to share (`?new=1`). */
export function shareableSlot(): string | null {
  void saveSlotsState.value;
  return playingSlot();
}

/** The playing game's id on the account's figures, if it has one yet. */
export function playingGameId(): string | null {
  const slot = playingSlot();
  if (!slot) return null;
  return saveSlotsState.peek().slots.find((x) => x.key === slot)?.cloud?.id ?? entryOf(slot)?.id ?? null;
}

/** T-3132: the stored game in this browser with this id (the leaderboards' and Community's), if any. */
export function slotOfGame(id: string): string | null {
  return saveSlotsState.peek().slots.find((x) => x.cloud?.id === id || entryOf(x.key)?.id === id)?.key ?? null;
}

/** T-2887: the id a share of the playing game goes up under, its figures sent first so the server copies the current
 * ones; null with no stored game or no session. A sandbox game's notice goes up even with no figures before it, or the
 * share would not be tagged. Remembered, so a share of a game later deleted here is taken down with it. */
export async function gameForShare(): Promise<string | null> {
  const s = session.peek(), slot = playingSlot(), m = moneyState.peek(), listed = saveSlotsState.peek().game;
  if (!s || !slot) return null;
  tidyEntries(); // an earlier game's entry in this slot is put aside first, never written over (T-3720)
  void dropDeleted();
  await sendScores(true);
  const cid = saveSlotsState.peek().slots.find((x) => x.key === slot)?.cloud?.id;
  let e = entryOf(slot) ?? { id: cid ?? newGameId() };
  if (cid && e.id !== cid) e = { id: cid };
  if (m && unranked(m) && !e.sandbox && m.game === listed) {
    try {
      await api("PUT", `trainworld/score/${e.id}`, latch(m));
      e.sandbox = true;
    } catch { /* shared untagged */ }
  }
  e.acct = s.account.id;
  e.shared = true;
  lsSet(GAMES_KEY, { ...games(), [slot]: stamped(slot, e) });
  return e.id;
}

/** T-3879: the most profit a day, either way, that a game sends (shonei-server's twMaxProfit), US$ */
const MAX_PROFIT = 1e14;

/** The game's figures now: the forecast weekday's points (T-3465) in all, its best cities, and (T-3879) the weekday's
 * profit, US$ (`dayProfit`: fares minus running costs and upkeep), from the same whole final. */
function figures(): { day: number; points: number; cities: Record<string, number>; profit?: number; profitCities?: Record<string, number> } | null {
  const m = moneyState.peek();
  const hour = m?.hours.at(-1)?.hour;
  const how = scoreSettled.peek();
  if (!m || hour === undefined || !how) return null;
  // the weekday row itself: `pointsScore` falls back to the past 24 hours, which a weekend or holiday lowers
  const per = [...pointsPerDay.peek()].map(([id, r]): [string, number] => [id, Math.round(r[0])]).filter(([id, v]) => id && v > 0).sort((a, b) => b[1] - a[1]);
  // the server refuses a figure over 1e12 (twMaxPoints), which would leave a bigger game stuck on its last accepted one
  const cap = (v: number) => Math.min(Math.round(v), 1e12);
  const points = cap(per.reduce((a, [, v]) => a + v, 0));
  // running lines whose points are not in yet; T-3657: unless fares pay none of the running cost, a real 0
  if (points <= 0 && how !== "idle" && !pointsBeforeCover.peek()) return null;
  const cities: Record<string, number> = {};
  for (const [id, v] of per.slice(0, MAX_CITIES)) cities[id] = cap(v);
  const inc = incomeByDay.peek()[0], w = moneyWorld.peek(), profit = inc ? Math.round(dayProfit(inc, m)) : NaN;
  if (!Number.isFinite(profit)) return { day: Math.floor(hour / 24), points, cities };
  const clamp = (v: number) => Math.max(-MAX_PROFIT, Math.min(MAX_PROFIT, Math.round(v)));
  // T-3885: the profit of each city the points name, so the profit boards take a game in the same cities as the points ones
  const byRegion = inc && w ? dayProfitByRegion(w, inc, m) : new Map<string, number>();
  const profitCities: Record<string, number> = {};
  for (const id of Object.keys(cities)) { const v = byRegion.get(id); if (v !== undefined && Number.isFinite(v)) profitCities[id] = clamp(v); }
  return { day: Math.floor(hour / 24), points, cities, profit: clamp(profit), profitCities };
}

/** A game whose figures never rank: a sandbox, (T-3115) a copy of a shared network, or (T-3707) a modded game, which
 * the server is told is a sandbox so it never ranks there either, the copy's own play notwithstanding. */
const unranked = (m: { mode: string; copy?: boolean; modded?: boolean }) => m.mode === "sandbox" || m.copy === true || m.modded === true;
/** T-3707: the latch the server keeps for good; a modded game says so (older servers read it as a sandbox's). T-3721: a
 * copy played as a normal game says so too, which its share shows as Copy. */
const latch = (m: { mode: string; copy?: boolean; modded?: boolean }) =>
  m.modded ? { mode: "sandbox", modded: true } : m.mode !== "sandbox" && m.copy === true ? { mode: "sandbox", copy: true } : { mode: "sandbox" };

/** T-3707: the server keeps scores only from the published game, and refused this page's (shonei-server
 * trainworld_builds.go): a dev page, or a tab open since a publish. Asked again after REFUSED_WAIT_MS, as a publish
 * that has just gone live can be refused for its first minute. */
export const scoresRefused = signal(false);
let refusedAt = 0;
const REFUSED_WAIT_MS = 10 * 60_000;
declare const __TW_BUILD__: string | undefined;
const BUILD = typeof __TW_BUILD__ === "string" ? __TW_BUILD__ : "dev";

let sending = false;
/** Send the playing game's figures if they changed and the gap has passed (`now` skips the gap). */
export async function sendScores(now = false): Promise<void> {
  const s = session.peek(), slot = playingSlot(), m = moneyState.peek();
  // T-3687: while a load is under way the money can be the incoming game's and the slot still the one before; a
  // sandbox or a copy opened from a normal game would latch that game for good. T-3734: so both must be one game's.
  if (sending || !s || accountState.peek() !== "on" || !slot || !m || m.game !== saveSlotsState.peek().game) return;
  const acct = s.account.id;
  tidyEntries(); // T-3720: an earlier game's entry in this slot is put aside, never written over or waited on
  // T-988: the slot's game id on the account names its figures too (a fork or a kept copy has a new one)
  const cid = saveSlotsState.peek().slots.find((x) => x.key === slot)?.cloud?.id;
  let e = entryOf(slot) ?? { id: cid ?? newGameId() };
  if (cid && e.id !== cid) e = { id: cid };
  let body: unknown;
  if (unranked(m)) {
    // only a game that may have figures on this account needs the notice; a modded one is latched anyway (T-3707), so
    // a build that does not read the mark (an open tab of an older one) can never list it
    if (e.sandbox || (!m.modded && (e.acct !== acct || !e.sent))) return;
    body = latch(m);
  } else {
    const f = figures();
    if (!f || (scoresRefused.peek() && Date.now() - refusedAt < REFUSED_WAIT_MS)) return;
    // (the server's profit flag too: a server that dropped `profit` is sent to again once it keeps it)
    const sig = JSON.stringify([f.points, f.cities, f.profit ?? null, f.profitCities ?? null, accountConfig.peek()?.trainworldProfit === true]);
    const before = e.acct === acct ? e.sent : undefined;
    if (sig === before || (!before && f.points <= 0)) return;
    if (!now && before && Date.now() - (e.at ?? 0) < SEND_GAP_MS) return;
    body = { mode: "normal", build: BUILD, ...f };
    e.sent = sig;
  }
  sending = true;
  try {
    await api("PUT", `trainworld/score/${e.id}`, body);
    if (unranked(m)) e.sandbox = true;
    else {
      delete e.sandbox; // listed, so not latched: a 403 remembered by mistake, or a latch taken off by hand
      if (scoresRefused.peek()) scoresRefused.value = false;
    }
    e.acct = acct;
    e.at = Date.now();
    // a share made while this went up (gameForShare) is kept: its mark takes the share down with the game
    if (games()[slot]?.id === e.id && games()[slot]?.shared) e.shared = true;
    lsSet(GAMES_KEY, { ...games(), [slot]: stamped(slot, e) });
    if (boards.peek().at) boards.value = { ...boards.peek(), at: 0 }; // the boards may have moved
  } catch (err) {
    // T-3707: not the published build (a dev page, or a tab open since a publish): none of this page's will be kept
    if ((err as ApiError).code === "unpublished") { scoresRefused.value = true; refusedAt = Date.now(); }
    // a game the server knows as a sandbox: it stays unlisted, so stop asking
    else if ((err as ApiError).status === 403) {
      const shared = e.shared || (games()[slot]?.id === e.id && games()[slot]?.shared) || undefined;
      lsSet(GAMES_KEY, { ...games(), [slot]: stamped(slot, { ...e, sandbox: true, acct, ...(shared ? { shared } : {}) }) });
    }
  } finally {
    sending = false;
  }
}

/** T-988: slots removed from this browser at sign-out whose games stay on the account, figures and all:
 * forgotten here, so they are not taken off as deleted. */
export function forgetScoreSlots(keys: string[]) {
  const all = games();
  for (const k of keys) delete all[k];
  lsSet(GAMES_KEY, all);
}

/** T-3720: each slot's entry matched to its game. One under the slot's account id takes the slot's origin, so a later new
 * id for the slot (`splitOff`) still finds it the slot's own; an earlier game's is put aside under `EARLIER`, so the
 * slot's game writes its own entry and the earlier one is still there to take off the account. */
function tidyEntries() {
  const all = games();
  let changed = false;
  for (const x of saveSlotsState.peek().slots) {
    const e = all[x.key];
    if (!e?.origin || !x.origin || e.origin === x.origin) continue;
    if (e.id === x.cloud?.id) all[x.key] = { ...e, origin: x.origin };
    else {
      const k = EARLIER + e.id, had = all[k];
      all[k] = had ? { ...e, ...had, shared: had.shared || e.shared } : e;
      delete all[x.key];
    }
    changed = true;
  }
  if (changed) lsSet(GAMES_KEY, all);
}

/** T-3720: the account's games as last listed in this page, for `goneFromAccount`. */
let listed: { acct: string; at: number; ids: Set<string> } | null = null;
const LISTED_MS = 60_000;
/** Whether game `id` is known to be off the account: a list fetched in this page lacks it (a deletion still owed counts
 * as off, `listGames`). An id the account has is another version of a game, maybe played on another device, and its
 * figures and share stay; with the account unreachable nothing is known. */
async function goneFromAccount(id: string): Promise<boolean> {
  const ctx = cloudContext(), cfg = accountConfig.peek();
  if (!ctx) return false;
  if (!ctx.on) return !!cfg && accountState.peek() === "on" && !cfg.trainworldGames; // a server that keeps no games
  if (!listed || listed.acct !== ctx.acct || Date.now() - listed.at > LISTED_MS) {
    const list = await listGames(ctx).catch(() => null);
    if (!list) return false;
    listed = { acct: ctx.acct, at: Date.now(), ids: new Set(list.games.map((g) => g.id)) };
  }
  return !listed.ids.has(id);
}

/** Entry `slot` forgotten here, if it still names game `id`; moved to `to` instead when that slot has none. */
function forgetEntry(slot: string, id: string, to?: string) {
  const now = games();
  if (now[slot]?.id !== id) return;
  if (to && to !== slot && !now[to]) now[to] = stamped(to, now[slot]);
  delete now[slot];
  lsSet(GAMES_KEY, now);
}

/** Slots gone from this browser take their figures off the account they were sent to. One run at a time, across tabs
 * too, so an entry's DELETE goes once; a call during a run gets another after it. */
let dropping: Promise<void> | null = null, dropAgain = false;
function dropDeleted(): Promise<void> {
  tidyEntries();
  if (dropping) { dropAgain = true; return dropping; }
  dropping = (async () => {
    try { do { dropAgain = false; await oneTabAtATime(dropOnce); } while (dropAgain); }
    finally { dropping = null; }
  })();
  return dropping;
}
/** Another tab's run in progress does this one's work: the entries are shared. */
const oneTabAtATime = (f: () => Promise<void>): Promise<void> =>
  typeof navigator !== "undefined" && navigator.locks ? navigator.locks.request("trainworld-score-drop", { ifAvailable: true }, (lock) => lock ? f() : undefined) : f();

async function dropOnce() {
  if (!saveSlotsState.peek().slots.length) return; // storage unread or blocked: nothing is known to be gone
  for (const slot of Object.keys(games())) {
    // read again for each: an earlier one's request may have taken a while
    const s = session.peek(), st = saveSlotsState.peek(), e = games()[slot];
    // a slot here, the game played (not written yet), or an earlier game's entry not put aside yet (`tidyEntries`)
    if (!e || st.slots.some((x) => x.key === slot) || slot === st.active) continue;
    // a game another slot here holds lives on there (a tab's change kept aside, keepOtherTabs)
    const holder = holderOf(e.id);
    if (holder) { forgetEntry(slot, e.id, holder.key); continue; }
    if ((e.sent && !e.sandbox) || e.shared) {
      if (!s || e.acct !== s.account.id || accountState.peek() !== "on") continue; // later, signed in
      if (!await goneFromAccount(e.id)) continue; // T-3720: later, once the account no longer has it
      if (games()[slot]?.id !== e.id || holderOf(e.id)) continue; // changed during the list's request
      try { await api("DELETE", `trainworld/score/${e.id}`); } catch (err) { if ((err as ApiError).status !== 404) continue; }
    }
    forgetEntry(slot, e.id); // unless the slot's entry changed under the request
  }
}

// ---------------------------------------------------------------- start

let started = false;
/** Ask the server once whether accounts are on; refresh the session; start sending. Idempotent. */
export function startAccount() {
  startGameTime();
  if (started || !BASE) return;
  started = true;
  // T-988: the clock worker gets the account on every change; a dead token found there signs out
  onTokenGone(tokenGone);
  onAccountGone(tokenGone);
  let posted = "";
  effect(() => {
    void session.value; void accountState.value; void accountConfig.value;
    const c = cloudContext(), k = JSON.stringify(c ? [c.token, c.acct, c.on] : null);
    if (k === posted) return;
    posted = k;
    postAccount(c);
  });
  const askConfig = async () => {
    try {
      const cfg = await api<AccountConfig>("GET", "noritetsu/config", undefined, { auth: false });
      lsSet(GAMES_ON_KEY, !!cfg.trainworldGames);
      accountConfig.value = cfg;
      accountState.value = cfg.signIn && cfg.trainworld ? "on" : "unavailable";
      return true;
    } catch {
      accountState.value = "unavailable";
      return false;
    }
  };
  void (async () => {
    // T-988: a page opened while the server could not be reached asks again, back online or every
    // CONFIG_RETRY_MS, so its account comes back without a reload
    if (!await askConfig()) await new Promise<void>((done) => {
      let busy = false;
      const again = async () => {
        if (busy) return;
        busy = true;
        if (await askConfig()) { clearInterval(t); removeEventListener("online", again); done(); }
        busy = false;
      };
      const t = setInterval(again, CONFIG_RETRY_MS);
      addEventListener("online", again);
    });
    // T-1771: plays and playtime, only against a server that says it takes them
    if (accountConfig.peek()?.trainworldPing) startPlaytime((p) => api("POST", "trainworld/ping", p));
    // T-4409: failure counts, without the session
    if (accountConfig.peek()?.trainworldCrash) startCrashCount((r) => api("POST", "trainworld/crash", r, { auth: false }));
    // T-3038: how many players have each achievement; sent without the session
    if (accountConfig.peek()?.trainworldAchievements) startAchievementCounts((method, path, body) => api(method, path, body, { auth: false }));
    if (accountState.peek() !== "on") { startFeedbackWatch(); return; }
    if (session.peek()) {
      try { storeSession(await api("POST", "noritetsu/refresh")); }
      catch { /* a 401 has signed out; offline keeps the token for next time */ }
    }
    startFeedbackWatch(); // T-2853: after the refresh, so a lapsed session is found out before a claim
    setInterval(() => void sendScores(), 15_000);
    effect(() => { void saveSlotsState.value; void session.value; void accountState.value; void dropDeleted(); });
    document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") void sendScores(true); });
  })();
  // another tab on this origin (noritetsu, or this game) signed in or out
  addEventListener("storage", (e) => {
    if (e.key !== AUTH_KEY) return;
    session.value = readSession();
    boardSetting.value = null;
    void loadBoardSetting();
  });
}
