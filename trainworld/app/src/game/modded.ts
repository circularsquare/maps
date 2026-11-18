// T-3707: a game played in a page that is not the published game, or with a mod loaded, is marked modded: the mark
// goes into its save (`SaveHeader.modded`) and stays there, and a modded game never reaches the leaderboards
// (account.ts sends the server's latch, which refuses its figures for good). The mark is for the page: once set, every
// game played in it afterwards is marked too, the one open now and any loaded or started later.
//
// Set here when the page can tell cheaply: it is served from a site other than anita.garden (or a local one), or it
// was built from the public repo (`unofficial-build.txt`, tools/opensource). Never from a build stamp the page does not
// find published: Anita's 8800 copy and agents' dev servers carry such stamps and their saves can reach her account; the
// server refuses those scores instead (shonei-server trainworld_builds.go).
//
// Mods call `trainsworld.markModded("<name>")` when they load (the mod loader of T-3377 will do it for them).

import { signal } from "@preact/signals";
import { postModded } from "../workers/clockClient";

declare const __TW_UNOFFICIAL__: boolean | undefined;

/** Why this page's games are modded, one entry each, as kept in the save; empty for the published game. */
export const moddedWhy = signal<string[]>([]);

const MAX_WHY = 8, MAX_LEN = 40;

/** Mark every game this page plays as modded, for good. `name` says why (a mod's name), kept in the save. */
export function markModded(name?: unknown) {
  const why = (typeof name === "string" && name.trim() ? name.trim() : "mod").replace(/[\u0000-\u001f\u007f]/g, "").slice(0, MAX_LEN);
  const now = moddedWhy.peek();
  if (now.includes(why) || now.length >= MAX_WHY) return; // already marked
  moddedWhy.value = [...now, why];
  postModded(moddedWhy.peek());
}

/** The cheap signs of a page that is not the published game, as reasons. */
function detected(): string[] {
  const out: string[] = [];
  const host = location.hostname;
  // a local, home-network or tailnet address is a test of our own (a phone on the dev server), never marked: a wrong
  // mark is permanent, so anything that isn't plainly a public site counts as local (a bare machine name, .lan, 100.64/10)
  const local = /^(localhost|\[::1\]|127\.\d+\.\d+\.\d+|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+|100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+|\[f[cd][0-9a-f:]*\]|\[fe80:[0-9a-f:]*\])$/i.test(host)
    || !host.includes(".") || /\.(local|lan|home|internal|localhost|home\.arpa|ts\.net)$/i.test(host);
  if (!/(^|\.)anita\.garden$/.test(host) && !local) out.push(("site:" + (host || location.protocol)).slice(0, MAX_LEN));
  if (typeof __TW_UNOFFICIAL__ === "boolean" && __TW_UNOFFICIAL__) out.push("unofficial build"); // i18n-ignore: kept in the save, not shown
  return out;
}

/** Once, at page start: the detected marks, and the entry point for mods. */
export function startModded() {
  (window as unknown as { trainsworld?: unknown }).trainsworld = Object.freeze({ markModded });
  for (const why of detected()) markModded(why);
}
