// The "Saved only in this browser" dot (T-1832, notes/T-1823.md fix 9; Anita's follow-up: a small
// yellow dot just beside the search glass, not a card; T-1868: right of it). Shown once a game kept only in this browser
// has had an hour of play (game/keepNudge.ts); it goes when the game is on the account or downloaded.
// Where accounts are on a click opens the account panel to sign in or make an account; where they are
// off (no account server) the wording is download-only and a click downloads the game, as the saved
// games panel's Download does.
// T-2614: the same dot says when the playing game's latest change has not reached the account for two
// minutes; a click saves it now, which sends it again.

import { accountOpen, accountState } from "../game/account";
import { cloudBehind, keepDue, startKeep } from "../game/keepNudge";
import { exportGame, manageSaveSlot } from "../workers/clockClient";
import { WHY_OFFLINE, WHY_UNREACHABLE } from "../workers/cloud";

const HEAD = "Saved only in this browser!\nIf your browser data is cleared, this game will be lost. ";
const TIP_ACCOUNT = HEAD + "You can make an account to save it online, or download a copy locally.";
const TIP_DOWNLOAD = HEAD + "You can download a copy to keep it.";

/** T-2614: the reason the account gave, then that the game is safe here; a try again only helps a connection */
function behindTip(why: string): string {
  const again = why === WHY_OFFLINE || why === WHY_UNREACHABLE ? " Click to try again." : "";
  return `Not on your account yet\n${why.replace(/([^.])$/, "$1.")} This game is saved in this browser.${again}`;
}

export function KeepDot() {
  const behind = cloudBehind.value;
  if (behind) return <button class="keep-dot" data-tip={behindTip(behind)} aria-label="Not on your account yet"
    onClick={() => manageSaveSlot("save")} />;
  if (!keepDue.value) return null;
  const accounts = accountState.value === "on";
  return <button class="keep-dot" data-tip={accounts ? TIP_ACCOUNT : TIP_DOWNLOAD} aria-label="Saved only in this browser"
    onClick={() => (accounts ? (accountOpen.value = true) : exportGame())} />;
}

/** Start counting play time; the dot itself sits beside the search glass (ui/DemandSelector.tsx). */
export function mountKeepDot() {
  startKeep();
}
