// The "Saved only in this browser" dot (T-1832, notes/T-1823.md fix 9; Anita's follow-up: a small
// yellow dot just left of the search glass, not a card). Shown once a game kept only in this browser
// has had an hour of play (game/keepNudge.ts); it goes when the game is on the account or downloaded.
// Where accounts are on a click opens the account panel to sign in or make an account; where they are
// off (no account server) the wording is download-only and a click downloads the game, as the saved
// games panel's Download does.

import { h, render } from "preact";
import { accountOpen, accountState } from "../game/account";
import { keepDue, startKeep } from "../game/keepNudge";
import { exportGame } from "../workers/clockClient";

const HEAD = "Saved only in this browser!\nIf your browser data is cleared, this game will be lost. ";
const TIP_ACCOUNT = HEAD + "You can make an account to save it online, or download a copy locally.";
const TIP_DOWNLOAD = HEAD + "You can download a copy to keep it.";

function KeepDot() {
  if (!keepDue.value) return null;
  const accounts = accountState.value === "on";
  return <button class="keep-dot" data-tip={accounts ? TIP_ACCOUNT : TIP_DOWNLOAD} aria-label="Saved only in this browser"
    onClick={() => (accounts ? (accountOpen.value = true) : exportGame())} />;
}

/** Mount the dot in the map's container and start counting play time. */
export function mountKeepDot(container: HTMLElement) {
  const root = document.createElement("div");
  container.appendChild(root);
  render(h(KeepDot, null), root);
  startKeep();
}
