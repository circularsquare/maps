// The "Saved in this browser only" card (T-1832, notes/T-1823.md fix 9): a post-it like the Getting
// started card, shown once a game kept only in this browser has had an hour of play (game/keepNudge.ts).
// Sign in opens the account panel, Download saves the game as a file; the x closes it for good.
// It waits while the Getting started card is up, and under the saves panel and any question.

import { h, render } from "preact";
import { accountOpen, accountState, session } from "../game/account";
import { addToAccount } from "../game/accountGames";
import { guideShown } from "../game/guide";
import { closeKeep, keepDue, startKeep } from "../game/keepNudge";
import { moneyState } from "../game/money";
import { ask } from "../game/ui";
import { exportGame, saveSlotsState } from "../workers/clockClient";
import { savesOpen } from "./SavesPanel";
import { Icon } from "./widgets";

function KeepCard() {
  const m = moneyState.value, g = m?.guide;
  if (!keepDue.value || ask.value || savesOpen.value || accountOpen.value || guideShown(g, m?.mode === "sandbox")) return null;
  const accounts = accountState.value === "on", signedIn = !!session.value;
  const { slots, active } = saveSlotsState.value, slot = slots.find((s) => s.key === active);
  const words = !accounts
    ? "Clearing your browser data deletes this game. Download a copy to keep it."
    : signedIn ? "Clearing your browser data deletes this game. Add it to your account, or download a copy."
      : "Clearing your browser data deletes this game. Sign in to keep it on your account, or download a copy.";
  return (
    <div class="keep-card" role="alert">
      <div class="keep-head">
        <strong>Saved in this browser only</strong>
        <button class="btn icon keep-x" data-tip="Close" aria-label="Close" onClick={closeKeep}><Icon.cross /></button>
      </div>
      <p>{words}</p>
      <div class="keep-buttons">
        {accounts && !signedIn && <button class="btn text" onClick={() => (accountOpen.value = true)}>Sign in</button>}
        {accounts && signedIn && slot && <button class="btn text" onClick={() => void addToAccount([slot])}>Add to account</button>}
        <button class="btn text" onClick={exportGame}>Download</button>
      </div>
    </div>
  );
}

/** Mount the card in the map's container and start counting play time. */
export function mountKeepCard(container: HTMLElement) {
  const root = document.createElement("div");
  container.appendChild(root);
  render(h(KeepCard, null), root);
  startKeep();
}
