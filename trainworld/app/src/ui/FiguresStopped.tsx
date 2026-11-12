// T-3314: the notice at the top centre of the map when the game's figures have stopped updating (game/figuresStopped.ts):
// the clock worker has not answered the page for minutes, or the demand pool is stuck waiting. It says how long ago the
// last autosave was (counting up while it shows; none yet, none said). Reload asks for the autosave first (the clock
// worker may be the stuck part, so it waits at most 1.5 s), then reloads.

import { useEffect, useState } from "preact/hooks";
import { t, tDuration } from "../i18n";
import { figuresStopped } from "../game/figuresStopped";
import { lastSaved } from "../game/money";

/** `save` writes the autosave and resolves when it is safe to leave the page. */
export function FiguresStopped({ save }: { save: () => Promise<void> }) {
  const on = figuresStopped.value;
  const [, tick] = useState(0);
  useEffect(() => {
    if (!on) return;
    const id = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, [on]);
  if (!on) return null;
  const at = lastSaved.value;
  const text = !at
    ? t("figuresStopped.note")
    : t("figuresStopped.noteSaved", { age: tDuration(Math.max(0, Math.round((Date.now() - at) / 1000))) });
  return <div class="demand-stopped-note" role="alert">
    <div class="dsn-row">
      <span>{text}</span>
      <button class="btn" onClick={() => { void save().finally(() => location.reload()); }}>{t("figuresStopped.reload")}</button>
    </div>
  </div>;
}
