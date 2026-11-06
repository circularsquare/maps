// T-2395: the city view's and the all cities view's home button, filled where this place is the
// game's home; a click makes it home, or on home clears it.

import { home } from "../../game/home";
import { t } from "../../i18n";
import { setHome } from "../../workers/clockClient";
import { WORLD_HOME } from "../../workers/saveFile";
import { Icon } from "../widgets";

/** `place`: a zone id or `WORLD_HOME`. */
export function HomeButton({ place }: { place: string }) {
  const on = home.value === place;
  const where = place === WORLD_HOME ? "world" : "city";
  const tip = t(on ? "homeButton.tipOn" : "homeButton.tipOff", { where });
  return (
    <button class="btn icon" data-tip={tip} aria-label={t(on ? "homeButton.on" : "homeButton.off")} aria-pressed={on}
      onClick={() => setHome(on ? null : place)}>
      {on ? <Icon.homeOn /> : <Icon.home />}
    </button>
  );
}
