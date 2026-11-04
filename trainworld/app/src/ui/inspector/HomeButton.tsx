// T-2395: the city view's and the all cities view's home button, filled where this place is the
// game's home; a click makes it home, or on home clears it.

import { home } from "../../game/home";
import { setHome } from "../../workers/clockClient";
import { WORLD_HOME } from "../../workers/saveFile";
import { Icon } from "../widgets";

/** `place`: a zone id or `WORLD_HOME`. */
export function HomeButton({ place }: { place: string }) {
  const on = home.value === place;
  const how = place === WORLD_HOME ? "start zoomed out on all cities" : "start here";
  const tip = on ? `Home\nYou ${how} whenever you open this game` : `Set as home\n${how[0].toUpperCase() + how.slice(1)} whenever you open this game`;
  return (
    <button class="btn icon" data-tip={tip} aria-label={on ? "Home" : "Set as home"} aria-pressed={on}
      onClick={() => setHome(on ? null : place)}>
      {on ? <Icon.homeOn /> : <Icon.home />}
    </button>
  );
}
