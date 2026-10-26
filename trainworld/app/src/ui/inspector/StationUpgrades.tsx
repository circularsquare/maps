// The station panel's row of upgrades (T-1281): each icon a toggle, outlined when not built, dashed
// while in the blueprint, filled when built. On a built station a click pays at once, as the
// junction inspector's flyover does. The passage (the player's link, T-197) starts a pick of the
// other station instead; its passages are listed below the row.

import { money, roughDistance } from "../../game/format";
import { LINK_MAX_M } from "../../game/complexes";
import { cancelLink, linking, linksAmong, startLink } from "../../game/links";
import type { StationInput } from "../../game/types";
import { hasUpgrade, PASSAGE_BASE, PASSAGE_PER_KM, UPGRADE, UPGRADE_BITS, UPGRADE_TEXT } from "../../game/upgrades";
import { edit, say, stationsById, world } from "../../game/world";
import { useClockQuery } from "../useClockQuery";

const ICONS: Record<number, () => preact.JSX.Element> = {
  // two sidings running on past a platform's end
  [UPGRADE.terminal]: () => <svg viewBox="0 0 16 16"><path d="M1.5 6h6M1.5 10h6M7.5 6l3 1.5h4M7.5 10l3-1.5h4" /><path class="solid" d="M1.5 7h5v2h-5z" /></svg>,
  [UPGRADE.parkRide]: () => <svg viewBox="0 0 16 16"><path d="M5.5 13V3h3.5a3 3 0 010 6H5.5" /></svg>,
  // a bench under a roof
  [UPGRADE.waitingRoom]: () => <svg viewBox="0 0 16 16"><path d="M2 6l6-3.5L14 6M3.5 10.5h9M4.5 10.5v3M11.5 10.5v3M4.5 8h7" /></svg>,
};
// two platforms and the walkway between them
const PassageIcon = () => <svg viewBox="0 0 16 16"><path class="solid" d="M1.5 3.5h3v9h-3zM11.5 3.5h3v9h-3z" /><path d="M4.5 8h7" /></svg>;

/** `passageOnly`: the row in complex view, where the passage is the one upgrade on offer. */
export function StationUpgrades({ s, passageOnly = false }: { s: StationInput; passageOnly?: boolean }) {
  const prices = useClockQuery({ what: "upgradeCost", node: s.num }, [s.num, s.level, s.built], (d) => Array.from(d));
  const toggle = async (bit: number, name: string, on: boolean) => {
    const a = await edit({ op: "stationUpgrade", nodes: [s.num], bit, on });
    if (!a.ok) return;
    if (on) say(a.charge > 0 ? `${name} built at ${s.name} for ${money(a.charge * 1e6)}.` : `${name} added to the blueprint.`);
    else if (s.built) say(`${name} removed from ${s.name}. Nothing is refunded.`);
  };
  return (
    <div class="upgrades" role="group" aria-label="Station upgrades">
      {!passageOnly && UPGRADE_TEXT.map((u) => {
        const on = hasUpgrade(s.upgrades, u.bit);
        const price = prices?.[UPGRADE_BITS.indexOf(u.bit)] ?? 0;
        const state = !on ? "" : s.built ? " on" : " planned";
        const act = on
          ? s.built ? "Built. Click to remove it; nothing is refunded." : "In the blueprint. Click to take it out."
          : s.built ? "Click to build it, paid at once." : "Click to add it to the blueprint.";
        const Ic = ICONS[u.bit];
        return (
          <button class={"btn icon upgrade" + state} aria-label={u.name} aria-pressed={on}
            data-tip={`${u.name}. ${u.does} ${money(price * 1e6)}.\n${act}`}
            onClick={() => void toggle(u.bit, u.name, !on)}>
            <Ic />
          </button>
        );
      })}
      <PassageButton s={s} />
    </div>
  );
}

function PassageButton({ s }: { s: StationInput }) {
  const picking = linking.value === s.id;
  const w = world.value;
  const byId = w ? stationsById(w) : new Map<string, StationInput>();
  const pairs = linksAmong([s.id]);
  const built = pairs.some(([a, b]) => byId.get(a)?.built && byId.get(b)?.built);
  const state = picking ? " on" : built ? " on" : pairs.length ? " planned" : "";
  const tip = `Passage. Two stations up to ${roughDistance(LINK_MAX_M)} apart joined by one are a complex, and riders change along the straight line on moving walkways. ${money(PASSAGE_BASE * 1e6)} plus ${money(PASSAGE_PER_KM * 1e6)} a km, paid once both stations are built.\n`
    + (picking ? "Click another station on the map, or here to stop." : "Click, then click the other station on the map.");
  return (
    <button class={"btn icon upgrade" + state} aria-label="Passage" aria-pressed={picking} data-tip={tip}
      onClick={() => (picking ? cancelLink() : startLink(s.id))}>
      <PassageIcon />
    </button>
  );
}
