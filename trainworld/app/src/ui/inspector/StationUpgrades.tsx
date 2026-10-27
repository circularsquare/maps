// The station panel's row of upgrades (T-1281): each icon a toggle, outlined when not built, dashed
// while in the blueprint, filled when built. On a built station a click pays at once, as the
// junction inspector's flyover does. The passage (the player's link, T-197) starts a pick of the
// other station instead; its passages are listed below the row.

import { money, roughDistance } from "../../game/format";
import { LINK_MAX_M } from "../../game/complexes";
import { cancelLink, linking, linksAmong, startLink } from "../../game/links";
import type { StationInput } from "../../game/types";
import { hasUpgrade, PASSAGE_BASE, PASSAGE_PER_KM, UPGRADE, UPGRADE_BITS, UPGRADE_TEXT, type UpgradeText } from "../../game/upgrades";
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

/** `members`: the complex's stations in complex view, where park and ride and the waiting room are
 * built for all of them at once as one shared facility (T-1528), and the passage. */
export function StationUpgrades({ s, members }: { s: StationInput; members?: string[] }) {
  const prices = useClockQuery({ what: "upgradeCost", node: s.num }, [s.num, s.level, s.built], (d) => Array.from(d));
  const toggle = async (bit: number, name: string, on: boolean) => {
    const a = await edit({ op: "stationUpgrade", nodes: [s.num], bit, on });
    if (!a.ok) return;
    if (on) say(a.charge > 0 ? `${name} built at ${s.name} for ${money(a.charge * 1e6)}.` : `${name} added to the blueprint.`);
    else if (s.built) say(`${name} removed from ${s.name}. Nothing is refunded.`);
  };
  if (members) return (
    <div class="upgrades" role="group" aria-label="Complex upgrades">
      {UPGRADE_TEXT.filter((u) => u.doesAll).map((u) => <ComplexUpgrade u={u} members={members} price={prices?.[UPGRADE_BITS.indexOf(u.bit)] ?? 0} />)}
      <PassageButton s={s} />
    </div>
  );
  return (
    <div class="upgrades" role="group" aria-label="Station upgrades">
      {UPGRADE_TEXT.map((u) => {
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

/** Park and ride or the waiting room for a whole complex (T-1528): on at every station or none, one
 * station's price at each station that has none yet (the same everywhere); a complex where only some
 * have it shows as not built, and a click builds it at the rest. */
function ComplexUpgrade({ u, members, price }: { u: UpgradeText; members: string[]; price: number }) {
  const w = world.value;
  const byId = w ? stationsById(w) : new Map<string, StationInput>();
  const st = members.map((id) => byId.get(id)).filter((x): x is StationInput => !!x);
  const have = st.filter((x) => hasUpgrade(x.upgrades, u.bit)), missing = st.filter((x) => !hasUpgrade(x.upgrades, u.bit));
  const all = missing.length === 0, built = st.some((x) => x.built), name = st[0]?.name ?? "";
  const state = !all ? "" : have.some((x) => x.built) ? " on" : " planned";
  const cost = (all ? st.length : missing.length) * price;
  const act = all
    ? built ? `Built. Click to remove it from all ${st.length} stations; nothing is refunded.` : "In the blueprint. Click to take it out."
    : (have.length ? `${have.length} of ${st.length} stations have one. Click to build it at the other ${missing.length}` : `Click to build it for all ${st.length} stations`)
      + (built ? ", paid at once." : " in the blueprint.");
  const click = async () => {
    const a = await edit({ op: "stationUpgrade", nodes: (all ? st : missing).map((x) => x.num), bit: u.bit, on: !all });
    if (!a.ok) return;
    if (!all) say(a.charge > 0 ? `${u.name} built at ${name} for ${money(a.charge * 1e6)}.` : `${u.name} added to the blueprint.`);
    else if (built) say(`${u.name} removed from ${name}. Nothing is refunded.`);
  };
  const Ic = ICONS[u.bit];
  return (
    <button class={"btn icon upgrade" + state} aria-label={u.name} aria-pressed={all}
      data-tip={`${u.name}. ${u.doesAll} ${money(cost * 1e6)}.\n${act}`} onClick={() => void click()}>
      <Ic />
    </button>
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
