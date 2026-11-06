// A city picked on the map zoomed far out (T-186, SPEC 8): its trips within the zone and its longer
// trips to and from other places, each a figure with a mode bar, and the three biggest places
// (T-600, Anita: brief statistics of what the hover has, no table of every place). The map draws
// the city outlined and one bubble per other city (map/intercityView.ts). A click on the city in the
// Cities tab opens it too, the city view (T-1438), with CityDetails.tsx's rows around these.

import { cityById, type CityId } from "../../game/cities";
import { cityDemandViews, eachCorridor, intercityDemand, INTERCITY_PLACES } from "../../game/demand";
import { count, exact } from "../../game/format";
import { zoneLoads } from "../../game/loading";
import { t, tSeries } from "../../i18n";
import { BUBBLE_ALPHA, bubbleColour, setBubbleColours, setBubbleShade } from "../../game/palette";
import { display, purposeFilter, shade, shownDayPhrase } from "../../game/ui";
import { localModes } from "../../map/modeBar";
import { goTo } from "../Search";
import { Icon, SegBar } from "../widgets";
import { pctOf } from "./tripText";
import type { City } from "../../game/cities";
import { builtCities } from "../../game/worldTotals";
import { GoalTicks } from "../GoalTicks";
import { CityFigures, CityNetwork } from "./CityDetails";
import { HomeButton } from "./HomeButton";
import { TripGraph } from "./TripGraph";

/** `note`: a second line for the part's hover */
type Part = { name: string; n: number; cls: string; note?: string };
/** A figure with its mode bar: train blue, walking green, car red, No route grey, as the map's bubbles.
 * `kind`: whether the trips are within the city or to and from other places. */
function TripsBar({ label, tip, kind, city, parts, goals }: { label: string; tip?: string; kind: "within" | "inter"; city: string; parts: Part[]; goals?: City }) {
  const all = parts.reduce((s, p) => s + p.n, 0);
  const phrase = shownDayPhrase.value;
  const bar = <SegBar kind="split-bar" label={tSeries(parts.map((p) => `${p.name} ${pctOf(p.n, all)}`))}
    parts={parts.map((p) => ({ share: p.n / all, cls: p.cls, tip: t("cityInspector.partTip", { name: p.name, count: exact(p.n), kind, city, phrase, share: pctOf(p.n, all) }) + (p.note ? `\n${p.note}` : "") }))} />;
  return (
    <li class={"zone-trips" + (goals ? " with-goals" : "")}>
      <div class="line">
        <span class="grow" data-tip={tip}>{label}</span>
        <span class="val" data-tip={t("cityInspector.totalTip", { count: exact(all), kind, city, phrase })}>{count(all)}</span>
      </div>
      {/* T-1437: the goals over the city's own split (T-1551: only here, not on the top bar) */}
      {goals ? <div class="goal-bar">{bar}<GoalTicks city={goals} /></div> : bar}
    </li>
  );
}

export function CityInspector({ id }: { id: string }) {
  const city = cityById(id);
  // T-421: a zone still loading (its bar on the map) has no figures to state yet
  const market = zoneLoads.value.has(id as CityId) ? null : intercityDemand.value;
  if (!city) return null;
  const byPlace = new Map<number, { all: number; rail: number; air: number }>();
  const c = market?.corridors;
  if (c) eachCorridor(c, (from, to) => from !== to && (from === city.place || to === city.place), (j, v) => {
    const other = c.table.from[j] === city.place ? c.table.to[j] : c.table.from[j];
    const p = byPlace.get(other) ?? { all: 0, rail: 0, air: 0 };
    p.all += v.trips;
    p.rail += v.rail;
    p.air += v.air;
    byPlace.set(other, p);
  });
  const places = [...byPlace].filter(([, p]) => p.all > 0).sort((a, b) => b[1].all - a[1].all);
  const interAll = places.reduce((s, [, p]) => s + p.all, 0);
  const interRail = places.reduce((s, [, p]) => s + p.rail, 0);
  const interAir = places.reduce((s, [, p]) => s + p.air, 0);
  // T-378: both ways, as the corridors count them
  const elsewhere = 2 * (market?.elsewhere?.[city.place] ?? 0);
  // within the zone: the open zone's modelled split, or only a count for a zone not opened
  const view = cityDemandViews.value.get(id as CityId);
  const lm = localModes(view);
  const localAll = view ? view.tripsPerDay : city.localTrips ?? 0;
  // the title mark is the zone's bubble colour (map/intercityView.ts): local and inter-region together
  const walk = lm?.walk ?? 0, none = lm?.none ?? 0;
  const rail = interRail + (lm?.train ?? 0);
  const d = display.value;
  setBubbleShade(shade.value);
  setBubbleColours(d.bubbleColours, d.rampFrom, d.rampTo);
  const mix = bubbleColour(rail, walk, Math.max(0, interAll + localAll - rail - walk - none), none);
  const filter = purposeFilter.value;
  const drawn = filter === "all";
  const top = places.slice(0, 3);
  const built = builtCities().has(city.id);
  return (
    <>
      <div class="title">
        <i class="mk" style={{ background: `rgba(${mix[0]}, ${mix[1]}, ${mix[2]}, ${BUBBLE_ALPHA})` }} />
        <span class="name">{city.name}</span>
        <HomeButton place={city.id} />
        <button class="btn icon" data-tip={t("cityInspector.goTo", { city: city.name })} aria-label={t("cityInspector.goTo", { city: city.name })} onClick={() => goTo({ kind: "city", name: city.name, city })}>
          <Icon.locate />
        </button>
      </div>
      <CityFigures city={city} built={built} />
      {!market && !lm ? (
        <p class="muted">{t("cityInspector.calculating")}</p>
      ) : (
        <>
          <ul class="rows">
            {lm ? (
              <TripsBar
                label={t("cityInspector.tripsWithin", { city: city.name })}
                kind="within"
                city={city.name}
                parts={[
                  { name: t("cityInspector.train"), n: lm.train, cls: "seg-train" },
                  { name: t("cityInspector.walk"), n: lm.walk, cls: "seg-walk" },
                  { name: t("cityInspector.car"), n: lm.drive, cls: "seg-drive" },
                  ...(lm.none > 0 ? [{ name: t("cityInspector.noRoute"), n: lm.none, cls: "seg-none", note: t("inspector.noRoadEnds") }] : []),
                ]}
                goals={city}
              />
            ) : localAll > 0 ? (
              <li class="zone-trips">
                <div class="line">
                  <span class="grow">{t("cityInspector.tripsWithin", { city: city.name })}</span>
                  <span class="val" data-tip={t("cityInspector.totalTip", { count: exact(localAll), kind: "within", city: city.name, phrase: shownDayPhrase.value })}>{count(localAll)}</span>
                </div>
              </li>
            ) : null}
            {market && (
              <TripsBar
                label={t("cityInspector.toOther")}
                tip={t("cityInspector.toOtherTip")}
                kind="inter"
                city={city.name}
                parts={[
                  { name: t("cityInspector.train"), n: interRail, cls: "seg-train" },
                  { name: t("cityInspector.car"), n: Math.max(0, interAll - interRail - interAir), cls: "seg-drive" },
                  ...(interAir > 0 ? [{ name: t("cityInspector.plane"), n: interAir, cls: "seg-other" }] : []),
                ]}
              />
            )}
          </ul>
          {/* T-2500: a zone still loading whose own figures are in (a load's round of its cluster) shows them
              and its graph; its trips to other places wait */}
          {!market && <p class="muted">{t("cityInspector.calculating")}</p>}
          {lm && <TripGraph city={city} />}
          {market && top.length > 0 && (
            <ul class="rows">
              <li class="muted small">{t("cityInspector.topDestinations")}</li>
              {top.map(([place, p]) => (
                <li key={place}>
                  <span class="grow">{INTERCITY_PLACES[place]}</span>
                  <span class="val" data-tip={t("cityInspector.destinationTip", { count: exact(p.all), phrase: shownDayPhrase.value, city: city.name, place: INTERCITY_PLACES[place] })}>{count(p.all)}</span>
                </li>
              ))}
            </ul>
          )}
          {market && !places.length && elsewhere <= 0 && <p class="muted small">{t("cityInspector.noInter")}</p>}
          {!drawn && <p class="muted small">{t("cityInspector.notDrawn")}</p>}
        </>
      )}
      {built && <CityNetwork city={city} />}
    </>
  );
}
