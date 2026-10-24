import { useState } from "preact/hooks";
import type { PurposeTrips } from "../../game/demand";
import { andList, capitalise, count } from "../../game/format";
import { PERIODS } from "../../game/periods";
import { shownDayPhrase } from "../../game/ui";
import { purposeColour, shownName } from "../../game/purposes";
import { Spinner } from "../Spinner";

const LABELS = PERIODS.map((p) => `${p.from}–${p.to}`);
const HOURS = PERIODS.map((p) => p.to - p.from);

type Row = { name: string; outward: number[]; returning: number[]; total: number };

function PurposePie({ rows, total, unit }: { rows: Row[]; total: number; unit: string }) {
  let angle = 0;
  const pct = (n: number) => (total > 0 ? Math.round(100 * n / total) : 0);
  // T-194: every slice and key entry gives its count, not only its share
  const words = (p: Row) => `${p.name}: ${count(p.total)} ${unit} ${shownDayPhrase.value}, ${pct(p.total)}%`;
  return <div class="purpose-summary">
    <svg class="purpose-pie" viewBox="0 0 100 100" role="img" aria-label={total > 0 ? `Purpose shares: ${rows.map(p => `${p.name} ${pct(p.total)}%`).join(", ")}` : "No trips in this scope"}>
      {total <= 0 ? <circle cx="50" cy="50" r="48" fill="var(--shade)" /> : rows.filter(p => p.total > 0).map(p => {
        const start = angle, sweep = 2*Math.PI*p.total/total;
        angle += sweep;
        const point = (a: number) => `${50+48*Math.sin(a)} ${50-48*Math.cos(a)}`;
        return sweep >= 2*Math.PI-1e-8
          ? <circle key={p.name} cx="50" cy="50" r="48" fill={purposeColour(p.name)} data-tip={words(p)} />
          : <path key={p.name} d={`M 50 50 L ${point(start)} A 48 48 0 ${sweep > Math.PI ? 1 : 0} 1 ${point(angle)} Z`} fill={purposeColour(p.name)} data-tip={words(p)} />;
      })}
    </svg>
    <ul class="purpose-legend">
      {rows.map(p => <li key={p.name} data-tip={words(p)}><i class="sw" style={{background: purposeColour(p.name)}} /><span>{p.name}</span><b>{pct(p.total)}%</b></li>)}
    </ul>
  </div>;
}

/**
 * One purpose's departures through the day (T-193): a bar per period, trips towards the
 * destination up from a shared baseline and trips home down from it, both as departures an hour
 * on one scale. Each bar's hover gives the count.
 */
function Timing({ row, colour, unit }: { row: Row; colour: string; unit: string }) {
  const max = Math.max(...row.outward.map((v, q) => v / HOURS[q]), ...row.returning.map((v, q) => v / HOURS[q]));
  const sumOut = row.outward.reduce((a, b) => a + b, 0), sumBack = row.returning.reduce((a, b) => a + b, 0);
  const tip = (dir: string, v: number, q: number, all: number) =>
    `${dir}, ${LABELS[q]}: ${count(v)} ${unit} (${count(v / HOURS[q])} an hour), ${all > 0 ? Math.round(100 * v / all) : 0}% of the day's`;
  const h = (v: number, q: number) => `${max > 0 ? 100 * (v / HOURS[q]) / max : 0}%`;
  return <>
    <div class="purpose-mirror">
      {LABELS.map((_, q) => <div class="slot" key={q}>
        <div class="up" data-tip={tip("To destination", row.outward[q], q, sumOut)}><i style={{ background: colour, height: h(row.outward[q], q) }} /></div>
        <div class="down" data-tip={tip("Return home", row.returning[q], q, sumBack)}><i style={{ background: colour, height: h(row.returning[q], q) }} /></div>
      </div>)}
    </div>
    <div class="purpose-axis">{LABELS.map(l => <span key={l}>{l}</span>)}</div>
  </>;
}

/** The selected bubble's actual departure-period mode counts, including crowding. */
export function PurposeBreakdown({ purposes }: { purposes?: PurposeTrips[] }) {
  const [railOnly, setRailOnly] = useState(false);
  const periodCounts = (v: number[][]) => LABELS.map((_, q) => v[0][q] + (railOnly ? 0 : v[1][q] + v[2][q]));
  // T-192: the model's school and university are one row, "School and university"
  let rows: Row[] | undefined;
  if (purposes) {
    const byName = new Map<string, Row>();
    for (const p of purposes) {
      const name = shownName(p.name), outward = periodCounts(p.outward), returning = periodCounts(p.returning);
      const r = byName.get(name);
      if (!r) byName.set(name, { name, outward, returning, total: 0 });
      else outward.forEach((v, q) => { r.outward[q] += v; r.returning[q] += returning[q]; });
    }
    rows = [...byName.values()];
    for (const r of rows) r.total = [...r.outward, ...r.returning].reduce((a, b) => a + b, 0);
  }
  const all = rows?.reduce((a,p) => a+p.total,0) ?? 0;
  /** what a purpose covers */
  const coverage = (name: string): string | undefined =>
    /^(school|university)/i.test(name) ? "Matched public schools and degree-granting colleges across the metro area. Private schools and exclusively online or nondegree students are not included. Homes, campus locations and typical-day attendance are estimated."
    : /^airport/i.test(name) ? "Air passengers to and from the metro area's commercial airports, not connecting passengers or airport staff. For an arriving passenger, the trip from the airport counts as the return. Homes and hotels at the other end are estimated."
    : /^medical/i.test(name) ? "Patients and visitors at hospitals, clinics, doctors and dentists mapped in OpenStreetMap. Each place's share of visits is estimated from its size."
    : /^attraction/i.test(name) ? "Visitors to museums, zoos, theme parks, temples, castles and other attractions in Wikidata. Visitors a year are Wikidata's figure, or estimated from how many Wikipedia languages write about the place. Homes and hotels at the other end are estimated."
    : undefined;
  const school = purposes?.some(p => /^school/i.test(p.name));
  const university = purposes?.some(p => /^university/i.test(p.name));
  // T-142: a city without an airport (Springfield) still lists the purpose, at zero
  const airport = purposes?.some(p => /^airport/i.test(p.name));
  const medical = purposes?.some(p => /^medical/i.test(p.name));
  const missing = [!school && "school attendance", !university && "university attendance", !airport && "airport passengers", !medical && "medical visits"].filter((s): s is string => !!s);
  const missingText = missing.length ? `${andList(missing)} ${missing.length > 1 ? "are" : "is"} not modelled here.` : "";
  const unit = railOnly ? "train trip legs" : "trip legs";
  return <>
    <div class="h"><span data-tip={[missingText && capitalise(missingText), (rows?.length ?? 0) > 1 && "Everyday timing and travel patterns are initial estimates."].filter(Boolean).join(" ") || undefined}>Trip purposes</span></div>
    <div class="purpose-scope">
      <button type="button" aria-pressed={!railOnly} class={!railOnly ? "on" : ""} onClick={() => setRailOnly(false)}>All modes</button>
      <button type="button" aria-pressed={railOnly} class={railOnly ? "on" : ""} onClick={() => setRailOnly(true)}>Train only</button>
    </div>
    {!rows ? <p class="muted"><Spinner tip="Working out the purposes" /></p> : <>
      <p class="muted small" data-tip="Bars above each line are departures an hour toward the destination, bars below it departures home">{count(all)} {railOnly ? "train" : "local"} trip legs {shownDayPhrase.value}</p>
      <PurposePie rows={rows} total={all} unit={unit} />
      {rows.map(p => <div class="purpose-item" key={p.name}>
        <div class="purpose-heading"><span data-tip={coverage(p.name)}><i class="sw" style={{background: purposeColour(p.name)}} /> {p.name}</span><span>{count(p.total)} · {all>0 ? Math.round(100*p.total/all) : 0}%</span></div>
        <Timing row={p} colour={purposeColour(p.name)} unit={unit} />
      </div>)}
    </>}
  </>;
}
