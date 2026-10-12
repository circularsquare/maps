import { useState } from "preact/hooks";
import type { PurposeTrips } from "../../game/demand";
import { count } from "../../game/format";

const LABELS = ["6–10", "10–16", "16–20", "20–24", "0–6"];
const HOURS = [4, 6, 4, 4, 6];

// Purpose colours are independent of the UI theme. Green is reserved for future events.
function purposeColour(name: string) {
  if (/^work/i.test(name)) return "#3978c6";
  if (/school/i.test(name)) return "#168b91";
  if (/university/i.test(name)) return "#c48225";
  if (/social|leisure/i.test(name)) return "#9460b4";
  if (/shopping|errands/i.test(name)) return "#d94a53";
  if (/event/i.test(name)) return "#439447";
  return "#8a817b";
}

function PurposePie({ rows, total }: { rows: {name: string; total: number}[]; total: number }) {
  let angle = 0;
  return <div class="purpose-summary">
    <svg class="purpose-pie" viewBox="0 0 100 100" role="img" aria-label={total > 0 ? `Purpose shares: ${rows.map(p => `${p.name} ${Math.round(100*p.total/total)}%`).join(", ")}` : "No trips in this scope"}>
      {total <= 0 ? <circle cx="50" cy="50" r="48" fill="var(--shade)" /> : rows.filter(p => p.total > 0).map(p => {
        const start = angle, sweep = 2*Math.PI*p.total/total;
        angle += sweep;
        const point = (a: number) => `${50+48*Math.sin(a)} ${50-48*Math.cos(a)}`;
        return sweep >= 2*Math.PI-1e-8
          ? <circle key={p.name} cx="50" cy="50" r="48" fill={purposeColour(p.name)}><title>{p.name}: 100%</title></circle>
          : <path key={p.name} d={`M 50 50 L ${point(start)} A 48 48 0 ${sweep > Math.PI ? 1 : 0} 1 ${point(angle)} Z`} fill={purposeColour(p.name)}><title>{p.name}: {Math.round(100*p.total/total)}%</title></path>;
      })}
    </svg>
    <ul class="purpose-legend">
      {rows.map(p => <li key={p.name}><i class="sw" style={{background: purposeColour(p.name)}} /><span>{p.name}</span><b>{total > 0 ? Math.round(100*p.total/total) : 0}%</b></li>)}
    </ul>
  </div>;
}

/** The selected bubble's actual departure-period mode counts, including crowding. */
export function PurposeBreakdown({ purposes }: { purposes?: PurposeTrips[] }) {
  const [railOnly, setRailOnly] = useState(false);
  const periodCounts = (v: number[][]) => LABELS.map((_, q) => v[0][q] + (railOnly ? 0 : v[1][q] + v[2][q]));
  const rows = purposes?.map(p => {
    const outward = periodCounts(p.outward), returning = periodCounts(p.returning);
    return {name: p.name, outward, returning, total: [...outward, ...returning].reduce((a,b) => a+b,0)};
  });
  const all = rows?.reduce((a,p) => a+p.total,0) ?? 0;
  const school = purposes?.some(p => /^school/i.test(p.name));
  const university = purposes?.some(p => /^university/i.test(p.name));
  const chart = (values: number[], label: string, maximum: number, colour: string) => (
    <div class="purpose-chart-row">
      <span class="muted small">{label}</span>
      <div class="purpose-bars">
        {values.map((v,q) => <div class="purpose-bar-slot" title={`${LABELS[q]}: ${count(v)} trips (${count(v/HOURS[q])}/hour); ${values.reduce((a,b)=>a+b,0)>0 ? Math.round(v/values.reduce((a,b)=>a+b,0)*100) : 0}% of this direction`}>
          <i style={{background: colour, height: `${maximum > 0 ? 100*(v/HOURS[q])/maximum : 0}%`}} />
        </div>)}
      </div>
    </div>
  );
  return <>
    <div class="h">Trip purposes</div>
    <div class="purpose-scope">
      <button type="button" aria-pressed={!railOnly} class={!railOnly ? "on" : ""} onClick={() => setRailOnly(false)}>All modes</button>
      <button type="button" aria-pressed={railOnly} class={railOnly ? "on" : ""} onClick={() => setRailOnly(true)}>Train only</button>
    </div>
    {!rows ? <p class="muted">Working it out.</p> : <>
      <p class="muted small">{count(all)} {railOnly ? "train" : "local"} trip legs a day. Bars show departures per hour.</p>
      <PurposePie rows={rows} total={all} />
      {rows.map(p => {
        const max = Math.max(...p.outward.map((v,q) => v/HOURS[q]), ...p.returning.map((v,q) => v/HOURS[q]));
        return <div class="purpose-item" key={p.name}>
          <div class="purpose-heading"><span><i class="sw" style={{background: purposeColour(p.name)}} /> {p.name}</span><span>{count(p.total)} · {all>0 ? Math.round(100*p.total/all) : 0}%</span></div>
          {chart(p.outward,"To destination",max,purposeColour(p.name))}
          {chart(p.returning,"Return home",max,purposeColour(p.name))}
          <div class="purpose-axis"><span />{LABELS.map(l => <span>{l}</span>)}</div>
        </div>;
      })}
      {school && <p class="muted small">School covers matched public-school sites across the New York metro. Private-school attendance is not included. Home locations and typical school-day attendance are estimated.</p>}
      {university && <p class="muted small">University covers matched degree-granting colleges across the New York metro. Exclusively online and nondegree undergraduate students are excluded. Campus locations, adult homes and typical-day attendance are approximate.</p>}
      <p class="muted small">{!university && "University attendance, "}{!school && "School attendance, "}{school ? "Airport" : "airport"} passengers and medical visits are not modelled yet. {rows.length>1 && "Everyday timing and travel patterns are initial estimates."}</p>
    </>}
  </>;
}
