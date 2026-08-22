// Pure chart renderers. Take numbers, return an SVG string. No DOM, no deps,
// no state — so they can be dropped into the hover card, the edit panel, or a
// search result without any of them knowing about the others.

const esc = s => String(s ?? '').replace(/[&<>"]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export const compact = n => {
  if (n == null) return '';
  const a = Math.abs(n);
  if (a >= 1e9) return (n / 1e9).toFixed(a >= 1e10 ? 0 : 1) + 'B';
  if (a >= 1e6) return (n / 1e6).toFixed(a >= 1e7 ? 0 : 1) + 'M';
  if (a >= 1e3) return (n / 1e3).toFixed(a >= 1e4 ? 0 : 1) + 'k';
  return String(Math.round(n));
};

/**
 * Population-history sparkline.
 *
 * `values` is one number (or null) per entry in `years`. Nulls are skipped
 * rather than drawn as zero — a gap in the series is not a population crash.
 *
 * Anything at or after `projFrom` is drawn DASHED. GHS runs to 2030, and a
 * projection rendered identically to observation is a quiet lie: the line would
 * claim to know 2030 as well as it knows 1990.
 *
 * The y-axis is zero-based, deliberately. Auto-scaling to min..max makes every
 * city look dramatic — a town that went 50k -> 52k gets the same swooping curve
 * as one that went 50k -> 5M, which is exactly backwards for a card whose job
 * is comparison across cities.
 */
export function sparkline(years, values, opts = {}) {
  const W = opts.w || 272, H = opts.h || 34, P = 3;
  const pts = years.map((y, i) => [y, values[i]])
                   .filter(([, v]) => v != null && isFinite(v));
  if (pts.length < 2) return '';

  const projFrom = opts.projFrom ?? Infinity;
  const x0 = pts[0][0], x1 = pts[pts.length - 1][0];
  const top = Math.max(...pts.map(p => p[1]));
  if (!(top > 0)) return '';

  const px = y => P + (y - x0) / (x1 - x0 || 1) * (W - 2 * P);
  const py = v => H - P - (v / top) * (H - 2 * P);
  const xy = pts.map(([y, v]) => `${px(y).toFixed(1)},${py(v).toFixed(1)}`);

  // Split at the projection boundary, repeating the boundary point so the solid
  // and dashed runs meet with no gap.
  const iSplit = pts.findIndex(([y]) => y >= projFrom);
  const solid = iSplit < 0 ? xy : xy.slice(0, iSplit + 1);
  const dashed = iSplit <= 0 ? [] : xy.slice(iSplit - 1);

  const stroke = opts.color || 'var(--accent)';
  const line = (p, dash) => p.length < 2 ? '' :
    `<polyline points="${p.join(' ')}" fill="none" stroke="${stroke}"
       stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"
       ${dash ? 'stroke-dasharray="2.5 2.5" opacity=".65"' : ''}/>`;

  const last = pts[pts.length - 1];
  return `<svg class="spark" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"
      preserveAspectRatio="none" role="img"
      aria-label="population ${esc(compact(pts[0][1]))} in ${x0} to ${esc(compact(last[1]))} in ${x1}">
    <line x1="${P}" y1="${H - P}" x2="${W - P}" y2="${H - P}"
          stroke="var(--line)" stroke-width="1"/>
    ${line(solid, false)}${line(dashed, true)}
    <circle cx="${px(last[0]).toFixed(1)}" cy="${py(last[1]).toFixed(1)}" r="1.8"
            fill="${stroke}" opacity=".65"/>
  </svg>`;
}
