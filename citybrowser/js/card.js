// The hover card. Pure rendering — takes a record, returns markup.
//
// Fields with no data say so explicitly rather than being omitted: a card that
// silently drops empty fields looks complete when it is not, and the whole
// point of this project is knowing what still needs work.

import { languageColor, elevColor, gdpColor, popColor, rgb } from './colors.js';
import { sparkline, compact } from './charts.js';
import { GHS_YEARS, GHS_PROJ_FROM } from './data.js';
import { koppenLabel } from './koppen.js';

const esc = s => String(s ?? '').replace(/[&<>"]/g,
  c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = n => (n == null ? null : Number(n).toLocaleString());
const none = '<span class="missing">&mdash;</span>';

function coord(lat, lon) {
  if (lat == null || lon == null) return none;
  return `${Math.abs(lat).toFixed(3)}&deg;${lat < 0 ? 'S' : 'N'} ` +
         `${Math.abs(lon).toFixed(3)}&deg;${lon < 0 ? 'W' : 'E'}`;
}

function swatch(color, text) {
  return `<span class="sw" style="background:${color}"></span>${esc(text)}`;
}

/** "Wise County · United States", but never "China · China".
 *
 * Wikidata's P131 is often the country itself for a directly-administered city,
 * so joining admin and country blindly prints the same name twice — which reads
 * as a rendering bug even though the data is fine. Chongqing did exactly that.
 */
export function subtitle(c) {
  const parts = [c.adminName, c.countryName].filter(Boolean);
  return parts.length === 2 && parts[0] === parts[1] ? parts[0] : parts.join(' · ');
}

// Climate, from the GHS CLIMATE theme. Per-blob, which matters far less here
// than it does for elevation — annual mean temperature over a 300 km² blob
// really is close to uniform, mountains excepted.
//
// This is NOT the "climate band (monthly min–max)" on the card wishlist. There
// is no monthly series anywhere in the GHS climate theme, only annual bioclim
// aggregates, so that still needs CHELSA. Labelled "annual" so the two are not
// confused when the band eventually lands beside it.
function climate(c) {
  const g = c.ghsCentre;
  if (!g || c.ghsConf === 'none' || (g.tempC == null && !g.koppen)) return '';
  const label = koppenLabel(g.koppen);
  const shifts = g.koppen2070 && g.koppen2070 !== g.koppen;
  return `<div class="sect">Climate <em class="seed">annual</em></div>
    ${g.koppen ? `<div class="kop"><b>${esc(g.koppen)}</b>${
      label ? ` <span>${esc(label)}</span>` : ''}</div>` : ''}
    <dl class="uc">
      ${g.tempC != null ? `<dt>Mean temp</dt><dd>${g.tempC}&thinsp;&deg;C</dd>` : ''}
      ${g.tempRange != null ? `<dt>Range</dt><dd>${g.tempRange}&thinsp;&deg;C</dd>` : ''}
      ${g.precipMm != null ? `<dt>Rainfall</dt><dd>${fmt(g.precipMm)} mm</dd>` : ''}
    </dl>
    ${shifts ? `<div class="kopshift">by 2070 &rarr; <b>${esc(g.koppen2070)}</b>${
      koppenLabel(g.koppen2070) ? ` ${esc(koppenLabel(g.koppen2070))}` : ''
    } <em>SSP2-4.5</em></div>` : ''}`;
}

// The GHS urban-centre block.
//
// The three roles say three genuinely different things and must not be phrased
// alike (see match_ghs.py):
//
//   centre  this city IS the centre        -- the figures describe it
//   member  this city is INSIDE the centre -- the figures describe its blob
//   near    a guess awaiting review        -- carries the "not matched" warning
//
// The population here is NEVER presented as the city's own. That is the whole
// finding in NOTES.md: GHS "Guangzhou" is 43M because it is four cities, and
// "Fort Worth" is 102k because it is a fragment. Shown as what the urban centre
// contains, both are interesting; shown as the city's population, both are
// simply wrong.
function urbanCentre(c) {
  const g = c.ghsCentre;
  if (!g || c.ghsConf === 'none') return '';
  const role = c.ghsRole || 'near';
  const dens = g.area ? Math.round(g.pop / g.area) : null;

  const head = role === 'centre' ? 'Urban centre'
    : role === 'member' ? 'Part of an urban centre'
    : 'Possible urban centre';

  // Only name the blob when it is not just this city's own name repeated back.
  const named = role === 'centre' && esc(g.name) === esc(c.name) ? ''
    : `<div class="ucname">${esc(g.name)}${g.nMembers > 1
        ? ` <em>&middot; ${g.nMembers} places</em>` : ''}</div>`;

  const hist = (g.hist && g.hist.length)
    ? `<div class="sparkwrap">${sparkline(GHS_YEARS, g.hist, { projFrom: GHS_PROJ_FROM })}
         <div class="sparklbl"><span>${GHS_YEARS[0]} ${esc(compact(g.hist[0]))}</span>
           <span>${esc(compact(g.hist[g.hist.length - 1]))} ${GHS_YEARS[GHS_YEARS.length - 1]}
             <em>proj.</em></span></div></div>`
    : '';

  // A handful of member names is the line that makes "43M" explain itself.
  const mem = (g.members && g.members.length > 1)
    ? `<div class="umem">${g.members.slice(0, 6).map(esc).join(' &middot; ')}${
        g.nMembers > 6 ? ` &middot; +${g.nMembers - 6}` : ''}</div>`
    : '';

  return `<div class="sect">${head}</div>
    ${named}
    <dl class="uc">
      <dt>Contains</dt><dd>${fmt(g.pop)}</dd>
      <dt>Area</dt><dd>${fmt(g.area)} km&sup2;</dd>
      ${dens ? `<dt>Density</dt><dd>${fmt(dens)} / km&sup2;</dd>` : ''}
      ${g.basin ? `<dt>River basin</dt><dd class="txt">${esc(g.basin)}</dd>` : ''}
    </dl>
    ${hist}${mem}`;
}

export function render(c, key) {
  if (!c) return '';
  const alt = (c.altNames || []).slice(0, 3);
  const facts = (c.facts || []).slice(0, 3);
  // Curated languages if present, otherwise the country-level seed — shown
  // differently, because "we know" and "we guessed from the country" must not
  // look the same on a reference card.
  const langs = c.languages || [];
  const seeds = (!langs.length && c.languageCandidates) ? c.languageCandidates : [];

  const pop = c.pop == null ? none
    : `<span style="color:${rgb(popColor(c.pop))}">${fmt(c.pop)}</span>`;
  // A borrowed per-blob average must never look like a surveyed figure, so it
  // is marked. `elevConflict` is the other direction: both sources have a value
  // and they disagree by more than 500 m, which means one of them is wrong.
  const elev = c.elev == null ? none
    : `<span style="color:${rgb(elevColor(c.elev))}">${fmt(c.elev)} m</span>` +
      (c.elevSrc === 'ghs' ? ' <em class="src">centre avg</em>' : '') +
      (c.elevConflict != null
        ? ` <em class="src warn-i" title="GHS urban centre average">&ne; ${fmt(c.elevConflict)}</em>` : '');
  const gdp = c.gdpPc == null ? none
    : `<span style="color:${rgb(gdpColor(c.gdpPc))}">$${fmt(c.gdpPc)}</span>`;

  const stale = (c._stale || []).length
    ? `<div class="warn">source changed under your edit:
       ${c._stale.map(esc).join(', ')}</div>` : '';
  // `low` means two different things and the copy has to distinguish them, or
  // a correctly-flagged fragment reads like a broken match.
  const unsure = c.ghsConf !== 'low' ? ''
    : c.ghsRole === 'centre'
      ? `<div class="warn">urban centre named for this city, but its population
         is far apart &mdash; check the match</div>`
      : `<div class="warn">not confidently matched</div>`;

  return `
    <h2>${esc(c.name || key)}</h2>
    ${alt.length ? `<div class="alt">${alt.map(esc).join(' &middot; ')}</div>` : ''}
    <div class="sub">${esc(subtitle(c))}</div>
    ${(c.typeNames || []).length ? `<div class="kinds">${
      (c.kind && c.kind !== 'city')
        ? `<span class="kindtag ${esc(c.kind)}">${c.kind === 'aggregate' ? 'metro area' : 'admin area'}</span>` : ''
      }${(c.typeNames || []).map(t => `<span class="ktype">${esc(t)}</span>`).join('')}</div>` : ''}
    ${stale}${unsure}
    <dl>
      <dt>Population</dt><dd>${pop}</dd>
      <dt>GDP / capita</dt><dd>${gdp}</dd>
      <dt>Altitude</dt><dd>${elev}</dd>
      <dt>Coordinates</dt><dd>${coord(c.lat, c.lon)}</dd>
    </dl>
    ${langs.length ? `<div class="sect">Languages</div>
      <div class="chips">${langs.slice(0, 6).map(l =>
        `<span class="chip">${swatch(languageColor(l.name || l), l.name || l)}</span>`
      ).join('')}</div>` : ''}
    ${seeds.length ? `<div class="sect">Languages <em class="seed">country-level</em></div>
      <div class="chips">${seeds.slice(0, 5).map(l =>
        `<span class="chip dim">${swatch(languageColor(l), l)}</span>`
      ).join('')}</div>` : ''}
    ${climate(c)}
    ${urbanCentre(c)}
    ${facts.length ? `<div class="sect">Facts</div>
      <ul class="facts">${facts.map(f => `<li>${esc(f)}</li>`).join('')}</ul>` : ''}
    <div class="qid">${esc(key)}${c._touched ? ' &middot; edited' : ''}</div>`;
}

export function place(el, x, y) {
  const w = el.offsetWidth || 300, h = el.offsetHeight || 200;
  el.style.left = Math.max(8, Math.min(x + 16, innerWidth - w - 10)) + 'px';
  el.style.top = Math.max(8, Math.min(y + 16, innerHeight - h - 10)) + 'px';
}
