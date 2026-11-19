// T-3738: a shared network's link preview. Link previewers (Discord, X, iMessage, Slack, WhatsApp) read the game's page
// without running it, so every `?share=` link showed the game's banner. This Worker sits on the page's route and, for a
// `?share=<id>` load only, puts that share's picture, title and counts in the page's og: and twitter: tags. Everything
// else (no share, any other file, the server slow or saying no) gets the page exactly as GitHub Pages sent it.
// The picture is painted by shonei-server (trainworld_community_preview.go); the card is its public
// GET /trainworld/community/<id>. Notes and deploy steps: trainworld/notes/T-3738.md.

const SHARE_ID = /^[a-z2-9]{8}$/;
/** the game's page, as the route sees it */
const PAGES = new Set(["/trainsworld/", "/trainsworld/index.html"]);
/** how long a previewer waits on us for the server's card before the page goes out unchanged */
const CARD_TIMEOUT_MS = 1500;
/** the card is cached at the edge this long; a share stopped meanwhile still 404s its picture */
const CARD_CACHE_S = 300;

/** 1756 -> "1,756" */
const count = (n) => String(Math.max(0, Math.floor(Number(n) || 0))).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
const plural = (n, one, many) => `${count(n)} ${Number(n) === 1 ? one : many}`;

/** the tags' text for a share's card: its title, else "<name>'s network"; the counts, and who made it when named */
export function tagsFor(card) {
  const name = typeof card.name === "string" ? card.name.trim() : "";
  const title = (typeof card.title === "string" && card.title.trim()) || (name ? `${name}'s network` : "");
  const counts = `${plural(card.lines, "line", "lines")}, ${plural(card.stations, "station", "stations")}`;
  return { title, description: name ? `${counts}, by ${name}` : counts };
}

async function getCard(api, id) {
  const r = await fetch(`${api}/trainworld/community/${id}`, {
    headers: { accept: "application/json" },
    signal: AbortSignal.timeout(CARD_TIMEOUT_MS),
    cf: { cacheTtlByStatus: { "200-299": CARD_CACHE_S, "404": 60, "500-599": 0 }, cacheEverything: true },
  });
  if (!r.ok) return null;
  const card = await r.json();
  return card && card.id === id && Number.isInteger(card.rev) && card.rev > 0 ? card : null;
}

function rewrite(page, card, id, api, site) {
  const { title, description } = tagsFor(card);
  const image = `${api}/trainworld/community/${id}/${card.rev}/preview.png`;
  const set = (value) => ({ element(e) { e.setAttribute("content", value); } });
  let r = new HTMLRewriter()
    .on('meta[property="og:image"]', set(image))
    .on('meta[name="twitter:image"]', set(image))
    .on('meta[property="og:description"]', set(description))
    .on('meta[name="twitter:description"]', set(description))
    .on('meta[property="og:url"]', set(`${site}/trainsworld/?share=${id}`));
  // no title and no name: the game's own title stays
  if (title) r = r.on('meta[property="og:title"]', set(title)).on('meta[name="twitter:title"]', set(title));
  return r.transform(page);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const id = url.searchParams.get("share") ?? "";
    if (!PAGES.has(url.pathname) || !SHARE_ID.test(id) || (request.method !== "GET" && request.method !== "HEAD")) return fetch(request);
    const api = (env.SHARE_API || "https://market.anita.garden").replace(/\/$/, "");
    const site = (env.SITE || "https://anita.garden").replace(/\/$/, "");
    const pending = fetch(request);
    let card = null;
    try { card = await getCard(api, id); } catch { /* slow, down or not JSON: the page as it is */ }
    const page = await pending;
    if (!card || !page.ok || !(page.headers.get("content-type") ?? "").includes("text/html")) return page;
    try { return rewrite(page, card, id, api, site); } catch { return page; }
  },
};
