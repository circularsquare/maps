// Name matching shared by the search box and its place worker (workers/placeSearch.worker.ts), with no
// imports, so the worker stays small. pipeline/place_index.py normalises the same way, step for step:
// a change here changes which shard a town is in, so rebuild the index with it (T-4004).

/** Letters NFD does not split into a base and an accent. */
const PLAIN: Record<string, string> = { ł: "l", ø: "o", đ: "d", ħ: "h", ı: "i", ß: "ss", æ: "ae", œ: "oe", þ: "th" };

/** Lower case, accents and punctuation dropped, spaces collapsed. */
export function normalise(s: string): string {
  return s
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[łøđħıßæœþ]/g, (ch) => PLAIN[ch])
    .replace(/&/g, " and ")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

/** How well `name` matches the query `q` (both normalised): 0 exact, 1 starts with it, 2 a word
 * starts with it, 3 contains it, 4 contains it ignoring spaces, 5 its words are clipped forms of
 * the name's (looseWords); -1 no match. */
export function matchScore(name: string, q: string): number {
  if (!q) return -1;
  if (name === q) return 0;
  if (name.startsWith(q)) return 1;
  if ((" " + name).includes(" " + q)) return 2;
  if (name.includes(q)) return 3;
  if (name.replace(/ /g, "").includes(q.replace(/ /g, ""))) return 4;
  if (looseWords(name, q)) return 5;
  return -1;
}

/** Equal place matches: bigger kinds first, then by population. */
export const PLACE_WEIGHT: Record<string, number> = {
  city: 1e8, town: 1e7, borough: 1e7, suburb: 5e6, quarter: 4e6, village: 3e6, neighbourhood: 2e6, hamlet: 0,
};

/** Nicknames made by clipping a word and adding an ending (T-177: "south philly" for South
 * Philadelphia): the query's words match consecutive words of the name, each either starting it
 * or sharing its first letters bar the query word's last two, at least four and not the whole
 * name word (so "southie" does not find every South Street). */
function looseWords(name: string, q: string): boolean {
  const qw = q.split(" ");
  if (!name.includes(qw[0].slice(0, 4))) return false; // the common case, cheaply
  const nw = name.split(" ");
  const like = (n: string, w: string) => {
    if (n.startsWith(w)) return true;
    let k = 0;
    while (k < n.length && k < w.length && n[k] === w[k]) k++;
    return k >= Math.max(4, w.length - 2) && k < n.length;
  };
  for (let i = 0; i + qw.length <= nw.length; i++) if (qw.every((w, j) => like(nw[i + j], w))) return true;
  return false;
}
