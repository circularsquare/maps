// The recorded building sounds (T-1474; sounds/ in app/public): fetched at start, decoded the moment the
// audio context opens (audioOut.ts), so a click can start one with nothing left to load or decode.

const FILES = { snap: "sounds/snap.wav", pop: "sounds/pop.wav" } as const;
export type SampleName = keyof typeof FILES;
const NAMES = Object.keys(FILES) as SampleName[];

const bytes: Partial<Record<SampleName, Promise<ArrayBuffer | null>>> = {};
const ready: Partial<Record<SampleName, { buf: AudioBuffer; peak: number }>> = {};

/** Start fetching the files (once); relative to the page, as the map's fonts are. */
export function fetchSamples() {
  for (const n of NAMES) {
    bytes[n] ??= fetch(new URL(FILES[n], document.baseURI).href).then((r) => (r.ok ? r.arrayBuffer() : null)).catch(() => null);
  }
}

/** Decode every file into an AudioBuffer of `ctx` and note its peak. */
export async function decodeSamples(ctx: BaseAudioContext) {
  fetchSamples();
  await Promise.all(NAMES.map(async (n) => {
    const raw = await bytes[n];
    if (!raw || ready[n]) return;
    try {
      const buf = await ctx.decodeAudioData(raw);
      let peak = 0;
      for (let c = 0; c < buf.numberOfChannels; c++) for (const v of buf.getChannelData(c)) peak = Math.max(peak, Math.abs(v));
      ready[n] = { buf, peak: peak || 1 };
    } catch {
      /* a file that will not decode stays silent */
    }
  }));
}

/** A decoded sample and its peak; null until decoded. */
export function sample(n: SampleName): { buf: AudioBuffer; peak: number } | null {
  return ready[n] ?? null;
}
