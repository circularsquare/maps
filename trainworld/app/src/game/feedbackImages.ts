// T-3326: screenshots attached to feedback. Each is shrunk to at most MAX_EDGE px on its long edge and
// re-encoded (WebP, JPEG where the browser cannot encode WebP) before it is sent; shonei-server takes at
// most 4 of 1.5 MB each, as image/webp, image/jpeg or image/png.

export const MAX_IMAGES = 4;
const MAX_EDGE = 1920;
/** the server's per-image cap (MaxTwImageBytes) */
const MAX_BYTES = 1.5 * 1024 * 1024;
const QUALITIES = [0.85, 0.7, 0.55, 0.4];

export interface FeedbackImage { id: number; blob: Blob; url: string }

export class ImageTooBig extends Error {}
export class NotAnImage extends Error {}

let nextId = 1;

const toBlob = (c: HTMLCanvasElement, type: string, q: number) => new Promise<Blob | null>((res) => c.toBlob(res, type, q));

async function decode(file: Blob): Promise<{ src: CanvasImageSource; w: number; h: number; done: () => void }> {
  try {
    const b = await createImageBitmap(file);
    return { src: b, w: b.width, h: b.height, done: () => b.close() };
  } catch {
    const url = URL.createObjectURL(file);
    const img = new Image();
    try {
      await new Promise<void>((res, rej) => { img.onload = () => res(); img.onerror = () => rej(new NotAnImage()); img.src = url; });
    } catch (e) { URL.revokeObjectURL(url); throw e; }
    return { src: img, w: img.naturalWidth, h: img.naturalHeight, done: () => URL.revokeObjectURL(url) };
  }
}

/** Whether a file or clipboard item is one the form takes. GIF and the rest are left out. */
export const acceptedType = (type: string) => type === "image/png" || type === "image/jpeg" || type === "image/webp";

/** Scale to the long edge cap (never up) and encode under the server's size cap. Throws NotAnImage or ImageTooBig. */
export async function prepareImage(file: Blob): Promise<FeedbackImage> {
  if (!acceptedType(file.type)) throw new NotAnImage();
  const d = await decode(file);
  try {
    if (!d.w || !d.h) throw new NotAnImage();
    const k = Math.min(1, MAX_EDGE / Math.max(d.w, d.h));
    const c = document.createElement("canvas");
    c.width = Math.max(1, Math.round(d.w * k));
    c.height = Math.max(1, Math.round(d.h * k));
    const ctx = c.getContext("2d");
    if (!ctx) throw new NotAnImage();
    ctx.drawImage(d.src, 0, 0, c.width, c.height);
    let type = "image/webp";
    for (const q of QUALITIES) {
      let b = await toBlob(c, type, q);
      // a browser that cannot encode WebP answers with a PNG: use JPEG, on white (it has no transparency)
      if (b && b.type !== type) {
        type = "image/jpeg";
        ctx.globalCompositeOperation = "destination-over";
        ctx.fillStyle = "#fff";
        ctx.fillRect(0, 0, c.width, c.height);
        b = await toBlob(c, type, q);
      }
      if (!b) throw new NotAnImage();
      if (b.size <= MAX_BYTES) return { id: nextId++, blob: b, url: URL.createObjectURL(b) };
    }
    throw new ImageTooBig();
  } finally { d.done(); }
}

export const dropImage = (im: FeedbackImage) => URL.revokeObjectURL(im.url);

/** The body field for sendFeedback: base64 without the data-URL prefix. */
export function imageField(im: FeedbackImage): Promise<{ type: string; data: string }> {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res({ type: im.blob.type, data: String(r.result).replace(/^data:[^,]*,/, "") });
    r.onerror = () => rej(r.error);
    r.readAsDataURL(im.blob);
  });
}
