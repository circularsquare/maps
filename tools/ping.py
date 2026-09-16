"""Play a quiet chime to get Anita's attention: a hand-back, or a question waiting in ask/OPEN.md.

    python C:/Users/anita/projects/maps/tools/ping.py

Anita, 2026-09-14: "when you try to contact me you can make a ping sound ... not too loud though."
It uses a quieter copy of shonei's Tools/attention.wav, written once next to this script as
ping.wav. Play it once per hand-back, not per message.
"""

import array
import sys
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUIET = HERE / "ping.wav"
SOURCE = Path(r"C:\Users\anita\projects\shonei\Tools\attention.wav")
GAIN = 0.35  # "not too loud"


def make_quiet():
    with wave.open(str(SOURCE), "rb") as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    if params.sampwidth != 2:
        raise ValueError(f"expected 16-bit PCM, got {params.sampwidth * 8}-bit")
    samples = array.array("h", frames)
    if sys.byteorder != "little":
        samples.byteswap()
    quiet = array.array("h", (int(s * GAIN) for s in samples))
    if sys.byteorder != "little":
        quiet.byteswap()
    with wave.open(str(QUIET), "wb") as w:
        w.setparams(params)
        w.writeframes(quiet.tobytes())


def main():
    try:
        import winsound
    except ImportError:
        print("[ping] winsound unavailable (not Windows)", file=sys.stderr)
        return 1
    if not QUIET.exists():
        try:
            make_quiet()
        except (OSError, ValueError, wave.Error) as e:
            print(f"[ping] could not make ping.wav ({e}); using the system beep", file=sys.stderr)
            winsound.MessageBeep()
            return 0
    winsound.PlaySound(str(QUIET), winsound.SND_FILENAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
