"""Print only the verdict lines of a review (T-4420): the lines that carry PASS or FAIL as a word. Run from the maps root:

  venv\\Scripts\\python.exe trainworld/tools/verdict.py <review file> [...]      PASS and FAIL lines only
  venv\\Scripts\\python.exe trainworld/tools/verdict.py <review file> --full     the whole file
  ... | venv\\Scripts\\python.exe trainworld/tools/verdict.py -                 read stdin

With no PASS or FAIL line it says so and prints the first 10 lines, so a review in another shape is not hidden."""
import argparse
import re
import sys

VERDICT = re.compile(r"\b(PASS|FAIL)\b")
FALLBACK_LINES = 10


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="review files, or - for stdin")
    ap.add_argument("--full", action="store_true", help="print every line")
    a = ap.parse_args()
    for path in a.files:
        if path == "-":
            text = sys.stdin.buffer.read().decode("utf-8", errors="replace")
        else:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        lines = text.splitlines()
        if len(a.files) > 1:
            print("== " + path)
        if a.full:
            print("\n".join(lines))
            continue
        hits = [l for l in lines if VERDICT.search(l)]
        if hits:
            print("\n".join(hits))
            print(f"({len(hits)} verdict lines of {len(lines)}; --full shows all)")
        else:
            print(f"no PASS or FAIL line in {len(lines)} lines; first {FALLBACK_LINES}:")
            print("\n".join(lines[:FALLBACK_LINES]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
