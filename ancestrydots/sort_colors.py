#!/usr/bin/env python3
"""Sort ancestry_colors.csv by [hue, sat, lit, population, label]."""
import re
from pathlib import Path

CSV = Path(__file__).parent / 'ancestry_colors.csv'

LINE_RE = re.compile(
    r'^\s*(?:"[^"]*"|[^,]+?)\s*,\s*\S+\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d,]+)\s*$'
)
LABEL_RE = re.compile(r'^\s*(?:"([^"]*)"|(.*?))\s*,')

def sort_key(line):
    m = LINE_RE.match(line)
    if not m:
        return (float('inf'),) * 5
    hue, sat, lit, pop = m.groups()
    lm = LABEL_RE.match(line)
    label = (lm.group(1) or lm.group(2)).strip() if lm else ''
    return float(hue), float(sat), float(lit), int(pop.replace(',', '')), label.lower()

def main():
    lines = CSV.read_text().splitlines()
    header = lines[0]
    data = [l for l in lines[1:] if l.strip()]
    data.sort(key=sort_key)
    CSV.write_text('\n'.join([header] + data) + '\n')
    print(f'Sorted {len(data)} rows.')

if __name__ == '__main__':
    main()
