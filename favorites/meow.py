import pandas as pd
import requests
import re
import os
import struct
import base64
import time

API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY', 'AIzaSyCiCzcGcB58rZ-OwK4GSvUiokMGyUjKxcE')
INPUT_FILE = 'favorites.csv'
OUTPUT_FILE = 'coords.csv'

def hex_to_place_id(hex_str):
    parts = hex_str.split(':')
    cell_id = int(parts[0], 16)
    cid = int(parts[1], 16)
    inner = struct.pack('<BQ', 0x09, cell_id) + struct.pack('<BQ', 0x11, cid)
    outer = bytes([0x0A, len(inner)]) + inner
    return base64.urlsafe_b64encode(outer).rstrip(b'=').decode()

def extract_hex_id(url):
    match = re.search(r'!1s([^!]+)', str(url))
    return match.group(1) if match else None

def lookup_place(place_id):
    resp = requests.get(
        'https://maps.googleapis.com/maps/api/place/details/json',
        params={'place_id': place_id, 'fields': 'geometry', 'key': API_KEY},
        timeout=10
    )
    result = resp.json().get('result', {})
    loc = result.get('geometry', {}).get('location', {})
    return loc.get('lat'), loc.get('lng')

# Load input
df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=['URL'])

# Load already-completed rows if output file exists
done = set()
if os.path.exists(OUTPUT_FILE):
    existing = pd.read_csv(OUTPUT_FILE)
    done = set(existing['URL'].dropna())
    print(f"Resuming — {len(done)} already done, {len(df) - len(done)} remaining.")
else:
    # Write header
    pd.DataFrame(columns=['Title', 'URL', 'lat', 'lon']).to_csv(OUTPUT_FILE, index=False)

total = len(df)
for i, row in df.iterrows():
    url = str(row['URL']).strip()
    title = row.get('Title', '')

    if url in done:
        continue

    hex_id = extract_hex_id(url)
    if not hex_id:
        print(f"[{i+1}/{total}] SKIP (no place ID): {title}")
        lat, lon = None, None
    else:
        place_id = hex_to_place_id(hex_id)
        lat, lon = lookup_place(place_id)
        print(f"[{i+1}/{total}] {title}: {lat}, {lon}")

    # Append this row immediately so progress is saved
    pd.DataFrame([{'Title': title, 'URL': url, 'lat': lat, 'lon': lon}]).to_csv(
        OUTPUT_FILE, mode='a', header=False, index=False
    )
    time.sleep(0.2)

result = pd.read_csv(OUTPUT_FILE)
found = result.dropna(subset=['lat', 'lon'])
print(f"\nDone. {len(found)}/{len(result)} places have coordinates.")
