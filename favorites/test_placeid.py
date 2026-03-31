import requests
import re
import os
import struct
import base64

API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY', 'AIzaSyCiCzcGcB58rZ-OwK4GSvUiokMGyUjKxcE')

test_url = 'https://www.google.com/maps/place/Ocean+Avenue+Pedestrian+Bridge/data=!4m2!3m1!1s0x89c24465689496b9:0xcba0063c7138d9e2'

def hex_to_place_id(hex_str):
    # Convert "0xAAAA:0xBBBB" hex pair to standard base64url Place ID
    parts = hex_str.split(':')
    cell_id = int(parts[0], 16)
    cid = int(parts[1], 16)
    # Build protobuf: outer field 1 (bytes) contains inner message with
    # field 1 (fixed64) = cell_id and field 3 (fixed64) = cid
    inner = struct.pack('<BQ', 0x09, cell_id) + struct.pack('<BQ', 0x11, cid)
    outer = bytes([0x0A, len(inner)]) + inner
    return base64.urlsafe_b64encode(outer).rstrip(b'=').decode()

def extract_hex_id(url):
    match = re.search(r'!1s([^!]+)', url)
    return match.group(1) if match else None

hex_id = extract_hex_id(test_url)
place_id = hex_to_place_id(hex_id)
print(f"Hex ID:      {hex_id}")
print(f"Place ID:    {place_id}")

resp = requests.get(
    'https://maps.googleapis.com/maps/api/place/details/json',
    params={'place_id': place_id, 'fields': 'name,geometry', 'key': API_KEY}
)
data = resp.json()
print(f"Status: {data['status']}")
if data.get('result'):
    loc = data['result']['geometry']['location']
    name = data['result'].get('name', '')
    print(f"Name:   {name}")
    print(f"Coords: {loc['lat']}, {loc['lng']}")
else:
    print(f"Response: {data}")
