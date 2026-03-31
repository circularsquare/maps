import requests
import re
import pandas as pd
import time

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

def resolve_by_cid(url):
    try:
        # 1. Get the long URL if it's a shortener
        r1 = session.get(url, allow_redirects=True, timeout=10)
        long_url = r1.url
        
        # 2. Extract the Hex CID (the part after the colon in the !1s0x... block)
        # e.g., ...!1s0x89c2f60b81f16a4b:0xd5134d0ea6764ac7...
        cid_match = re.search(r'0x[0-9a-f]+:(0x[0-9a-f]+)', long_url)
        
        if cid_match:
            hex_cid = cid_match.group(1)
            # Convert hex to signed 64-bit decimal string
            dec_cid = str(int(hex_cid, 16))
            
            # 3. Hit the CID redirector
            cid_url = f"https://www.google.com/maps?cid={dec_cid}"
            r2 = session.get(cid_url, allow_redirects=True, timeout=10)
            final_url = r2.url
            
            # 4. Extract coordinates from the final landing URL
            coord_match = re.search(r'@([-]?\d+\.\d+),([-]?\d+\.\d+)', final_url)
            if coord_match:
                return float(coord_match.group(1)), float(coord_match.group(2)), "CID Match"
        
        # Fallback: check if @lat,lon is already in the long_url
        coord_match = re.search(r'@([-]?\d+\.\d+),([-]?\d+\.\d+)', long_url)
        if coord_match:
            return float(coord_match.group(1)), float(coord_match.group(2)), "Direct URL"

    except Exception as e:
        return None, None, f"Error: {str(e)[:15]}"
    
    return None, None, "No ID Found"

# Run the 10-row test
df = pd.read_csv('favorites.csv').dropna(subset=['URL']).head(10).copy()
print(f"{'Index':<6} | {'Status':<12} | {'Lat':<10} | {'Lon':<10} | {'Method'}")
print("-" * 75)

for i, row in df.iterrows():
    lat, lon, method = resolve_by_cid(row['URL'])
    # Success check: Coordinates should NOT be exactly 40.6978 / -73.9246
    is_nyc_default = (lat == 40.69785 and lon == -73.9246)
    status = "✅ Unique" if lat and not is_nyc_default else "❌ Failed/Center"
    print(f"{i:<6} | {status:<12} | {str(lat)[:8]:<10} | {str(lon)[:8]:<10} | {method}")
    time.sleep(0.5)