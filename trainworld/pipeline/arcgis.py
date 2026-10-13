"""One cached ArcGIS REST feature query, paged (T-277: Canada's census geography, whose StatCan
downloads sit behind a bot check). Pages are kept in `data/raw/<name>_<offset>.json` as GeoJSON in
lon/lat; delete them to refetch."""
import json
import os

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
UA = {"User-Agent": "trainworld-pipeline/0.1 (rail game data build; python-requests)"}


def features(name, layer_url, where="1=1", fields="*", geometry=True, bbox=None, page=1000):
    """Every feature of `layer_url` matching `where` (and meeting bbox = (w, s, e, n) lon/lat),
    as GeoJSON feature dicts."""
    out, offset = [], 0
    while True:
        path = os.path.join(RAW, f"{name}_{offset}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        else:
            q = {"where": where, "outFields": fields, "returnGeometry": "true" if geometry else "false",
                 "outSR": 4326, "f": "geojson", "resultOffset": offset, "resultRecordCount": page,
                 "orderByFields": next((k for k in ("OBJECTID", "FID") if k in fields.split(",")), "")}
            if bbox is not None:
                q.update(geometry=",".join(str(v) for v in bbox), geometryType="esriGeometryEnvelope",
                         inSR=4326, spatialRel="esriSpatialRelIntersects")
            r = requests.post(layer_url + "/query", data=q, headers=UA, timeout=300)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(f"{name}: {data['error']}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".part", "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(path + ".part", path)
            print(f"  {name}: {offset + len(data.get('features', []))} features", flush=True)
        feats = data.get("features", [])
        out += feats
        more = data.get("exceededTransferLimit") or data.get("properties", {}).get("exceededTransferLimit")
        if not feats or (not more and len(feats) < page):
            return out
        offset += len(feats)
