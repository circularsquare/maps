# Global Crop Dot Map

Interactive dot map of global crop production, where each dot represents 10,000 hectares of harvested area. Built with MapLibre GL.

## Data

Uses the [CROPGRIDS](https://doi.org/10.6084/m9.figshare.22491997) dataset (Tang et al., 2024) — a global georeferenced dataset of 173 crops at 0.05° resolution (~5.6 km), year 2020. Published in *Scientific Data*.

167 crops produce dots at the 10k ha threshold, totaling ~145k dots across 11 groups (Cereals, Oilseeds, Pulses, Roots & Tubers, Fruits, Vegetables, Nuts, Stimulants & Spices, Sugar, Fibre, Fodder).

## Features

- Dots colored by crop group (HSL spread within each group)
- Collapsible legend with group-level and individual crop toggles
- Tooltip on hover showing crop name
- Dot size slider
- ~1.4 MB binary payload

## Data pipeline

1. Download CROPGRIDS NetCDF files into `data/cropgrids/`
2. Unzip `CROPGRIDSv1.08_NC_maps.zip`
3. Run `python make_cropgrids_dots.py`
   - Reads each `.nc` file's `harvarea` grid
   - Merges fodder variants into parent crops (e.g. maize + maize fodder)
   - Walks a Hilbert curve for spatial coherence, accumulates area, drops a dot every 10k ha
   - Outputs `data/all_dots.bin` (10 bytes/dot: float32 lon, float32 lat, uint16 crop_id) and `data/legend.json`

Requires `netCDF4` (`pip install netCDF4`). Processing takes ~5 minutes for all 173 crops.

## Serving

Open `index.html` with any local server (e.g. `python -m http.server`). No build step needed.
