import os
import json
import glob
import gc
import resource
import faulthandler
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from processing import process_tile_file  # new helper that reads an entire tile LAS

# === DEBUG: crash on OOM with traceback ===
faulthandler.enable()
MEM_LIMIT = 0  # no extra limit here
# =====================

# === CONFIG ===
TILE_DIR           = "tiles/1"
OUTPUT_FILE        = "output/geojson/buildings.geojson"
MIN_BUILDING_AREA  = 20      # m²
HEIGHT_THRESHOLD   = 2.0     # meters above ground
WORKERS            = max(1, cpu_count() - 2)
# ===============

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    # Collect tile files
    tiles = sorted(glob.glob(os.path.join(TILE_DIR, "*.las")))
    print(f"Found {len(tiles)} tile files in '{TILE_DIR}'.")

    with open(OUTPUT_FILE, "w") as out:
        out.write('{"type":"FeatureCollection","features":[')
        first = True

        with Pool(WORKERS, maxtasksperchild=1) as pool:
            for feats in tqdm(
                pool.imap_unordered(process_tile_file, tiles),
                total=len(tiles),
                desc="Processing tiles"
            ):
                for feat in feats:
                    if not first: out.write(",")
                    out.write(json.dumps(feat))
                    first = False
                gc.collect()

        out.write("]}")

    print(f"✅ Done: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()