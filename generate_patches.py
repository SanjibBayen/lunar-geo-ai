import os
import numpy as np
import rasterio
import rasterio.windows
import cv2
from tqdm import tqdm

PATCH_SIZE = 32
STRIDE = 16
CONF_THRESH = 0.8

stack_path = "data/stack.tif"
out_dir = "patches"

os.makedirs(f"{out_dir}/boulders", exist_ok=True)
os.makedirs(f"{out_dir}/non_boulders", exist_ok=True)
os.makedirs(f"{out_dir}/landslides", exist_ok=True)
os.makedirs(f"{out_dir}/non_landslides", exist_ok=True)

with rasterio.open(stack_path) as src:
    width, height = src.width, src.height
    max_slope = None
    max_curv = None
    min_curv = None

    # Precompute curvature stats from sampled tiles (OPTIONAL)
    sample = src.read(window=rasterio.windows.Window(0, 0, 512, 512))
    slope_sample = sample[1].astype(np.float32)
    curv_sample = sample[2].astype(np.float32)
    max_slope = np.max(slope_sample)
    max_curv = np.max(curv_sample)
    min_curv = np.min(curv_sample)

    id_counter = 0
    for y in tqdm(range(0, height - PATCH_SIZE, STRIDE)):
        for x in range(0, width - PATCH_SIZE, STRIDE):
            window = rasterio.windows.Window(x, y, PATCH_SIZE, PATCH_SIZE)
            patch = src.read([1, 2, 3], window=window).astype(np.float32)  # shape: (3, 32, 32)
            patch_tmc, patch_slope, patch_curv = patch[0], patch[1], patch[2]

            if np.isnan(patch_tmc).any() or np.std(patch_tmc) < 5:
                continue

            # --- Weak label for boulder ---
            norm = (patch_tmc - np.percentile(patch_tmc, 2)) / (np.percentile(patch_tmc, 98) - np.percentile(patch_tmc, 2))
            norm = np.clip(norm, 0, 1)
            bright = cv2.threshold((norm * 255).astype(np.uint8), 230, 255, cv2.THRESH_BINARY)[1]
            bright_area = np.sum(bright > 0)

            if bright_area >= 3:
                cv2.imwrite(f"{out_dir}/boulders/patch_{id_counter}.png", patch_tmc.astype(np.uint8))
            else:
                cv2.imwrite(f"{out_dir}/non_boulders/patch_{id_counter}.png", patch_tmc.astype(np.uint8))

            # --- Weak label for landslide ---
            slope_score = np.mean(patch_slope) / max(max_slope, 1e-5)
            curv_score = (np.mean(patch_curv) - min_curv) / max((max_curv - min_curv), 1e-5)
            landslide_conf = 0.5 * slope_score + 0.5 * curv_score

            patch_stack = np.stack([patch_tmc, patch_slope, patch_curv], axis=0)  # shape (3,32,32)
            save_path = f"{out_dir}/landslides/patch_{id_counter}.npy" if landslide_conf > CONF_THRESH else f"{out_dir}/non_landslides/patch_{id_counter}.npy"
            np.save(save_path, patch_stack)

            id_counter += 1
