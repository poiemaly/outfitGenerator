import os
import shutil
import random
import pandas as pd

# Load dataset paths
DATA_ROOT = r"E:\school\vscode_projects\outfitGenerator\Deepfashion_dataset"

LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
IMG_ROOT = DATA_ROOT

# Images per bottom type
MAX_PER_BOTTOM = 20

# Point CSV category name to new folder name
BOTTOM_CATEGORY_MAP = {
    "Jeans": "jeans",
    "Shorts": "shorts",
    "Skirt": "skirt",
    "Joggers": "joggers",
    "Dress": "dress", 
}

GALLERY_ROOT = "bottom_gallery"


def main():

    os.makedirs(GALLERY_ROOT, exist_ok=True)
    df = pd.read_csv(LABELS_CSV)

    # Keep categories included in map
    df_bottoms = df[df["category_name"].isin(BOTTOM_CATEGORY_MAP.keys())]

    print("Found bottoms rows:", len(df_bottoms))

    for csv_cat, simple_label in BOTTOM_CATEGORY_MAP.items():
        subset = df_bottoms[df_bottoms["category_name"] == csv_cat]

        if subset.empty:
            print(f"[WARN] No rows found for bottom category '{csv_cat}'")
            continue

        n = min(MAX_PER_BOTTOM, len(subset))
        subset = subset.sample(n=n, random_state=42)

        out_dir = os.path.join(GALLERY_ROOT, simple_label)
        os.makedirs(out_dir, exist_ok=True)

        print(f"Creating gallery for {csv_cat} -> {simple_label} ({n} images)")

        for _, row in subset.iterrows():
            rel_path = row["image_name"].replace("\\", "/")
            src_path = os.path.join(IMG_ROOT, rel_path)

            if not os.path.exists(src_path):
                continue

            dst_path = os.path.join(out_dir, os.path.basename(src_path))

            # Avoid overwriting if it already exists
            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)

    print("Done. bottom_gallery created")

if __name__ == "__main__":
    main()
