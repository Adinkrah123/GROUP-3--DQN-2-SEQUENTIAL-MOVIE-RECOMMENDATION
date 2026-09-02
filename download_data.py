"""
Downloads the MovieLens Latest-Small dataset from the official GroupLens
source into ./data/. Run this once before training/evaluation.

    python download_data.py
"""

import os
import zipfile
import urllib.request

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = "data"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "ml-latest-small.zip")

    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, zip_path)

    print("Extracting movies.csv and ratings.csv ...")
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith("movies.csv") or name.endswith("ratings.csv"):
                target_name = os.path.basename(name)
                with z.open(name) as src, open(os.path.join(DATA_DIR, target_name), "wb") as dst:
                    dst.write(src.read())

    os.remove(zip_path)
    print(f"Done. Files saved to {DATA_DIR}/movies.csv and {DATA_DIR}/ratings.csv")


if __name__ == "__main__":
    main()
