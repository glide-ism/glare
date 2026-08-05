"""Download gridded model inputs for the Wrangell example."""

import os
import shutil
import urllib.request

BASE_URL = "https://examples.glide-ism.org/wrangells"
DEST_DIR = os.path.join(os.path.dirname(__file__), "model_inputs")

FILES = [
    "gridded_dem.nc",
    "gridded_climate.nc",
]


def download(name):
    url = f"{BASE_URL}/{name}"
    dest = os.path.join(DEST_DIR, name)
    print(f"Downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Wget/1.21"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    for name in FILES:
        download(name)
    print("Done.")


if __name__ == "__main__":
    main()
