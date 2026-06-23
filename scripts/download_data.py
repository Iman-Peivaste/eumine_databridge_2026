"""
Download Bridge Dataset files from Google Drive into data/raw/.
Uses gdown (pip install gdown if needed).
Run: python scripts/download_data.py
"""

import subprocess
import sys
import os
from pathlib import Path

# Install gdown if not present
try:
    import gdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown

RAW = Path(__file__).parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# Google Drive folder ID
FOLDER_ID = "1CAF8_rymTdr-2PM9z-xy2RnSYJKi8XS2"

print(f"Downloading Bridge Dataset to {RAW} ...")
gdown.download_folder(
    id=FOLDER_ID,
    output=str(RAW),
    quiet=False,
    use_cookies=False,
)
print("Download complete.")
