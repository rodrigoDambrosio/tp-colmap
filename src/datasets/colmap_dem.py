"""
COLMAP demo datasets from https://demuc.de/colmap/datasets/
Each scene ships with images + a pre-computed sparse reconstruction.
We re-use only the images and run our own hloc pipeline on them.
DB/query split: 80 % / 20 % (sorted order, reproducible).
"""
from pathlib import Path
from typing import Callable, List, Optional
import zipfile
import requests

from .base import Dataset, DATA_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

BASE = "https://github.com/colmap/colmap/releases/download/3.11.1"

# scene_key -> list of part URLs to download, in concatenation order
# Single-part: just the .zip. Multi-part: .z01, .z02, ..., .zip (last).
SCENES = {
    "south-building": {
        "label": "South Building",
        "parts": [f"{BASE}/south-building.zip"],
    },
    "gerrard-hall": {
        "label": "Gerrard Hall",
        "parts": [f"{BASE}/gerrard-hall.zip"],
    },
    "person-hall": {
        "label": "Person Hall",
        "parts": [
            f"{BASE}/person-hall.z01",
            f"{BASE}/person-hall.zip",
        ],
    },
    "graham-hall": {
        "label": "Graham Hall",
        "parts": [
            f"{BASE}/graham-hall.z01",
            f"{BASE}/graham-hall.z02",
            f"{BASE}/graham-hall.z03",
            f"{BASE}/graham-hall.z04",
            f"{BASE}/graham-hall.z05",
            f"{BASE}/graham-hall.zip",
        ],
    },
}


class ColmapDemDataset(Dataset):
    """COLMAP demo dataset (demuc.de). Images only — no GT poses."""

    def __init__(self, scene: str):
        if scene not in SCENES:
            raise ValueError(f"Escena desconocida: {scene}. Opciones: {list(SCENES)}")
        self._scene = scene
        self._meta  = SCENES[scene]
        self._root  = DATA_DIR / "colmap_dem" / scene

    @property
    def name(self) -> str:
        return f"colmap_{self._scene}"

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self, log_fn: Callable[[str], None] = print) -> None:
        if self._images_dir().exists() and any(self._images_dir().iterdir()):
            log_fn(f"[{self._meta['label']}] Dataset ya existe, saltando descarga.")
            return

        self._root.mkdir(parents=True, exist_ok=True)
        parts = self._meta["parts"]
        label = self._meta["label"]

        if len(parts) == 1:
            # Single-part: download directly as the zip
            zip_path = self._root.parent / f"{self._scene}.zip"
            log_fn(f"[{label}] Descargando...")
            self._download(parts[0], zip_path, log_fn)
            log_fn(f"[{label}] Extrayendo...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self._root.parent)
            zip_path.unlink(missing_ok=True)
        else:
            # Multi-part: download each part then concatenate z01+z02+...+zip
            part_paths = []
            for url in parts:
                fname = self._root.parent / Path(url).name
                log_fn(f"[{label}] Descargando parte: {Path(url).name}")
                self._download(url, fname, log_fn)
                part_paths.append(fname)

            combined = self._root.parent / f"{self._scene}_combined.zip"
            log_fn(f"[{label}] Combinando {len(part_paths)} partes...")
            with open(combined, "wb") as out:
                for p in part_paths:
                    out.write(p.read_bytes())
                    p.unlink()

            log_fn(f"[{label}] Extrayendo...")
            with zipfile.ZipFile(combined, "r") as zf:
                zf.extractall(self._root.parent)
            combined.unlink(missing_ok=True)

        log_fn(f"[{label}] Listo en {self._root}")

    def _download(self, url: str, dest: Path, log_fn: Callable) -> None:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total      = int(resp.headers.get("content-length", 0))
        downloaded = 0
        last_pct   = -1
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=131072):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    if pct != last_pct and pct % 10 == 0:
                        log_fn(f"  {pct}%  ({downloaded // 1_048_576} MB)")
                        last_pct = pct

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def _images_dir(self) -> Path:
        # COLMAP demo datasets extract to <scene>/images/
        return self._root / "images"

    def get_images_dir(self) -> Path:
        return self._images_dir()

    def _all_images(self) -> List[str]:
        d = self._images_dir()
        if not d.exists():
            return []
        return sorted(
            f.name for f in d.iterdir()
            if f.is_file() and f.suffix in IMAGE_EXTS
        )

    def get_db_images(self) -> List[str]:
        imgs  = self._all_images()
        split = max(1, int(len(imgs) * 0.8))
        return imgs[:split]

    def get_query_images(self) -> List[str]:
        imgs  = self._all_images()
        split = max(1, int(len(imgs) * 0.8))
        return imgs[split:]

    def get_ground_truth(self) -> Optional[dict]:
        return None
