from pathlib import Path
from typing import Callable, List, Optional
from .base import Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class CustomDataset(Dataset):
    """
    Dataset propio: carpeta con imágenes del usuario.
    Divide automáticamente 80% DB / 20% query.
    Si la carpeta tiene subcarpetas 'db/' y 'query/', las usa directamente.
    """

    def __init__(self, folder: Path):
        self._folder = Path(folder)

    @property
    def name(self) -> str:
        return f"custom_{self._folder.name}"

    def download(self, log_fn: Callable[[str], None] = print) -> None:
        if not self._folder.exists():
            raise FileNotFoundError(f"La carpeta no existe: {self._folder}")
        count = len(self._all_images())
        log_fn(f"[Custom] Carpeta: {self._folder}")
        log_fn(f"[Custom] {count} imágenes encontradas.")
        if count < 10:
            log_fn("  Advertencia: menos de 10 imágenes puede dar una reconstrucción pobre.")

    def get_images_dir(self) -> Path:
        return self._folder

    def _all_images(self) -> List[str]:
        return sorted(
            f.name for f in self._folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )

    def get_db_images(self) -> List[str]:
        if (self._folder / "db").exists():
            return sorted(
                f.name for f in (self._folder / "db").iterdir()
                if f.suffix.lower() in IMAGE_EXTS
            )
        imgs = self._all_images()
        split = max(1, int(len(imgs) * 0.8))
        return imgs[:split]

    def get_query_images(self) -> List[str]:
        if (self._folder / "query").exists():
            return sorted(
                f.name for f in (self._folder / "query").iterdir()
                if f.suffix.lower() in IMAGE_EXTS
            )
        imgs = self._all_images()
        split = max(1, int(len(imgs) * 0.8))
        return imgs[split:]

    def get_queries_txt(self) -> Optional[Path]:
        return None
