from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

DATA_DIR = Path("data/datasets")


class Dataset(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def download(self, log_fn: Callable[[str], None] = print) -> None:
        pass

    @abstractmethod
    def get_images_dir(self) -> Path:
        pass

    @abstractmethod
    def get_db_images(self) -> List[str]:
        """Returns image names relative to get_images_dir()."""
        pass

    @abstractmethod
    def get_query_images(self) -> List[str]:
        """Returns image names relative to get_images_dir()."""
        pass

    def get_queries_txt(self) -> Optional[Path]:
        """Path to a file with query poses (COLMAP format), if available."""
        return None

    def get_ground_truth(self) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """Returns {image_name: (translation_world, quaternion_w2c)} or None."""
        return None
