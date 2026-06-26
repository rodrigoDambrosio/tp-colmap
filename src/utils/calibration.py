"""Camera calibration utilities — EXIF-based and manual intrinsics."""
from pathlib import Path
from typing import Optional, Dict, Any


def read_exif_calibration(image_path: Path) -> Optional[Dict[str, Any]]:
    """
    Estimate SIMPLE_RADIAL intrinsics from EXIF embedded in a JPEG/HEIC.
    Returns None if EXIF is missing or focal length cannot be determined.

    Focal length in pixels is derived from FocalLengthIn35mmFilm (most
    reliable tag on modern phones) using the 36 mm full-frame reference:
        f_px = f_35mm * image_width / 36
    """
    try:
        from PIL import Image

        img = Image.open(image_path)
        width, height = img.size

        exif_raw = img._getexif()
        if exif_raw is None:
            return None

        from PIL.ExifTags import TAGS
        exif = {TAGS.get(k, k): v for k, v in exif_raw.items()}

        focal_px: Optional[float] = None

        # Best source: 35 mm film equivalent focal length
        f35 = exif.get("FocalLengthIn35mmFilm")
        if f35 and float(f35) > 0:
            focal_px = float(f35) * width / 36.0

        # Fallback: FocalLength (mm) — needs sensor width which we approximate
        if focal_px is None:
            fl = exif.get("FocalLength")
            if fl is not None:
                fl_mm = float(fl.numerator) / float(fl.denominator) if hasattr(fl, "numerator") else float(fl)
                # Common phone sensor: 1/2.55"  ≈ 5.7 mm diagonal, ~4.8 mm wide
                # Use conservative 5 mm estimate — better than nothing
                if fl_mm > 0:
                    focal_px = fl_mm * width / 5.0

        if focal_px is None or focal_px <= 0:
            return None

        return {
            "model":  "SIMPLE_RADIAL",
            "width":  width,
            "height": height,
            "params": [round(focal_px, 2), width / 2.0, height / 2.0, 0.0],
            "source": "EXIF",
        }

    except Exception:
        return None


def make_manual_calibration(
    focal_px: float,
    width: int,
    height: int,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    k1: float = 0.0,
) -> Dict[str, Any]:
    return {
        "model":  "SIMPLE_RADIAL",
        "width":  width,
        "height": height,
        "params": [focal_px, cx if cx is not None else width / 2.0,
                   cy if cy is not None else height / 2.0, k1],
        "source": "manual",
    }


def calibration_to_image_options(cal: Dict[str, Any]) -> Dict[str, Any]:
    """Convert calibration dict to pycolmap ImageReaderOptions fields."""
    params_str = ",".join(f"{p:.4f}" for p in cal["params"])
    return {
        "camera_model":  cal["model"],
        "camera_params": params_str,
        "single_camera": True,
    }
