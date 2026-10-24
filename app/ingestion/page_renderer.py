import fitz  # PyMuPDF
from PIL import Image


def render_page(page: "fitz.Page", dpi: int = 150) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def resize_for_model(img: Image.Image, max_side: int = 1568) -> Image.Image:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    return img


def crop_region(img: Image.Image, bbox: list[float], pad: float = 0.01) -> Image.Image:
    """Crop using a normalized [x0, y0, x1, y1] box, with a little padding."""
    w, h = img.size
    x0, y0, x1, y1 = bbox
    box = (max(0, (x0 - pad) * w), max(0, (y0 - pad) * h),
           min(w, (x1 + pad) * w), min(h, (y1 + pad) * h))
    return img.crop(tuple(int(v) for v in box))