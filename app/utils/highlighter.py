from PIL import Image, ImageDraw

COLORS = {"table": "#e63946", "chart": "#2a9d8f", "diagram": "#f4a261", "figure": "#8e44ad"}


def draw_regions(img: Image.Image, regions: list[dict], width: int = 4) -> Image.Image:
    """Draw normalized (0-1) bounding boxes on a copy of the page image."""
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for r in regions:
        x0, y0, x1, y1 = r["bbox"]
        draw.rectangle([x0 * w, y0 * h, x1 * w, y1 * h],
                       outline=COLORS.get(r.get("type"), "#e63946"), width=width)
    return img