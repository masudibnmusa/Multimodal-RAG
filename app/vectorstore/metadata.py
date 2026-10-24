import json
from dataclasses import dataclass
from pathlib import Path

from app import config


@dataclass
class ElementMeta:
    doc_id: str
    page: int
    element_type: str            # text | table | chart | diagram | figure
    bbox: list[float] | None = None  # normalized [x0, y0, x1, y1]

    def to_chroma(self) -> dict:
        return {"doc_id": self.doc_id, "page": self.page,
                "element_type": self.element_type, "bbox": json.dumps(self.bbox)}

    @staticmethod
    def from_chroma(d: dict) -> dict:
        return {**d, "bbox": json.loads(d["bbox"]) if d.get("bbox") else None}


def page_image_path(doc_id: str, page: int) -> Path:
    return config.PAGE_IMG_DIR / doc_id / f"page_{page:04d}.png"


def parsed_path(doc_id: str, page: int) -> Path:
    return config.PARSED_DIR / doc_id / f"page_{page:04d}.json"