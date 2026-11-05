from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import FixtureManifest, FixtureManifestEntry


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
IMAGES_DIR = FIXTURES_DIR / "images"

SCENES = {
    "desk_static": ("DESK", "Notebook, mug, and blue pen", (211, 188, 142)),
    "label_small_text": ("SOLUTION Y", "Use 12 mL before 18:30", (210, 219, 199)),
    "reach_keys": ("KEYS", "Hand approaching keys", (186, 174, 155)),
    "pickup_keys": ("KEYS", "Hand lifting keys", (182, 169, 148)),
    "putdown_keys": ("KEYS", "Keys placed on counter", (194, 181, 159)),
    "kitchen_scene": ("KITCHEN", "Counter, kettle, and keys", (177, 188, 170)),
    "hallway_scene": ("HALLWAY", "Coat rack and closed door", (177, 167, 150)),
    "keys_counter": ("COUNTER", "Keys beside the green bowl", (196, 183, 152)),
    "keys_pocket": ("POCKET", "Keys entering jacket pocket", (110, 123, 128)),
    "painting_detail": ("PAINTING", "Woman blowing pink bubblegum", (201, 145, 129)),
    "toothbrush": ("BATHROOM", "Toothbrush beside dental floss", (164, 205, 204)),
}


def _font(size: int):
    try:
        from PIL import ImageFont

        # Pillow's bundled font keeps fixture pixels independent of host fonts.
        return ImageFont.load_default(size=size)
    except ImportError as exc:
        raise RuntimeError("Install the eval extra: pip install -e '.[eval]'") from exc


def generate() -> FixtureManifest:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Install the eval extra: pip install -e '.[eval]'") from exc

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    title_font = _font(42)
    detail_font = _font(25)
    entries: list[FixtureManifestEntry] = []
    for fixture_id, (title, detail, background) in SCENES.items():
        image = Image.new("RGB", (640, 360), background)
        draw = ImageDraw.Draw(image)
        draw.rectangle((28, 28, 612, 332), outline=(42, 46, 43), width=4)
        draw.ellipse((474, 52, 570, 148), fill=(236, 222, 169), outline=(42, 46, 43))
        draw.line((76, 258, 564, 258), fill=(42, 46, 43), width=5)
        draw.text((62, 74), title, fill=(24, 28, 26), font=title_font)
        draw.text((62, 171), detail, fill=(24, 28, 26), font=detail_font)
        path = IMAGES_DIR / f"{fixture_id}.jpg"
        image.save(
            path,
            format="JPEG",
            quality=88,
            optimize=False,
            progressive=False,
            subsampling=0,
        )
        frame_bytes = path.read_bytes()
        entries.append(
            FixtureManifestEntry(
                id=fixture_id,
                path=f"images/{path.name}",
                sha256=hashlib.sha256(frame_bytes).hexdigest(),
                width=640,
                height=360,
                description=detail,
            )
        )
    manifest = FixtureManifest(
        version=1,
        generator="evals.perception_memory.generate_fixtures:v2-pillow-11.3.0",
        entries=entries,
    )
    (FIXTURES_DIR / "manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    manifest = generate()
    print(json.dumps({"fixtures": len(manifest.entries), "path": str(FIXTURES_DIR)}))


if __name__ == "__main__":
    main()
