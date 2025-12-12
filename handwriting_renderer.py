import io
import math
import os
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from matplotlib import font_manager


DEFAULT_FONTS = [
    "Segoe Script",
    "Bradley Hand ITC",
    "Lucida Handwriting",
    "Comic Sans MS",
    "Brush Script MT",
    "Freestyle Script",
    "Segoe Print",
    "Comic Neue",
]


def _collect_font_catalog() -> Dict[str, str]:
    """Return a mapping {font_name_lower: path} for available system fonts."""
    catalog: Dict[str, str] = {}
    for path in font_manager.findSystemFonts(fontpaths=None, fontext="ttf"):
        try:
            name = font_manager.FontProperties(fname=path).get_name()
        except Exception:
            continue
        if not name:
            continue
        key = name.strip().lower()
        if key not in catalog:
            catalog[key] = path
    return catalog


@lru_cache(maxsize=1)
def _cached_font_catalog() -> Dict[str, str]:
    return _collect_font_catalog()


def _pil_font_from_name(name: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    catalog = _cached_font_catalog()
    if name:
        key = name.strip().lower()
        if key in catalog:
            return ImageFont.truetype(catalog[key], size=size)
    for fallback in DEFAULT_FONTS:
        key = fallback.lower()
        if key in catalog:
            return ImageFont.truetype(catalog[key], size=size)
    # Fall back to the default DejaVu font shipped with PIL
    return ImageFont.load_default()


@dataclass
class PaperStyle:
    name: str
    background_rgb: Tuple[int, int, int]
    line_rgb: Tuple[int, int, int] = (173, 173, 173)
    show_guides: bool = False


PAPER_PRESETS: Dict[str, PaperStyle] = {
    "plain": PaperStyle(name="plain", background_rgb=(252, 246, 232)),
    "ruled": PaperStyle(
        name="ruled",
        background_rgb=(255, 253, 247),
        line_rgb=(195, 213, 233),
        show_guides=True,
    ),
    "grid": PaperStyle(
        name="grid",
        background_rgb=(250, 248, 238),
        line_rgb=(210, 210, 210),
        show_guides=True,
    ),
}


@dataclass
class RenderConfig:
    width: int = 1400
    height: int = 900
    margin: int = 80
    font_name: Optional[str] = None
    font_size: int = 64
    ink_color: Tuple[int, int, int] = (32, 32, 32)
    paper_style: str = "plain"
    line_spacing: float = 1.35
    max_chars_per_line: Optional[int] = None
    jitter_px: float = 1.8
    jitter_prob: float = 0.6
    tilt_degrees: float = -2.5
    noise_strength: float = 0.08
    shadow_alpha: float = 0.08
    seed: Optional[int] = None
    custom_font_path: Optional[str] = None
    custom_background_path: Optional[str] = None

    def paper(self) -> PaperStyle:
        return PAPER_PRESETS.get(self.paper_style, PAPER_PRESETS["plain"])


class HandwritingRenderer:
    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        self._rng = random.Random(self.config.seed)

    def _reset_rng(self):
        self._rng = random.Random(self.config.seed)

    def render(self, text: str, **overrides) -> Image.Image:
        cfg = self._merged_config(overrides)
        self.config = cfg  # keep latest for download convenience
        self._reset_rng()

        if cfg.custom_background_path and os.path.exists(cfg.custom_background_path):
            try:
                img = Image.open(cfg.custom_background_path).convert("RGB")
                img = img.resize((cfg.width, cfg.height))
            except Exception:
                img = Image.new(
                    "RGB", (cfg.width, cfg.height), color=cfg.paper().background_rgb
                )
        else:
            img = Image.new(
                "RGB", (cfg.width, cfg.height), color=cfg.paper().background_rgb
            )
        
        draw = ImageDraw.Draw(img)

        if cfg.paper().show_guides and not cfg.custom_background_path:
            self._draw_guides(draw, cfg)

        if cfg.custom_font_path and os.path.exists(cfg.custom_font_path):
            try:
                font = ImageFont.truetype(cfg.custom_font_path, size=cfg.font_size)
            except Exception:
                font = _pil_font_from_name(cfg.font_name, cfg.font_size)
        else:
            font = _pil_font_from_name(cfg.font_name, cfg.font_size)

        self._draw_text(draw, font, text, cfg)

        if cfg.tilt_degrees:
            img = self._apply_tilt(img, cfg.tilt_degrees, cfg.paper().background_rgb)

        if cfg.shadow_alpha > 0:
            img = self._add_shadow(img, cfg.shadow_alpha)

        if cfg.noise_strength > 0:
            img = self._add_paper_noise(img, cfg.noise_strength)

        return img

    def to_bytes(self, image: Image.Image, fmt: str = "PNG") -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        buffer.seek(0)
        return buffer.read()

    def available_fonts(self) -> List[str]:
        return sorted(
            {
                name.title()
                for name in _cached_font_catalog().keys()
                if any(token in name for token in ["script", "hand", "comic", "cursive"])
            }
        )

    def _merged_config(self, overrides: Dict) -> RenderConfig:
        cfg_dict = {**self.config.__dict__, **overrides}
        return RenderConfig(**cfg_dict)

    def _draw_text(
        self,
        draw: ImageDraw.ImageDraw,
        font: ImageFont.FreeTypeFont,
        text: str,
        cfg: RenderConfig,
    ) -> None:
        max_width = cfg.width - 2 * cfg.margin
        x_cursor = cfg.margin
        y_cursor = cfg.margin
        base_line_height = font.getbbox("Hg")[3] - font.getbbox("Hg")[1]

        for char in text:
            if char == "\n":
                x_cursor = cfg.margin
                y_cursor += int(base_line_height * cfg.line_spacing)
                continue

            bbox = font.getbbox(char)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]

            if cfg.max_chars_per_line and (x_cursor - cfg.margin) // max(
                1, char_width
            ) >= cfg.max_chars_per_line:
                x_cursor = cfg.margin
                y_cursor += int(base_line_height * cfg.line_spacing)

            if x_cursor + char_width > cfg.margin + max_width:
                x_cursor = cfg.margin
                y_cursor += int(base_line_height * cfg.line_spacing)

            jitter_x, jitter_y = self._char_jitter(cfg, char_height)
            draw.text(
                (x_cursor + jitter_x, y_cursor + jitter_y),
                char,
                font=font,
                fill=cfg.ink_color,
            )
            x_cursor += char_width

    def _char_jitter(self, cfg: RenderConfig, char_height: int) -> Tuple[float, float]:
        if self._rng.random() > cfg.jitter_prob:
            return 0.0, 0.0
        jitter = cfg.jitter_px
        return (
            self._rng.uniform(-jitter, jitter),
            self._rng.uniform(-jitter * 0.5, jitter * 0.5),
        )

    @staticmethod
    def _apply_tilt(
        img: Image.Image, degrees: float, fill_color: Tuple[int, int, int]
    ) -> Image.Image:
        radians = math.radians(degrees)
        shear = math.tan(radians)
        width, height = img.size
        xshift = abs(shear) * height
        new_width = width + int(xshift)
        return img.transform(
            (new_width, height),
            Image.AFFINE,
            (1, shear, -xshift if degrees < 0 else 0, 0, 1, 0),
            Image.BICUBIC,
            fillcolor=fill_color,
        )

    @staticmethod
    def _add_shadow(img: Image.Image, alpha: float) -> Image.Image:
        shadow = img.copy().convert("RGBA").filter(ImageFilter.GaussianBlur(radius=4))
        r, g, b, _ = shadow.split()
        new_alpha = Image.new("L", img.size, int(255 * alpha))
        shadow = Image.merge("RGBA", (r, g, b, new_alpha))
        composite = Image.alpha_composite(
            Image.new("RGBA", img.size, (0, 0, 0, 0)), shadow
        )
        composite = Image.alpha_composite(composite, img.convert("RGBA"))
        return composite.convert("RGB")

    @staticmethod
    def _add_paper_noise(img: Image.Image, strength: float) -> Image.Image:
        arr = np.asarray(img).astype(np.float32)
        noise = np.random.normal(0.0, 255 * strength, arr.shape[:2])
        noise = noise[:, :, None]
        arr = arr + noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def _draw_guides(self, draw: ImageDraw.ImageDraw, cfg: RenderConfig) -> None:
        style = cfg.paper()
        spacing = int(cfg.font_size * cfg.line_spacing)
        for y in range(cfg.margin, cfg.height - cfg.margin, spacing):
            draw.line(
                [(cfg.margin, y), (cfg.width - cfg.margin, y)],
                fill=style.line_rgb,
                width=1,
            )

        if cfg.paper_style == "grid":
            spacing_x = cfg.font_size // 2
            for x in range(cfg.margin, cfg.width - cfg.margin, spacing_x):
                draw.line(
                    [(x, cfg.margin), (x, cfg.height - cfg.margin)],
                    fill=style.line_rgb,
                    width=1,
                )

