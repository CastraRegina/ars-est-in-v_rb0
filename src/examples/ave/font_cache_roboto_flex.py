"""Example to build an AvFont and store a font cache for Roboto Flex.

This script demonstrates how to:

1. Build an AvFont for a specific variation of Roboto Flex.
2. Warm up the glyph cache by requesting glyphs.
3. Serialize the font cache using AvFont.to_cache_dict.
4. Store the cache as a JSON file in fonts/cache.
5. Load the cache again using AvFont.from_cache_dict.

Run with:

    PYTHONPATH=./src python3 -m examples.ave.font_cache_roboto_flex
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict

from fontTools.ttLib import TTFont

from ave.font import AvFont, AvFontProperties
from ave.fonttools import FontHelper
from ave.glyph import AvGlyphFromTTFontFactory, AvGlyphPolygonizeFactory

ROBOTO_FLEX_FILENAME = "fonts/RobotoFlex-VariableFont_GRAD,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght.ttf"

CACHE_DIR: Path = Path("fonts/cache")
CACHE_FILE: Path = CACHE_DIR / "RobotoFlex_variable_font_cache.json"
ZIP_CACHE_FILE: Path = CACHE_DIR / "RobotoFlex_variable_font_cache_example.json.zip"


def build_avfont(font_path: str, axes_values: Dict[str, float]) -> AvFont:
    """Build an AvFont for the given font file and variation axes.

    Args:
        font_path: Path to the variable TrueType font file.
        axes_values: Mapping of axis tags to values, for example
            {"wght": 400.0}.

    Returns:
        AvFont instance backed by an AvGlyphPolygonizeFactory.
    """
    variable_font = TTFont(font_path)
    ttfont = FontHelper.instantiate_ttfont(variable_font, axes_values)

    polygonize_steps = 0
    glyph_factory_ttfont = AvGlyphFromTTFontFactory(ttfont)
    glyph_factory_polygonized = AvGlyphPolygonizeFactory(glyph_factory_ttfont, polygonize_steps)

    font_properties = AvFontProperties.from_ttfont(ttfont)
    return AvFont(glyph_factory_polygonized, font_properties)


def warm_up_cache(avfont: AvFont, text: str) -> None:
    """Populate the glyph cache by requesting glyphs for all characters.

    Args:
        avfont: Font whose glyph cache will be warmed up.
        text: Text whose characters will be requested from the font.
    """
    for character in text:
        avfont.get_glyph(character)


def save_font_cache(avfont: AvFont, cache_path: Path) -> None:
    """Save the font cache of avfont to cache_path as JSON.

    Args:
        avfont: Font whose cache should be stored.
        cache_path: Target path for the JSON cache file.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_dict = avfont.to_cache_dict()
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(cache_dict, file, ensure_ascii=True, indent=2)


def zip_font_cache(cache_path: Path, zip_path: Path) -> None:
    """Create a zip archive containing the given cache JSON file.

    Args:
        cache_path: Path to the JSON cache file.
        zip_path: Path to the resulting zip file.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(cache_path, arcname=cache_path.name)


def load_font_cache(cache_path: Path) -> AvFont:
    """Load a font cache from cache_path and return an AvFont.

    The cache file can be either a plain JSON file or a .zip archive
    containing a single JSON file created by save_font_cache.

    Args:
        cache_path: Path to a JSON or .zip file created by save_font_cache
            and optionally zipped by zip_font_cache.

    Returns:
        AvFont instance reconstructed from the cache.
    """
    if cache_path.suffix == ".zip":
        with zipfile.ZipFile(cache_path, "r") as zip_file:
            names = zip_file.namelist()
            if not names:
                raise ValueError("Zip cache file is empty")
            inner_name = names[0]
            with zip_file.open(inner_name) as file:
                cache_bytes = file.read()
        cache_dict = json.loads(cache_bytes.decode("utf-8"))
    else:
        with cache_path.open("r", encoding="utf-8") as file:
            cache_dict = json.load(file)

    return AvFont.from_cache_dict(cache_dict)


def main() -> None:
    """Main entry point for the font cache example."""
    print("Building AvFont for Roboto Flex variable font ...")

    axes_values: Dict[str, float] = {"wght": 400.0}
    avfont = build_avfont(ROBOTO_FLEX_FILENAME, axes_values)

    text = "I Roboto Flex cache example text 0123456789"
    print("Warming up glyph cache ...")
    warm_up_cache(avfont, text)

    print(f"Saving font cache to {CACHE_FILE} ...")
    save_font_cache(avfont, CACHE_FILE)

    print(f"Saving zipped font cache to {ZIP_CACHE_FILE} ...")
    zip_font_cache(CACHE_FILE, ZIP_CACHE_FILE)

    print("Loading font cache again from zip file ...")
    cached_font = load_font_cache(ZIP_CACHE_FILE)

    if cached_font.props.to_dict() != avfont.props.to_dict():
        raise RuntimeError("Cached font properties do not match original")

    for character in text:
        if avfont.get_glyph(character).to_dict() != cached_font.get_glyph(character).to_dict():
            raise RuntimeError(f"Cached glyph for {character!r} does not match original")

    print("Font info from cached font:")
    print(cached_font.get_info_string())


if __name__ == "__main__":
    main()
