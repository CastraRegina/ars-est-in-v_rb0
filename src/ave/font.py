"""Font handling utilities and typography processing for OpenType and SVG fonts."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont

from ave.font_support import AvFontProperties
from ave.glyph import (
    AvGlyph,
    AvGlyphCachedFactory,
    AvGlyphFactory,
    AvGlyphFromTTFontFactory,
)


###############################################################################
# AvFont
###############################################################################
@dataclass
class AvFont:
    """Font abstraction with cached glyph access and font metrics.

    Uses an AvGlyphCachedFactory to store glyphs and provides access to
    font properties. The cached factory can optionally have a source_factory
    for fallback glyph loading when glyphs are not in the cache.
    """

    _glyph_factory: AvGlyphCachedFactory
    _font_properties: AvFontProperties = field(default_factory=AvFontProperties)

    def __init__(
        self,
        glyph_factory: AvGlyphFactory,
        font_properties: AvFontProperties,
    ) -> None:
        if isinstance(glyph_factory, AvGlyphCachedFactory):
            self._glyph_factory = glyph_factory
        else:
            self._glyph_factory = AvGlyphCachedFactory(source_factory=glyph_factory)
        self._font_properties = font_properties

    @property
    def glyph_factory(self) -> AvGlyphCachedFactory:
        """Returns the glyph factory used by this font."""
        return self._glyph_factory

    @property
    def props(self) -> AvFontProperties:
        """Returns the AvFontProperties object associated with this font."""
        return self._font_properties

    def get_glyph(self, character: str) -> AvGlyph:
        """Returns the AvGlyph for the given character from the factory."""
        return self._glyph_factory.get_glyph(character)

    def to_dict(self) -> dict:
        """Return a dictionary representing the font.

        The dictionary contains font properties and all cached glyphs
        nested under a "Font" key.

        Returns:
            dict: Dictionary suitable for JSON serialization.
        """
        return {
            "Font": {
                "format_version": 1,
                "font_properties": self._font_properties.to_dict(),
                **self._glyph_factory.to_cache_dict(),  # Merge glyphs from factory
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> AvFont:
        """Create an AvFont instance from a dictionary.

        Args:
            data: Dictionary created by to_dict() with "Font" key.

        Returns:
            AvFont: Font instance backed by a cached glyph factory.
        """
        font_data = data["Font"]
        font_properties = AvFontProperties.from_dict(font_data.get("font_properties", {}))
        glyph_factory = AvGlyphCachedFactory.from_cache_dict(font_data)
        return cls(glyph_factory=glyph_factory, font_properties=font_properties)

    def to_cache_file(self, cache_file_path: str) -> None:
        """Save font data to compressed file.

        Args:
            cache_file_path: Path to save the compressed font data
        """
        cache_data = self.to_dict()

        target_path = Path(cache_file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(target_path, "wt", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_cache_file(
        cls, cache_file_path: Optional[str] = None, fallback_factory: Optional[AvGlyphFactory] = None
    ) -> AvFont:
        """
        Load font from cache file with optional fallback factory.

        At least one of cache_file_path or fallback_factory must be specified.
        If only cache_file_path is provided, loads cached glyphs only.
        If only fallback_factory is provided, loads glyphs from that factory only.
        If both are provided, loads cached glyphs with fallback factory for missing glyphs.

        Args:
            cache_file_path: Optional path to the cache file
            fallback_factory: Optional factory for loading missing glyphs

        Returns:
            AvFont: Font instance with cached glyphs and/or fallback

        Raises:
            ValueError: If neither cache_file_path nor fallback_factory is specified
        """
        if cache_file_path is None and fallback_factory is None:
            raise ValueError("At least one of cache_file_path or fallback_factory must be specified")

        # Load from cache file if provided
        if cache_file_path is not None:
            # Load cache data
            path = Path(cache_file_path)
            if not path.exists():
                raise FileNotFoundError(f"Cache file not found: {cache_file_path}")

            try:
                # Try gzip first
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    cache_data = json.load(f)
            except (gzip.BadGzipFile, OSError):
                # Fall back to regular JSON
                with open(path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in cache file {cache_file_path}: {e}") from e

            # Extract font data
            font_data = cache_data["Font"]
            font_properties = AvFontProperties.from_dict(font_data.get("font_properties", {}))

            # Create glyph factory
            if fallback_factory is not None:
                # Load cached glyphs with fallback factory
                glyph_factory = AvGlyphCachedFactory.from_cache_dict(font_data)
                glyph_factory.source_factory = fallback_factory
            else:
                # Load cached glyphs only
                glyph_factory = AvGlyphCachedFactory.from_cache_dict(font_data)
        else:
            # No cache file - use fallback factory only
            # Try to get font properties from the factory
            font_properties = fallback_factory.get_font_properties()
            if font_properties is None:
                # Factory doesn't provide properties, use defaults
                font_properties = AvFontProperties()
            glyph_factory = AvGlyphCachedFactory(source_factory=fallback_factory)

        return cls(glyph_factory=glyph_factory, font_properties=font_properties)

    def get_info_string(self) -> str:
        """
        Return a string containing information about the font.
        The string is formatted for display in a text box or similar.
        """
        font_info_string = self._font_properties.info_string()
        font_info_string += "-----Glyphs in cache:-----\n"
        glyph_count = 0

        for glyph_character in self._glyph_factory.glyphs:
            glyph_count += 1
            font_info_string += f"{glyph_character}"
            if glyph_count % 20 == 0:
                font_info_string += "\n"
        if font_info_string[-1] != "\n":
            font_info_string += "\n"
        return font_info_string

    def actual_ascender(self):
        """Returns the overall maximum ascender by iterating over all glyphs in the cache (positive value)."""
        ascender: float = 0.0

        for glyph in self._glyph_factory.glyphs.values():
            ascender = max(ascender, glyph.ascender)
        return ascender

    def actual_descender(self):
        """Returns the overall minimum descender by iterating over all glyphs in the cache (negative value)."""
        descender: float = 0.0

        for glyph in self._glyph_factory.glyphs.values():
            descender = min(descender, glyph.descender)
        return descender


###############################################################################
# Main
###############################################################################


def main():
    """Main"""


if __name__ == "__main__":
    main()
