"""Font handling utilities and typography processing for OpenType and SVG fonts."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont

from ave.glyph import (
    AvGlyph,
    AvGlyphCachedFactory,
    AvGlyphFactory,
    AvGlyphFromTTFontFactory,
)


###############################################################################
# AvFontProperties
###############################################################################
@dataclass
class AvFontProperties:
    """
    Represents the properties of a font.

    The properties are as follows:

    - `ascender`: The highest y-coordinate above the baseline (mostly positive value).
    - `descender`: The lowest y-coordinate below the baseline (usually negative value).
    - `line_gap`: Additional spacing between lines.
    - `x_height`: Height of lowercase 'x'.
    - `cap_height`: Height of uppercase 'H'.
    - `dash_thickness`: Height/thickness of dash '-' character.
    - `units_per_em`: Units per em.
    - `family_name`: Font family name.
    - `subfamily_name`: Style name (Regular, Bold, etc.).
    - `full_name`: Full font name.
    - `license_description`: License text.
    - `line_height`: Computed line height of the font (ascender - descender + line_gap).
    """

    ascender: float
    descender: float
    line_gap: float

    x_height: float
    cap_height: float
    dash_thickness: float
    units_per_em: float

    family_name: str
    subfamily_name: str
    full_name: str
    license_description: str

    @property
    def line_height(self) -> float:
        """Computed line height of the font (ascender - descender + line_gap)."""
        return self.ascender - self.descender + self.line_gap

    def __init__(
        self,
        ascender: float = 0,
        descender: float = 0,
        line_gap: float = 0,
        x_height: float = 0,
        cap_height: float = 0,
        dash_thickness: float = 0,
        units_per_em: float = 1000,
        family_name: str = "",
        subfamily_name: str = "",
        full_name: str = "",
        license_description: str = "",
    ) -> None:
        super().__init__()
        self.ascender = ascender
        self.descender = descender
        self.line_gap = line_gap
        self.x_height = x_height
        self.cap_height = cap_height
        self.dash_thickness = dash_thickness
        self.units_per_em = units_per_em
        self.family_name = family_name
        self.subfamily_name = subfamily_name
        self.full_name = full_name
        self.license_description = license_description

    @classmethod
    def from_dict(cls, data: dict) -> AvFontProperties:
        """Create an AvFontProperties instance from a dictionary."""
        return cls(
            ascender=data.get("ascender", 0),
            descender=data.get("descender", 0),
            line_gap=data.get("line_gap", 0),
            x_height=data.get("x_height", 0),
            cap_height=data.get("cap_height", 0),
            dash_thickness=data.get("dash_thickness", 0),
            units_per_em=data.get("units_per_em", 1000),
            family_name=data.get("family_name", ""),
            subfamily_name=data.get("subfamily_name", ""),
            full_name=data.get("full_name", ""),
            license_description=data.get("license_description", ""),
        )

    def to_dict(self) -> dict:
        """Convert the AvFontProperties instance to a dictionary."""
        return {
            "ascender": self.ascender,
            "descender": self.descender,
            "line_gap": self.line_gap,
            "x_height": self.x_height,
            "cap_height": self.cap_height,
            "dash_thickness": self.dash_thickness,
            "units_per_em": self.units_per_em,
            "family_name": self.family_name,
            "subfamily_name": self.subfamily_name,
            "full_name": self.full_name,
            "license_description": self.license_description,
        }

    @classmethod
    def _glyph_visual_height(cls, font: TTFont, char: str) -> float:
        """Return yMax - yMin of the glyph for `char`, or 0.0 if missing/empty."""
        cmap = font.getBestCmap()
        if not cmap or ord(char) not in cmap:
            return 0.0

        glyph_name = cmap[ord(char)]
        glyph_set = font.getGlyphSet()

        pen = BoundsPen(glyph_set)
        try:
            glyph_set[glyph_name].draw(pen)
        except KeyError:
            return 0.0

        if pen.bounds is None:
            return 0.0

        _, y_min, _, y_max = pen.bounds
        return max(0.0, float(y_max - y_min))

    @classmethod
    def _get_name_safe(cls, name_table, name_id: int) -> str:
        """
        Extract a name string with multiple fallbacks.
        This is the most robust way to read name records from real-world fonts.
        """
        # 1. Fast path - getDebugName handles most cases correctly
        name = name_table.getDebugName(name_id)
        if name is not None:
            return name

        # 2. Try Windows Unicode English (platform 3, encoding 1, lang 0x409)
        record = name_table.getName(name_id, 3, 1, 0x409)
        if record is not None:
            try:
                return record.toUnicode()
            except (UnicodeDecodeError, ValueError):
                pass

        # 3. Any Windows record
        record = name_table.getName(name_id, 3, 1)
        if record is not None:
            try:
                return record.toUnicode()
            except (UnicodeDecodeError, ValueError):
                pass

        # 4. Last resort: iterate all records manually
        for record in name_table.names:
            if record.nameID == name_id:
                try:
                    return record.toUnicode()
                except (UnicodeDecodeError, ValueError, AttributeError):
                    continue

        return ""

    @classmethod
    def from_ttfont(cls, ttfont: TTFont) -> AvFontProperties:
        """
        Create AvFontProperties from a fontTools TTFont object.
        Works with static and variable fonts (as long as the correct variation is active).
        """
        hhea = ttfont["hhea"]
        head = ttfont["head"]
        name_table = ttfont["name"]

        return cls(
            ascender=float(hhea.ascender),  # type: ignore
            descender=float(hhea.descender),  # type: ignore
            line_gap=float(hhea.lineGap),  # type: ignore
            # line_height is automatically computed via @computed_field
            x_height=cls._glyph_visual_height(ttfont, "x"),
            cap_height=cls._glyph_visual_height(ttfont, "H"),
            dash_thickness=cls._glyph_visual_height(ttfont, "-"),
            units_per_em=float(head.unitsPerEm),  # type: ignore
            family_name=cls._get_name_safe(name_table, 1),
            subfamily_name=cls._get_name_safe(name_table, 2),
            full_name=cls._get_name_safe(name_table, 4),
            license_description=cls._get_name_safe(name_table, 13),
        )

    def info_string(self) -> str:
        """
        Return a string containing information about the font properties.
        The string is formatted for display in a text box or similar.
        """

        info_string = (
            "-----Font Information:-----\n"
            f"ascender:      {self.ascender:>5.0f} (max distance above baseline = highest y-coord, positive value)\n"
            f"descender:     {self.descender:>5.0f} (max distance below baseline = lowest y-coord, negative value)\n"
            f"line_gap:      {self.line_gap:>5.0f} (additional spacing between lines of text)\n"
            f"line_height:   {self.line_height:>5.0f} (ascender - descender + line_gap)\n"
            f"x_height:      {self.x_height:>5.0f} (height of lowercase 'x')\n"
            f"cap_height:    {self.cap_height:>5.0f} (height of uppercase 'H')\n"
            f"dash_thickness:{self.dash_thickness:>5.0f} (height/thickness of dash '-')\n"
            f"units_per_em:  {self.units_per_em:>5.0f} (number of units per EM)\n"
            f"family_name:         {self.family_name}\n"
            f"subfamily_name:      {self.subfamily_name}\n"
            f"full_name:           {self.full_name}\n"
            f"license_description: {self.license_description}\n"
        )
        return info_string

    def __repr__(self) -> str:
        return f"AvFontProperties({self.family_name} {self.subfamily_name}, {self.units_per_em}upem)"


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

    @classmethod
    def from_cache_file(cls, cache_file_path: str, ttfont_fallback: Optional[TTFont] = None) -> AvFont:
        """
        Load font from cache file with optional TTFont fallback for missing glyphs.

        Creates an AvGlyphCachedFactory with the cached glyphs. If ttfont_fallback
        is provided, it's set as the source_factory for automatic fallback loading.

        Args:
            cache_file_path: Path to the cache file
            ttfont_fallback: Optional TTFont for loading missing glyphs

        Returns:
            AvFont: Font instance with cached glyphs and optional fallback
        """
        # Load cache data
        cache_data = cls._load_cache_file(cache_file_path)

        # Extract font data from nested structure
        font_data = cache_data["Font"]

        # Load font properties
        font_properties = AvFontProperties.from_dict(font_data.get("font_properties", {}))

        # Create glyph factory
        if ttfont_fallback is not None:
            # Create cached factory with TTFont as source
            ttfont_factory = AvGlyphFromTTFontFactory(ttfont_fallback)

            # Load glyphs from cache data
            glyph_entries = font_data.get("glyphs", {})
            glyphs: Dict[str, AvGlyph] = {}

            for character, glyph_data in glyph_entries.items():
                try:
                    glyphs[character] = AvGlyph.from_dict(glyph_data)
                except (ValueError, KeyError, TypeError) as e:
                    print(f"Warning: Failed to load glyph for '{character}': {e}")
                except Exception as e:
                    raise RuntimeError(f"Unexpected error loading glyph '{character}': {e}") from e

            # Use AvGlyphCachedFactory with TTFont as source factory
            glyph_factory = AvGlyphCachedFactory(glyphs=glyphs, source_factory=ttfont_factory)
        else:
            # Simple cached factory without source
            glyph_factory = AvGlyphCachedFactory.from_cache_dict(font_data)

        return cls(glyph_factory=glyph_factory, font_properties=font_properties)

    @classmethod
    def _load_cache_file(cls, cache_file_path: str) -> dict:
        """Load cache data from compressed or regular JSON file."""
        path = Path(cache_file_path)

        if not path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_file_path}")

        try:
            # Try gzip first
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        except (gzip.BadGzipFile, OSError):
            # Fall back to regular JSON
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in cache file {cache_file_path}: {e}") from e

    def get_glyph(self, character: str) -> AvGlyph:
        """Returns the AvGlyph for the given character from the factory."""
        return self._glyph_factory.get_glyph(character)

    def save_cache(self, cache_file_path: str) -> None:
        """Save current font state to file."""
        cache_data = self.to_dict()

        target_path = Path(cache_file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(target_path, "wt", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

    def append_glyph_to_cache(self, glyph: AvGlyph, cache_file_path: str) -> None:
        """Append a single glyph to the cache file."""
        # Load existing cache
        try:
            cache_data = self._load_cache_file(cache_file_path)
        except FileNotFoundError:
            cache_data = {"format_version": 1, "font_properties": self._font_properties.to_dict(), "glyphs": {}}

        # Add new glyph
        cache_data["glyphs"][glyph.character] = glyph.to_dict()

        # Save back
        target_path = Path(cache_file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(target_path, "wt", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

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
