"""Font handling utilities and typography processing for OpenType and SVG fonts."""

from __future__ import annotations

from dataclasses import dataclass

from ave.font_support import AvFontProperties
from ave.glyph import AvGlyph, AvGlyphCachedFactory


###############################################################################
# AvFont
###############################################################################
@dataclass
class AvFont:
    """Font abstraction with cached glyph access and font metrics.

    Uses an AvGlyphCachedFactory to store glyphs and provides access to
    font properties. The cached factory can optionally have a second_source
    for fallback glyph loading when glyphs are not in the cache.
    """

    _glyph_factory: AvGlyphCachedFactory

    def __init__(self, glyph_factory: AvGlyphCachedFactory) -> None:
        self._glyph_factory = glyph_factory

    @property
    def glyph_factory(self) -> AvGlyphCachedFactory:
        """Returns the glyph factory used by this font."""
        return self._glyph_factory

    @property
    def props(self) -> AvFontProperties:
        """Returns the AvFontProperties object associated with this font."""
        return self._glyph_factory.get_font_properties()

    def get_glyph(self, character: str) -> AvGlyph:
        """Returns the AvGlyph for the given character from the factory."""
        return self._glyph_factory.get_glyph(character)

    def get_info_string(self) -> str:
        """
        Return a string containing information about the font.
        The string is formatted for display in a text box or similar.
        """
        font_info_string = self.props.info_string()
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

    def actual_ascender(self) -> float:
        """Returns the overall maximum ascender by iterating over all glyphs.

        Positive value representing the highest y-coordinate among all glyphs.
        Returns negative infinity if no glyphs are cached.
        """
        return max((g.ascender for g in self._glyph_factory.glyphs.values()), default=float("-inf"))

    def actual_descender(self) -> float:
        """Returns the overall minimum descender by iterating over all glyphs.

        Negative value representing the lowest y-coordinate among all glyphs.
        Returns positive infinity if no glyphs are cached.
        """
        return min((g.descender for g in self._glyph_factory.glyphs.values()), default=float("inf"))


###############################################################################
# Main
###############################################################################


def main() -> None:
    """Main entry point for the font module."""


if __name__ == "__main__":
    main()
