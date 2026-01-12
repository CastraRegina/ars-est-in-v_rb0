"""Multi-glyph container for managing collections of AvGlyph objects."""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from ave.common import Align
from ave.geom import AvBox
from ave.glyph import (
    AvGlyph,
    AvGlyphCachedFactory,
    AvGlyphFactory,
    AvGlyphPersistentFactory,
)

###############################################################################
# Letter
###############################################################################


@dataclass
class AvLetter:
    """
    A Letter is a Glyph which is scaled to real dimensions with a position and alignment.
    """

    _glyph: AvGlyph
    _scale: float  # = font_size / units_per_em
    _xpos: float  # left-to-right
    _ypos: float  # bottom-to-top
    _align: Optional[Align] = None  # LEFT, RIGHT, BOTH. Defaults to None.

    def __init__(
        self,
        glyph: AvGlyph,
        scale: float,
        xpos: float = 0.0,
        ypos: float = 0.0,
        align: Optional[Align] = None,
    ) -> None:
        self._glyph = glyph
        self._scale = scale
        self._xpos = xpos
        self._ypos = ypos
        self._align = align

    @property
    def xpos(self) -> float:
        """The x position of the letter in real dimensions."""
        return self._xpos

    @xpos.setter
    def xpos(self, xpos: float) -> None:
        """Sets the x position of the letter in real dimensions."""
        self._xpos = xpos

    @property
    def ypos(self) -> float:
        """The y position of the letter in real dimensions."""
        return self._ypos

    @ypos.setter
    def ypos(self, ypos: float) -> None:
        """Sets the y position of the letter in real dimensions."""
        self._ypos = ypos

    @property
    def scale(self) -> float:
        """Returns the scale factor for the letter which is used to transform the glyph to real dimensions."""
        return self._scale

    @property
    def align(self) -> Optional[Align]:
        """The alignment of the letter; None, LEFT, RIGHT, BOTH."""
        return self._align

    @property
    def trafo(self) -> List[float]:
        """
        Returns the affine transformation matrix for the letter to transform the glyph to real dimensions.
        Returns: [scale, 0, 0, scale, xpos, ypos] or [scale, 0, 0, scale, xpos-lsb, ypos] if alignment is LEFT or BOTH.
        """
        if self.align in (Align.LEFT, Align.BOTH):
            lsb_scaled = self.scale * self._glyph.left_side_bearing
            return [self.scale, 0, 0, self.scale, self.xpos - lsb_scaled, self.ypos]
        return [self.scale, 0, 0, self.scale, self.xpos, self.ypos]

    @property
    def width(self) -> float:
        """
        Returns the width calculated considering the alignment.
        """
        return self.scale * self._glyph.width(self.align)

    @property
    def height(self) -> float:
        """
        The height of the Letter, i.e. the height of the bounding box.
        """
        return self.scale * self._glyph.height

    @property
    def ascender(self) -> float:
        """
        The maximum distance above the baseline, i.e. the highest y-coordinate of a Letter (positive value).
        """
        return self.scale * self._glyph.ascender

    @property
    def descender(self) -> float:
        """
        The maximum distance below the baseline, i.e. the lowest y-coordinate of a Letter (negative value).
        """
        return self.scale * self._glyph.descender

    @property
    def left_side_bearing(self) -> float:
        """
        LSB: The horizontal space on the left side of the Letter taking alignment into account.
        """
        if self.align in (Align.LEFT, Align.BOTH):
            return 0.0
        return self.scale * self._glyph.left_side_bearing

    @property
    def right_side_bearing(self) -> float:
        """
        RSB: The horizontal space on the right side of the Letter taking alignment into account.
        """
        if self.align in (Align.RIGHT, Align.BOTH):
            return 0.0
        return self.scale * self._glyph.right_side_bearing

    def bounding_box(self) -> AvBox:
        """
        Returns bounding box (tightest box around Letter) in real dimensions.
        Coordinates are relative to baseline-origin (0,0) with orientation left-to-right, bottom-to-top
        Returns:
            AvBox: The bounding box of the letter.
        """
        return self._glyph.bounding_box().transform_affine(self.trafo)

    def svg_path_string(self) -> str:
        """
        Returns the SVG path representation of the letter in real dimensions.
        The SVG path is a string that defines the outline of the letter using
        SVG path commands. This path can be used to render the letter as a
        vector graphic.
        Returns:
            str: The SVG path string representing the letter.
        """
        scale, _, _, _, translate_x, translate_y = self.trafo
        return self._glyph.path.svg_path_string(scale, translate_x, translate_y)

    def svg_path_string_debug_polyline(self, stroke_width: float = 1.0) -> str:
        """
        Returns a debug SVG path representation of the letter using only polylines.
        This method converts curves (Q, C) to straight lines between control points,
        making it useful for debugging the path structure.

        Supported commands:
            M (move-to), L (line-to), Z (close-path)
            Q (quadratic bezier) -> converted to L commands
            C (cubic bezier) -> converted to L commands

        Args:
            stroke_width (float): The stroke width used to determine marker sizes.

        Returns:
            str: The debug SVG path string using only polylines with markers.
                Markers include:
                - Squares: Regular points (L commands)
                - Circles: Control points (intermediate points in Q and C commands)
                - Triangles (pointing right): M command points (segment starts)
                - Triangles (pointing left): Points before Z commands (segment ends)
                Note: Triangles are drawn in addition to the base markers (squares/circles).
        """
        scale, _, _, _, translate_x, translate_y = self.trafo
        return self._glyph.path.svg_path_string_debug_polyline(scale, translate_x, translate_y, stroke_width)


###############################################################################
# MultiWeightLetter
###############################################################################


@dataclass
class AvMultiWeightLetter:
    """
    MultiWeightLetter: Container for managing collections of AvGlyph objects with weight support.
    Can handle multiple glyphs of the same character with different weights, all at the same position.
    Internal weights are normalized from 0 to 1 with equal spacing.
    """

    _glyphs: List[AvGlyph] = field(default_factory=list)
    _x_positions: List[float] = field(default_factory=list)  # X positions for each glyph (for fine-tuning)
    _scale: float = 1.0  # = font_size / units_per_em
    _xpos: float = 0.0  # left-to-right
    _ypos: float = 0.0  # bottom-to-top
    _align: Optional[Align] = None  # LEFT, RIGHT, BOTH. Defaults to None.

    def __init__(
        self,
        glyphs: List[AvGlyph],
        x_positions: List[float],
        scale: float,
        xpos: float = 0.0,
        ypos: float = 0.0,
        align: Optional[Align] = None,
    ) -> None:
        self._scale = scale
        self._xpos = xpos
        self._ypos = ypos
        self._align = align
        self._glyphs = glyphs

        # Validate and set x positions
        if len(x_positions) != len(glyphs):
            if len(x_positions) == 0 and len(glyphs) > 0:
                # Initialize with 0.0 for all glyphs if empty list provided (stacked at same position)
                self._x_positions = [0.0] * len(glyphs)
            else:
                raise ValueError("x_positions must have the same length as glyphs")
        else:
            self._x_positions = x_positions

        # Validate all glyphs are AvGlyph objects
        for i, glyph in enumerate(self._glyphs):
            if not isinstance(glyph, AvGlyph):
                raise TypeError(f"Glyph at index {i} is not an AvGlyph object")

        # Validate all glyphs have the same character (for multi-weight use case)
        if len(self._glyphs) > 1:
            first_char = self._glyphs[0].character
            for i, glyph in enumerate(self._glyphs[1:], 1):
                if glyph.character != first_char:
                    print(
                        f"Warning: Glyph at index {i} has character '{glyph.character}' "
                        f"but expected '{first_char}' for multi-weight letter"
                    )

    @classmethod
    def from_factories(
        cls,
        character: str,
        factories: List[AvGlyphFactory],
        scale: float = 1.0,
        xpos: float = 0.0,
        ypos: float = 0.0,
        align: Optional[Align] = None,
        x_positions: Optional[List[float]] = None,
    ) -> "AvMultiWeightLetter":
        """
        Create AvMultiWeightLetter from multiple factories.
        Internal weights will be automatically assigned from 0 to 1 with equal spacing.

        Args:
            character: The character to create glyphs for
            factories: List of factories (ordered from lightest to heaviest weight)
            scale: Scale factor
            xpos: X position
            ypos: Y position
            align: Alignment
            x_positions: Optional X positions for each glyph (defaults to all 0.0 for stacked)
        """
        glyphs = []

        for factory in factories:
            glyph = factory.get_glyph(character)
            glyphs.append(glyph)

        if x_positions is None:
            x_positions = [0.0] * len(glyphs)  # Stack all at same position

        return cls(
            glyphs=glyphs,
            x_positions=x_positions,
            scale=scale,
            xpos=xpos,
            ypos=ypos,
            align=align,
        )

    @property
    def character(self) -> str:
        """Get the character (from first glyph)."""
        if not self._glyphs:
            return ""
        return self._glyphs[0].character

    # @property
    # def weights(self) -> List[float]:
    #     """Get internal normalized weights (0 to 1 with equal spacing)."""
    #     if not self._glyphs:
    #         return []

    #     if len(self._glyphs) == 1:
    #         return [0.5]

    #     return [i / (len(self._glyphs) - 1) for i in range(len(self._glyphs))]

    @property
    def xpos(self) -> float:
        """The x position of the letter in real dimensions."""
        return self._xpos

    @xpos.setter
    def xpos(self, xpos: float) -> None:
        """Sets the x position of the letter in real dimensions."""
        self._xpos = xpos

    @property
    def ypos(self) -> float:
        """The y position of the letter in real dimensions."""
        return self._ypos

    @ypos.setter
    def ypos(self, ypos: float) -> None:
        """Sets the y position of the letter in real dimensions."""
        self._ypos = ypos

    @property
    def scale(self) -> float:
        """Returns the scale factor for the letter which is used to transform the glyph to real dimensions."""
        return self._scale

    @property
    def align(self) -> Optional[Align]:
        """The alignment of the letter; None, LEFT, RIGHT, BOTH."""
        return self._align

    @property
    def glyphs(self) -> List[AvGlyph]:
        """Get the glyphs list."""
        return self._glyphs

    @property
    def x_positions(self) -> List[float]:
        """Get the x positions list."""
        return self._x_positions

    # TODO: implement correct implementation
    @property
    def trafo(self) -> List[float]:
        """
        Returns the affine transformation matrix for the letter to transform the glyph to real dimensions.
        Returns: [scale, 0, 0, scale, xpos, ypos] or [scale, 0, 0, scale, xpos-lsb, ypos] if alignment is LEFT or BOTH.
        """
        if len(self._glyphs) == 0:
            return [self.scale, 0, 0, self.scale, self.xpos, self.ypos]
        if self.align in (Align.LEFT, Align.BOTH):
            lsb_scaled = self.scale * self._glyphs[0].left_side_bearing
            return [self.scale, 0, 0, self.scale, self.xpos - lsb_scaled, self.ypos]
        return [self.scale, 0, 0, self.scale, self.xpos, self.ypos]

    # TODO: implement correct implementation
    @property
    def width(self) -> float:
        """
        Returns the width calculated considering the alignment.
        """
        if len(self._glyphs) == 0:
            return 0.0
        return self.scale * self._glyphs[0].width(self.align)

    # TODO: implement correct implementation
    @property
    def height(self) -> float:
        """
        The height of the Letter, i.e. the height of the bounding box.
        """
        if len(self._glyphs) == 0:
            return 0.0
        return self.scale * self._glyphs[0].height

    # TODO: implement correct implementation
    @property
    def ascender(self) -> float:
        """
        The maximum distance above the baseline, i.e. the highest y-coordinate of a Letter (positive value).
        """
        if len(self._glyphs) == 0:
            return 0.0
        return self.scale * self._glyphs[0].ascender

    # TODO: implement correct implementation
    @property
    def descender(self) -> float:
        """
        The maximum distance below the baseline, i.e. the lowest y-coordinate of a Letter (negative value).
        """
        if len(self._glyphs) == 0:
            return 0.0
        return self.scale * self._glyphs[0].descender

    # TODO: implement correct implementation
    @property
    def left_side_bearing(self) -> float:
        """
        LSB: The horizontal space on the left side of the Letter taking alignment into account.
        """
        if len(self._glyphs) == 0:
            return 0.0
        if self.align in (Align.LEFT, Align.BOTH):
            return 0.0
        return self.scale * self._glyphs[0].left_side_bearing

    # TODO: implement correct implementation
    @property
    def right_side_bearing(self) -> float:
        """
        RSB: The horizontal space on the right side of the Letter taking alignment into account.
        """
        if len(self._glyphs) == 0:
            return 0.0
        if self.align in (Align.RIGHT, Align.BOTH):
            return 0.0
        return self.scale * self._glyphs[0].right_side_bearing

    # TODO: implement correct implementation
    def bounding_box(self):
        """Get bounding box that encompasses all glyphs."""
        if not self._glyphs:
            return AvBox(0.0, 0.0, 0.0, 0.0)

        if len(self._glyphs) == 1:
            bbox = self._glyphs[0].bounding_box()
            x_pos = self._x_positions[0]
            return AvBox(bbox.xmin + x_pos, bbox.ymin, bbox.xmax + x_pos, bbox.ymax)

        # Calculate combined bounding box using pre-calculated x positions
        # Initialize with first glyph's bounding box
        first_bbox = self._glyphs[0].bounding_box()
        first_x = self._x_positions[0]
        min_x = first_bbox.xmin + first_x
        max_x = first_bbox.xmax + first_x
        min_y = first_bbox.ymin
        max_y = first_bbox.ymax

        # Process remaining glyphs
        for i, glyph in enumerate(self._glyphs[1:], 1):
            bbox = glyph.bounding_box()
            x_pos = self._x_positions[i]

            # Transform bbox to its position
            glyph_min_x = bbox.xmin + x_pos
            glyph_max_x = bbox.xmax + x_pos
            glyph_min_y = bbox.ymin
            glyph_max_y = bbox.ymax

            min_x = min(min_x, glyph_min_x)
            max_x = max(max_x, glyph_max_x)
            min_y = min(min_y, glyph_min_y)
            max_y = max(max_y, glyph_max_y)

        return AvBox(min_x, min_y, max_x, max_y)

    # TODO: implement correct implementation
    def svg_path_string(self) -> str:
        """SVG path string of the letter in real dimensions."""
        if len(self._glyphs) == 0:
            return "M 0 0"

        scale, _, _, _, translate_x, translate_y = self.trafo
        return self._glyphs[0].path.svg_path_string(scale, translate_x, translate_y)


def main():
    """Test function for AvMultiWeightLetter."""

    def load_cached_fonts(path_name: str, font_fn_base: str) -> List[AvGlyphCachedFactory]:
        """
        Load cached font files from directory.

        Args:
            path_name: Directory path where cached files are stored (e.g., "fonts/cache")
            font_fn_base: Base filename pattern (e.g., "RobotoFlex-VariableFont_AA_")

        Returns:
            List[AvGlyphCachedFactory]: Ordered list of font factories (lightest to heaviest)
        """
        cache_dir = Path(path_name)
        if not cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {path_name}")

        # Find all matching files
        pattern = f"{font_fn_base}*_wght*.json.zip"
        file_paths = list(cache_dir.glob(pattern))

        if not file_paths:
            raise FileNotFoundError(f"No cached font files found with pattern: {pattern}")

        # Load factories (files are already in light to heavy order)
        factories = []

        for file_path in file_paths:
            try:
                factory = AvGlyphPersistentFactory.load_from_file(str(file_path))
                factories.append(factory)
                print(f"Loaded: {file_path.name}")
            except (FileNotFoundError, ValueError, RuntimeError, OSError, gzip.BadGzipFile) as e:
                print(f"Warning: Failed to load {file_path.name}: {e}")

        if not factories:
            raise ValueError("No valid cached font files could be loaded")

        return factories

    # Example usage
    try:
        factories = load_cached_fonts("fonts/cache", "RobotoFlex-VariableFont_AA_")
        print(f"Loaded {len(factories)} fonts")

        # Print font properties for each factory
        for i, factory in enumerate(factories):
            font_props = factory.get_font_properties()
            print(
                f"Font {i+1}: "
                f"ascender={font_props.ascender}, "
                f"descender={font_props.descender}, "
                f"units_per_em={font_props.units_per_em}"
            )

        # Example: Create AvMultiWeightLetter with different weights stacked
        print("\nCreating AvMultiWeightLetter with stacked weights...")
        multi_letter = AvMultiWeightLetter.from_factories(
            character="I",
            factories=factories,
            scale=50.0 / 2048.0,  # 50pt font
            xpos=100,
            ypos=100,
        )
        print(f"Created multi-weight letter for character '{multi_letter.character}'")
        print(f"Number of weight variants: {len(multi_letter.glyphs)}")
        print(f"Initial X positions (for fine-tuning): {multi_letter.x_positions}")

    except (FileNotFoundError, ValueError, RuntimeError, OSError, gzip.BadGzipFile) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
