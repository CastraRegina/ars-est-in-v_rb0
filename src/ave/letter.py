"""Multi-weight letter implementation using AvSingleGlyphLetter."""

from __future__ import annotations

import gzip
import re
from abc import ABC, abstractmethod
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
# AvLetter
###############################################################################


class AvLetter(ABC):
    """
    Abstract base class for letter representations.

    A Letter is a visual representation of one or more glyphs positioned
    in real-world coordinates with scaling, alignment, and positioning.
    """

    @property
    @abstractmethod
    def xpos(self) -> float:
        """The x position of the letter in real dimensions."""
        raise NotImplementedError

    @xpos.setter
    @abstractmethod
    def xpos(self, xpos: float) -> None:
        """Sets the x position of the letter in real dimensions."""
        raise NotImplementedError

    @property
    @abstractmethod
    def ypos(self) -> float:
        """The y position of the letter in real dimensions."""
        raise NotImplementedError

    @ypos.setter
    @abstractmethod
    def ypos(self, ypos: float) -> None:
        """Sets the y position of the letter in real dimensions."""
        raise NotImplementedError

    @property
    @abstractmethod
    def scale(self) -> float:
        """Returns the scale factor for the letter."""
        raise NotImplementedError

    @scale.setter
    @abstractmethod
    def scale(self, scale: float) -> None:
        """Sets the scale factor for the letter."""
        raise NotImplementedError

    @property
    @abstractmethod
    def align(self) -> Optional[Align]:
        """The alignment of the letter; None, LEFT, RIGHT, BOTH."""
        raise NotImplementedError

    @align.setter
    @abstractmethod
    def align(self, align: Optional[Align]) -> None:
        """Sets the alignment of the letter."""
        raise NotImplementedError

    @property
    @abstractmethod
    def ascender(self) -> float:
        """The maximum distance above the baseline."""
        raise NotImplementedError

    @property
    @abstractmethod
    def descender(self) -> float:
        """The maximum distance below the baseline."""
        raise NotImplementedError

    @property
    @abstractmethod
    def height(self) -> float:
        """The height of the letter."""
        raise NotImplementedError

    @property
    @abstractmethod
    def advance_width(self) -> float:
        """Returns the advance width of the letter."""
        raise NotImplementedError

    @abstractmethod
    def width(self) -> float:
        """Returns the width calculated considering the alignment."""
        raise NotImplementedError

    @abstractmethod
    def left_side_bearing(self) -> float:
        """The horizontal space on the left side of the letter."""
        raise NotImplementedError

    @abstractmethod
    def right_side_bearing(self) -> float:
        """The horizontal space on the right side of the letter."""
        raise NotImplementedError

    @abstractmethod
    def bounding_box(self) -> AvBox:
        """Returns the bounding box around the letter's outline."""
        raise NotImplementedError

    @abstractmethod
    def letter_box(self) -> AvBox:
        """Returns the letter's advance box."""
        raise NotImplementedError

    @abstractmethod
    def svg_path_string(self) -> str:
        """Returns the SVG path representation of the letter."""
        raise NotImplementedError


###############################################################################
# AvSingleGlyphLetter
###############################################################################


@dataclass
class AvSingleGlyphLetter(AvLetter):
    """
    A Letter is a Glyph which is scaled to real dimensions with a position, alignment and horizontal offset.
    """

    _glyph: AvGlyph
    _scale: float  # scale from glyph-coordinates to real dimensions (font_size / units_per_em)
    _xpos: float  # left-to-right, value in real dimensions
    _ypos: float  # bottom-to-top, value in real dimensions
    _align: Optional[Align] = None  # LEFT, RIGHT, BOTH. Defaults to None.
    _x_offset: float = 0.0  # horizontal offset in real dimensions

    def __init__(
        self,
        glyph: AvGlyph,
        scale: float = 1.0,  # scale from glyph-coordinates to real dimensions (font_size / units_per_em)
        xpos: float = 0.0,  # left-to-right, value in real dimensions
        ypos: float = 0.0,  # bottom-to-top, value in real dimensions
        align: Optional[Align] = None,  # LEFT, RIGHT, BOTH. Defaults to None.
        x_offset: float = 0.0,  # horizontal offset in real dimensions
    ) -> None:
        self._glyph = glyph
        self._scale = scale
        self._xpos = xpos
        self._ypos = ypos
        self._align = align
        self._x_offset = x_offset

    @property
    def glyph(self) -> AvGlyph:
        """The glyph of the letter."""
        return self._glyph

    @property
    def scale(self) -> float:
        """Returns the scale factor for the letter which is used to transform the glyph to real dimensions."""
        return self._scale

    @scale.setter
    def scale(self, scale: float) -> None:
        """Sets the scale factor for the letter which is used to transform the glyph to real dimensions."""
        self._scale = scale

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
    def align(self) -> Optional[Align]:
        """Returns the alignment of the letter; None, LEFT, RIGHT, BOTH."""
        return self._align

    @align.setter
    def align(self, align: Optional[Align]) -> None:
        """Sets the alignment of the letter; None, LEFT, RIGHT, BOTH."""
        self._align = align

    @property
    def x_offset(self) -> float:
        """Horizontal offset in real dimensions."""
        return self._x_offset

    @x_offset.setter
    def x_offset(self, x_offset: float) -> None:
        """Sets the horizontal offset in real dimensions."""
        self._x_offset = x_offset

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
    def height(self) -> float:
        """
        The height of the Letter, i.e. the height of the bounding box.
        """
        return self.scale * self._glyph.height

    @property
    def trafo(self) -> List[float]:
        """
        Returns the affine transformation matrix for the letter to transform the glyph to real dimensions.
        Returns:    [scale, 0, 0, scale, xpos + x_offset, ypos] or
                    [scale, 0, 0, scale, xpos - lsb + x_offset, ypos] if alignment is LEFT or BOTH.
        """
        if self.align in (Align.LEFT, Align.BOTH):
            lsb_scaled = self.scale * self._glyph.left_side_bearing()
            return [self.scale, 0, 0, self.scale, self.xpos - lsb_scaled + self._x_offset, self.ypos]
        return [self.scale, 0, 0, self.scale, self.xpos + self._x_offset, self.ypos]

    def centroid(self) -> Tuple[float, float]:
        """
        Returns the centroid of the letter in real dimensions.

        The centroid is the geometric center of the letter's outline,
        calculated from the actual path geometry (not from the bounding box)
        and transformed to world coordinates.

        Returns:
            Tuple[float, float]: The (x, y) coordinates of the centroid.
        """
        # Get the glyph's centroid and transform it using the letter's transformation
        glyph_centroid = self._glyph.centroid()
        scale, _, _, _, translate_x, translate_y = self.trafo

        # Transform the centroid coordinates
        centroid_x = glyph_centroid[0] * scale + translate_x
        centroid_y = glyph_centroid[1] * scale + translate_y

        return (centroid_x, centroid_y)

    @property
    def advance_width(self) -> float:
        """
        Returns the advance width of the letter in real dimensions.
        """
        return self.scale * self._glyph.advance_width

    def width(self) -> float:
        """
        Returns the width calculated considering the alignment.
        """
        return self.scale * self._glyph.width(self.align)

    def left_side_bearing(self) -> float:
        """
        LSB: The horizontal space on the left side of the Letter taking alignment into account (sign varies +/-).
        Positive values when the glyph is placed to the right of the origin (i.e. positive bounding_box.xmin).
        Negative values when the glyph is placed to the left of the origin (i.e. negative bounding_box.xmin).
        Note: Returns 0.0 for LEFT or BOTH alignment as the letter is positioned at the origin.
        """
        # actually: return self.bounding_box.xmin - self.letter_box().xmin
        return self.scale * self._glyph.left_side_bearing

    def right_side_bearing(self) -> float:
        """
        RSB: The horizontal space on the right side of the Letter taking alignment into account (sign varies +/-).
        Positive values when the glyph's bounding box is inside the advance box (i.e. positive bounding_box.xmax).
        Negative values when the glyph's bounding box extends to the right of the glyph box
                (i.e. bounding_box.xmax > advance_width).
        Note: Returns 0.0 for RIGHT or BOTH alignment as the letter is positioned to end at the advance width.
        """
        # actually: self.advance_width - ( self.bounding_box.xmax - self.letter_box().xmin )
        return self.scale * self._glyph.right_side_bearing

    @property
    def bounding_box(self) -> AvBox:
        """
        Returns the tightest bounding box around the letter's outline in real dimensions.

        The box is transformed to world coordinates including position, scale, and x_offset.
        Coordinates are relative to baseline-origin (0,0) with orientation left-to-right, bottom-to-top.

        Returns:
            AvBox: The bounding box in world coordinates.
        """
        return self._glyph.bounding_box.transform_affine(self.trafo)

    def letter_box(self) -> AvBox:
        """
        Returns the glyph's advance box in real dimensions.

        The glyph box (advance box) spans from the transformed position to
        position + advanceWidth, and from descender to ascender.
        Unlike the outline bounding box, this includes the full advance width.
        The box is transformed to world coordinates including position, scale, and x_offset.

        Returns:
            AvBox: The transformed glyph advance box.
        """
        return self._glyph.glyph_box().transform_affine(self.trafo)

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
        Returns a debug SVG path representation of the letter using polylines with visual markers.

        This method converts the letter's path to straight lines and adds markers to visualize
        the path structure, making it ideal for debugging complex letter shapes and understanding
        control point relationships in curves.

        Command conversion:
            M, L, Z: Preserved as-is (move-to, line-to, close-path)
            Q: Converted to lines connecting all three points (start, control, end)
            C: Converted to lines connecting all four points (start, control1, control2, end)

        Visual markers (size based on stroke_width):
            - Squares: Path points (L commands and curve endpoints)
            - Circles: Control points (intermediate points in Q and C commands)
            - Right triangles: Move-to points (M commands - segment starts)
            - Left triangles: Points before close-path (Z commands - segment ends)

        Args:
            stroke_width: Determines marker sizes (default: 1.0)

        Returns:
            str: Complete SVG path string with polylines and markers.
                    Uses the letter's transformation matrix for positioning.
        """
        scale, _, _, _, translate_x, translate_y = self.trafo
        return self._glyph.path.svg_path_string_debug_polyline(scale, translate_x, translate_y, stroke_width)


###############################################################################
# MultiWeightLetter
###############################################################################


@dataclass
class AvMultiWeightLetter(AvLetter):
    """
    MultiWeightLetter: Container for managing collections of AvSingleGlyphLetter objects with weight support.
    Can handle multiple letters of the same character with different weights, all at the same position.

    Weight Convention:
    - Factories/letters are ordered from HEAVIEST to LIGHTEST (index 0 = heavy/black)
    - Normalized weights: 0.0 = lightest (white), 1.0 = heaviest (black)
    - Index 0 corresponds to normalized weight 1.0 (heaviest)
    - Index N-1 corresponds to normalized weight 0.0 (lightest)
    - This matches image convention: black (0) = heavy, white (255) = light
    """

    _letters: List[AvSingleGlyphLetter] = field(default_factory=list)

    def __init__(
        self,
        letters: List[AvSingleGlyphLetter],
    ) -> None:
        self._letters = letters

        # Validate all letters are AvSingleGlyphLetter objects
        for i, letter in enumerate(self._letters):
            if not isinstance(letter, AvSingleGlyphLetter):
                raise TypeError(f"Letter at index {i} is not an AvSingleGlyphLetter object")

        # Validate all letters have the same character (for multi-weight use case)
        if len(self._letters) > 1:
            first_char = self._letters[0]._glyph.character
            for i, letter in enumerate(self._letters[1:], 1):
                glyph = letter._glyph
                if glyph.character != first_char:
                    print(
                        f"Warning: Letter at index {i} has character '{glyph.character}' "
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
        x_offsets: Optional[List[float]] = None,
    ) -> "AvMultiWeightLetter":
        """
        Create AvMultiWeightLetter from multiple factories.

        Weight Convention:
        - Factories must be ordered from HEAVIEST to LIGHTEST
        - Index 0 = heaviest weight (black, normalized weight 1.0)
        - Index N-1 = lightest weight (white, normalized weight 0.0)
        - Normalized weights are evenly spaced: [1.0, ..., 0.0]

        Args:
            character: The character to create glyphs for
            factories: List of factories (ordered from HEAVIEST to LIGHTEST weight)
            scale: Scale factor
            xpos: X position
            ypos: Y position
            align: Alignment
            x_offsets: Optional X offsets for each glyph (defaults to all 0.0 for stacked)
        """
        letters = []

        if x_offsets is None:
            x_offsets = [0.0] * len(factories)  # Stack all at same position
        elif len(x_offsets) != len(factories):
            raise ValueError("x_offsets must have the same length as factories")

        for factory, x_offset in zip(factories, x_offsets):
            glyph = factory.get_glyph(character)
            letter = AvSingleGlyphLetter(glyph=glyph, scale=scale, xpos=xpos, ypos=ypos, align=align, x_offset=x_offset)
            letters.append(letter)

        return cls(letters=letters)

    @property
    def character(self) -> str:
        """Get the character (from first letter)."""
        if not self._letters:
            return ""
        return self._letters[0].glyph.character

    @property
    def xpos(self) -> float:
        """The x position of the letter in real dimensions."""
        if not self._letters:
            return 0.0
        return self._letters[0].xpos

    @xpos.setter
    def xpos(self, xpos: float) -> None:
        """Sets the x position of the letter in real dimensions."""
        for letter in self._letters:
            letter.xpos = xpos

    @property
    def ypos(self) -> float:
        """The y position of the letter in real dimensions."""
        if not self._letters:
            return 0.0
        return self._letters[0].ypos

    @ypos.setter
    def ypos(self, ypos: float) -> None:
        """Sets the y position of the letter in real dimensions."""
        for letter in self._letters:
            letter.ypos = ypos

    @property
    def scale(self) -> float:
        """Returns the scale factor for the letter which is used to transform the glyph to real dimensions."""
        if not self._letters:
            return 1.0
        return self._letters[0].scale

    @scale.setter
    def scale(self, scale: float) -> None:
        """Sets the scale factor for the letter."""
        for letter in self._letters:
            letter.scale = scale

    @property
    def align(self) -> Optional[Align]:
        """The alignment of the letter; None, LEFT, RIGHT, BOTH."""
        if not self._letters:
            return None
        return self._letters[0].align

    @align.setter
    def align(self, align: Optional[Align]) -> None:
        """Sets the alignment of the letter."""
        for letter in self._letters:
            letter.align = align

    @property
    def letters(self) -> List[AvSingleGlyphLetter]:
        """Get the letters list."""
        return self._letters

    @property
    def x_offsets(self) -> List[float]:
        """Get the x offsets for each letter."""
        return [letter.x_offset for letter in self._letters]

    @property
    def glyphs(self) -> List[AvGlyph]:
        """Get the glyphs list."""
        return [letter.glyph for letter in self._letters]

    @property
    def num_weights(self) -> int:
        """Get the number of weight variants.

        Returns:
            Number of weight variants (letters) in this multi-weight letter.
        """
        return len(self._letters)

    def get_normalized_weight(self, index: int) -> float:
        """Get the normalized weight value for a given index.

        Weight Convention:
        - Index 0 = heaviest (normalized weight 1.0)
        - Index N-1 = lightest (normalized weight 0.0)
        - Weights are evenly spaced between 1.0 and 0.0

        Args:
            index: Letter index (0 to N-1)

        Returns:
            Normalized weight in range [0.0, 1.0] where 1.0 = heaviest, 0.0 = lightest

        Raises:
            IndexError: If index is out of range
        """
        if not 0 <= index < len(self._letters):
            raise IndexError(f"Index {index} out of range [0, {len(self._letters) - 1}]")

        if len(self._letters) == 1:
            return 1.0  # Single weight is considered heaviest

        # Index 0 = weight 1.0 (heaviest), Index N-1 = weight 0.0 (lightest)
        return 1.0 - (index / (len(self._letters) - 1))

    def get_letter_by_weight(self, normalized_weight: float) -> AvSingleGlyphLetter:
        """Get the letter closest to the specified normalized weight.

        Weight Convention:
        - normalized_weight 1.0 = heaviest (black) → returns letter at index 0
        - normalized_weight 0.0 = lightest (white) → returns letter at index N-1
        - Intermediate values interpolate and round to nearest available weight

        Args:
            normalized_weight: Weight value in range [0.0, 1.0]

        Returns:
            The letter with the closest matching weight

        Raises:
            ValueError: If normalized_weight is out of range or no letters available
        """
        if not self._letters:
            raise ValueError("No letters available")

        if not 0.0 <= normalized_weight <= 1.0:
            raise ValueError(f"normalized_weight must be in [0.0, 1.0], got {normalized_weight}")

        if len(self._letters) == 1:
            return self._letters[0]

        # Convert normalized weight to index
        # weight 1.0 → index 0, weight 0.0 → index N-1
        index = int((1.0 - normalized_weight) * (len(self._letters) - 1) + 0.5)
        index = max(0, min(index, len(self._letters) - 1))  # Clamp to valid range

        return self._letters[index]

    @staticmethod
    def gray_to_weight_index(gray_value: float, num_weights: int) -> int:
        """Convert image gray value (0-255) to weight index.

        Maps image gray values to glyph weight indices consistently:
        - Black (0) → index 0 (heaviest weight)
        - White (255) → index N-1 (lightest weight)
        - Gray values interpolate linearly

        Args:
            gray_value: Gray value from image (0=black, 255=white)
            num_weights: Number of available weight variants

        Returns:
            Weight index (0=heaviest, N-1=lightest)

        Raises:
            ValueError: If parameters are out of valid range

        Example:
            >>> AvMultiWeightLetter.gray_to_weight_index(0, 9)    # black
            0  # heaviest
            >>> AvMultiWeightLetter.gray_to_weight_index(255, 9)  # white
            8  # lightest
            >>> AvMultiWeightLetter.gray_to_weight_index(127.5, 9)  # mid-gray
            4  # medium weight
        """
        if not 0 <= gray_value <= 255:
            raise ValueError(f"gray_value must be in [0, 255], got {gray_value}")
        if num_weights < 1:
            raise ValueError(f"num_weights must be >= 1, got {num_weights}")

        if num_weights == 1:
            return 0

        # Direct mapping: darker (lower) values → lower indices (heavier)
        normalized = gray_value / 255.0  # 0.0 to 1.0 (0=black, 1=white)
        index = int(normalized * (num_weights - 1) + 0.5)  # Round to nearest
        return min(index, num_weights - 1)  # Clamp to valid range

    @staticmethod
    def gray_normalized_to_weight_index(gray_normalized: float, num_weights: int) -> int:
        """Convert normalized gray value (0.0-1.0) to weight index.

        Maps normalized gray values to glyph weight indices:
        - Black (0.0) → index 0 (heaviest weight)
        - White (1.0) → index N-1 (lightest weight)
        - Gray values interpolate linearly

        Args:
            gray_normalized: Normalized gray value (0.0=black, 1.0=white)
            num_weights: Number of available weight variants

        Returns:
            Weight index (0=heaviest, N-1=lightest)

        Raises:
            ValueError: If parameters are out of valid range

        Example:
            >>> AvMultiWeightLetter.gray_normalized_to_weight_index(0.0, 9)  # black
            0  # heaviest
            >>> AvMultiWeightLetter.gray_normalized_to_weight_index(1.0, 9)  # white
            8  # lightest
            >>> AvMultiWeightLetter.gray_normalized_to_weight_index(0.5, 9)  # mid-gray
            4  # medium weight
        """
        if not 0.0 <= gray_normalized <= 1.0:
            raise ValueError(f"gray_normalized must be in [0.0, 1.0], got {gray_normalized}")
        if num_weights < 1:
            raise ValueError(f"num_weights must be >= 1, got {num_weights}")

        if num_weights == 1:
            return 0

        # Direct mapping: darker (lower) values → lower indices (heavier)
        index = int(gray_normalized * (num_weights - 1) + 0.5)  # Round to nearest
        return min(index, num_weights - 1)  # Clamp to valid range

    @property
    def ascender(self) -> float:
        """
        The maximum distance above the baseline, i.e. the highest y-coordinate of a Letter (positive value).
        """
        if not self._letters:
            return 0.0
        return max(letter.ascender for letter in self._letters)

    @property
    def descender(self) -> float:
        """
        The maximum distance below the baseline, i.e. the lowest y-coordinate of a Letter (negative value).
        """
        if not self._letters:
            return 0.0
        return min(letter.descender for letter in self._letters)

    @property
    def height(self) -> float:
        """
        The total height of all letters.
        """
        if not self._letters:
            return 0.0
        return max(letter.height for letter in self._letters)

    @property
    def advance_width(self) -> float:
        """Returns the maximum advance width of all letters."""
        if len(self._letters) == 0:
            return 0.0
        return max(letter.advance_width for letter in self._letters)

    def width(self) -> float:
        """
        Returns width considering the letter's alignment, or advance_width if alignment is None.

        For MultiWeightLetter, uses the maximum advance width of all letters and applies
        alignment adjustments based on self.align.

        Returns:
            float: The calculated width based on alignment:
                - None:  advance_width (maximum of all letters)
                - LEFT:  advance_width - left_side_bearing
                - RIGHT: advance_width - right_side_bearing
                - BOTH:  bounding_box.width (combined width of all letters)
        """
        # LSB = return self.bounding_box.xmin - self.letter_box().xmin
        # RSB = return self.advance_width - (self.bounding_box.xmax - self.letter_box().xmin)
        if len(self._letters) == 0:
            return 0.0

        if self.align is None:
            return self.advance_width

        if self.align == Align.LEFT:
            return self.advance_width - self.left_side_bearing()
        if self.align == Align.RIGHT:
            return self.advance_width - self.right_side_bearing()
        if self.align == Align.BOTH:
            return self.bounding_box.width

    def left_side_bearing(self) -> float:
        """
        LSB: The horizontal space on the left side of the Letter taking alignment into account (sign varies +/-).
        Positive values when the glyph is placed to the right of the origin (i.e. positive bounding_box.xmin).
        Negative values when the glyph is placed to the left of the origin (i.e. negative bounding_box.xmin).
        Note: For MultiWeightLetter, this is calculated from the combined bounding box of all letters.
        """
        if len(self._letters) == 0:
            return 0.0
        return self.bounding_box.xmin - self.letter_box().xmin

    def right_side_bearing(self) -> float:
        """
        RSB: The horizontal space on the right side of the Letter taking alignment into account (sign varies +/-).
        Positive values when the glyph's bounding box is inside the advance box (i.e. positive bounding_box.xmax).
        Negative values when the glyph's bounding box extends to the right of the glyph box
                (i.e. bounding_box.xmax > advance_width).
        Note: For MultiWeightLetter, this is calculated using the maximum advance width and combined bounding box.
        """
        if len(self._letters) == 0:
            return 0.0
        return self.advance_width - (self.bounding_box.xmax - self.letter_box().xmin)

    @property
    def bounding_box(self):
        """Returns the combined bounding box encompassing all letters' outlines.

        Computes the tightest box that contains the transformed outlines of all letters.

        Returns:
            AvBox: Combined bounding box of all letters, or empty box if no letters.
        """
        if not self._letters:
            return AvBox(0.0, 0.0, 0.0, 0.0)

        # Get bounding boxes from all letters
        bounding_boxes = [letter.bounding_box for letter in self._letters]

        # Use AvBox.combine to get the overall bounding box
        return AvBox.combine(*bounding_boxes)

    def letter_box(self) -> AvBox:
        """Returns the combined letter box encompassing all letters' advance boxes.

        Computes the box that contains the transformed advance boxes of all letters,
        including their full advance widths.

        Returns:
            AvBox: Combined letter box of all letters, or empty box if no letters.
        """
        if not self._letters:
            return AvBox(0.0, 0.0, 0.0, 0.0)

        # Get letter boxes from all letters
        letter_boxes = [letter.letter_box() for letter in self._letters]

        # Use AvBox.combine to get the overall letter box
        return AvBox.combine(*letter_boxes)

    def svg_path_string(self) -> str:
        """SVG path string of the letter in real dimensions."""
        if not self._letters:
            return "M 0 0"

        # Merge all letter path strings
        path_strings = []
        for letter in self._letters:
            path_strings.append(letter.svg_path_string())

        return " ".join(path_strings)

    # @staticmethod
    # def align_x_offsets_by_centroid(
    #     multi_weight_letter: AvMultiWeightLetter,
    # ) -> None:
    #     """
    #     Adapt x_offset of all letters so their glyph centroids align horizontally.

    #     Uses the geometric centroid of each glyph (calculated from the actual shape)
    #     rather than bounding box center, providing more accurate visual alignment
    #     especially for asymmetric characters.

    #     Processes glyphs from heaviest to lightest, with the heaviest glyph
    #     remaining fixed as the reference point.

    #     Args:
    #         multi_weight_letter: The AvMultiWeightLetter to modify
    #     """
    #     if not multi_weight_letter.letters or len(multi_weight_letter.letters) < 2:
    #         return

    #     # Process from heaviest to lightest (last to first)
    #     # The heaviest glyph stays fixed as reference
    #     for i in range(len(multi_weight_letter.letters) - 2, -1, -1):
    #         current = multi_weight_letter.letters[i]
    #         heavier = multi_weight_letter.letters[i + 1]

    #         # Get centroids
    #         current_centroid = current.centroid()
    #         heavier_centroid = heavier.centroid()

    #         # Calculate the offset needed to align centroids horizontally
    #         offset_adjustment: float = heavier_centroid[0] - current_centroid[0]

    #         # Update the current letter's x_offset
    #         current.x_offset += offset_adjustment

    @staticmethod
    def align_x_offsets_by_centering(
        multi_weight_letter: AvMultiWeightLetter,
    ) -> None:
        """
        Align the horizontal position of multi-weight letters by centering each lighter glyph
        within its heavier neighbor.

        This algorithm processes glyphs from heaviest to lightest, keeping the heaviest
        glyph fixed as reference. Each lighter glyph is centered horizontally within
        the next heavier glyph, regardless of relative widths.

        The centering is based on bounding box centers, providing visual alignment
        that works well for most character types.

        Args:
            multi_weight_letter: The AvMultiWeightLetter to modify
        """
        if not multi_weight_letter.letters or len(multi_weight_letter.letters) < 2:
            return

        # Process from heaviest to lightest (last to first)
        # The heaviest glyph stays fixed as reference
        for i in range(len(multi_weight_letter.letters) - 2, -1, -1):
            current = multi_weight_letter.letters[i]
            heavier = multi_weight_letter.letters[i + 1]

            # Get bounding boxes
            current_bbox = current.bounding_box
            heavier_bbox = heavier.bounding_box

            # Calculate centers
            heavier_center = (heavier_bbox.xmin + heavier_bbox.xmax) / 2
            current_center = (current_bbox.xmin + current_bbox.xmax) / 2

            # Center the current glyph within the heavier glyph
            offset_adjustment = heavier_center - current_center
            current.x_offset += offset_adjustment


###############################################################################
# Letter Factories
###############################################################################

###############################################################################
# AvLetterFactory
###############################################################################


class AvLetterFactory(ABC):
    """Abstract base class for letter factories.

    Letter factories create AvLetter instances from stream items (characters or syllables).
    This class-based approach allows for stateful factories and better extensibility
    compared to simple callables.
    """

    @abstractmethod
    def create_letter(
        self,
        item: str,
        scale: float,
        xpos: float,
        ypos: float,
        align: Optional[Align] = None,
    ) -> AvLetter:
        """Create a letter from the given parameters.

        Args:
            item: The item from the stream (character or syllable)
            scale: Scale factor for the letter
            xpos: X position of the letter
            ypos: Y position of the letter
            align: Alignment setting for the letter

        Returns:
            An AvLetter instance
        """
        raise NotImplementedError


###############################################################################
# AvSingleGlyphLetterFactory
###############################################################################


class AvSingleGlyphLetterFactory(AvLetterFactory):
    """Factory for creating AvSingleGlyphLetter objects.

    This factory uses a glyph factory to create single glyph letters.
    It's the most common factory type for simple text layout.
    """

    def __init__(self, glyph_factory: AvGlyphFactory):
        """Initialize the factory with a glyph factory.

        Args:
            glyph_factory: Factory to create glyphs for characters
        """
        self._glyph_factory = glyph_factory

    def create_letter(
        self,
        item: str,
        scale: float,
        xpos: float,
        ypos: float,
        align: Optional[Align] = None,
    ) -> AvLetter:
        """Create an AvSingleGlyphLetter from the given parameters.

        Args:
            item: The character to create a letter for
            scale: Scale factor for the letter
            xpos: X position of the letter
            ypos: Y position of the letter
            align: Alignment setting for the letter

        Returns:
            An AvSingleGlyphLetter instance
        """
        glyph = self._glyph_factory.get_glyph(item)
        return AvSingleGlyphLetter(
            glyph=glyph,
            scale=scale,
            xpos=xpos,
            ypos=ypos,
            align=align,
        )


###############################################################################
# AvMultiWeightLetterFactory
###############################################################################


class AvMultiWeightLetterFactory(AvLetterFactory):
    """Factory for creating AvMultiWeightLetter objects.

    This factory uses multiple glyph factories to create letters with different
    weights (e.g., light, regular, bold). Each glyph factory represents a
    different weight, and the resulting letter contains all weight variants.
    """

    def __init__(self, glyph_factories: List[AvGlyphFactory]):
        """Initialize the factory with multiple glyph factories.

        Args:
            glyph_factories: List of glyph factories for different weights
        """
        self._glyph_factories = glyph_factories

    def create_letter(
        self,
        item: str,
        scale: float,
        xpos: float,
        ypos: float,
        align: Optional[Align] = None,
    ) -> AvLetter:
        """Create an AvMultiWeightLetter from the given parameters.

        Args:
            item: The character to create letters for
            scale: Scale factor for the letters
            xpos: X position of the letters
            ypos: Y position of the letters
            align: Alignment setting for the letters

        Returns:
            An AvMultiWeightLetter instance containing all weight variants
        """
        letters = []
        for glyph_factory in self._glyph_factories:
            glyph = glyph_factory.get_glyph(item)
            letter = AvSingleGlyphLetter(
                glyph=glyph,
                scale=scale,
                xpos=xpos,
                ypos=ypos,
                align=align,
            )
            letters.append(letter)

        return AvMultiWeightLetter(letters=letters)


###############################################################################
# main
###############################################################################


def main():
    """Test function for AvMultiWeightLetter."""
    from ave.page import AvSvgPage  # pylint: disable=import-outside-toplevel

    def discover_font_basenames(path_name: str) -> List[str]:
        """
        Discover all unique font basenames in the cache directory.

        This groups font files by their base name (everything before the _XX weight suffix).
        For example, "Grandstander-VariableFont_AA_01_wght0100.json.zip" and
        "Grandstander-VariableFont_AA_02_wght0200.json.zip" both belong to
        "Grandstander-VariableFont_AA_" base font.

        Args:
            path_name: Directory path where cached files are stored (e.g., "fonts/cache")

        Returns:
            List[str]: List of unique font basenames
        """
        cache_dir = Path(path_name)
        if not cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {path_name}")

        # Find all font files
        pattern = "*_wght*.json.zip"
        file_paths = list(cache_dir.glob(pattern))

        if not file_paths:
            raise FileNotFoundError(f"No cached font files found with pattern: {pattern}")

        # Extract unique basenames (everything before _XX_wght)
        basenames = set()
        for file_path in file_paths:
            # Extract basename before _XX_wght (where XX is the weight number)
            # Pattern: FontName_XX_wghtYYYY.json.zip
            match = re.match(r"(.+?)_\d+_wght\d+", file_path.name)
            if match:
                basenames.add(match.group(1) + "_")

        # Return sorted list
        return sorted(list(basenames))

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

    def render_letter_with_boxes(multi_letter, colors, stroke_width):
        """Render a multi-weight letter with bounding boxes."""
        # Render all weight variants (already in heavy-to-light order)
        for letter, color in zip(multi_letter.letters, colors):
            svg_path = svg_page.drawing.path(letter.svg_path_string(), fill=color, stroke="none")
            svg_page.add(svg_path)

        # Add bounding box in yellow
        bbox = multi_letter.bounding_box
        svg_bbox = svg_page.drawing.path(
            f"M {bbox.xmin:g} {bbox.ymin:g} "
            f"L {bbox.xmax:g} {bbox.ymin:g} "
            f"L {bbox.xmax:g} {bbox.ymax:g} "
            f"L {bbox.xmin:g} {bbox.ymax:g} Z",
            fill="none",
            stroke="yellow",
            stroke_width=stroke_width,
        )
        svg_page.add(svg_bbox, True)

        # Add letter box in green
        lbox = multi_letter.letter_box()
        svg_lbox = svg_page.drawing.path(
            f"M {lbox.xmin:g} {lbox.ymin:g} "
            f"L {lbox.xmax:g} {lbox.ymin:g} "
            f"L {lbox.xmax:g} {lbox.ymax:g} "
            f"L {lbox.xmin:g} {lbox.ymax:g} Z",
            fill="none",
            stroke="green",
            stroke_width=stroke_width,
        )
        svg_page.add(svg_lbox, True)

    def render_alignment_test():
        """Render alignment test letters (U and A) at both edges."""
        # U at x-pos=0 with LEFT alignment
        multi_letter_u_left = AvMultiWeightLetter.from_factories(
            character="U",
            factories=one_font_factories,
            scale=scale,
            xpos=0.0,
            ypos=test_ypos,
        )
        multi_letter_u_left.align = Align.LEFT
        AvMultiWeightLetter.align_x_offsets_by_centering(multi_letter_u_left)
        render_letter_with_boxes(multi_letter_u_left, colors, stroke_width)

        # U at x-pos=1.0 with RIGHT alignment
        multi_letter_u_right = AvMultiWeightLetter.from_factories(
            character="U",
            factories=one_font_factories,
            scale=scale,
            xpos=1.0,
            ypos=test_ypos,
        )
        multi_letter_u_right.align = Align.RIGHT
        AvMultiWeightLetter.align_x_offsets_by_centering(multi_letter_u_right)
        multi_letter_u_right.xpos = 1.0 - multi_letter_u_right.width()
        render_letter_with_boxes(multi_letter_u_right, colors, stroke_width)

        # A at x-pos=0 with LEFT alignment (one font-size above)
        multi_letter_a_left = AvMultiWeightLetter.from_factories(
            character="A",
            factories=one_font_factories,
            scale=scale,
            xpos=0.0,
            ypos=test_ypos + font_size,
        )
        multi_letter_a_left.align = Align.LEFT
        AvMultiWeightLetter.align_x_offsets_by_centering(multi_letter_a_left)
        render_letter_with_boxes(multi_letter_a_left, colors, stroke_width)

        # A at x-pos=1.0 with RIGHT alignment (one font-size above)
        multi_letter_a_right = AvMultiWeightLetter.from_factories(
            character="A",
            factories=one_font_factories,
            scale=scale,
            xpos=1.0,
            ypos=test_ypos + font_size,
        )
        multi_letter_a_right.align = Align.RIGHT
        AvMultiWeightLetter.align_x_offsets_by_centering(multi_letter_a_right)
        multi_letter_a_right.xpos = 1.0 - multi_letter_a_right.width()
        render_letter_with_boxes(multi_letter_a_right, colors, stroke_width)

    # Example usage
    try:
        # Colors for multi-weight rendering (heavy to light, black to light gray)
        # Index 0 = heaviest (black), Index N-1 = lightest (light gray)
        colors = [
            "#000000",  # Index 0: Heaviest weight (black)
            "#202020",
            "#404040",
            "#606060",
            "#808080",
            "#A0A0A0",
            "#C0C0C0",
            "#D0D0D0",
            "#E0E0E0",  # Index 8: Lightest weight (light gray)
        ]

        # Discover all available fonts in cache
        font_basenames = discover_font_basenames("fonts/cache")
        print(f"Discovered {len(font_basenames)} fonts:")
        for basename in font_basenames:
            print(f"  - {basename}")

        # Characters to display
        characters = ""
        characters += "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        characters += "abcdefghijklmnopqrstuvwxyz "
        characters += "0123456789 "
        # characters += ',.;:+-*#_<> !"§$%&/()=?{}[] '
        # # NON-ASCII EXCEPTION: German characters and special symbols for comprehensive font testing
        # characters += "ÄÖÜ äöü ß€µ@²³~^°\\ '`"

        # Create SVG page for multi-weight letters
        # Setup the page with A4 dimensions
        viewbox_width = 180  # viewbox width in mm
        viewbox_height = 120  # viewbox height in mm
        vb_scale = 1.0 / viewbox_width  # scale viewbox so that x-coordinates are between 0 and 1
        font_size = vb_scale * 2.7  # in mm (already in viewbox units)

        # Load the first font to get units_per_em
        one_font_factories = load_cached_fonts("fonts/cache", font_basenames[0])
        print(f"Loaded {len(one_font_factories)} fonts for units_per_em calculation")

        # Get units_per_em from the first factory
        units_per_em = one_font_factories[0].get_font_properties().units_per_em if one_font_factories else 2048.0
        scale = font_size / units_per_em  # proper scale calculation

        # Create the SVG page
        svg_page = AvSvgPage.create_page_a4(viewbox_width, viewbox_height, vb_scale)

        # Draw viewbox border
        svg_page.add(
            svg_page.drawing.path(
                d=(
                    f"M 0 0 "
                    f"L {vb_scale * viewbox_width} 0 "
                    f"L {vb_scale * viewbox_width} {vb_scale * viewbox_height} "
                    f"L 0 {vb_scale * viewbox_height} "
                    f"Z"
                ),
                stroke="blue",
                stroke_width=0.1 * vb_scale,
                fill="none",
            ),
            False,
        )

        # Render characters with multi-weight for each font
        line_height = 0.04  # Space between font lines
        current_ypos = 0.02

        for font_idx, font_basename in enumerate(font_basenames):
            print(f"Rendering font {font_idx + 1}/{len(font_basenames)}: {font_basename}...", end="", flush=True)

            # Load font for this line
            one_font_factories = load_cached_fonts("fonts/cache", font_basename)

            # Get units_per_em from the current font factory
            units_per_em = one_font_factories[0].get_font_properties().units_per_em if one_font_factories else 2048.0
            scale = font_size / units_per_em  # proper scale calculation for the current font

            # Reset x position for each font line
            current_xpos = 0.05

            # Calculate stroke width based on font dash thickness
            font_props = one_font_factories[0].get_font_properties()
            stroke_width = 0.1 * font_props.dash_thickness * font_size / font_props.units_per_em

            for char in characters:
                # Create multi-weight letter for this character
                multi_letter = AvMultiWeightLetter.from_factories(
                    character=char,
                    factories=one_font_factories,
                    scale=scale,  # Use proper scale, not font_size
                    xpos=current_xpos,
                    ypos=current_ypos,
                )

                AvMultiWeightLetter.align_x_offsets_by_centering(multi_letter)
                # AvMultiWeightLetter.align_x_offsets_by_centroid(multi_letter)

                # Render each weight variant with different opacity
                render_letter_with_boxes(multi_letter, colors, stroke_width)

                # Move to next position
                current_xpos += multi_letter.width() + 0.002

            # Render alignment test letters
            test_ypos = current_ypos  # Same line as the characters
            render_alignment_test()
            # Add font name at the end of the line
            font_display_name = font_basename.replace("-VariableFont_AA_", "")
            font_text = f" -- {font_display_name}"

            for char in font_text:
                # Create multi-weight letter for font name character
                multi_letter = AvMultiWeightLetter.from_factories(
                    character=char,
                    factories=one_font_factories,
                    scale=scale,
                    xpos=current_xpos,
                    ypos=current_ypos,
                )

                AvMultiWeightLetter.align_x_offsets_by_centering(multi_letter)

                # Render heaviest weight in black (bottom layer) - index 0 is now heaviest
                svg_path_heavy = svg_page.drawing.path(
                    multi_letter.letters[0].svg_path_string(), fill="#000000", stroke="none"
                )
                svg_page.add(svg_path_heavy)

                # Render lightest weight in light gray (top layer) - index -1 is now lightest
                svg_path_light = svg_page.drawing.path(
                    multi_letter.letters[-1].svg_path_string(), fill="#E0E0E0", stroke="none"
                )
                svg_page.add(svg_path_light)

                # Move to next position
                current_xpos += multi_letter.width() + 0.001

            # Move to next line for next font
            current_ypos += line_height
            print(" done")

        # Save the SVG
        svg_filename = "data/output/example/svg/multi_weight_letters.svgz"
        print(f"\nSaving to {svg_filename} ...")
        svg_page.save_as(svg_filename, include_debug_layer=True, pretty=True, compressed=True)
        print(f"Saved to {svg_filename}")

    except (FileNotFoundError, ValueError, RuntimeError, OSError, gzip.BadGzipFile) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()


# ───────────────────────────────────────────────────────────────────────────────
# advance_width
# width()
# left_side_bearing
# right_side_bearing
# bounding_box()
# letter_box()
# ───────────────────────────────────────────────────────────────────────────────

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                        METHOD DEPENDENCY TREE                               │
# └─────────────────────────────────────────────────────────────────────────────┘

# AvGlyph.width(align=None)
#     calls:
#     - self.advance_width
#     - self.left_side_bearing()  [if align == LEFT]
#     - self.right_side_bearing() [if align == RIGHT]
#     - self.bounding_box       [if align == BOTH]
#     - self.align                [to determine which calculation]

# AvGlyph.advance_width
#     (property, no dependencies)

# AvGlyph.left_side_bearing()
#     calls:
#     - self.bounding_box
#     - self.align                [returns 0.0 for LEFT/BOTH]

# AvGlyph.right_side_bearing()
#     calls:
#     - self.advance_width
#     - self.bounding_box
#     - self.align                [returns 0.0 for RIGHT/BOTH]

# AvGlyph.bounding_box
#     calls:
#     - self._path.bounding_box

# AvGlyph.letter_box()
#     calls:
#     - self.bounding_box
#     - self.advance_width

# ───────────────────────────────────────────────────────────────────────────────

# AvSingleGlyphLetter.width()
#     calls:
#     - self._glyph.width(self.align)
#     - self.align                [parameter passed to glyph]

# AvSingleGlyphLetter.advance_width
#     calls:
#     - self._glyph.advance_width

# AvSingleGlyphLetter.left_side_bearing()
#     calls:
#     - self._glyph.left_side_bearing()
#     - self.scale

# AvSingleGlyphLetter.right_side_bearing()
#     calls:
#     - self._glyph.right_side_bearing()
#     - self.scale

# AvSingleGlyphLetter.bounding_box
#     calls:
#     - self._glyph.bounding_box
#     - self.trafo

# AvSingleGlyphLetter.letter_box()
#     calls:
#     - self._glyph.glyph_box()
#     - self.trafo

# ───────────────────────────────────────────────────────────────────────────────

# AvMultiWeightLetter.width()
#     calls:
#     - self.advance_width          [if align is None]
#     - self.advance_width          [if align == LEFT]
#     - self.left_side_bearing()    [if align == LEFT]
#     - self.advance_width          [if align == RIGHT]
#     - self.right_side_bearing()   [if align == RIGHT]
#     - self.bounding_box         [if align == BOTH]
#     - self.align                  [to determine which calculation]

# AvMultiWeightLetter.advance_width
#     calls:
#     - Letter.advance_width (for each letter, takes max)

# AvMultiWeightLetter.left_side_bearing()
#     calls:
#     - self.bounding_box
#     - self.letter_box()

# AvMultiWeightLetter.right_side_bearing()
#     calls:
#     - self.advance_width
#     - self.bounding_box
#     - self.letter_box()

# AvMultiWeightLetter.bounding_box
#     calls:
#     - Letter.bounding_box (for each letter)
#     - AvBox.combine()

# AvMultiWeightLetter.letter_box()
#     calls:
#     - Letter.letter_box() (for each letter)
#     - AvBox.combine()

# ───────────────────────────────────────────────────────────────────────────────

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                            SUMMARY                                          │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │                                                                             │
# │  AvGlyph:      Base implementation, depends on path and advance_width       │
# │  AvSingleGlyphLetter:     Delegates to AvGlyph, adds transformation (trafo) │
# │  AvMultiWeight: Aggregates from multiple AvSingleGlyphLetter instances      │
# │                                                                             │
# │  Key Pattern:                                                               │
# │  - bounding_box() is fundamental - needed by most other methods             │
# │  - advance_width is a base property used by width() and side bearings       │
# │  - MultiWeightLetter aggregates, SingleLetter delegates                     │
# └─────────────────────────────────────────────────────────────────────────────┘
