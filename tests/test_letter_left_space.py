"""Tests for AvLetter.left_space implementations."""

from __future__ import annotations

import numpy as np
import pytest

from ave.common import Align
from ave.glyph import AvGlyph
from ave.letter import AvMultiWeightLetter, AvSingleGlyphLetter
from ave.letter_support import LetterSpacing
from ave.path import AvPath


def _rectangle_path(width: float = 1.0, height: float = 1.0) -> AvPath:
    """Create a simple rectangular AvPath with clockwise winding."""
    points = np.array(
        [
            (0.0, 0.0),
            (width, 0.0),
            (width, height),
            (0.0, height),
        ],
        dtype=float,
    )
    commands = ["M", "L", "L", "L", "Z"]
    return AvPath(points, commands)


def _make_letter(
    xpos: float = 0.0,
    glyph_width: float = 1.0,
    glyph_height: float = 1.0,
    ypos: float = 0.0,
) -> AvSingleGlyphLetter:
    """Create a left-aligned AvSingleGlyphLetter positioned at *xpos*."""
    glyph = AvGlyph(
        character="x",
        advance_width=glyph_width,
        path=_rectangle_path(glyph_width, glyph_height),
    )
    return AvSingleGlyphLetter(
        glyph=glyph,
        scale=1.0,
        xpos=xpos,
        ypos=ypos,
        align=Align.LEFT,
    )


class TestAvSingleGlyphLetterLeftSpace:
    """Edge cases for AvSingleGlyphLetter.left_space."""

    def test_returns_zero_without_left_neighbor(self) -> None:
        """Return 0.0 when there is no left neighbor."""
        letter = _make_letter()
        assert LetterSpacing.space_between(letter.left_letter, letter) == pytest.approx(0.0)

    def test_returns_zero_when_geometry_missing(self) -> None:
        """Return 0.0 when this letter has no geometry."""
        empty_glyph = AvGlyph(character="x", advance_width=1.0, path=AvPath())
        left = _make_letter()
        right = AvSingleGlyphLetter(
            glyph=empty_glyph,
            scale=1.0,
            xpos=1.5,
            ypos=0.0,
            align=Align.LEFT,
        )
        right.left_letter = left
        assert LetterSpacing.space_between(right.left_letter, right) == pytest.approx(0.0)

    def test_returns_zero_when_left_geometry_missing(self) -> None:
        """Return 0.0 when the left neighbor has no geometry."""
        empty_glyph = AvGlyph(character="x", advance_width=1.0, path=AvPath())
        left = AvSingleGlyphLetter(
            glyph=empty_glyph,
            scale=1.0,
            xpos=0.0,
            ypos=0.0,
            align=Align.LEFT,
        )
        right = _make_letter(xpos=1.5)
        right.left_letter = left
        assert LetterSpacing.space_between(right.left_letter, right) == pytest.approx(0.0)

    def test_positive_gap_matches_geometry_distance(self) -> None:
        """Return a positive value equal to the horizontal gap between letters."""
        left = _make_letter(xpos=0.0)
        right = _make_letter(xpos=1.5)
        right.left_letter = left
        assert LetterSpacing.space_between(right.left_letter, right) == pytest.approx(0.5, abs=1e-3)

    def test_returns_near_zero_when_letters_touch(self) -> None:
        """Return near 0.0 when the letter outlines touch."""
        left = _make_letter(xpos=0.0)
        right = _make_letter(xpos=1.0)
        right.left_letter = left
        assert abs(LetterSpacing.space_between(right.left_letter, right)) <= 1e-3

    def test_returns_zero_when_gap_below_default_tolerance(self) -> None:
        """Return 0.0 when the gap is below the default tolerance."""
        left = _make_letter(xpos=0.0)
        right = _make_letter(xpos=1.0005)
        right.left_letter = left
        assert LetterSpacing.space_between(right.left_letter, right) == pytest.approx(0.0)

    def test_negative_value_when_letters_overlap(self) -> None:
        """Return a negative shift magnitude when outlines overlap."""
        left = _make_letter(xpos=0.0)
        right = _make_letter(xpos=0.75)
        right.left_letter = left
        # Right letter overlaps left by 0.25 units, so shift must be positive.
        assert LetterSpacing.space_between(right.left_letter, right) == pytest.approx(-0.25, abs=1e-3)


class TestAvMultiWeightLetterLeftSpace:
    """Ensure multi-weight letters delegate spacing to the heaviest glyph."""

    def test_returns_zero_when_no_letters(self) -> None:
        """Return 0.0 when the multi-weight container is empty."""
        left = _make_letter(xpos=0.0)
        multi = AvMultiWeightLetter([])
        multi.left_letter = left
        assert LetterSpacing.space_between(multi.left_letter, multi) == pytest.approx(0.0)

    def test_uses_heaviest_letter_for_spacing(self) -> None:
        """Use the index 0 letter for spacing regardless of other weights."""
        left = _make_letter(xpos=0.0)

        heavy = _make_letter(xpos=1.5)
        light = _make_letter(xpos=0.75)
        multi = AvMultiWeightLetter([heavy, light])
        multi.left_letter = left

        assert LetterSpacing.space_between(multi.left_letter, multi) == pytest.approx(0.5, abs=1e-3)


class TestLetterLeftSpaceEdgeCases:
    """Additional edge cases for left_space calculations."""

    def test_interlocking_shapes(self) -> None:
        """Test letters that interlock (concave shapes)."""
        # Create an 'L' shape for the left letter
        # 1.0 x 1.0 total, missing top-right 0.5x0.5
        l_points = np.array(
            [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0.0, 1.0)],
            dtype=float,
        )
        l_cmds = ["M", "L", "L", "L", "L", "L", "Z"]
        l_path = AvPath(l_points, l_cmds)
        l_glyph = AvGlyph("L", 1.0, l_path)
        left = AvSingleGlyphLetter(l_glyph, xpos=0.0)

        # Create an inverted 'L' (7-like) shape for the right letter
        # It fits into the empty space of the left 'L'
        # 0.5 width stem on top, 1.0 width total.
        inv_l_points = np.array(
            [(0.0, 0.5), (0.5, 0.5), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            dtype=float,
        )
        inv_l_cmds = ["M", "L", "L", "L", "L", "L", "Z"]
        inv_l_path = AvPath(inv_l_points, inv_l_cmds)
        inv_l_glyph = AvGlyph("7", 1.0, inv_l_path)

        # Position right letter at x=0.6.
        # Left 'L' occupies x=[0, 1]. Right '7' occupies x=[0.6, 1.6].
        # Limiting gap is 0.1.

        # NOTE: We add a small y-offset to avoid exact overlap of horizontal segments
        # (L's shelf at y=0.5 and 7's shelf at y=0.5).
        right = AvSingleGlyphLetter(inv_l_glyph, xpos=0.6, ypos=0.01)
        right.left_letter = left

        # 0.6 (pos) - 0.5 (max extent of L at relevant height) = 0.1
        # With ypos=0.01, the horizontal clearance dominates.
        assert LetterSpacing.space_between(right.left_letter, right) == pytest.approx(0.1, abs=1e-3)

    def test_differing_scales(self) -> None:
        """Test spacing with different scales."""
        # Left: 1x1 rect scaled by 2.0 -> occupies [0, 2]
        left = _make_letter(xpos=0.0)
        left.scale = 2.0

        # Right: 1x1 rect scaled by 0.5. positioned at x=2.5.
        # Occupies [2.5, 3.0].
        right = _make_letter(xpos=2.5)
        right.scale = 0.5
        right.left_letter = left

        # Gap = 2.5 - 2.0 = 0.5
        assert LetterSpacing.space_between(right.left_letter, right) == pytest.approx(0.5, abs=1e-3)

    def test_touching_letters(self) -> None:
        """Test letters that exactly touch."""
        left = _make_letter(xpos=0.0)
        # Right starts exactly where left ends (x=1.0)
        right = _make_letter(xpos=1.0)
        right.left_letter = left

        # Touching counts as intersection, so it might try to separate them.
        # The separation distance should be negligible.
        assert LetterSpacing.space_between(right.left_letter, right) == pytest.approx(0.0, abs=1e-3)

    def test_vertically_overlapping(self) -> None:
        """Test letters that overlap vertically but are offset in Y."""
        # Left: 0.0 to 1.0 in Y.
        left = _make_letter(xpos=0.0, ypos=0.0)

        # Right: 0.5 to 1.5 in Y. Overlaps y=[0.5, 1.0].
        right = _make_letter(xpos=1.5, ypos=0.5)
        right.left_letter = left

        # Gap is still 1.5 - 1.0 = 0.5
        assert LetterSpacing.space_between(right.left_letter, right) == pytest.approx(0.5, abs=1e-3)

    def test_vertically_disjoint(self) -> None:
        """Test letters that do not overlap vertically."""
        # Left: 0.0 to 1.0 in Y.
        left = _make_letter(xpos=0.0, ypos=0.0)

        # Right: 2.0 to 3.0 in Y. Completely above.
        right = _make_letter(xpos=0.0, ypos=2.0)
        right.left_letter = left

        # Since they don't overlap vertically, the 'left space' (collision distance)
        # is effectively infinite or limited by search bounds.
        space = LetterSpacing.space_between(right.left_letter, right)
        assert space > 1.0

    def test_alignment_affects_position(self) -> None:
        """Test that alignment changes effective position."""
        # Glyph 1x1.
        # Align.LEFT: xpos is left edge.
        # Align.RIGHT: xpos is origin (usually).
        # We need to check how LEFT alignment shifts the shape if LSB > 0.

        # Create a glyph with LSB > 0
        # Rect from x=0.5 to x=1.5 (width=1.0, LSB=0.5)
        points = np.array([(0.5, 0.0), (1.5, 0.0), (1.5, 1.0), (0.5, 1.0)], dtype=float)
        commands = ["M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)
        glyph = AvGlyph("x", 2.0, path)  # Advance width 2.0

        # Case 1: No alignment (default) -> Origin at xpos.
        # Shape is at [xpos+0.5, xpos+1.5].
        l1 = AvSingleGlyphLetter(glyph, xpos=0.0, align=None)

        # Case 2: LEFT alignment -> Shift by -LSB.
        # Trafo shifts x by -LSB (-0.5).
        # Shape should be at [xpos, xpos+1.0] -> [0.0, 1.0].
        l2 = AvSingleGlyphLetter(glyph, xpos=0.0, align=Align.LEFT)

        # Check LSB logic
        assert l1.bounding_box.xmin == pytest.approx(0.5)
        assert l2.bounding_box.xmin == pytest.approx(0.0)

        # Now test spacing between an aligned letter and another
        # l2 occupies [0, 1].
        # Place another rect at x=1.5.
        r = _make_letter(xpos=1.5)
        r.left_letter = l2

        # Gap = 1.5 - 1.0 = 0.5
        assert LetterSpacing.space_between(r.left_letter, r) == pytest.approx(0.5, abs=1e-3)
