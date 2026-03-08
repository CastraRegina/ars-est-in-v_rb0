"""Tests for AvLetter.left_space implementations."""

from __future__ import annotations

import numpy as np
import pytest

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
    ypos: float = 0.0,
    scale: float = 1.0,
) -> AvSingleGlyphLetter:
    """Create a simple 1x1 square letter for testing."""
    glyph = AvGlyph(character="x", advance_width=1.0, path=_rectangle_path())
    return AvSingleGlyphLetter(
        glyph=glyph,
        scale=scale,
        xpos=xpos,
        ypos=ypos,
    )


class TestLetterSpacing:
    """Tests for LetterSpacing.find_transition_shift implementation."""

    def test_returns_zero_without_left_neighbor(self) -> None:
        """Return 0.0 when there is no left neighbor."""
        letter = _make_letter()
        assert LetterSpacing.find_transition_shift(letter.left_letter, letter) == pytest.approx(0.0)

    def test_returns_zero_when_geometry_missing(self) -> None:
        """Return 0.0 when this letter has no geometry."""
        empty_glyph = AvGlyph(character="x", advance_width=1.0, path=AvPath())
        left = _make_letter()
        right = AvSingleGlyphLetter(
            glyph=empty_glyph,
            scale=1.0,
            xpos=1.5,
            ypos=0.0,
        )
        right.left_letter = left
        assert LetterSpacing.find_transition_shift(right.left_letter, right) == pytest.approx(0.0)

    def test_returns_zero_when_left_geometry_missing(self) -> None:
        """Return 0.0 when the left neighbor has no geometry."""
        empty_glyph = AvGlyph(character="x", advance_width=1.0, path=AvPath())
        left = AvSingleGlyphLetter(
            glyph=empty_glyph,
            scale=1.0,
            xpos=0.0,
            ypos=0.0,
        )
        right = _make_letter(xpos=1.5)
        right.left_letter = left
        assert LetterSpacing.find_transition_shift(right.left_letter, right) == pytest.approx(0.0)

    def test_positive_gap_matches_geometry_distance(self) -> None:
        """Return a negative value equal to the horizontal gap between letters."""
        left = _make_letter(xpos=0.0)
        right = _make_letter(xpos=1.5)
        right.left_letter = left
        assert LetterSpacing.find_transition_shift(right.left_letter, right) == pytest.approx(-0.5, abs=1e-3)

    def test_returns_near_zero_when_letters_touch(self) -> None:
        """Return near 0.0 when the letter outlines touch."""
        left = _make_letter(xpos=0.0)
        right = _make_letter(xpos=1.0)
        right.left_letter = left
        assert abs(LetterSpacing.find_transition_shift(right.left_letter, right)) <= 5e-3

    def test_negative_value_when_letters_overlap(self) -> None:
        """Return a positive shift magnitude when outlines overlap."""
        left = _make_letter(xpos=0.0)
        right = _make_letter(xpos=0.75)
        right.left_letter = left
        # Right letter overlaps left by 0.25 units, so shift must be positive.
        assert LetterSpacing.find_transition_shift(right.left_letter, right) == pytest.approx(0.25, abs=5e-3)

    def test_silhouette_properties_consistency(self) -> None:
        """Test that silhouette properties provide consistent results."""
        left = _make_letter(xpos=0.0)
        right = _make_letter(xpos=1.5)
        right.left_letter = left

        # Test with silhouette properties (fixed internal steps)
        result1 = LetterSpacing.find_transition_shift(right.left_letter, right)
        result2 = LetterSpacing.find_transition_shift(right.left_letter, right)

        # Results should be identical since silhouette properties use cached/fixed steps
        assert result1 == result2
        assert result1 == pytest.approx(-0.5, rel=1e-3)

    def test_tolerance_parameter(self) -> None:
        """Test tolerance parameter behavior."""
        left = _make_letter(xpos=0.0)
        right = _make_letter(xpos=1.5)
        right.left_letter = left

        # Test with different tolerances
        result_loose = LetterSpacing.find_transition_shift(right.left_letter, right, tolerance=0.01)
        result_tight = LetterSpacing.find_transition_shift(right.left_letter, right, tolerance=0.0001)

        # Both should give similar results for this simple case
        assert result_loose == pytest.approx(-0.5, rel=1e-2)
        assert result_tight == pytest.approx(-0.5, rel=1e-3)

    def test_max_iterations_parameter(self) -> None:
        """Test max_iterations parameter."""
        left = _make_letter(xpos=0.0)
        right = _make_letter(xpos=1.5)
        right.left_letter = left

        # Test with very low iterations
        result_low_iter = LetterSpacing.find_transition_shift(right.left_letter, right, max_iterations=5)

        # Should still give reasonable result
        assert result_low_iter == pytest.approx(-0.5, rel=1e-1)

    def test_edge_case_very_far_apart(self) -> None:
        """Test letters beyond normal spacing (edge case protection)."""
        left = _make_letter(xpos=0.0)
        right = _make_letter(xpos=100.0)
        right.left_letter = left

        # Should handle large gaps correctly
        result = LetterSpacing.find_transition_shift(right.left_letter, right)
        assert result < -50.0  # Should be large negative

    def test_edge_case_vertically_disjoint(self) -> None:
        """Test vertically disjoint letters."""
        left = _make_letter(xpos=0.0, ypos=0.0)
        right = _make_letter(xpos=0.0, ypos=2.0)
        right.left_letter = left

        # Should handle vertical disjoint case
        result = LetterSpacing.find_transition_shift(right.left_letter, right)
        assert result <= -1.0  # Should be large negative (<= due to edge case protection)

    def test_edge_case_complex_shapes(self) -> None:
        """Test with complex non-rectangular shapes."""
        # Create an 'L' shape for the left letter
        l_points = np.array(
            [(0.0, 0.0), (1.0, 0.0), (1.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0.0, 1.0)],
            dtype=float,
        )
        l_cmds = ["M", "L", "L", "L", "L", "L", "Z"]
        l_path = AvPath(l_points, l_cmds)
        l_glyph = AvGlyph("L", 1.0, l_path)
        left = AvSingleGlyphLetter(l_glyph, xpos=0.0)

        # Create an inverted 'L' shape for the right letter
        inv_l_points = np.array(
            [(0.0, 0.5), (0.5, 0.5), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            dtype=float,
        )
        inv_l_cmds = ["M", "L", "L", "L", "L", "L", "Z"]
        inv_l_path = AvPath(inv_l_points, inv_l_cmds)
        inv_l_glyph = AvGlyph("7", 1.0, inv_l_path)
        right = AvSingleGlyphLetter(inv_l_glyph, xpos=0.6, ypos=0.01)
        right.left_letter = left

        result = LetterSpacing.find_transition_shift(right.left_letter, right)
        assert result == pytest.approx(-0.1, abs=1e-3)

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
        assert LetterSpacing.find_transition_shift(right.left_letter, right) == pytest.approx(-0.5, abs=1e-3)

    def test_multi_weight_letters(self) -> None:
        """Test with AvMultiWeightLetter."""
        left = _make_letter(xpos=0.0)
        heavy = _make_letter(xpos=1.5)
        light = _make_letter(xpos=0.75)
        multi = AvMultiWeightLetter([heavy, light])
        multi.left_letter = left

        assert LetterSpacing.find_transition_shift(multi.left_letter, multi) == pytest.approx(-0.5, abs=1e-3)

    def test_performance_optimizations(self) -> None:
        """Test that optimizations don't affect correctness."""
        # Create overlapping letters
        left = _make_letter(xpos=0.0)
        right = _make_letter(xpos=0.75)
        right.left_letter = left

        # The result should be the same regardless of optimizations
        result = LetterSpacing.find_transition_shift(right.left_letter, right)
        assert result == pytest.approx(0.25, abs=5e-3)

        # Test with letters that are far apart (tests bbox pre-filter)
        right_far = _make_letter(xpos=10.0)
        right_far.left_letter = left
        result_far = LetterSpacing.find_transition_shift(right_far.left_letter, right_far)
        assert result_far < -8.0

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

        # Create an inverted 'L' shape for the right letter
        # 1.0 x 1.0 total, missing bottom-left 0.5x0.5
        inv_l_points = np.array(
            [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.5), (0.5, 0.5)],
            dtype=float,
        )
        inv_l_cmds = ["M", "L", "L", "L", "L", "L", "Z"]
        inv_l_path = AvPath(inv_l_points, inv_l_cmds)
        inv_l_glyph = AvGlyph("7", 1.0, inv_l_path)

        # Left 'L' occupies x=[0, 1]. Right '7' occupies x=[0.6, 1.6].
        # Limiting gap is 0.1.

        # NOTE: We add a small y-offset to avoid exact overlap of horizontal segments
        # (L's shelf at y=0.5 and 7's shelf at y=0.5).
        right = AvSingleGlyphLetter(inv_l_glyph, xpos=0.6, ypos=0.01)
        right.left_letter = left

        # 0.6 (pos) - 0.5 (max extent of L at relevant height) = 0.1
        # With ypos=0.01, the horizontal clearance dominates.
        assert LetterSpacing.find_transition_shift(right.left_letter, right) == pytest.approx(-0.1, abs=1e-3)

    def test_alignment_affects_position(self) -> None:
        """Test that position affects bounding box."""
        # Glyph 1x1.
        # Rect from x=0.5 to x=1.5 (width=1.0, LSB=0.5)
        points = np.array([(0.5, 0.0), (1.5, 0.0), (1.5, 1.0), (0.5, 1.0)], dtype=float)
        commands = ["M", "L", "L", "L", "Z"]
        path = AvPath(points, commands)
        glyph = AvGlyph("x", 2.0, path)  # Advance width 2.0

        # Case 1: xpos=0.0
        # Shape is at [xpos+0.5, xpos+1.5] -> [0.5, 1.5].
        l1 = AvSingleGlyphLetter(glyph, xpos=0.0)

        # Case 2: xpos=-0.5
        # Shape is at [xpos+0.5, xpos+1.5] -> [0.0, 1.0].
        l2 = AvSingleGlyphLetter(glyph, xpos=-0.5)

        # Check position logic
        assert l1.bounding_box.xmin == pytest.approx(0.5)
        assert l2.bounding_box.xmin == pytest.approx(0.0)

        # Now test spacing between letters
        # l2 occupies [0, 1].
        # Place another rect at x=1.5.
        r = _make_letter(xpos=1.5)
        r.left_letter = l2

        # Gap = 1.5 - 1.0 = 0.5
        assert LetterSpacing.find_transition_shift(r.left_letter, r) == pytest.approx(-0.5, abs=1e-3)

    def test_vertically_overlapping(self) -> None:
        """Test letters that overlap vertically but are offset in Y."""
        # Left: 0.0 to 1.0 in Y.
        left = _make_letter(xpos=0.0, ypos=0.0)

        # Right: 0.5 to 1.5 in Y. Overlaps y=[0.5, 1.0].
        right = _make_letter(xpos=1.5, ypos=0.5)
        right.left_letter = left

        # Gap is still 1.5 - 1.0 = 0.5
        assert LetterSpacing.find_transition_shift(right.left_letter, right) == pytest.approx(-0.5, abs=1e-3)

    def test_vertically_disjoint(self) -> None:
        """Test letters that do not overlap vertically."""
        # Left: 0.0 to 1.0 in Y.
        left = _make_letter(xpos=0.0, ypos=0.0)

        # Right: 2.0 to 3.0 in Y. Completely above.
        right = _make_letter(xpos=0.0, ypos=2.0)
        right.left_letter = left

        # Since they don't overlap vertically, the 'left space' (collision distance)
        # is effectively infinite or limited by search bounds.
        space = LetterSpacing.find_transition_shift(right.left_letter, right)
        assert space <= -1.0

    def test_touching_letters(self) -> None:
        """Test letters that exactly touch."""
        left = _make_letter(xpos=0.0)
        # Right starts exactly where left ends (x=1.0)
        right = _make_letter(xpos=1.0)
        right.left_letter = left

        # Touching counts as intersection, so it might try to separate them.
        # The separation distance should be negligible.
        assert abs(LetterSpacing.find_transition_shift(right.left_letter, right)) <= 5e-3
