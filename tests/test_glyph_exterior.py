"""Tests for AvGlyph.exterior method."""

from __future__ import annotations

import numpy as np

from ave.glyph import AvGlyph
from ave.path import SINGLE_POLYGON_CONSTRAINTS, AvPath


def test_exterior_empty_path():
    """Test exterior with empty path."""
    glyph = AvGlyph(character="A", advance_width=100.0, path=AvPath())
    result = glyph.exterior(steps=10)
    assert not result


def test_exterior_simple_square():
    """Test exterior with simple square (no holes)."""
    points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
    commands = ["M", "L", "L", "L", "Z"]
    path = AvPath(points, commands)
    glyph = AvGlyph(character="A", advance_width=100.0, path=path)

    result = glyph.exterior(steps=10)

    assert len(result) == 1
    assert result[0].constraints == SINGLE_POLYGON_CONSTRAINTS
    assert len(result[0].points) >= 3
    assert result[0].commands[-1] == "Z"


def test_exterior_square_with_hole():
    """Test exterior with square containing a hole - hole should be removed."""
    outer_points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
    outer_commands = ["M", "L", "L", "L", "Z"]

    inner_points = np.array([[25, 25], [25, 75], [75, 75], [75, 25]], dtype=np.float64)
    inner_commands = ["M", "L", "L", "L", "Z"]

    all_points = np.vstack([outer_points, inner_points])
    all_commands = outer_commands + inner_commands

    path = AvPath(all_points, all_commands)
    glyph = AvGlyph(character="O", advance_width=100.0, path=path)

    result = glyph.exterior(steps=10)

    assert len(result) == 1
    assert result[0].constraints == SINGLE_POLYGON_CONSTRAINTS
    assert len(result[0].points) >= 3
    assert result[0].commands[-1] == "Z"


def test_exterior_two_separate_squares():
    """Test exterior with two separate squares - should return two polygons."""
    square1_points = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=np.float64)
    square1_commands = ["M", "L", "L", "L", "Z"]

    square2_points = np.array([[100, 100], [150, 100], [150, 150], [100, 150]], dtype=np.float64)
    square2_commands = ["M", "L", "L", "L", "Z"]

    all_points = np.vstack([square1_points, square2_points])
    all_commands = square1_commands + square2_commands

    path = AvPath(all_points, all_commands)
    glyph = AvGlyph(character="i", advance_width=200.0, path=path)

    result = glyph.exterior(steps=10)

    assert len(result) == 2
    for polygon in result:
        assert polygon.constraints == SINGLE_POLYGON_CONSTRAINTS
        assert len(polygon.points) >= 3
        assert polygon.commands[-1] == "Z"


def test_exterior_overlapping_squares():
    """Test exterior with overlapping squares - should merge into one polygon."""
    square1_points = np.array([[0, 0], [60, 0], [60, 60], [0, 60]], dtype=np.float64)
    square1_commands = ["M", "L", "L", "L", "Z"]

    square2_points = np.array([[40, 40], [100, 40], [100, 100], [40, 100]], dtype=np.float64)
    square2_commands = ["M", "L", "L", "L", "Z"]

    all_points = np.vstack([square1_points, square2_points])
    all_commands = square1_commands + square2_commands

    path = AvPath(all_points, all_commands)
    glyph = AvGlyph(character="X", advance_width=100.0, path=path)

    result = glyph.exterior(steps=10)

    assert len(result) == 1
    assert result[0].constraints == SINGLE_POLYGON_CONSTRAINTS
    assert len(result[0].points) >= 3
    assert result[0].commands[-1] == "Z"


def test_exterior_with_curves():
    """Test exterior with curved path - should polygonize curves."""
    points = np.array(
        [[0, 0], [50, -20], [100, 0], [120, 50], [100, 100], [50, 120], [0, 100], [-20, 50], [0, 0]], dtype=np.float64
    )
    commands = ["M", "Q", "Q", "Q", "Q", "Z"]

    path = AvPath(points, commands)
    glyph = AvGlyph(character="O", advance_width=100.0, path=path)

    result = glyph.exterior(steps=20)

    assert len(result) == 1
    assert result[0].constraints == SINGLE_POLYGON_CONSTRAINTS
    assert len(result[0].points) >= 3
    assert result[0].commands[-1] == "Z"
    assert all(cmd in ["M", "L", "Z"] for cmd in result[0].commands)
