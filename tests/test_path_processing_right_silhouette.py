"""Tests for right exterior silhouette functionality."""

from __future__ import annotations

import numpy as np
import shapely.geometry

from ave.path import SINGLE_POLYGON_CONSTRAINTS, AvSinglePolygonPath
from ave.path_exterior import AvPathExterior


def _rotate_180(coords: np.ndarray) -> np.ndarray:
    rotated = coords.copy()
    rotated[:, 0] *= -1.0
    rotated[:, 1] *= -1.0
    return rotated


def test_right_silhouette_simple_rectangle() -> None:
    """Test right silhouette of a simple rectangle."""
    points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 5.0],
            [0.0, 5.0],
        ]
    )
    commands = ["M", "L", "L", "L", "Z"]
    rect = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

    result = AvPathExterior.right_exterior_silhouette_list([rect])
    assert len(result) == 1
    silhouette = result[0]

    assert len(silhouette.points) >= 4
    assert silhouette.area > 0

    min_x = np.min(silhouette.points[:, 0])
    assert np.isclose(min_x, 0.0)

    shapely_poly = shapely.geometry.Polygon(silhouette.points[:, :2])
    assert shapely_poly.is_valid
    assert shapely_poly.is_simple


def test_right_silhouette_convex_polygon() -> None:
    """Test right silhouette of a convex polygon."""
    points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [5.0, 8.0],
        ]
    )
    commands = ["M", "L", "L", "Z"]
    triangle = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

    result = AvPathExterior.right_exterior_silhouette_list([triangle])
    assert len(result) == 1
    silhouette = result[0]

    assert silhouette.area > 0
    assert len(silhouette.points) >= 3


def test_right_silhouette_concave_polygon_with_left_notch() -> None:
    """Test right silhouette of a concave polygon with a left-side notch."""
    points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
            [0.0, 7.0],
            [5.0, 7.0],
            [5.0, 3.0],
            [0.0, 3.0],
        ]
    )
    commands = ["M"] + ["L"] * 7 + ["Z"]
    concave = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

    result = AvPathExterior.right_exterior_silhouette_list([concave])
    assert len(result) == 1
    silhouette = result[0]

    assert silhouette.area > 0
    assert silhouette.area >= concave.area

    min_x = np.min(silhouette.points[:, 0])
    assert np.isclose(min_x, 0.0)


def test_right_silhouette_matches_rotated_left_silhouette() -> None:
    """Test right silhouette through 180-degree left-silhouette symmetry."""
    points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 3.0],
            [7.0, 3.0],
            [7.0, 6.0],
            [10.0, 6.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ]
    )
    commands = ["M"] + ["L"] * 7 + ["Z"]
    polygon = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

    right_result = AvPathExterior.right_exterior_silhouette(polygon)

    rotated_polygon = AvSinglePolygonPath(
        _rotate_180(polygon.points),
        commands,
        SINGLE_POLYGON_CONSTRAINTS,
    )
    rotated_left = AvPathExterior.left_exterior_silhouette(rotated_polygon)
    expected_points = _rotate_180(rotated_left.points)

    assert np.allclose(right_result.points, expected_points)
    assert right_result.commands == rotated_left.commands
    assert right_result.area > 0
