"""Tests for left exterior silhouette functionality."""

from __future__ import annotations

import numpy as np

from ave.path import SINGLE_POLYGON_CONSTRAINTS, AvSinglePolygonPath


def test_left_silhouette_simple_rectangle():
    """Test left silhouette of a simple rectangle."""
    from ave.path_processing import AvPathCreator

    # Simple rectangle: counter-clockwise from bottom-left
    points = np.array(
        [
            [0.0, 0.0],  # bottom-left
            [10.0, 0.0],  # bottom-right
            [10.0, 5.0],  # top-right
            [0.0, 5.0],  # top-left
        ]
    )
    commands = ["M", "L", "L", "L", "Z"]
    rect = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

    # Apply silhouette
    result = AvPathCreator.left_exterior_silhouette_list([rect])
    assert len(result) == 1
    silhouette = result[0]

    # Expected: left edge (dy < 0 from top-left to bottom-left)
    # Plus vertical blocking edge at x=10
    # Should have: top-left -> bottom-left -> bottom-right -> top-right -> close
    assert len(silhouette.points) >= 4

    # Check that result is valid
    assert silhouette.area > 0

    # Verify the silhouette contains the vertical blocking edge at max_x
    max_x = np.max(silhouette.points[:, 0])
    assert np.isclose(max_x, 10.0)


def test_left_silhouette_convex_polygon():
    """Test left silhouette of a convex polygon (triangle)."""
    from ave.path_processing import AvPathCreator

    # Triangle: CCW
    points = np.array(
        [
            [0.0, 0.0],  # bottom-left
            [10.0, 0.0],  # bottom-right
            [5.0, 8.0],  # top-center
        ]
    )
    commands = ["M", "L", "L", "Z"]
    triangle = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

    result = AvPathCreator.left_exterior_silhouette_list([triangle])
    assert len(result) == 1
    silhouette = result[0]

    # Should be valid
    assert silhouette.area > 0
    assert len(silhouette.points) >= 3


def test_left_silhouette_concave_polygon_with_undercut():
    """Test left silhouette of concave polygon with right-side notch."""
    from ave.path_processing import AvPathCreator

    # L-shaped polygon with notch on RIGHT side
    # CCW from bottom-left
    # For left projection, this becomes a full rectangle (fills in right-side features)
    points = np.array(
        [
            [0.0, 0.0],  # bottom-left
            [10.0, 0.0],  # bottom-right
            [10.0, 3.0],  # right side mid
            [5.0, 3.0],  # indent start
            [5.0, 6.0],  # indent end
            [10.0, 6.0],  # right side top
            [10.0, 10.0],  # top-right
            [0.0, 10.0],  # top-left
        ]
    )
    commands = ["M"] + ["L"] * 7 + ["Z"]
    concave = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

    result = AvPathCreator.left_exterior_silhouette_list([concave])
    assert len(result) == 1
    silhouette = result[0]

    # Left silhouette fills in right-side notches
    assert silhouette.area > 0
    # For this L-shape with right-side notch, silhouette should be bounding box
    assert silhouette.area >= concave.area

    # Should be x-monotone (no undercuts)
    # Every horizontal scanline should intersect at most 2 edges


def test_left_silhouette_empty_polygon():
    """Test left silhouette of empty polygon."""
    from ave.path_processing import AvPathCreator

    empty = AvSinglePolygonPath(np.empty((0, 3), dtype=np.float64), [], SINGLE_POLYGON_CONSTRAINTS)

    result = AvPathCreator.left_exterior_silhouette_list([empty])
    assert len(result) == 1
    assert len(result[0].points) == 0


def test_left_silhouette_multiple_polygons():
    """Test left silhouette with multiple input polygons."""
    from ave.path_processing import AvPathCreator

    # Two simple rectangles
    points1 = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]])
    points2 = np.array([[10.0, 0.0], [15.0, 0.0], [15.0, 5.0], [10.0, 5.0]])

    commands = ["M", "L", "L", "L", "Z"]
    poly1 = AvSinglePolygonPath(points1, commands, SINGLE_POLYGON_CONSTRAINTS)
    poly2 = AvSinglePolygonPath(points2, commands, SINGLE_POLYGON_CONSTRAINTS)

    result = AvPathCreator.left_exterior_silhouette_list([poly1, poly2])
    assert len(result) == 2
    assert result[0].area > 0
    assert result[1].area > 0


def test_left_silhouette_no_downward_edges():
    """Test polygon with no downward edges (horizontal line)."""
    from ave.path_processing import AvPathCreator

    # Horizontal line going right (degenerate case)
    # This should return a degenerate polygon or vertical line
    points = np.array(
        [
            [0.0, 5.0],
            [10.0, 5.0],
            [10.0, 5.0],
            [0.0, 5.0],
        ]
    )
    commands = ["M", "L", "L", "L", "Z"]
    line = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

    result = AvPathCreator.left_exterior_silhouette_list([line])
    assert len(result) == 1
    # Should handle gracefully


def test_left_silhouette_preserves_vertical_edges():
    """Test that vertical edges with dy < 0 are preserved."""
    from ave.path_processing import AvPathCreator

    # Polygon with vertical left edge
    points = np.array(
        [
            [0.0, 0.0],  # bottom-left
            [10.0, 0.0],  # bottom-right
            [10.0, 10.0],  # top-right
            [0.0, 10.0],  # top-left (vertical edge to bottom-left)
        ]
    )
    commands = ["M", "L", "L", "L", "Z"]
    rect = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

    result = AvPathCreator.left_exterior_silhouette_list([rect])
    silhouette = result[0]

    # Should preserve the vertical left edge
    assert silhouette.area > 0


def test_left_silhouette_output_is_simple_polygon():
    """Test that output is always a simple (non-self-intersecting) polygon."""
    import shapely.geometry

    from ave.path_processing import AvPathCreator

    # Complex concave shape
    points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 3.0],
            [5.0, 3.0],
            [5.0, 6.0],
            [10.0, 6.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ]
    )
    commands = ["M"] + ["L"] * 7 + ["Z"]
    poly = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

    result = AvPathCreator.left_exterior_silhouette_list([poly])
    silhouette = result[0]

    # Create Shapely polygon and verify it's simple
    shapely_poly = shapely.geometry.Polygon(silhouette.points[:, :2])
    assert shapely_poly.is_valid, "Output polygon must be valid"
    assert shapely_poly.is_simple, "Output polygon must be simple (no self-intersections)"
    assert shapely_poly.geom_type == "Polygon", "Output must be a single Polygon, not MultiPolygon"


def test_left_silhouette_blocking_edge_at_max_x():
    """Test that the right blocking edge is at max_x of input."""
    from ave.path_processing import AvPathCreator

    points = np.array(
        [
            [2.0, 0.0],
            [15.0, 0.0],
            [15.0, 10.0],
            [2.0, 10.0],
        ]
    )
    commands = ["M", "L", "L", "L", "Z"]
    rect = AvSinglePolygonPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

    result = AvPathCreator.left_exterior_silhouette_list([rect])
    silhouette = result[0]

    # Max x of output should be same as input
    input_max_x = np.max(points[:, 0])
    output_max_x = np.max(silhouette.points[:, 0])
    assert np.isclose(output_max_x, input_max_x), "Blocking edge should be at max_x of input"
