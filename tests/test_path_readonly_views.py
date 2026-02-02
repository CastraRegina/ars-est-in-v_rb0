"""Tests for read-only view enforcement on AvPath properties."""

from __future__ import annotations

import numpy as np
import pytest

from ave.path import AvPath
from ave.path_support import SINGLE_POLYGON_CONSTRAINTS


class TestPointsReadOnlyView:
    """Test that points property returns read-only view."""

    def test_points_returns_view_not_copy(self):
        """Verify points returns a view (shares memory with internal array)."""
        points = np.array([[0, 0, 0], [10, 0, 1], [10, 10, 1], [0, 10, 1]], dtype=np.float64)
        commands = ["M", "L", "L", "L", "Z"]
        path = AvPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        view = path.points
        # View should share memory with internal array
        assert view.base is not None or np.shares_memory(view, path._points)

    def test_points_is_read_only(self):
        """Verify points array cannot be modified."""
        points = np.array([[0, 0, 0], [10, 0, 1], [10, 10, 1], [0, 10, 1]], dtype=np.float64)
        commands = ["M", "L", "L", "L", "Z"]
        path = AvPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        view = path.points
        assert not view.flags.writeable

    def test_modifying_points_raises_error(self):
        """Verify attempting to modify points raises ValueError."""
        points = np.array([[0, 0, 0], [10, 0, 1], [10, 10, 1], [0, 10, 1]], dtype=np.float64)
        commands = ["M", "L", "L", "L", "Z"]
        path = AvPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        view = path.points
        with pytest.raises(ValueError, match="assignment destination is read-only"):
            view[0, 0] = 999.0

    def test_modifying_points_slice_raises_error(self):
        """Verify attempting to modify points slice raises ValueError."""
        points = np.array([[0, 0, 0], [10, 0, 1], [10, 10, 1], [0, 10, 1]], dtype=np.float64)
        commands = ["M", "L", "L", "L", "Z"]
        path = AvPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        view = path.points
        with pytest.raises(ValueError, match="assignment destination is read-only"):
            view[:, 0] = 0.0

    def test_read_operations_work_normally(self):
        """Verify read operations on points work as expected."""
        points = np.array([[0, 0, 0], [10, 0, 1], [10, 10, 1], [0, 10, 1]], dtype=np.float64)
        commands = ["M", "L", "L", "L", "Z"]
        path = AvPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        view = path.points

        # All read operations should work
        assert view.shape == (4, 3)
        assert view[0, 0] == 0.0
        assert view[1, 0] == 10.0
        assert np.allclose(view[:, 0], [0, 10, 10, 0])
        assert view.mean() == pytest.approx(3.583333, abs=1e-6)

    def test_copying_points_allows_modification(self):
        """Verify that explicitly copying points allows modification."""
        points = np.array([[0, 0, 0], [10, 0, 1], [10, 10, 1], [0, 10, 1]], dtype=np.float64)
        commands = ["M", "L", "L", "L", "Z"]
        path = AvPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        # Explicit copy should be writable
        copy = path.points.copy()
        assert copy.flags.writeable
        copy[0, 0] = 999.0
        assert copy[0, 0] == 999.0

        # Original should be unchanged
        assert path.points[0, 0] == 0.0

    def test_multiple_accesses_return_consistent_views(self):
        """Verify multiple accesses return consistent read-only views."""
        points = np.array([[0, 0, 0], [10, 0, 1], [10, 10, 1], [0, 10, 1]], dtype=np.float64)
        commands = ["M", "L", "L", "L", "Z"]
        path = AvPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        view1 = path.points
        view2 = path.points

        # Both should be read-only
        assert not view1.flags.writeable
        assert not view2.flags.writeable

        # Both should have same data
        assert np.array_equal(view1, view2)


class TestCommandsDirectReference:
    """Test that commands property returns direct reference for performance."""

    def test_commands_returns_list(self):
        """Verify commands returns a list as expected by API."""
        points = np.array([[0, 0, 0], [10, 0, 1], [10, 10, 1], [0, 10, 1]], dtype=np.float64)
        commands = ["M", "L", "L", "L", "Z"]
        path = AvPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = path.commands
        assert isinstance(result, list)
        assert result == ["M", "L", "L", "L", "Z"]

    def test_read_operations_work_normally(self):
        """Verify read operations on commands work as expected."""
        points = np.array([[0, 0, 0], [10, 0, 1], [10, 10, 1], [0, 10, 1]], dtype=np.float64)
        commands = ["M", "L", "L", "L", "Z"]
        path = AvPath(points, commands, SINGLE_POLYGON_CONSTRAINTS)

        result = path.commands
        assert len(result) == 5
        assert result[0] == "M"
        assert result[-1] == "Z"
        assert "L" in result
        assert result.count("L") == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
