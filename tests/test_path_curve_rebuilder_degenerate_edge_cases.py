"""Additional edge case tests for AvPathCurveRebuilder degenerate curve handling."""

from __future__ import annotations

import numpy as np

from ave.path import AvPath
from ave.path_processing import AvPathCurveRebuilder
from ave.path_support import PathCommandProcessor


def validate_path_consistency(path: AvPath):
    """Validate that points and commands are consistent."""
    point_idx = 0

    for cmd in path.commands:
        consumed = PathCommandProcessor.get_point_consumption(cmd)

        if cmd == "Z":
            # Z commands don't consume points
            continue

        # Check we have enough points
        assert point_idx + consumed <= len(path.points), (
            f"Not enough points for {cmd} command at index {point_idx}. "
            f"Need {consumed}, have {len(path.points) - point_idx}"
        )

        point_idx += consumed

    # All points should be consumed
    assert point_idx == len(path.points), (
        f"Points left over after processing commands. " f"Consumed {point_idx}, have {len(path.points)}"
    )


class TestDegenerateCurveEdgeCases:
    """Test edge cases for degenerate curve detection and fixing."""

    def test_q_without_preceding_m(self):
        """Test Q command without preceding M point."""
        # Create path directly with Q (no M)
        points = np.array(
            [
                [0.0, 0.0, 2.0],  # Q - control
                [10.0, 0.0, 0.0],  # Q - end
            ]
        )
        commands = ["Q"]
        # Use internal constructor to bypass validation
        path = AvPath.__new__(AvPath)
        path._points = points  # pylint: disable=protected-access
        path._commands = commands  # pylint: disable=protected-access
        path._constraints = None  # pylint: disable=protected-access
        path._bounding_box = None  # pylint: disable=protected-access
        path._polygonized_path = None  # pylint: disable=protected-access

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)  # pylint: disable=protected-access

        # Should be empty (Q without start is skipped)
        assert len(result.points) == 0
        assert len(result.commands) == 0
        validate_path_consistency(result)

    def test_c_without_preceding_m(self):
        """Test C command without preceding M point."""
        # Create path directly with C (no M)
        points = np.array(
            [
                [3.0, 10.0, 3.0],  # C - ctrl1
                [7.0, 10.0, 3.0],  # C - ctrl2
                [10.0, 0.0, 0.0],  # C - end
            ]
        )
        commands = ["C"]
        # Use internal constructor to bypass validation
        path = AvPath.__new__(AvPath)
        path._points = points  # pylint: disable=protected-access
        path._commands = commands  # pylint: disable=protected-access
        path._constraints = None  # pylint: disable=protected-access
        path._bounding_box = None  # pylint: disable=protected-access
        path._polygonized_path = None  # pylint: disable=protected-access

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)  # pylint: disable=protected-access

        # Should be empty (C without start is skipped)
        assert len(result.points) == 0
        assert len(result.commands) == 0
        validate_path_consistency(result)

    def test_consecutive_degenerate_quadratic_curves(self):
        """Test multiple consecutive degenerate Q curves."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [0.0, 1e-15, 2.0],  # Q1 - control (degenerate)
                [10.0, 0.0, 0.0],  # Q1 - end
                [10.0, 1e-15, 2.0],  # Q2 - control (degenerate)
                [20.0, 0.0, 0.0],  # Q2 - end
            ]
        )
        commands = ["M", "Q", "Q"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)  # pylint: disable=protected-access

        # Both Q should be replaced with L
        assert result.commands == ["M", "L", "L"]
        assert len(result.points) == 3
        validate_path_consistency(result)

    def test_cubic_ctrl1_ctrl2_both_coincide_with_start(self):
        """Test C where both ctrl1 and ctrl2 coincide with start point."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [0.0, 1e-15, 3.0],  # C - ctrl1 (coincides with start)
                [0.0, 2e-15, 3.0],  # C - ctrl2 (also coincides with start)
                [10.0, 0.0, 0.0],  # C - end
            ]
        )
        commands = ["M", "C"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)  # pylint: disable=protected-access

        # Should be simplified to Q with start point as control
        assert result.commands == ["M", "Q"]
        assert len(result.points) == 3
        validate_path_consistency(result)

    def test_cubic_ctrl1_ctrl2_both_coincide_with_end(self):
        """Test C where both ctrl1 and ctrl2 coincide with end point."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [10.0, 1e-15, 3.0],  # C - ctrl1 (coincides with end)
                [10.0, 2e-15, 3.0],  # C - ctrl2 (also coincides with end)
                [10.0, 0.0, 0.0],  # C - end
            ]
        )
        commands = ["M", "C"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)  # pylint: disable=protected-access

        # Should be simplified to Q with end point as control
        assert result.commands == ["M", "Q"]
        assert len(result.points) == 3
        validate_path_consistency(result)

    def test_mixed_valid_and_degenerate_curves(self):
        """Test path with mix of valid and degenerate curves."""
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [5.0, 10.0, 2.0],  # Q1 - control (valid)
                [10.0, 0.0, 0.0],  # Q1 - end
                [10.0, 1e-15, 2.0],  # Q2 - control (degenerate)
                [15.0, 0.0, 0.0],  # Q2 - end
                [15.0, 0.0, 3.0],  # C - ctrl1 (degenerate)
                [20.0, 10.0, 3.0],  # C - ctrl2
                [25.0, 0.0, 0.0],  # C - end
                [25.0, 5.0, 3.0],  # C2 - ctrl1 (valid)
                [30.0, 10.0, 3.0],  # C2 - ctrl2
                [35.0, 0.0, 0.0],  # C2 - end
            ]
        )
        commands = ["M", "Q", "Q", "C", "C"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)  # pylint: disable=protected-access

        # Q1 valid, Q2→L, C→Q, C2 valid
        assert result.commands == ["M", "Q", "L", "Q", "C"]
        validate_path_consistency(result)

    def test_path_consistency_after_all_transformations(self):
        """Test that all transformations maintain valid path structure."""
        test_cases = [
            # (description, points, commands, expected_commands)
            ("Q ctrl=start", [[0, 0, 0], [0, 1e-15, 2], [10, 0, 0]], ["M", "Q"], ["M", "L"]),
            ("Q ctrl=end", [[0, 0, 0], [10, 1e-15, 2], [10, 0, 0]], ["M", "Q"], ["M", "L"]),
            ("Q start=end", [[0, 0, 0], [5, 10, 2], [0, 1e-15, 0]], ["M", "Q"], ["M"]),
            ("Q collinear", [[0, 0, 0], [5, 0, 2], [10, 0, 0]], ["M", "Q"], ["M", "L"]),
            ("C ctrl1=start", [[0, 0, 0], [0, 1e-15, 3], [7, 10, 3], [10, 0, 0]], ["M", "C"], ["M", "Q"]),
            ("C ctrl2=end", [[0, 0, 0], [3, 10, 3], [10, 1e-15, 3], [10, 0, 0]], ["M", "C"], ["M", "Q"]),
            ("C both degenerate", [[0, 0, 0], [0, 1e-15, 3], [10, 1e-15, 3], [10, 0, 0]], ["M", "C"], ["M", "L"]),
            ("C ctrl1=ctrl2", [[0, 0, 0], [5, 10, 3], [5, 10 + 1e-15, 3], [10, 0, 0]], ["M", "C"], ["M", "Q"]),
            ("C start=end", [[0, 0, 0], [3, 10, 3], [7, 10, 3], [0, 1e-15, 0]], ["M", "C"], ["M"]),
            ("C collinear", [[0, 0, 0], [3, 0, 3], [7, 0, 3], [10, 0, 0]], ["M", "C"], ["M", "L"]),
        ]

        for desc, points, commands, expected_cmds in test_cases:
            path = AvPath(np.array(points, dtype=float), commands)
            result = AvPathCurveRebuilder._fix_degenerate_curves(path)  # pylint: disable=protected-access

            assert result.commands == expected_cmds, f"Failed: {desc}"
            validate_path_consistency(result)

    def test_point_types_preserved_correctly(self):
        """Test that point types are preserved correctly in fixed paths."""
        # Test degenerate Q that gets replaced with L
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - type 0
                [0.0, 1e-15, 2.0],  # Q - control (type 2, degenerate)
                [10.0, 0.0, 0.0],  # Q - end (type 0)
            ]
        )
        commands = ["M", "Q"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)  # pylint: disable=protected-access

        # After fixing degenerate Q, should have M and L with type 0
        assert len(result.points) == 2
        assert result.points[0, 2] == 0.0  # M point type
        assert result.points[1, 2] == 0.0  # L point type

    def test_tolerance_edge_cases(self):
        """Test behavior at tolerance boundaries."""
        # Exactly at tolerance - should be considered degenerate
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [1e-9, 0.0, 2.0],  # Q - control (at tolerance)
                [10.0, 0.0, 0.0],  # Q - end
            ]
        )
        commands = ["M", "Q"]
        path = AvPath(points, commands)

        result = AvPathCurveRebuilder._fix_degenerate_curves(path)  # pylint: disable=protected-access

        # Should be replaced with L since distance <= tolerance
        assert result.commands == ["M", "L"]

        # Further above tolerance and not collinear
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # M - start
                [1e-6, 5.0, 2.0],  # Q - control (well above tolerance and not collinear)
                [10.0, 0.0, 0.0],  # Q - end
            ]
        )
        path = AvPath(points, commands)
        result = AvPathCurveRebuilder._fix_degenerate_curves(path)  # pylint: disable=protected-access

        # Should remain Q since distance > tolerance and not collinear
        assert result.commands == ["M", "Q"]
