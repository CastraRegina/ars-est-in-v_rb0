# file: serialize_example.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from ave.common import AvGlyphCmds
from ave.geom import AvBox


class Parent(ABC):
    """
    Abstract base class defining the glyph interface.
    """

    @property
    @abstractmethod
    def character(self) -> str:
        """Returns the character"""
        raise NotImplementedError

    @property
    @abstractmethod
    def points(self) -> NDArray[np.float64]:
        """Returns the points as an array of shape (n_points, 2)"""
        raise NotImplementedError

    @property
    @abstractmethod
    def commands(self) -> List[AvGlyphCmds]:
        """Returns the commands as a list of AvGlyphCmds"""
        raise NotImplementedError

    @property
    @abstractmethod
    def width(self) -> float:
        """Returns the width"""
        raise NotImplementedError


class SerializeExample(Parent):
    """
    Concrete class implementing the Parent interface.
    """

    _character: str
    _width: float
    _points: NDArray[np.float64]
    _commands: List[AvGlyphCmds]
    _bounding_box_cache: Optional[AvBox]

    def __init__(
        self,
        character: str,
        width: float,
        points: NDArray[np.float64],
        commands: List[AvGlyphCmds],
    ) -> None:
        # backing fields
        self._character = character
        self._width = width
        self._points = points
        self._commands = commands
        self._bounding_box_cache = None  # internal-only cache

        # essential validation (why: downstream geometry depends on correctness)
        if not isinstance(self._points, np.ndarray):
            raise TypeError("points must be a numpy ndarray")
        if self._points.ndim != 2 or self._points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        if not isinstance(self._commands, list):
            raise TypeError("commands must be a list")

    # ===== Required abstract property implementations =====

    @property
    def character(self) -> str:
        return self._character

    @property
    def points(self) -> NDArray[np.float64]:
        return self._points

    @property
    def commands(self) -> List[AvGlyphCmds]:
        return self._commands

    @property
    def width(self) -> float:
        return self._width

    # ===== Serialization =====

    def to_dict(self) -> dict:
        return {
            "character": self._character,
            "width": self._width,
            "points": self._points.tolist(),
            "commands": list(self._commands),
        }

    # ===== Debug representation =====

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return (
            f"{cls}(character={self._character!r}, "
            f"width={self._width}, "
            f"points=array(shape={self._points.shape}), "
            f"commands={self._commands})"
        )


if __name__ == "__main__":
    example = SerializeExample(
        "A",
        1.0,
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        ["M", "L", "Z"],
    )
    print(example)
    print("Serialized:", example.to_dict())
