"""Handling Paths for SVG"""

from __future__ import annotations

import re
from typing import Callable, ClassVar, List, Optional, Tuple


class AvSvgPath:
    """This class provides a collection of functions for manipulation of SVG-paths.
    A SVG-path is characterized by a string describing a sequence of points.
    The points' connection types are according to their commands.
    Commands (command : number of values : command-character):
        MoveTo:           2: Mm
        LineTo:           2: Ll   1: Hh(x)   1:Vv(y)
        CubicBezier:      6: Cc   4: Ss
        QuadraticBezier:  4: Qq   2: Tt
        ArcCurve:         7: Aa
        ClosePath:        0: Zz
    """

    # Command letters:
    SVG_CMDS: ClassVar[str] = "MmLlHhVvCcSsQqTtAaZz"
    # Definition of a number:
    SVG_ARGS: ClassVar[str] = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"

    @staticmethod
    def beautify_commands(path_string: str, round_func: Optional[Callable] = None) -> str:
        """Takes the given _path_string_ and rounds (mathematical) each point of the path
            by using the given _round_func_.
            If _round_func_ is None just a cast by "float()" is done.

        Args:
            path_string (str): _description_
            round_func (Optional[Callable], optional): _description_. Defaults to None.

        Returns:
            str: the beautified path_string
        """
        org_commands = re.findall(f"[{AvSvgPath.SVG_CMDS}][^{AvSvgPath.SVG_CMDS}]*", path_string)
        ret_commands = []
        for command in org_commands:
            command_letter = command[0]
            args = re.findall(AvSvgPath.SVG_ARGS, command[1:])
            batch_size = len(args)
            if command_letter in "MmLlTt":
                batch_size = 2
            elif command_letter in "SsQq":
                batch_size = 4
            elif command_letter in "Cc":
                batch_size = 6
            elif command_letter in "HhVv":
                batch_size = 1
            elif command_letter in "Aa":
                batch_size = 7

            if batch_size == 0:  # e.g. for command "Z"
                ret_commands.append(command_letter)
            else:
                for i, arg in enumerate(args):
                    if not i % batch_size:
                        ret_commands.append(command_letter)
                    if round_func:
                        ret_commands.append(f"{(round_func(float(arg))):g}")
                    else:
                        ret_commands.append(f"{(float(arg)):g}")

        ret_path_string = " ".join(ret_commands)
        return ret_path_string

    @staticmethod
    def convert_relative_to_absolute(path_string: str) -> str:
        """Take the give SVG _path_string_ and check the commands and coordinates.
        Relative coordinates will be converted into absolute ones,
        i.e. lower case letter commands will be replaced by upper case letter commands.
        The representation (i.e. geometry) of the path is still the same.
        Use this function before doing a transform.

        Args:
            path_string (str): SVG path string input

        Returns:
            str: path_string using absolute coordinates
        """
        org_commands = re.findall(f"[{AvSvgPath.SVG_CMDS}][^{AvSvgPath.SVG_CMDS}]*", path_string)
        ret_commands = []
        # Store the first point of each path (absolute):
        first_point: list[float] = []  # acts like None
        # Keep track of the last (iterating) point (absolute):
        last_point: list[float] = [0.0, 0.0]

        for command in org_commands:
            command_letter = command[0]
            args = re.findall(AvSvgPath.SVG_ARGS, command[1:])

            if command_letter.isupper():
                if command_letter in "MLCSQTA":
                    last_point = [float(args[-2]), float(args[-1])]
                elif command_letter in "H":
                    last_point[0] = float(args[-1])
                elif command_letter in "V":
                    last_point[1] = float(args[-1])
            else:
                if command_letter in "mlt":
                    for i in range(0, len(args), 2):
                        args[i + 0] = f"{(float(args[i+0]) + last_point[0]):g}"
                        args[i + 1] = f"{(float(args[i+1]) + last_point[1]):g}"
                        last_point = [float(args[i + 0]), float(args[i + 1])]
                elif command_letter in "sq":
                    for i in range(0, len(args), 4):
                        args[i + 0] = f"{(float(args[i+0]) + last_point[0]):g}"
                        args[i + 1] = f"{(float(args[i+1]) + last_point[1]):g}"
                        args[i + 2] = f"{(float(args[i+2]) + last_point[0]):g}"
                        args[i + 3] = f"{(float(args[i+3]) + last_point[1]):g}"
                        last_point = [float(args[i + 2]), float(args[i + 3])]
                elif command_letter in "c":
                    for i in range(0, len(args), 6):
                        args[i + 0] = f"{(float(args[i+0]) + last_point[0]):g}"
                        args[i + 1] = f"{(float(args[i+1]) + last_point[1]):g}"
                        args[i + 2] = f"{(float(args[i+2]) + last_point[0]):g}"
                        args[i + 3] = f"{(float(args[i+3]) + last_point[1]):g}"
                        args[i + 4] = f"{(float(args[i+4]) + last_point[0]):g}"
                        args[i + 5] = f"{(float(args[i+5]) + last_point[1]):g}"
                        last_point = [float(args[i + 4]), float(args[i + 5])]
                elif command_letter in "h":
                    for i, arg in enumerate(args):
                        args[i] = f"{(float(arg) + last_point[0]):g}"
                        last_point[0] = float(args[i])
                elif command_letter in "v":
                    for i, arg in enumerate(args):
                        args[i] = f"{(float(arg) + last_point[1]):g}"
                        last_point[1] = float(args[i])
                elif command_letter in "a":
                    for i in range(0, len(args), 7):
                        args[i + 5] = f"{(float(args[i+5]) + last_point[0]):g}"
                        args[i + 6] = f"{(float(args[i+6]) + last_point[1]):g}"
                        last_point = [float(args[i + 5]), float(args[i + 6])]

            ret_commands.append(command_letter.upper() + " ".join(args))

            if command_letter in "Mm" and not first_point:
                first_point = [float(args[0]), float(args[1])]
            if command_letter in "Zz":
                last_point = first_point
                first_point = []  # acts like None

        ret_path_string = " ".join(ret_commands)
        return ret_path_string

    @staticmethod
    def transform_path_string(path_string: str, affine_trafo: List[float]) -> str:
        """Transform the given SVG-_path_string_ by using the given _affine_trafo_.
        Make sure the _path_string_ uses absolute coordinates.

        Affine transform (see also shapely - Affine Transformations)
            affine_trafo = [a00, a01, a10, a11, b0, b1]
                | x' | = | a00 a01 b0 |   | x |
                | y' | = | a10 a11 b1 | * | y |
                | 1  | = |  0   0  1  |   | 1 |

        Args:
            path_string (str): SVG-path-string input
            affine_trafo (List[float]): Affine transformation

        Returns:
            str: the transformed _path_string_
        """

        def transform(x_str: str, y_str: str) -> Tuple[str, str]:
            x_new = affine_trafo[0] * float(x_str) + affine_trafo[1] * float(y_str) + affine_trafo[4]
            y_new = affine_trafo[2] * float(x_str) + affine_trafo[3] * float(y_str) + affine_trafo[5]
            return f"{x_new:g}", f"{y_new:g}"

        org_commands = re.findall(f"[{AvSvgPath.SVG_CMDS}][^{AvSvgPath.SVG_CMDS}]*", path_string)
        ret_commands = []

        for command in org_commands:
            command_letter = command[0]
            args = re.findall(AvSvgPath.SVG_ARGS, command[1:])

            if command_letter in "MLCSQT":  # (x,y) once or several times
                for i in range(0, len(args), 2):
                    (args[i + 0], args[i + 1]) = transform(args[i + 0], args[i + 1])
            elif command_letter in "H":  # (x) once or several times
                for i, _ in enumerate(args):
                    (args[i], _) = transform(args[i], "1")
            elif command_letter in "V":  # (y) once or several times
                for i, _ in enumerate(args):
                    (_, args[i]) = transform("1", args[i])
            elif command_letter in "A":  # (rx ry angle flag flag x y)+
                for i in range(0, len(args), 7):
                    args[i + 0] = f"{float(args[i+0])*affine_trafo[0]:g}"
                    args[i + 1] = f"{float(args[i+1])*affine_trafo[3]:g}"
                    (args[i + 5], args[i + 6]) = transform(args[i + 5], args[i + 6])
            ret_commands.append(command_letter.upper() + " ".join(args))

        ret_path_string = " ".join(ret_commands)
        return ret_path_string


if __name__ == "__main__":
    A_LIST_NONE = None
    a_list_empty = []
    a_list_filled = [0, 0]

    if A_LIST_NONE:  # false
        print("a_list_none true", A_LIST_NONE)
    else:
        print("a_list_none false", A_LIST_NONE)

    if a_list_empty:  # false
        print("a_list_empty true", a_list_empty)
    else:
        print("a_list_empty false", a_list_empty)

    if a_list_filled:  # true
        print("a_list_filled true", a_list_filled)
    else:
        print("a_list_filled false", a_list_filled)

    a_string_empty: str = ""
    a_string_filled: str = "toto"

    if a_string_empty:  # false
        print("a_string_empty true", a_string_empty)
    else:
        print("a_string_empty false", a_string_empty)

    if a_string_filled:  # true
        print("a_string_filled true", a_string_filled)
    else:
        print("a_string_filled false", a_string_filled)
