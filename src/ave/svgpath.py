"""Handling Paths for SVG"""

from __future__ import annotations

import re
from typing import Callable, ClassVar, Optional, Sequence, Tuple, Union

# Type-Definition for SvgPath-Commands; uppercase = absolute coordinates; lowercase = relative.
# SvgPathCmds = Literal[
#     # MoveTo (2) - start a new subpath and move the current point to (x,y)
#     "M",
#     "m",
#     # LineTo (2) - draw a straight line from the current point to (x,y)
#     "L",
#     "l",
#     # Horizontal LineTo (1) - draw a horizontal line to the given x coordinate (y stays unchanged)
#     "H",
#     "h",
#     # Vertical LineTo (1) - draw a vertical line to the given y coordinate (x stays unchanged)
#     "V",
#     "v",
#     # Cubic Bezier To (6) - draw a cubic Bézier curve with two control points and an endpoint (x,y)
#     "C",
#     "c",
#     # Smooth cubic Bezier To (4) - Cubic curve to (x,y) using the reflection of the previous cubic control point
#     "S",
#     "s",
#     # Quadratic Bezier To (4) - draw a quadratic Bézier curve with one control point and an endpoint (x,y)
#     "Q",
#     "q",
#     # Smooth quadratic Bezier To (2) - Quadratic curve to (x,y) using the reflection of the previous control point
#     "T",
#     "t",
#     # Arc (7) - draw an elliptical arc with parameters (rx ry x-axis-rotation large-arc-flag sweep-flag x y)
#     "A",
#     "a",
#     # ClosePath (0) - close subpath by drawing a line from the current point to start point
#     "Z",
#     "z",
# ]


class AvSvgPath:
    """
    This class provides a collection of static methods for manipulation of SVG-paths.
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
        """
        Takes the given _path_string_ and rounds (mathematical) each point of the path
            by using the given _round_func_.
            If _round_func_ is None just a cast by "float()" is done.

        Args:
            path_string (str): a SVG path string
            round_func (Optional[Callable], optional):
                a function that takes a float and returns a float. Defaults to None.

        Returns:
            str: the beautified path_string
        """
        # Find all commands (with their arguments) in the given path string
        org_commands = re.findall(f"[{AvSvgPath.SVG_CMDS}][^{AvSvgPath.SVG_CMDS}]*", path_string)
        # Initialize the list of commands to be returned
        ret_commands = []
        # Iterate over all commands
        for command in org_commands:
            # Determine the letter of the command
            command_letter = command[0]
            # Determine the arguments of the command
            args = re.findall(AvSvgPath.SVG_ARGS, command[1:])
            # Determine the batch size of the command
            batch_size = len(args)
            # Special cases for the commands
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

            # If the command has no arguments (like "Z"), just append the command letter
            if batch_size == 0:
                ret_commands.append(command_letter)
            else:
                # Iterate over all arguments in batches of the batch size
                for i, arg in enumerate(args):
                    # If it is the first argument of a batch, append the command letter
                    if not i % batch_size:
                        ret_commands.append(command_letter)
                    # Append the argument (rounded by the round_func if given)
                    if round_func:
                        ret_commands.append(f"{(round_func(float(arg))):g}")
                    else:
                        ret_commands.append(f"{(float(arg)):g}")

        # Join the commands to a string and return
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
    def transform_path_string(path_string: str, affine_trafo: Sequence[Union[int, float]]) -> str:
        """Transform the given SVG-_path_string_ by using the given _affine_trafo_.
        Make sure the _path_string_ uses absolute coordinates.

        The given _affine_trafo_ is a list of 6 floats, performing an affine transformation.
        The transformation is defined as:
            | x' | = | a00 a01 b0 |   | x |
            | y' | = | a10 a11 b1 | * | y |
            | 1  | = |  0   0  1  |   | 1 |
        with
            affine_trafo = [a00, a01, a10, a11, b0, b1]
        See also shapely - Affine Transformations

        Args:
            path_string (str): SVG-path-string input
            affine_trafo (List[float]): Affine transformation

        Returns:
            str: the transformed _path_string_
        """

        def transform(x_str: str, y_str: str) -> Tuple[str, str]:
            """Perform an affine transformation on the given (x,y) point.

            Args:
                x_str (str): x-coordinate of the point
                y_str (str): y-coordinate of the point

            Returns:
                Tuple[str, str]: the transformed (x,y) point
            """
            # Perform the transformation
            x_new = affine_trafo[0] * float(x_str) + affine_trafo[1] * float(y_str) + affine_trafo[4]
            y_new = affine_trafo[2] * float(x_str) + affine_trafo[3] * float(y_str) + affine_trafo[5]
            # Round the result to the number of decimal places given by the SVG standard
            return f"{x_new:g}", f"{y_new:g}"

        # Split the path string into commands
        org_commands = re.findall(f"[{AvSvgPath.SVG_CMDS}][^{AvSvgPath.SVG_CMDS}]*", path_string)
        ret_commands = []

        # Iterate over the commands
        for command in org_commands:
            command_letter = command[0]
            args = re.findall(AvSvgPath.SVG_ARGS, command[1:])

            # Check the type of command
            if command_letter in "MLCSQT":  # (x,y) once or several times
                # Iterate over the arguments
                for i in range(0, len(args), 2):
                    # Perform the transformation
                    (args[i + 0], args[i + 1]) = transform(args[i + 0], args[i + 1])
            elif command_letter in "H":  # (x) once or several times
                # Iterate over the arguments
                for i, _ in enumerate(args):
                    # Perform the transformation
                    (args[i], _) = transform(args[i], "1")
            elif command_letter in "V":  # (y) once or several times
                # Iterate over the arguments
                for i, _ in enumerate(args):
                    # Perform the transformation
                    (_, args[i]) = transform("1", args[i])
            elif command_letter in "A":  # (rx ry angle flag flag x y)+
                # Iterate over the arguments
                for i in range(0, len(args), 7):
                    # Perform the transformation
                    args[i + 0] = f"{float(args[i+0])*affine_trafo[0]:g}"
                    args[i + 1] = f"{float(args[i+1])*affine_trafo[3]:g}"
                    (args[i + 5], args[i + 6]) = transform(args[i + 5], args[i + 6])
            # Append the transformed command to the output
            ret_commands.append(command_letter.upper() + " ".join(args))

        # Join the commands together
        ret_path_string = " ".join(ret_commands)
        return ret_path_string


# TODO: add rect_to_path function here: Provide a rectangle as SVG path (polygon)
# TODO: add circle_to_path function here: Provide a circle as SVG path (polygon)

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

    print("------------------------------------------------------------------")
    print(AvSvgPath.beautify_commands("M 1 2 L 3 4"))
    print(AvSvgPath.beautify_commands("M1 2 L3 4"))
    print(AvSvgPath.beautify_commands("M 1.0 2.0 L 3.0 4.0"))
