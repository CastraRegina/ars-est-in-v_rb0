from __future__ import annotations

import gzip
import io
import math
import os
import re
import sys
from enum import Enum, auto
from typing import Callable, ClassVar, Dict, List, Optional, Tuple

import matplotlib.path
import numpy
import shapely
import shapely.geometry
import shapely.wkt
import svgpath2mpl
import svgpathtools
import svgpathtools.path
import svgpathtools.paths2svg
import svgwrite
import svgwrite.base
import svgwrite.container
import svgwrite.elementfactory

from av.consts import (
    POLYGONIZE_ANGLE_MAX_DEG,
    POLYGONIZE_ANGLE_MAX_STEPS,
    POLYGONIZE_UNIFORM_NUM_POINTS,
)

# if __name__ == "__main__":
#     sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# POLYGONIZE_UNIFORM_NUM_POINTS = 10  # minimum 2 = (start, end)
# POLYGONIZE_ANGLE_MAX_DEG = 5  # 2 # difference of two derivatives less than
# POLYGONIZE_ANGLE_MAX_STEPS = 9  # 9
# POLYGONIZE_TYPE = Polygonize.BY_ANGLE


class AVsvgPath:
    SVG_CMDS: ClassVar[str] = "MmLlHhVvCcSsQqTtAaZz"
    SVG_ARGS: ClassVar[str] = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
    # Commands (number of values : command-character):
    #     MoveTo            2: Mm
    #     LineTo            2: Ll   1: Hh(x)   1:Vv(y)
    #     CubicBezier:      6: Cc   4: Ss
    #     QuadraticBezier:  4: Qq   2: Tt
    #     ArcCurve:         7: Aa
    #     ClosePath:        0: Zz

    @staticmethod
    def beautify_commands(path_string: str, round_func: Optional[Callable] = None) -> str:
        org_commands = re.findall(f"[{AVsvgPath.SVG_CMDS}][^{AVsvgPath.SVG_CMDS}]*", path_string)
        ret_commands = []
        for command in org_commands:
            command_letter = command[0]
            args = re.findall(AVsvgPath.SVG_ARGS, command[1:])
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
        org_commands = re.findall(f"[{AVsvgPath.SVG_CMDS}][^{AVsvgPath.SVG_CMDS}]*", path_string)
        ret_commands = []
        first_point = None  # Store the first point of each path (absolute)
        # Keep track of the last (iterating) point (absolute)
        last_point: list[float] = [0, 0]

        for command in org_commands:
            command_letter = command[0]
            args = re.findall(AVsvgPath.SVG_ARGS, command[1:])

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
                first_point = None

        ret_path_string = " ".join(ret_commands)
        return ret_path_string

    @staticmethod
    def transform_path_string(path_string: str, affine_trafo: List[float]) -> str:
        # Affine transform (see also shapely - Affine Transformations)
        #     affine_transform = [a00, a01, a10, a11, b0, b1]
        #       | x' | = | a00 a01 b0 |   | x |
        #       | y' | = | a10 a11 b1 | * | y |
        #       | 1  | = |  0   0  1  |   | 1 |

        def transform(x_str: str, y_str: str) -> Tuple[str, str]:
            x_new = (
                affine_trafo[0] * float(x_str) + affine_trafo[1] * float(y_str) + affine_trafo[4]
            )
            y_new = (
                affine_trafo[2] * float(x_str) + affine_trafo[3] * float(y_str) + affine_trafo[5]
            )
            return f"{x_new:g}", f"{y_new:g}"

        org_commands = re.findall(f"[{AVsvgPath.SVG_CMDS}][^{AVsvgPath.SVG_CMDS}]*", path_string)
        ret_commands = []

        for command in org_commands:
            command_letter = command[0]
            args = re.findall(AVsvgPath.SVG_ARGS, command[1:])

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


class AVPathPolygon:
    @staticmethod
    def multipolygon_to_path_string(
        multipolygon: shapely.geometry.MultiPolygon,
    ) -> List[str]:
        svg_string = multipolygon.svg()
        path_strings = re.findall(r'd="([^"]+)"', svg_string)
        return path_strings

    @staticmethod
    def deepcopy(
        geometry: shapely.geometry.base.BaseGeometry,
    ) -> shapely.geometry.base.BaseGeometry:
        return shapely.wkt.loads(geometry.wkt)

    @staticmethod
    def polygonize_uniform(segment, num_points: int = POLYGONIZE_UNIFORM_NUM_POINTS) -> str:
        # *segment* most likely of type QuadraticBezier or CubicBezier
        # create points ]start,...,end]
        ret_string = ""
        poly = segment.poly()
        points = [poly(i / (num_points - 1)) for i in range(1, num_points - 1)]
        for point in points:
            ret_string += f"L{point.real:g},{point.imag:g}"
        ret_string += f"L{segment.end.real:g},{segment.end.imag:g}"
        return ret_string

    @staticmethod
    def polygonize_by_angle(
        segment,
        max_angle_degree: float = POLYGONIZE_ANGLE_MAX_DEG,
        max_steps: int = POLYGONIZE_ANGLE_MAX_STEPS,
    ) -> str:
        # *segment* most likely of type QuadraticBezier or CubicBezier
        # create points ]start,...,end]
        params = [0, 0.5, 1]  # [0, 1/3, 0.5, 2/3, 1]
        points = [segment.point(t) for t in params]
        tangents = [segment.unit_tangent(t) for t in params]
        angle_limit = math.cos(max_angle_degree * math.pi / 180)

        for _ in range(1, max_steps):
            (new_params, new_points, new_tangents) = ([], [], [])
            updated = False
            for param, point, tangent in zip(params, points, tangents):
                if not new_points:  # nps is empty, i.e. first iteration
                    (new_params, new_points, new_tangents) = (
                        [param],
                        [point],
                        [tangent],
                    )
                else:
                    dot_product = (
                        new_tangents[-1].real * tangent.real + new_tangents[-1].imag * tangent.imag
                    )
                    if dot_product < angle_limit:
                        new_param = (new_params[-1] + param) / 2
                        new_params.append(new_param)
                        new_points.append(segment.point(new_param))
                        new_tangents.append(segment.unit_tangent(new_param))
                        updated = True
                    new_params.append(param)
                    new_points.append(point)
                    new_tangents.append(tangent)
            params = new_params
            points = new_points
            tangents = new_tangents
            if not updated:
                break
        ret_string = ""
        for point in points[1:]:
            ret_string += f"L{point.real:g},{point.imag:g}"
        return ret_string

    @staticmethod
    def polygonize_path(path_string: str, polygonize_segment_func: Callable) -> str:
        def moveto(coord: complex) -> str:
            return f"M{coord.real:g},{coord.imag:g}"

        def lineto(coord: complex) -> str:
            return f"L{coord.real:g},{coord.imag:g}"

        ret_path_string = ""
        path_collection = svgpathtools.parse_path(path_string)
        for sub_path in path_collection.continuous_subpaths():
            ret_path_string += moveto(sub_path.start)
            for segment in sub_path:
                if isinstance(segment, svgpathtools.CubicBezier) or isinstance(
                    segment, svgpathtools.QuadraticBezier
                ):
                    ret_path_string += polygonize_segment_func(segment)
                elif isinstance(segment, svgpathtools.Line):
                    ret_path_string += lineto(segment.end)
                else:
                    print("ERROR during polygonizing: " + "not supported segment: " + segment)
                    ret_path_string += lineto(segment.end)
            if sub_path.isclosed():
                ret_path_string += "Z "
        return ret_path_string

    @staticmethod
    def rect_to_path(rect: Tuple[float, float, float, float]) -> str:
        (x_pos, y_pos, width, height) = rect
        (x00, y00) = (x_pos, y_pos)
        (x10, y10) = (x_pos + width, y_pos)
        (x11, y11) = (x_pos + width, y_pos + height)
        (x01, y01) = (x_pos, y_pos + height)
        ret_path = (
            f"M{x00:g} {y00:g} "
            + f"L{x10:g} {y10:g} "
            + f"L{x11:g} {y11:g} "
            + f"L{x01:g} {y01:g} Z"
        )
        return ret_path

    @staticmethod
    def circle_to_path(
        x_pos: float,
        y_pos: float,
        radius: float,
        angle_degree: float = POLYGONIZE_ANGLE_MAX_DEG,
    ) -> str:
        ret_path = ""
        num_points = math.ceil(360 / angle_degree)
        for i in range(num_points):
            angle_rad = 2 * math.pi * i / num_points
            x_circ = x_pos + radius * math.sin(angle_rad)
            y_circ = y_pos + radius * math.cos(angle_rad)
            if i <= 0:
                ret_path += f"M{x_circ:g} {y_circ:g} "
            else:
                ret_path += f"L{x_circ:g} {y_circ:g} "
        ret_path += "Z"
        return ret_path

    def __init__(self, multipolygon: Optional[shapely.geometry.MultiPolygon] = None):
        self.multipolygon: shapely.geometry.MultiPolygon = shapely.geometry.MultiPolygon()
        if multipolygon:
            self.multipolygon = multipolygon

    def add_polygon_arrays(self, polygon_arrays: list[numpy.ndarray]):
        # first polygon_array is always additive.
        # All other arrays are additive, if same orientation like first array.
        first_is_ccw = True
        for index, polygon_array in enumerate(polygon_arrays):
            polygon = shapely.Polygon(polygon_array)
            polygon_ccw = polygon.exterior.is_ccw
            polygon = polygon.buffer(0)  # get rid of self-intersections (4,9)
            if index == 0:  # first array, so store its orientation
                first_is_ccw = polygon_ccw
            if self.multipolygon.is_empty:  # just add first polygon
                self.multipolygon = shapely.geometry.MultiPolygon([polygon])
            else:
                if polygon_ccw == first_is_ccw:  # same orient --> add to...
                    self.multipolygon = self.multipolygon.union(polygon)
                else:  # different orient --> substract from existing...
                    self.multipolygon = self.multipolygon.difference(polygon)

    def add_path_string(self, path_string: str):
        mpl_path: matplotlib.path.Path = svgpath2mpl.parse_path(path_string)
        polygon_arrays: numpy.ndarray = mpl_path.to_polygons()
        self.add_polygon_arrays(polygon_arrays)

    def path_strings(self) -> List[str]:
        return AVPathPolygon.multipolygon_to_path_string(self.multipolygon)

    def svg_paths(
        self, dwg: svgwrite.Drawing, **svg_properties
    ) -> List[svgwrite.elementfactory.ElementBuilder]:
        svg_paths = []
        path_strings = self.path_strings()
        for path_string in path_strings:
            svg_paths.append(dwg.path(path_string, **svg_properties))
        return svg_paths
