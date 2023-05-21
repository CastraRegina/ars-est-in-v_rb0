from typing import Dict, List, Tuple, Callable, ClassVar
from enum import Enum, auto
import io
import gzip
import re
import math
import numpy
import svgwrite
import svgwrite.base
import svgwrite.container
import svgwrite.elementfactory
from svgwrite.extensions import Inkscape
import shapely
import shapely.geometry
import shapely.wkt
import svgpathtools
import svgpathtools.path
import svgpathtools.paths2svg
import svgpath2mpl
import matplotlib.path
from fontTools.ttLib import TTFont
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.varLib import instancer
# from fontTools.pens.transformPen import TransformPen


INPUT_FILE_LOREM_IPSUM = "data/input/example/txt/Lorem_ipsum_10000.txt"
OUTPUT_FILE = "data/output/example/svg/din_a4_page_rectangle_and_text.svg"

CANVAS_UNIT = "mm"  # Units for CANVAS dimensions
CANVAS_WIDTH = 210  # DIN A4 page width in mm
CANVAS_HEIGHT = 297  # DIN A4 page height in mm

RECT_WIDTH = 150  # rectangle width in mm
RECT_HEIGHT = 150  # rectangle height in mm

VB_RATIO = 1 / RECT_WIDTH  # multiply each dimension with this ratio

FONT_FILENAME = "fonts/RobotoFlex-VariableFont_GRAD,XTRA," + \
                "YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght.ttf"
# FONT_FILENAME = "fonts/Recursive-VariableFont_CASL,CRSV,MONO,slnt,wght.ttf"
# FONT_FILENAME = "fonts/NotoSansMono-VariableFont_wdth,wght.ttf"

FONT_SIZE = VB_RATIO * 3  # in mm


class Polygonize(Enum):
    BY_ANGLE = auto()
    UNIFORM = auto()


class Align(Enum):
    LEFT = auto()
    RIGHT = auto()
    BOTH = auto()


POLYGONIZE_UNIFORM_NUM_POINTS = 10  # minimum 2 = (start, end)
POLYGONIZE_ANGLE_MAX_DEG = 5  # 2 # difference of two derivatives less than
POLYGONIZE_ANGLE_MAX_STEPS = 9  # 9
POLYGONIZE_TYPE = Polygonize.BY_ANGLE


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
    def beautify_commands(path_string: str,
                          round_func: Callable = None) -> str:
        org_commands = re.findall(
            f'[{AVsvgPath.SVG_CMDS}][^{AVsvgPath.SVG_CMDS}]*', path_string)
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
                    if not (i % batch_size):
                        ret_commands.append(command_letter)
                    if round_func:
                        ret_commands.append(f'{(round_func(float(arg))):g}')
                    else:
                        ret_commands.append(f'{(float(arg)):g}')

        ret_path_string = ' '.join(ret_commands)
        return ret_path_string

    @staticmethod
    def convert_relative_to_absolute(path_string: str) -> str:
        org_commands = re.findall(
            f'[{AVsvgPath.SVG_CMDS}][^{AVsvgPath.SVG_CMDS}]*', path_string)
        ret_commands = []
        first_point = None  # Store the first point of each path (absolute)
        # Keep track of the last (iterating) point (absolute)
        last_point: list[float] = [0, 0]

        for command in org_commands:
            command_letter = command[0]
            args = re.findall(AVsvgPath.SVG_ARGS, command[1:])

            if command_letter.isupper():
                if command_letter in 'MLCSQTA':
                    last_point = [float(args[-2]), float(args[-1])]
                elif command_letter in 'H':
                    last_point[0] = float(args[-1])
                elif command_letter in 'V':
                    last_point[1] = float(args[-1])
            else:
                if command_letter in "mlt":
                    for i in range(0, len(args), 2):
                        args[i+0] = f'{(float(args[i+0]) + last_point[0]):g}'
                        args[i+1] = f'{(float(args[i+1]) + last_point[1]):g}'
                        last_point = [float(args[i+0]), float(args[i+1])]
                elif command_letter in "sq":
                    for i in range(0, len(args), 4):
                        args[i+0] = f'{(float(args[i+0]) + last_point[0]):g}'
                        args[i+1] = f'{(float(args[i+1]) + last_point[1]):g}'
                        args[i+2] = f'{(float(args[i+2]) + last_point[0]):g}'
                        args[i+3] = f'{(float(args[i+3]) + last_point[1]):g}'
                        last_point = [float(args[i+2]), float(args[i+3])]
                elif command_letter in "c":
                    for i in range(0, len(args), 6):
                        args[i+0] = f'{(float(args[i+0]) + last_point[0]):g}'
                        args[i+1] = f'{(float(args[i+1]) + last_point[1]):g}'
                        args[i+2] = f'{(float(args[i+2]) + last_point[0]):g}'
                        args[i+3] = f'{(float(args[i+3]) + last_point[1]):g}'
                        args[i+4] = f'{(float(args[i+4]) + last_point[0]):g}'
                        args[i+5] = f'{(float(args[i+5]) + last_point[1]):g}'
                        last_point = [float(args[i+4]), float(args[i+5])]
                elif command_letter in "h":
                    for i, arg in enumerate(args):
                        args[i] = f'{(float(arg) + last_point[0]):g}'
                        last_point[0] = float(args[i])
                elif command_letter in "v":
                    for i, arg in enumerate(args):
                        args[i] = f'{(float(arg) + last_point[1]):g}'
                        last_point[1] = float(args[i])
                elif command_letter in "a":
                    for i in range(0, len(args), 7):
                        args[i+5] = f'{(float(args[i+5]) + last_point[0]):g}'
                        args[i+6] = f'{(float(args[i+6]) + last_point[1]):g}'
                        last_point = [float(args[i+5]), float(args[i+6])]

            ret_commands.append(command_letter.upper() + ' '.join(args))

            if command_letter in 'Mm' and not first_point:
                first_point = [float(args[0]), float(args[1])]
            if command_letter in 'Zz':
                last_point = first_point
                first_point = None

        ret_path_string = ' '.join(ret_commands)
        return ret_path_string

    @staticmethod
    def transform_path_string(path_string: str,
                              affine_trafo: List[float]) -> str:
        # Affine transform (see also shapely - Affine Transformations)
        #     affine_transform = [a00, a01, a10, a11, b0, b1]
        #       | x' | = | a00 a01 b0 |   | x |
        #       | y' | = | a10 a11 b1 | * | y |
        #       | 1  | = |  0   0  1  |   | 1 |

        def transform(x_str: str, y_str: str) -> Tuple[str, str]:
            x_new = affine_trafo[0] * float(x_str) + \
                affine_trafo[1] * float(y_str) + \
                affine_trafo[4]
            y_new = affine_trafo[2] * float(x_str) + \
                affine_trafo[3] * float(y_str) + \
                affine_trafo[5]
            return f'{x_new:g}', f'{y_new:g}'

        org_commands = re.findall(
            f'[{AVsvgPath.SVG_CMDS}][^{AVsvgPath.SVG_CMDS}]*', path_string)
        ret_commands = []

        for command in org_commands:
            command_letter = command[0]
            args = re.findall(AVsvgPath.SVG_ARGS, command[1:])

            if command_letter in 'MLCSQT':  # (x,y) once or several times
                for i in range(0, len(args), 2):
                    (args[i+0], args[i+1]) = transform(args[i+0], args[i+1])
            elif command_letter in 'H':  # (x) once or several times
                for i, _ in enumerate(args):
                    (args[i], _) = transform(args[i], 1)
            elif command_letter in 'V':  # (y) once or several times
                for i, _ in enumerate(args):
                    (_, args[i]) = transform(1, args[i])
            elif command_letter in 'A':  # (rx ry angle flag flag x y)+
                for i in range(0, len(args), 7):
                    args[i+0] = f'{float(args[i+0])*affine_trafo[0]:g}'
                    args[i+1] = f'{float(args[i+1])*affine_trafo[3]:g}'
                    (args[i+5], args[i+6]) = transform(args[i+5], args[i+6])
            ret_commands.append(command_letter.upper() + ' '.join(args))

        ret_path_string = ' '.join(ret_commands)
        return ret_path_string


class AVGlyph:  # pylint: disable=function-redefined
    @staticmethod
    def svg_rect(dwg: svgwrite.Drawing,
                 rect: Tuple[float, float, float, float],
                 stroke: str, stroke_width: float, **svg_properties) \
            -> svgwrite.elementfactory.ElementBuilder:
        pass

    @staticmethod
    def polygonize_path_string(path_string: str) -> str:
        pass

    def real_width(self, font_size: float, align: Align = None) -> float:
        pass

    def real_dash_thickness(self, font_size: float) -> float:
        pass

    def real_sidebearing_left(self, font_size: float) -> float:
        pass

    def real_sidebearing_right(self, font_size: float) -> float:
        pass

    def real_path_string(self, x_pos: float, y_pos: float,
                         font_size: float) -> str:
        pass

    def svg_path(self, dwg: svgwrite.Drawing,
                 x_pos: float, y_pos: float,
                 font_size: float, **svg_properties) \
            -> svgwrite.elementfactory.ElementBuilder:
        pass

    def svg_text(self, dwg: svgwrite.Drawing,
                 x_pos: float, y_pos: float,
                 font_size: float, **svg_properties) \
            -> svgwrite.elementfactory.ElementBuilder:
        pass

    def rect_em(self, x_pos: float, y_pos: float,
                ascent: float, descent: float,
                real_width: float, font_size: float) \
            -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        pass

    def rect_em_width(self, x_pos: float, y_pos: float,
                      ascent: float, descent: float,
                      font_size: float) \
            -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        pass

    def rect_given_ascent_descent(self, x_pos: float, y_pos: float,
                                  ascent: float, descent: float,
                                  font_size: float) \
            -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        pass

    def rect_font_ascent_descent(self, x_pos: float, y_pos: float,
                                 font_size: float) \
            -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        pass

    def rect_bounding_box(self, x_pos: float, y_pos: float, font_size: float) \
            -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        pass


class AVFont:
    def __init__(self, ttfont: TTFont):
        # ttfont is already configured with the given axes_values
        self.ttfont = ttfont
        self.ascender = self.ttfont['hhea'].ascender  # in unitsPerEm
        self.descender = self.ttfont['hhea'].descender  # in unitsPerEm
        self.line_gap = self.ttfont['hhea'].lineGap  # in unitsPerEm
        self.x_height = self.ttfont["OS/2"].sxHeight  # in unitsPerEm
        self.cap_height = self.ttfont["OS/2"].sCapHeight  # in unitsPerEm
        self.units_per_em = self.ttfont['head'].unitsPerEm
        self.family_name = self.ttfont['name'].getDebugName(1)
        self.subfamily_name = self.ttfont['name'].getDebugName(2)
        self.full_name = self.ttfont['name'].getDebugName(4)
        self.license_description = self.ttfont['name'].getDebugName(13)
        self._glyph_cache: Dict[str, AVGlyph] = {}  # character->AVGlyph

    # def real_ascender(self, font_size: float) -> float:
    #     return self.ascender * font_size / self.units_per_em

    # def real_descender(self, font_size: float) -> float:
    #     return self.descender * font_size / self.units_per_em

    # def real_line_gap(self, font_size: float) -> float:
    #     return self.line_gap * font_size / self.units_per_em

    # def real_x_height(self, font_size: float) -> float:
    #     return self.x_height * font_size / self.units_per_em

    # def real_cap_height(self, font_size: float) -> float:
    #     return self.cap_height * font_size / self.units_per_em

    def glyph(self, character: str) -> AVGlyph:
        glyph = self._glyph_cache.get(character, None)
        if not glyph:
            glyph = AVGlyph(self, character)
            self._glyph_cache[character] = glyph
        return glyph

    def glyph_ascent_descent_of(self, characters: str) -> Tuple[float, float]:
        (ascent, descent) = (0.0, 0.0)
        for char in characters:
            if bounding_box := self.glyph(char).bounding_box:
                (_, descent, _, ascent) = bounding_box
                break
        for char in characters:
            if bounding_box := self.glyph(char).bounding_box:
                (_, y_min, _, y_max) = bounding_box
                ascent = max(ascent, y_max)
                descent = min(descent, y_min)
        return (ascent, descent)

    # def real_dash_thickness(self, font_size: float) -> float:
    #     glyph = self.glyph("-")
    #     if glyph.bounding_box:
    #         thickness = glyph.bounding_box[3] - glyph.bounding_box[1]
    #         return thickness * font_size / self.units_per_em
    #     return 0.0

    @staticmethod
    def default_axes_values(ttfont: TTFont) -> Dict[str, float]:
        axes_values: Dict[str, float] = {}
        for axis in ttfont['fvar'].axes:
            axes_values[axis.axisTag] = axis.defaultValue
        return axes_values

    @staticmethod
    def real_value(ttfont: TTFont, font_size: float, value: float) -> float:
        units_per_em = ttfont['head'].unitsPerEm
        return value * font_size / units_per_em


class AVPathPolygon:
    @staticmethod
    def multipolygon_to_path_string(multipolygon: shapely.geometry.MultiPolygon
                                    ) -> List[str]:
        svg_string = multipolygon.svg()
        path_strings = re.findall(r'd="([^"]+)"', svg_string)
        return path_strings

    @staticmethod
    def deepcopy(geometry: shapely.geometry.base.BaseGeometry) \
            -> shapely.geometry.base.BaseGeometry:
        return shapely.wkt.loads(geometry.wkt)

    @staticmethod
    def polygonize_uniform(segment,
                           num_points: int =
                           POLYGONIZE_UNIFORM_NUM_POINTS) -> str:
        # *segment* most likely of type QuadraticBezier or CubicBezier
        # create points ]start,...,end]
        ret_string = ""
        poly = segment.poly()
        points = [poly(i/(num_points-1)) for i in range(1, num_points-1)]
        for point in points:
            ret_string += f'L{point.real:g},{point.imag:g}'
        ret_string += f'L{segment.end.real:g},{segment.end.imag:g}'
        return ret_string

    @staticmethod
    def polygonize_by_angle(segment,
                            max_angle_degree: float =
                            POLYGONIZE_ANGLE_MAX_DEG,
                            max_steps: int =
                            POLYGONIZE_ANGLE_MAX_STEPS) -> str:
        # *segment* most likely of type QuadraticBezier or CubicBezier
        # create points ]start,...,end]
        params = [0, 0.5, 1]  # [0, 1/3, 0.5, 2/3, 1]
        points = [segment.point(t) for t in params]
        tangents = [segment.unit_tangent(t) for t in params]
        angle_limit = math.cos(max_angle_degree * math.pi/180)

        for _ in range(1, max_steps):
            (new_params, new_points, new_tangents) = ([], [], [])
            updated = False
            for (param, point, tangent) in zip(params, points, tangents):
                if not new_points:  # nps is empty, i.e. first iteration
                    (new_params, new_points, new_tangents) = (
                        [param], [point], [tangent])
                else:
                    dot_product = new_tangents[-1].real*tangent.real + \
                        new_tangents[-1].imag*tangent.imag
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
            ret_string += f'L{point.real:g},{point.imag:g}'
        return ret_string

    @staticmethod
    def polygonize_path(path_string: str,
                        polygonize_segment_func: Callable) -> str:
        def moveto(coord: complex) -> str:
            return f'M{coord.real:g},{coord.imag:g}'

        def lineto(coord: complex) -> str:
            return f'L{coord.real:g},{coord.imag:g}'

        ret_path_string = ""
        path_collection = svgpathtools.parse_path(path_string)
        for sub_path in path_collection.continuous_subpaths():
            ret_path_string += moveto(sub_path.start)
            for segment in sub_path:
                if isinstance(segment, svgpathtools.CubicBezier) or \
                        isinstance(segment, svgpathtools.QuadraticBezier):
                    ret_path_string += polygonize_segment_func(segment)
                elif isinstance(segment, svgpathtools.Line):
                    ret_path_string += lineto(segment.end)
                else:
                    print("ERROR during polygonizing: " +
                          "not supported segment: " + segment)
                    ret_path_string += lineto(segment.end)
            if sub_path.isclosed():
                ret_path_string += "Z "
        return ret_path_string

    @staticmethod
    def rect_to_path(rect: Tuple[float, float, float, float]) -> str:
        (x_pos, y_pos, width, height) = rect
        (x00, y00) = (x_pos, y_pos)
        (x10, y10) = (x_pos+width, y_pos)
        (x11, y11) = (x_pos+width, y_pos+height)
        (x01, y01) = (x_pos, y_pos+height)
        ret_path = f"M{x00:g} {y00:g} " + \
                   f"L{x10:g} {y10:g} " + \
                   f"L{x11:g} {y11:g} " + \
                   f"L{x01:g} {y01:g} Z"
        return ret_path

    @staticmethod
    def circle_to_path(x_pos: float, y_pos: float, radius: float,
                       angle_degree: float = POLYGONIZE_ANGLE_MAX_DEG) -> str:
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

    def __init__(self, multipolygon: shapely.geometry.MultiPolygon = None):
        self.multipolygon: shapely.geometry.MultiPolygon = \
            shapely.geometry.MultiPolygon()
        if multipolygon:
            self.multipolygon = multipolygon

    def add_polygon_arrays(self, polygon_arrays: list[numpy.ndarray]):
        # first polygon_array is always additive.
        # All other arrays are additive, if same orient like first array.
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

    def svg_paths(self, dwg: svgwrite.Drawing, **svg_properties) \
            -> List[svgwrite.elementfactory.ElementBuilder]:
        svg_paths = []
        path_strings = self.path_strings()
        for path_string in path_strings:
            svg_paths.append(dwg.path(path_string, **svg_properties))
        return svg_paths


class AVGlyph:  # pylint: disable=function-redefined
    @staticmethod
    def svg_rect(dwg: svgwrite.Drawing,
                 rect: Tuple[float, float, float, float],
                 stroke: str, stroke_width: float, **svg_properties) \
            -> svgwrite.elementfactory.ElementBuilder:
        (x_pos, y_pos, width, height) = rect
        rect_properties = {"insert": (x_pos, y_pos),
                           "size": (width, height),
                           "stroke": stroke,  # color
                           "stroke_width": stroke_width,
                           "fill": "none"}
        rect_properties.update(svg_properties)
        return dwg.rect(**rect_properties)

    @staticmethod
    def polygonize_path_string(path_string: str) -> str:
        if not path_string:
            path_string = "M 0 0"
        else:
            polygon = AVPathPolygon()
            poly_func = None
            match POLYGONIZE_TYPE:
                case Polygonize.UNIFORM:
                    poly_func = AVPathPolygon.polygonize_uniform
                case Polygonize.BY_ANGLE:
                    poly_func = AVPathPolygon.polygonize_by_angle
            path_string = AVPathPolygon.polygonize_path(path_string, poly_func)

            polygon.add_path_string(path_string)
            path_strings = polygon.path_strings()
            path_string = " ".join(path_strings)
        return path_string

    def __init__(self, avfont: AVFont, character: str):
        self._avfont: AVFont = avfont
        self.character: str = character
        bounds_pen = BoundsPen(self._avfont.ttfont.getGlyphSet())
        glyph_name = self._avfont.ttfont.getBestCmap()[ord(character)]
        self._glyph_set = self._avfont.ttfont.getGlyphSet()[glyph_name]
        self._glyph_set.draw(bounds_pen)
        self.bounding_box = bounds_pen.bounds  # (x_min, y_min, x_max, y_max)
        self.width = self._glyph_set.width
        # create and store a polygonized_path_string:
        svg_pen = SVGPathPen(self._avfont.ttfont.getGlyphSet())
        self._glyph_set.draw(svg_pen)
        self.path_string = svg_pen.getCommands()
        self.polygonized_path_string = \
            AVGlyph.polygonize_path_string(self.path_string)

    def real_width(self, font_size: float, align: Align = None) -> float:
        real_width = self.width * font_size / self._avfont.units_per_em
        if not align:
            return real_width
        (bb_x_pos, _, bb_width, _) = self.rect_bounding_box(0, 0, font_size)

        if align == Align.LEFT:
            return real_width - bb_x_pos
        elif align == Align.RIGHT:
            return bb_x_pos + bb_width
        elif align == Align.BOTH:
            return bb_width
        else:
            print("ERROR in real_width(): align-value not implemented", align)
            return real_width

    def real_dash_thickness(self, font_size: float) -> float:
        glyph = self._avfont.glyph("-")
        if glyph.bounding_box:
            thickness = glyph.bounding_box[3] - glyph.bounding_box[1]
            return thickness * font_size / self._avfont.units_per_em
        return 0.0

    def real_sidebearing_left(self, font_size: float) -> float:
        if self.bounding_box:
            return self.bounding_box[0] * font_size / self._avfont.units_per_em
        return 0.0

    def real_sidebearing_right(self, font_size: float) -> float:
        if self.bounding_box:
            sidebearing_right = self.width - self.bounding_box[2]
            return sidebearing_right * font_size / self._avfont.units_per_em
        return 0.0

    def real_path_string(self, x_pos: float, y_pos: float,
                         font_size: float) -> str:
        scale = font_size / self._avfont.units_per_em
        path_string = AVsvgPath.transform_path_string(
            self.polygonized_path_string,
            (scale, 0, 0, -scale, x_pos, y_pos))
        return path_string

    def svg_path(self, dwg: svgwrite.Drawing,
                 x_pos: float, y_pos: float,
                 font_size: float, **svg_properties) \
            -> svgwrite.elementfactory.ElementBuilder:
        path_string = self.real_path_string(x_pos, y_pos, font_size)
        svg_path = dwg.path(path_string, **svg_properties)
        return svg_path

    def svg_text(self, dwg: svgwrite.Drawing,
                 x_pos: float, y_pos: float,
                 font_size: float, **svg_properties) \
            -> svgwrite.elementfactory.ElementBuilder:
        text_properties = {"insert": (x_pos, y_pos),
                           "font_family": self._avfont.family_name,
                           "font_size": font_size}
        text_properties.update(svg_properties)
        ret_text = dwg.text(self.character, **text_properties)
        return ret_text

    def rect_em(self, x_pos: float, y_pos: float,
                ascent: float, descent: float,
                real_width: float, font_size: float) \
            -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        units_per_em = self._avfont.units_per_em
        middle_of_em = 0.5 * (ascent + descent) * font_size / units_per_em

        rect = (x_pos,
                y_pos - middle_of_em - 0.5 * font_size,
                real_width,
                font_size)
        return rect

    def rect_em_width(self, x_pos: float, y_pos: float,
                      ascent: float, descent: float,
                      font_size: float) \
            -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        return self.rect_em(x_pos, y_pos, ascent, descent,
                            self.real_width(font_size), font_size)

    def rect_given_ascent_descent(self, x_pos: float, y_pos: float,
                                  ascent: float, descent: float,
                                  font_size: float) \
            -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        units_per_em = self._avfont.units_per_em
        rect = (x_pos,
                y_pos - ascent * font_size / units_per_em,
                self.real_width(font_size),
                font_size - descent * font_size / units_per_em)
        return rect

    def rect_font_ascent_descent(self, x_pos: float, y_pos: float,
                                 font_size: float) \
            -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        ascent = self._avfont.ascender
        descent = self._avfont.descender
        return self.rect_given_ascent_descent(x_pos, y_pos,
                                              ascent, descent,
                                              font_size)

    def rect_bounding_box(self, x_pos: float, y_pos: float, font_size: float) \
            -> Tuple[float, float, float, float]:
        # returns (x_pos_left_corner, y_pos_top_corner, width, height)
        rect = (0.0, 0.0, 0.0, 0.0)
        if self.bounding_box:
            units_per_em = self._avfont.units_per_em
            (x_min, y_min, x_max, y_max) = self.bounding_box
            rect = (x_pos + x_min * font_size / units_per_em,
                    y_pos - y_max * font_size / units_per_em,
                    (x_max - x_min) * font_size / units_per_em,
                    (y_max - y_min) * font_size / units_per_em)
        return rect

    def area_coverage(self, ascent: float, descent: float,
                      font_size: float) -> float:
        glyph_string = self.real_path_string(0, 0, font_size)
        glyph_polygon = AVPathPolygon()
        glyph_polygon.add_path_string(glyph_string)

        rect = self.rect_em_width(0, 0, ascent, descent, font_size)
        rect_string = AVPathPolygon.rect_to_path(rect)
        rect_polygon = AVPathPolygon()
        rect_polygon.add_path_string(rect_string)

        inter = rect_polygon.multipolygon.intersection(
            glyph_polygon.multipolygon)
        rect_area = rect_polygon.multipolygon.area

        return inter.area / rect_area


class SVGoutput:
    def __init__(self,
                 canvas_width_mm: float, canvas_height_mm: float,
                 viewbox_x: float, viewbox_y: float,
                 viewbox_width: float, viewbox_height: float):
        self.drawing: svgwrite.Drawing = svgwrite.Drawing(
            size=(f"{canvas_width_mm}mm", f"{canvas_height_mm}mm"),
            viewBox=(f"{viewbox_x} {viewbox_y} " +
                     f"{viewbox_width} {viewbox_height}"),
            profile='full')
        self._inkscape: Inkscape = Inkscape(self.drawing)
        # main   -- editable->locked=False  --  hidden->display="block"
        # debug  -- editable->locked=False  --  hidden->display="none"
        #    glyph
        #       bounding_box
        #       em_width
        #       font_ascent_descent
        #       sidebearing
        #    background
        self.layer_debug: \
            svgwrite.container.Group = self._inkscape.layer(
                label="Layer debug", locked=False, display="none")
        self.drawing.add(self.layer_debug)

        self.layer_debug_glyph_background: \
            svgwrite.container.Group = self._inkscape.layer(
                label="Layer background", locked=True)
        self.layer_debug.add(self.layer_debug_glyph_background)

        self.layer_debug_glyph: \
            svgwrite.container.Group = self._inkscape.layer(
                label="Layer glyph", locked=True)
        self.layer_debug.add(self.layer_debug_glyph)

        self.layer_debug_glyph_sidebearing: \
            svgwrite.container.Group = self._inkscape.layer(
                label="Layer sidebearing", locked=True)  # yellow, orange
        self.layer_debug_glyph.add(self.layer_debug_glyph_sidebearing)

        self.layer_debug_glyph_font_ascent_descent: \
            svgwrite.container.Group = self._inkscape.layer(
                label="Layer font_ascent_descent", locked=True)  # green
        self.layer_debug_glyph.add(self.layer_debug_glyph_font_ascent_descent)

        self.layer_debug_glyph_em_width: \
            svgwrite.container.Group = self._inkscape.layer(
                label="Layer em_width", locked=True)  # blue
        self.layer_debug_glyph.add(self.layer_debug_glyph_em_width)

        self.layer_debug_glyph_bounding_box: \
            svgwrite.container.Group = self._inkscape.layer(
                label="Layer bounding_box", locked=True)  # red
        self.layer_debug_glyph.add(self.layer_debug_glyph_bounding_box)

        self.layer_main: \
            svgwrite.container.Group = self._inkscape.layer(
                label="Layer main", locked=False)
        self.drawing.add(self.layer_main)

    def draw_path(self, path_string: str, **svg_properties) \
            -> svgwrite.elementfactory.ElementBuilder:
        return self.drawing.path(path_string, **svg_properties)

    def saveas(self, filename: str, pretty: bool = False, indent: int = 2,
               compressed: bool = False):
        svg_buffer = io.StringIO()
        self.drawing.write(svg_buffer, pretty=pretty, indent=indent)
        output_data = svg_buffer.getvalue().encode('utf-8')
        if compressed:
            output_data = gzip.compress(output_data)
        with open(filename, 'wb') as svg_file:
            svg_file.write(output_data)

    def add_glyph_sidebearing(self, glyph: AVGlyph,
                              x_pos: float, y_pos: float, font_size: float):
        sb_left = glyph.real_sidebearing_left(font_size)
        sb_right = glyph.real_sidebearing_right(font_size)

        rect_bb = glyph.rect_bounding_box(x_pos, y_pos, font_size)
        rect = (x_pos, rect_bb[1], sb_left, rect_bb[3])
        self.layer_debug_glyph_sidebearing.add(
            AVGlyph.svg_rect(self.drawing, rect, "none", 0, fill="yellow"))

        rect = (x_pos + glyph.real_width(font_size) - sb_right,
                rect_bb[1], sb_right, rect_bb[3])
        self.layer_debug_glyph_sidebearing.add(
            AVGlyph.svg_rect(self.drawing, rect, "none", 0, fill="orange"))

    def add_glyph_font_ascent_descent(self, glyph: AVGlyph,
                                      x_pos: float, y_pos: float,
                                      font_size: float):
        stroke_width = glyph.real_dash_thickness(font_size)
        rect = glyph.rect_font_ascent_descent(x_pos, y_pos, font_size)
        self.layer_debug_glyph_font_ascent_descent.add(
            AVGlyph.svg_rect(self.drawing, rect, "green", 0.3*stroke_width))

    def add_glyph_em_width(self, glyph: AVGlyph, x_pos: float, y_pos: float,
                           font_size: float, ascent: float, descent: float):
        stroke_width = glyph.real_dash_thickness(font_size)
        rect = glyph.rect_em_width(x_pos, y_pos, ascent, descent, font_size)
        self.layer_debug_glyph_em_width.add(
            AVGlyph.svg_rect(self.drawing, rect, "blue", 0.2*stroke_width))

    def add_glyph_bounding_box(self, glyph: AVGlyph,
                               x_pos: float, y_pos: float, font_size: float):
        stroke_width = glyph.real_dash_thickness(font_size)
        rect = glyph.rect_bounding_box(x_pos, y_pos, font_size)
        self.layer_debug_glyph_bounding_box.add(
            AVGlyph.svg_rect(self.drawing, rect, "red", 0.1*stroke_width))

    def add_glyph(self, glyph: AVGlyph,
                  x_pos: float, y_pos: float, font_size: float,
                  ascent: float = None, descent: float = None):
        if ascent and descent:
            self.add_glyph_em_width(glyph, x_pos, y_pos,
                                    font_size, ascent, descent)
        self.add_glyph_sidebearing(glyph, x_pos, y_pos, font_size)
        self.add_glyph_font_ascent_descent(glyph, x_pos, y_pos, font_size)
        self.add_glyph_bounding_box(glyph, x_pos, y_pos, font_size)
        self.add(glyph.svg_path(self.drawing, x_pos, y_pos, font_size))

    def add(self, element: svgwrite.base.BaseElement):
        return self.layer_main.add(element)


class Potpourri:
    @staticmethod
    def print_glyph_coverage(avfont: AVFont, ascent: float, descent: float,
                             font_size: float, text: str) -> None:
        # how much of the space (width*(ascent+descent)) is covered by glyph?
        for character in text:
            glyph = avfont.glyph(character)
            area_ratio = glyph.area_coverage(ascent, descent, font_size)
            print(character, area_ratio)

    @staticmethod
    def print_glyph_number_of_paths(avfont: AVFont, text: str) -> None:
        # which glyphs are constructed using several paths (add & sub)?
        for character in text:
            glyph = avfont.glyph(character)
            glyph_path_string = glyph.real_path_string(0, 0, 1)
            parsed_path = svgpathtools.parse_path(glyph_path_string)
            num_parsed_sub_paths = len(parsed_path.continuous_subpaths())
            if num_parsed_sub_paths > 1:
                areas = [p.area() for p in parsed_path.continuous_subpaths()]
                areas = [f"{(a):+04.2f}" for a in areas]
                print(f"{character:1} : {num_parsed_sub_paths:2} - {areas}")


class SimpleLineLayouter:
    def __init__(self, svg_output: SVGoutput,
                 avfont: AVFont, font_size: float):
        self.svg_output = svg_output
        self.avfont = avfont
        self.font_size = font_size

    def end_pos_text(self, x_left: float, text: str) -> float:
        x_pos = x_left
        for index, character in enumerate(text):
            if character.isspace():
                character = " "
            glyph = self.avfont.glyph(character)
            if not index:  # first character:
                x_pos += glyph.real_width(self.font_size, Align.LEFT)
            elif index < len(text)-1:  # "middle" character:
                x_pos += glyph.real_width(self.font_size)
            else:  # last character:
                x_pos += glyph.real_width(self.font_size, Align.RIGHT)
        return x_pos

    def layout_line(self, y_pos: float, x_left: float, x_right: float,
                    text: str) -> str:
        line_text = text
        ret_text = ""
        index_last_space = 0
        for index, character in enumerate(text):
            if character.isspace():
                if self.end_pos_text(x_left, text[:index]) > x_right:
                    line_text = text[:index_last_space].rstrip()
                    ret_text = text[index_last_space:].lstrip()
                    break
                index_last_space = index
        line_end_pos = self.end_pos_text(x_left, line_text)
        each_delta = 0
        if ret_text:
            each_delta = (x_right - line_end_pos) / (len(line_text)-1)
        # print(f"line_text:  _{line_text}_ _{line_end_pos}_<_{x_right}_")
        # print(f"ret_text:   _{ret_text}_")
        # print(f"delta= _{(x_right - line_end_pos)}_")

        # do the layout:
        x_pos = x_left
        for index, character in enumerate(line_text):
            glyph = self.avfont.glyph(character)
            if not index:  # first character:
                x_sb = glyph.real_sidebearing_left(self.font_size)
                self.svg_output.add_glyph(
                    glyph, x_pos - x_sb, y_pos, self.font_size)
                x_pos += glyph.real_width(self.font_size, Align.LEFT)
            else:
                self.svg_output.add_glyph(
                    glyph, x_pos, y_pos, self.font_size)
                x_pos += glyph.real_width(self.font_size)
            x_pos += each_delta
        # circ_path = AVPathPolygon.circle_to_path(x_pos, y_pos,
        #                                          0.5*VB_RATIO, 2)
        # svg_path = self.svg_output.draw_path(circ_path, stroke="red",
        #                                      stroke_width=0.01 * VB_RATIO,
        #                                      fill="blue")
        # self.svg_output.add(svg_path)

        return ret_text


def main():
    # Center the rectangle horizontally and vertically on the page
    vb_w = VB_RATIO * CANVAS_WIDTH
    vb_h = VB_RATIO * CANVAS_HEIGHT
    vb_x = -VB_RATIO * (CANVAS_WIDTH - RECT_WIDTH) / 2
    vb_y = -VB_RATIO * (CANVAS_HEIGHT - RECT_HEIGHT) / 2

    # Set up the SVG canvas:
    # Define viewBox so that "1" is the width of the rectangle
    # Multiply a dimension with "VB_RATIO" to get the size regarding viewBox
    svg_output = SVGoutput(CANVAS_WIDTH, CANVAS_HEIGHT, vb_x, vb_y, vb_w, vb_h)
    # Draw the rectangle
    svg_output.add(
        svg_output.drawing.rect(
            insert=(0, 0),
            size=(VB_RATIO*RECT_WIDTH, VB_RATIO*RECT_HEIGHT),  # = (1.0, xxxx)
            stroke="black",
            stroke_width=0.1*VB_RATIO,
            fill="none"
        )
    )

    ttfont = TTFont(FONT_FILENAME)
    avfont = AVFont(ttfont)

    # x_pos = VB_RATIO * 10  # in mm
    # y_pos = VB_RATIO * 10  # in mm

    # text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ " + \
    #        "abcdefghijklmnopqrstuvwxyz " + \
    #        "ÄÖÜ äöü ß€µ@²³~^°\\ 1234567890 " + \
    #        ',.;:+-*#_<> !"§$%&/()=?{}[]'

    # (ascent, descent) = avfont.glyph_ascent_descent_of(
    #     "ABCDEFGHIJKLMNOPQRSTUVWXYZ " +
    #     "abcdefghijklmnopqrstuvwxyz ")

    # c_x_pos = x_pos
    # c_y_pos = y_pos
    # for character in text:
    #     glyph = avfont.glyph(character)
    #     svg_output.add_glyph(glyph, c_x_pos, c_y_pos, FONT_SIZE,
    #                          ascent, descent)
    #     c_x_pos += glyph.real_width(FONT_SIZE)

    # c_x_pos = x_pos
    # c_y_pos = y_pos - FONT_SIZE
    # for character in text:
    #     glyph = avfont.glyph(character)
    #     svg_output.add_glyph(glyph, c_x_pos, c_y_pos, FONT_SIZE)
    #     c_x_pos += glyph.real_width(FONT_SIZE)

    # c_x_pos = x_pos
    # c_y_pos = y_pos + FONT_SIZE
    # for character in text:
    #     glyph = avfont.glyph(character)
    #     svg_output.add_glyph(glyph, c_x_pos, c_y_pos, FONT_SIZE)
    #     c_x_pos += glyph.real_width(FONT_SIZE)

    # c_x_pos = x_pos
    # c_y_pos = y_pos + 3 * FONT_SIZE
    # for character in text:
    #     glyph = avfont.glyph(character)
    #     svg_output.add_glyph(glyph, c_x_pos, c_y_pos, FONT_SIZE)
    #     c_x_pos += glyph.real_width(FONT_SIZE)

    # circ_path = AVPathPolygon.circle_to_path(
    #     20*VB_RATIO, 20*VB_RATIO, 15*VB_RATIO, 2)
    # svg_path = svg_output.draw_path(circ_path, stroke="red",
    #                                 stroke_width=0.1 * VB_RATIO, fill="blue")
    # svg_output.add(svg_path)

    # Potpourri.print_glyph_coverage(avfont, ascent, descent, FONT_SIZE, text)
    # Potpourri.print_glyph_number_of_paths(avfont, text)

    # -------------------------------------------------------------------------
    # # check an instantiated font:
    # axes_values = AVFont.default_axes_values(ttfont)
    # axes_values.update({"wght": 700, "wdth": 25, "GRAD": 100})
    # ttfont = instancer.instantiateVariableFont(ttfont, axes_values)
    # font = AVFont(ttfont)
    # c_x_pos = x_pos
    # c_y_pos = y_pos + 4 * FONT_SIZE
    # for character in text:
    #     glyph = font.glyph(character)
    #     svg_output.add_glyph(glyph, c_x_pos, c_y_pos, FONT_SIZE,
    #                          ascent, descent)
    #     c_x_pos += glyph.real_width(FONT_SIZE)

    # -------------------------------------------------------------------------
    # # Take a look on Ä:
    # glyph = font.glyph("Ä")
    # print(type(glyph._avfont.ttfont.getGlyphSet()))
    # print(dir(glyph._avfont.ttfont.getGlyphSet()))
    # print(vars(glyph._avfont.ttfont.getGlyphSet()))
    # print("--------------------------------------------------------")
    # glyph_name = glyph._avfont.ttfont.getBestCmap()[ord(character)]
    # glyph_set = glyph._avfont.ttfont.getGlyphSet()[glyph_name]
    # print(vars(glyph_set))

    def instantiate_font(ttfont: TTFont, values: Dict[str, float]) -> AVFont:
        # values {"wght": 700, "wdth": 25, "GRAD": 100}
        axes_values = AVFont.default_axes_values(ttfont)
        axes_values.update(values)
        ttfont = instancer.instantiateVariableFont(ttfont, axes_values)
        return AVFont(ttfont)

    y_pos = VB_RATIO * -0.7 + 1 * FONT_SIZE  # in mm
    x_left = 0
    x_right = 1
    font_weight = 100
    font_weight_delta = 18
    with open(INPUT_FILE_LOREM_IPSUM, 'r', encoding="utf-8") as file:
        lorem_ipsum = file.read()
        # lorem_ipsum = lorem_ipsum[:400]
        # print(f"input-text: _{lorem_ipsum}_")
        # y_pos = y_pos + 6 * FONT_SIZE
        text = lorem_ipsum
        while y_pos < (RECT_HEIGHT*VB_RATIO):
            print(y_pos, font_weight)
            avfont = instantiate_font(ttfont, {"wght": font_weight})
            layouter = SimpleLineLayouter(svg_output, avfont, FONT_SIZE)
            text = layouter.layout_line(y_pos, x_left, x_right, text)
            # print(f"remaining-text: _{text}_")
            if not text:
                break
            y_pos += 1 * FONT_SIZE
            font_weight += font_weight_delta
            font_weight = min(font_weight, 1000)

    # Save the SVG file
    print("save...")
    svg_output.saveas(OUTPUT_FILE+"z", pretty=True, indent=2, compressed=True)
    print("save done.")

    print(RECT_HEIGHT*VB_RATIO)


if __name__ == "__main__":
    main()
