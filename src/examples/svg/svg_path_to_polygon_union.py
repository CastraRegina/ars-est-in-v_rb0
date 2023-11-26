from typing import List
import re
import numpy
import svgwrite
import svgwrite.base
import svgwrite.container
import svgwrite.elementfactory
import shapely
import shapely.geometry
import shapely.wkt
import svgpathtools.paths2svg
import svgpathtools.path
import svgpath2mpl
import matplotlib.path


SVG_PATH_11_ADD = "M  7 10 L  7 19 L 19 19 L 19 10 Z"  # CW  (lb)
SVG_PATH_12_SUB = "M  8 11 L 13 11 L 13 13 L  8 13 Z"  # CCW (lb)
SVG_PATH_13_SUB = "M 15 15 L 18 15 L 18 18 L 15 18 Z"  # CCW (lb)

SVG_PATH_21_ADD = "M 20 20 L 20 29 L 29 29 L 29 20 Z"  # CW  (lb)
SVG_PATH_22_SUB = "M 21 21 L 23 21 L 23 23 L 21 23 Z"  # CCW (lb)
SVG_PATH_23_SUB = "M 25 25 L 28 25 L 28 28 L 25 28 Z"  # CCW (lb)

SVG_PATH_31_ADD = "M -11 0 L 0 11 L 11 0 L 10 0 L 0 10 L -10 0 Z"  # CW (lb)
SVG_PATH_32_ADD = "M -12 6 L 12 6 L 12 5 L -12 5 Z"  # CW (lb)

SVG_PATH_41_ADD = "M0 1 C20 1 0 21 21 21 L20 20 C1 20 21 0 1 0 Z"  # CW (lb)
SVG_PATH_42_ADD = "M1 21 Q1 1 21 1 L 20 0 Q0 0 0 20 Z"  # CW (lb)

OUTPUT_FILE = "data/output/example/svg/path_to_polygon_union.svg"


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
    def polygonize_path_uniform(path_string: str, num_points: int = 100) -> str:
        def moveto(coord) -> str:
            return f"M{coord.real:g},{coord.imag:g}"

        def lineto(coord) -> str:
            return f"L{coord.real:g},{coord.imag:g}"

        def polygonize_segment(segment, num_points) -> str:
            # *segment* most likely of type QuadraticBezier or CubicBezier
            # create points ]start,...,end]
            ret_string = ""
            poly = segment.poly()
            points = [poly(i / (num_points - 1)) for i in range(1, num_points - 1)]
            for point in points:
                ret_string += lineto(point)
            ret_string += lineto(segment.end)
            return ret_string

        ret_path_string = ""
        path_collection = svgpathtools.parse_path(path_string)
        for sub_path in path_collection.continuous_subpaths():
            ret_path_string += moveto(sub_path.start)
            for segment in sub_path:
                if isinstance(segment, svgpathtools.CubicBezier) or isinstance(
                    segment, svgpathtools.QuadraticBezier
                ):
                    ret_path_string += polygonize_segment(segment, num_points)
                elif isinstance(segment, svgpathtools.Line):
                    ret_path_string += lineto(segment.end)
                else:
                    print(
                        "ERROR during polygonizing: "
                        + "not supported segment: "
                        + segment
                    )
                    ret_path_string += lineto(segment.end)
            if sub_path.isclosed():
                ret_path_string += "Z "
        return ret_path_string

    def __init__(self, multipolygon: shapely.geometry.MultiPolygon = None):
        self.multipolygon: shapely.geometry.MultiPolygon = (
            shapely.geometry.MultiPolygon()
        )
        if multipolygon:
            self.multipolygon = multipolygon

    def add_polygon_arrays(self, polygon_arrays: list[numpy.ndarray]):
        # first polygon_array is always additive.
        # All other arrays are additive, if same orient like first array.
        first_is_ccw = True
        for index, polygon_array in enumerate(polygon_arrays):
            polygon = shapely.Polygon(polygon_array)
            polygon_ccw = polygon.exterior.is_ccw
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


print("- 1 ------------------------------------------------------------------")

polygon_11 = shapely.geometry.Polygon(
    svgpath2mpl.parse_path(SVG_PATH_11_ADD).to_polygons()[0]
)
polygon_12 = shapely.geometry.Polygon(
    svgpath2mpl.parse_path(SVG_PATH_12_SUB).to_polygons()[0]
)
polygon_13 = shapely.geometry.Polygon(
    svgpath2mpl.parse_path(SVG_PATH_13_SUB).to_polygons()[0]
)

polygon_10 = polygon_11.difference(polygon_12)
polygon_10 = polygon_10.difference(polygon_13)

print(polygon_11.area)
print(polygon_12.area)
print(polygon_13.area)
print(polygon_10.area)
print(polygon_10)

print("- 2 ------------------------------------------------------------------")

polygon_31 = shapely.geometry.Polygon(
    svgpath2mpl.parse_path(SVG_PATH_31_ADD).to_polygons()[0]
)
polygon_32 = shapely.geometry.Polygon(
    svgpath2mpl.parse_path(SVG_PATH_32_ADD).to_polygons()[0]
)

polygon_30 = polygon_31.union(polygon_32)

print(polygon_31.area)
print(polygon_32.area)
print(polygon_30.area)
print(polygon_30)

print("- 3 ------------------------------------------------------------------")

# AVPathPolygon.polygonize_path(SVG_PATH_42_ADD, None)

print("- 4 ------------------------------------------------------------------")

SVG_PATH_STRING = " ".join(
    [
        SVG_PATH_11_ADD,
        SVG_PATH_12_SUB,
        SVG_PATH_13_SUB,
        SVG_PATH_31_ADD,
        SVG_PATH_32_ADD,
        SVG_PATH_41_ADD,
        SVG_PATH_42_ADD,
    ]
)
print("A1:", SVG_PATH_STRING)
SVG_PATH_STRING = AVPathPolygon.polygonize_path_uniform(SVG_PATH_STRING)

p_shape = AVPathPolygon()
print("B:", p_shape.multipolygon)
p_shape.add_path_string(SVG_PATH_STRING)

print("C:", p_shape.multipolygon)
print("D:", p_shape.path_strings())

drawing = svgwrite.Drawing(OUTPUT_FILE, viewBox="-14 -2 37 25")
for path in p_shape.svg_paths(
    drawing, stroke="black", stroke_width="0.03", fill="none"
):
    drawing.add(path)
drawing.save()

print("E:", p_shape.multipolygon)
new_multipolygon = shapely.wkt.loads(p_shape.multipolygon.wkt)
print("F:", new_multipolygon)
new_multipolygon2 = AVPathPolygon.deepcopy(p_shape.multipolygon)
print("G:", new_multipolygon2)
