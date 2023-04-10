import svgpathtools
import svgwrite

# Define the SVG path string
PATH_STRING_INPUT = "M0 1 C20 1 0 21 21 21 L21 20 C1 20 21 0 1 0 Z" + \
                    "M1 21 Q1 1 21 1 L 20 0 Q0 0 0 20 Z"
POLYGONIZE_NUM_POINTS = 50  # minimum 2 = (start, end)
OUTPUT_FILE = "data/output/example/svg/polygonize_path_uniform.svg"


def polygonize_path_uniform(path_string: str,
                            num_points: int = 100) -> str:
    def moveto(coord) -> str:
        return f'M{coord.real:g},{coord.imag:g}'

    def lineto(coord) -> str:
        return f'L{coord.real:g},{coord.imag:g}'

    def polygonize_segment(segment, num_points) -> str:
        ret_string = ""
        poly = segment.poly()
        points = [poly(i/(num_points-1)) for i in range(1, num_points-1)]
        for point in points:
            ret_string += lineto(point)
        ret_string += lineto(segment.end)
        return ret_string

    ret_path_string = ""
    path_collection = svgpathtools.parse_path(path_string)
    for sub_path in path_collection.continuous_subpaths():
        ret_path_string += moveto(sub_path.start)
        for segment in sub_path:
            if isinstance(segment, svgpathtools.CubicBezier) or \
                    isinstance(segment, svgpathtools.QuadraticBezier):
                ret_path_string += polygonize_segment(segment, num_points)
            elif isinstance(segment, svgpathtools.Line):
                ret_path_string += lineto(segment.end)
            else:
                print("ERROR during polygonizing: " +
                      "not supported segment: " + segment)
                ret_path_string += lineto(segment.end)
        if sub_path.isclosed():
            ret_path_string += "Z "
    return ret_path_string


polygonized_path = polygonize_path_uniform(PATH_STRING_INPUT,
                                           POLYGONIZE_NUM_POINTS)
print("Input :", PATH_STRING_INPUT)
print("Output:", polygonized_path)
print(f'  ...using "{POLYGONIZE_NUM_POINTS}" points to polygonize.')

dwg = svgwrite.Drawing(OUTPUT_FILE, viewBox='-1 -1 23 23')
path1 = dwg.path(PATH_STRING_INPUT, stroke="black",
                 stroke_width="0.06", fill="none")
path2 = dwg.path(polygonized_path, stroke="red",
                 stroke_width="0.03", fill="none")
dwg.add(path1)
dwg.add(path2)
dwg.save()
