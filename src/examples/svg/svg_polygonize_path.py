from typing import Callable
import math
import svgpathtools
import svgwrite

# Define the SVG path string
PATH_STRING_INPUT = (
    "M0 1 C20 1 0 21 21 21 L21 20 C1 20 21 0 1 0 Z"
    + "M1 21 Q1 1 21 1 L 20 0 Q0 0 0 20 Z"
)

POLYGONIZE_UNIFORM_NUM_POINTS = 100  # minimum 2 = (start, end)
POLYGONIZE_ANGLE_MAX_DEGREE = 2  # difference of two derivatives less than
POLYGONIZE_ANGLE_MAX_STEPS = 9

OUTPUT_FILE = "data/output/example/svg/polygonize_path.svg"


def polygonize_uniform(segment, num_points: int = 100) -> str:
    # *segment* most likely of type QuadraticBezier or CubicBezier
    # create points ]start,...,end]
    ret_string = ""
    poly = segment.poly()
    points = [poly(i / (num_points - 1)) for i in range(1, num_points - 1)]
    for point in points:
        ret_string += f"L{point.real:g},{point.imag:g}"
    ret_string += f"L{segment.end.real:g},{segment.end.imag:g}"
    return ret_string


def polygonize_by_angle(
    segment, max_angle_degree: float = 2, max_steps: int = 9
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
                (new_params, new_points, new_tangents) = ([param], [point], [tangent])
            else:
                dot_product = (
                    new_tangents[-1].real * tangent.real
                    + new_tangents[-1].imag * tangent.imag
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
                print(
                    "ERROR during polygonizing: " + "not supported segment: " + segment
                )
                ret_path_string += lineto(segment.end)
        if sub_path.isclosed():
            ret_path_string += "Z "
    return ret_path_string


def poly_func_uniform(segment):
    return polygonize_uniform(segment, num_points=POLYGONIZE_UNIFORM_NUM_POINTS)


def poly_func_by_angle(segment):
    return polygonize_by_angle(
        segment,
        max_angle_degree=POLYGONIZE_ANGLE_MAX_DEGREE,
        max_steps=POLYGONIZE_ANGLE_MAX_STEPS,
    )


polygon_path_uniform = polygonize_path(PATH_STRING_INPUT, poly_func_uniform)
polygon_path_by_angle = polygonize_path(PATH_STRING_INPUT, poly_func_by_angle)

print("Input :", PATH_STRING_INPUT)
print("Output:", polygon_path_uniform)
print(f'  ...using "{POLYGONIZE_UNIFORM_NUM_POINTS}" points to polygonize.')
print("Output:", polygon_path_by_angle)
print(
    f'  ...using "{POLYGONIZE_ANGLE_MAX_DEGREE}" max-angle [degree] and '
    + f'a maximum of approximately "{POLYGONIZE_ANGLE_MAX_STEPS}" points.'
)

dwg = svgwrite.Drawing(OUTPUT_FILE, viewBox="-1 -1 23 23")

path1 = dwg.path(PATH_STRING_INPUT, stroke="black", stroke_width="0.09", fill="none")
path2 = dwg.path(polygon_path_uniform, stroke="red", stroke_width="0.06", fill="none")
path3 = dwg.path(
    polygon_path_by_angle, stroke="green", stroke_width="0.03", fill="none"
)

dwg.add(path1)
dwg.add(path2)
dwg.add(path3)

dwg.save()
