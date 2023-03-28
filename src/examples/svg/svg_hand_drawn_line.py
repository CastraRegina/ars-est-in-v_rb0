
from typing import Tuple
import math
import random
import svgwrite

OUTPUT_FILE = "data/output/example/svg/din_a5_page_hand_drawn_line.svg"

CANVAS_UNIT = "mm"  # Units for CANVAS dimensions
CANVAS_WIDTH = 297/2  # DIN A5 page width in mm
CANVAS_HEIGHT = 210  # DIN A5 page height in mm

RECT_WIDTH = 297/2-30  # rectangle width in mm
RECT_HEIGHT = 210-30  # rectangle height in mm

VB_RATIO = 1 / RECT_WIDTH  # multiply each dimension with this ratio

C_BEZIER = 0.5  # factor to set the bezier control points


def dist_delta_normal(point1: Tuple[float, float],
                      point2: Tuple[float, float]) \
        -> Tuple[float, Tuple[float, float], Tuple[float, float]]:
    delta = (point2[0]-point1[0], point2[1]-point1[1])
    dist = math.sqrt(delta[0]**2 + delta[1]**2)
    normal = (delta[1], -delta[0])
    if dist:  # do not divide by zero
        delta = (delta[0]/dist, delta[1]/dist)
        normal = (normal[0]/dist, normal[1]/dist)
    return (dist, delta, normal)


def svg_hand_drawn_line(dwg: svgwrite.Drawing,
                        point1: Tuple[float, float],
                        point2: Tuple[float, float],
                        width: float,
                        min_width: float,  # path not smaller than min_width
                        width_deviation_ratio: float,  # +/- ratio*width
                        straight_deviation_ratio: float,  # +/- ratio*width
                        num_supports: int,  # # of points between end points
                        **svg_properties) \
        -> svgwrite.elementfactory.ElementBuilder:

    (dist, delta, normal) = dist_delta_normal(point1, point2)
    positions = [i/(num_supports+1) for i in range(1, num_supports+1)]

    points_up = [(point1[0]+width/2*normal[0],
                  point1[1]+width/2*normal[1])]
    points_dn = [(point1[0]-width/2*normal[0],
                  point1[1]-width/2*normal[1])]
    for pos in positions:
        rwidth = width + width * random.uniform(
            max(-(1-min_width/width), - width_deviation_ratio),
            width_deviation_ratio)
        rwidth = max(rwidth, min_width)
        rstraight = random.uniform(-1, 1) * width*straight_deviation_ratio
        support = (point1[0] + pos*dist*delta[0],
                   point1[1] + pos*dist*delta[1])
        points_up.append((support[0] + (+rwidth/2 + rstraight)*normal[0],
                          support[1] + (+rwidth/2 + rstraight)*normal[1]))
        points_dn.append((support[0] + (-rwidth/2 + rstraight)*normal[0],
                          support[1] + (-rwidth/2 + rstraight)*normal[1]))
    points_up.append((point2[0]+width/2*normal[0],
                      point2[1]+width/2*normal[1]))
    points_dn.append((point2[0]-width/2*normal[0],
                      point2[1]-width/2*normal[1]))
    points_dn.reverse()

    d_str = ""
    for i, point in enumerate(points_up):
        if i == 0:  # first point
            d_str += f"M {point[0]} {point[1]} "
        else:
            pt0 = points_up[i-1]
            pt3 = point
            (dist, _, _) = dist_delta_normal(pt0, pt3)
            pt1 = (pt0[0] + C_BEZIER*dist*delta[0],
                   pt0[1] + C_BEZIER*dist*delta[1])
            pt2 = (pt3[0] - C_BEZIER*dist*delta[0],
                   pt3[1] - C_BEZIER*dist*delta[1])
            d_str += f"C {pt1[0]} {pt1[1]} " + \
                f"{pt2[0]} {pt2[1]} {pt3[0]} {pt3[1]} "

    for i, point in enumerate(points_dn):
        if i == 0:  # first point
            d_str += f"L {point[0]} {point[1]} "
        else:
            pt0 = points_dn[i-1]
            pt3 = point
            (dist, _, _) = dist_delta_normal(pt0, pt3)
            pt1 = (pt0[0] - C_BEZIER*dist*delta[0],
                   pt0[1] - C_BEZIER*dist*delta[1])
            pt2 = (pt3[0] + C_BEZIER*dist*delta[0],
                   pt3[1] + C_BEZIER*dist*delta[1])
            d_str += f"C {pt1[0]} {pt1[1]} " + \
                f"{pt2[0]} {pt2[1]} {pt3[0]} {pt3[1]} "
    d_str += "Z"

    path_properties = {"fill": "black"}
    path_properties.update(svg_properties)
    path = dwg.path(d=d_str, **path_properties)
    return path


def main():
    # Center the rectangle horizontally and vertically on the page
    vb_w = VB_RATIO * CANVAS_WIDTH
    vb_h = VB_RATIO * CANVAS_HEIGHT
    vb_x = -VB_RATIO * (CANVAS_WIDTH - RECT_WIDTH) / 2
    vb_y = -VB_RATIO * (CANVAS_HEIGHT - RECT_HEIGHT) / 2

    # Set up the SVG canvas:
    # Define viewBox so that "1" is the width of the rectangle
    # Multiply a dimension with "VB_RATIO" to get the size regarding viewBox
    dwg = svgwrite.Drawing(OUTPUT_FILE,
                           size=(f"{CANVAS_WIDTH}mm", f"{CANVAS_HEIGHT}mm"),
                           viewBox=(f"{vb_x} {vb_y} {vb_w} {vb_h}")
                           )

    # Draw the rectangle
    dwg.add(
        dwg.rect(
            insert=(0, 0),
            size=(VB_RATIO*RECT_WIDTH, VB_RATIO*RECT_HEIGHT),  # = (1.0, xxxx)
            stroke="black",
            stroke_width=0.1*VB_RATIO,
            fill="none"
        )
    )

    # horizontal
    dwg.add(svg_hand_drawn_line(dwg, (0.1, 0.5), (0.9, 0.5),
                                0.25*VB_RATIO, 0.1*VB_RATIO,
                                0.4, 2.0, 9))
    # vertical
    dwg.add(svg_hand_drawn_line(dwg, (0.5, 0.1), (0.5, 0.9),
                                0.25*VB_RATIO, 0.1*VB_RATIO,
                                0.4, 2.0, 9))
    # TL
    dwg.add(svg_hand_drawn_line(dwg, (0.45, 0.45), (0.1, 0.1),
                                0.25*VB_RATIO, 0.1*VB_RATIO,
                                0.4, 1.0, 9))
    # BL
    dwg.add(svg_hand_drawn_line(dwg, (0.45, 0.55), (0.1, 0.9),
                                0.25*VB_RATIO, 0.1*VB_RATIO,
                                0.4, 1.0, 9))
    # TR
    dwg.add(svg_hand_drawn_line(dwg, (0.55, 0.45), (0.9, 0.1),
                                0.25*VB_RATIO, 0.1*VB_RATIO,
                                0.4, 1.0, 9))
    # BR
    dwg.add(svg_hand_drawn_line(dwg, (0.55, 0.55), (0.9, 0.9),
                                0.25*VB_RATIO, 0.1*VB_RATIO,
                                0.4, 1.0, 9,
                                fill="green"))

    height = RECT_HEIGHT/RECT_WIDTH
    dwg.add(svg_hand_drawn_line(dwg, (0.0, 0.0), (1.0, 0.0),
                                0.25*VB_RATIO, 0.1*VB_RATIO,
                                0.4, 1.8, 9))
    dwg.add(svg_hand_drawn_line(dwg, (1.0, 0.0), (1.0, height),
                                0.25*VB_RATIO, 0.1*VB_RATIO,
                                0.4, 1.8, 12))
    dwg.add(svg_hand_drawn_line(dwg, (1.0, height), (0.0, height),
                                0.25*VB_RATIO, 0.1*VB_RATIO,
                                0.4, 1.8, 9))
    dwg.add(svg_hand_drawn_line(dwg, (0.0, height), (0.0, 0.0),
                                0.25*VB_RATIO, 0.1*VB_RATIO,
                                0.4, 1.8, 12))

    # Save the SVG file
    dwg.saveas(OUTPUT_FILE, pretty=True, indent=2)


if __name__ == "__main__":
    main()
