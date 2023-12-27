"""An image representation "Pillow Type L (8-bit pixels, grayscale)" using Numpy ndarray"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import PIL.Image

# from typing import Optional


@dataclass
class AVImageGrey:
    """An image representation "Pillow Type L (8-bit pixels, grayscale)" using Numpy ndarray
    Coordinates start at (0,0) at top left corner.
    x-Axis from left to right.
    y-Axis from top to bottom.
    Methods with "px" are pixel-based absolute coordinate values.
    Methods with "rel" are relative coordinate values with width=1.0 and height=1.0
    """

    image_np_array: np.ndarray

    def __init__(self, image_np_array: np.ndarray):
        self.image_np_array = image_np_array

    @property
    def width(self) -> int:
        """Width of the image as an integer

        Returns:
            int: width
        """
        return self.image_np_array.shape[1]

    @property
    def height(self) -> int:
        """Height of the image as an integer

        Returns:
            int: height
        """
        return self.image_np_array.shape[0]

    def crop_width_px(self, x_px: int, y_px: int, width_px: int, height_px: int) -> AVImageGrey:
        """Crop a rectangular portion defined by pixels.
           Coordinates start at (0,0) at top left corner.
           E.g. whole image would be (0, 0, width, height)

        Args:
            x_px (int): start point (left, inclusive)
            y_px (int): start point (top, inclusive)
            width_px (int): width to the right in pixel
            height_px (int): height to the bottom in pixel

        Returns:
            AVImageGrey: cropped image
        """
        image_np_array = self.image_np_array[y_px : (y_px + height_px), x_px : (x_px + width_px)]
        return AVImageGrey(image_np_array)

    def crop_point_px(self, left: int, top: int, right: int, bottom: int) -> AVImageGrey:
        """Crop a rectangular portion defined by pixel-positions
            including pixels of right and bottom.
            Coordinates start at (0,0) at top left corner.
            E.g. whole image would be (0, 0, width-1, height-1)

        Args:
            left (int): start point (left, inclusive)
            top (int): start point (top, inclusive)
            right (int): end point (right, inclusive)
            bottom (int): end point (bottom, inclusive)

        Returns:
            AVImageGrey: cropped image
        """
        image_np_array = self.image_np_array[top : (bottom + 1), left : (right + 1)]
        return AVImageGrey(image_np_array)

    def getpixel_px(self, x_px: int, y_px: int) -> int:
        """Returns the value of a pixel at x_px, y_px.
            Coordinates start at (0,0) at top left corner.
            Right-bottom-corner at coordinate [ (width-1), (height-1) ]

        Args:
            x_px (int): x-position (left-to-right)
            y_px (int): y-position (top-to-bottom)

        Returns:
            int: The value of a pixel at x_px, y_px
        """
        return self.image_np_array[y_px, x_px]

    def getpixel_rel(self, x_rel: float, y_rel: float) -> int:
        """Returns the value of a pixel at relative coordinate x_rel, y_rel.
            Coordinate (0,0) at top left corner.
            0 <= x_rel <= 1  and   0 <= y_rel <= 1

        Args:
            x_rel (float): x-position (left-to-right)
            y_rel (float): y-position (top-to-bottom)

        Returns:
            int: The value of a pixel at relative coordinate x_rel, y_rel
        """
        x_px = round(x_rel * (self.width - 1))
        y_px = round(y_rel * (self.height - 1))
        return self.getpixel_px(x_px, y_px)

    @classmethod
    def load_image(cls, filename: str) -> AVImageGrey:
        """Loads image and converts it if not in the correct format of "L (8-bit pixels, grayscale)"

        Args:
            filename (str): path and filename

        Returns:
            AVImageGrey: the loaded image
        """
        with PIL.Image.open(filename) as pil_image:
            if pil_image.mode != "L":
                # convert to "L (8-bit pixels, grayscale)"
                pil_image = pil_image.convert("L")
            return AVImageGrey(np.array(pil_image))

    @classmethod
    def save_image(cls, image: AVImageGrey, filename: str):
        """Saves an image using format of "L (8-bit pixels, grayscale)"

        Args:
            image (AVImageGrey): image to save
            filename (str): path and filename
        """
        pil_image = PIL.Image.fromarray(image.image_np_array, mode="L")
        pil_image.save(filename)


def main():
    """Some examples to show the functionality of this module"""
    filename_input = "data/output/example/png/test_board_10grays.png"
    filename_output = "section_output.png"
    image_in = AVImageGrey.load_image(filename_input)
    print("width :", image_in.width)
    print("height:", image_in.height)

    image1 = image_in.crop_width_px(199, 99, 202, 402)
    print("width :", image1.width)
    print("height:", image1.height)

    image2 = image_in.crop_point_px(199, 99, 400, 500)
    print("width :", image2.width)
    print("height:", image2.height)

    print("Value(  0,  0)", image_in.getpixel_px(0, 0))
    print("Value(150,150)", image_in.getpixel_px(150, 150))
    print("Value(15/70,15/70)", image_in.getpixel_rel(150 / 700, 150 / 700))

    print(np.array_equal(image1.image_np_array, image2.image_np_array))
    AVImageGrey.save_image(image2, filename_output)


if __name__ == "__main__":
    main()

# class AVImageOld:
#     def __init__(self, filename: str, convert: str = "L"):
#         self.image = None
#         self.width = 0
#         self.height = 0
#         with PIL.Image.open(filename) as self.image:
#             if convert:
#                 self.image = self.image.convert(convert)
#             (self.width, self.height) = self.image.size

#     def section(self, x: float, y: float, width: float, height: float) -> PIL.Image:
#         # relative to width 0..1
#         if not self.image:
#             return None
#         # TODO: x,y,width,height min(0) max(1)
#         # TODO: round()

#     def avg_value(self, x: float, y: float, width: float, height: float) -> float:
#         pass
#         # TODO

#     def max_value(self, x: float, y: float, width: float, height: float) -> float:
#         pass
#         # TODO

#     def min_value(self, x: float, y: float, width: float, height: float) -> float:
#         pass
#         # TODO


# def extract_section(image_path, x, y, width, height):
#     # Open the image file
#     image = PIL.Image.open(image_path).convert("L")  # Convert to grayscale

#     # Extract the section based on coordinates
#     section = image.crop((x, y, x + width, y + height))

#     return section


# def calculate_average_gray_value(image_path, x, y, width, height):
#     # Extract the section from the image
#     section = extract_section(image_path, x, y, width, height)

#     # Convert the section to a numpy array
#     section_array = np.array(section)

#     # Calculate the average gray value
#     average_gray_value = np.mean(section_array)

#     return average_gray_value


# def save_section_as_image(section, output_path):
#     # Save the section as a PNG file
#     section.save(output_path)


# # Example usage
# image_path = "data/output/example/png/test_board_10grays.png"
# x = 0  # X-coordinate of the top-left corner of the section
# y = 0  # Y-coordinate of the top-left corner of the section
# width = 700  # Width of the section
# height = 700  # Height of the section
# output_path = "section_output.png"

# # Extract the section from the image
# section = extract_section(image_path, x, y, width, height)

# # Save the section as a PNG file
# save_section_as_image(section, output_path)

# # Calculate the average gray value of the section
# average_gray_value = calculate_average_gray_value(image_path, x, y, width, height)
# print(f"The average gray value of the section is: {average_gray_value}")
# print(f"The section has been saved as: {output_path}")

# image = AVImageOld(image_path)
