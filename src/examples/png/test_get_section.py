"""An image representation "Pillow Type L (8-bit pixels, grayscale)" using Numpy ndarray"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import PIL.Image

# from typing import Optional


@dataclass
class AVImageGrey:
    """An image representation "Pillow Type L (8-bit pixels, grayscale)" using Numpy ndarray"""

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

    def crop_px(self, x_px: int, y_px: int, width_px: int, height_px: int) -> AVImageGrey:
        """Crop a rectangular portion defined by pixels. Coordinate (0,0) at top left corner.

        Args:
            x_px (int): start point (left)
            y_px (int): start point (top)
            width_px (int): width to the right
            height_px (int): height to the bottom

        Returns:
            AVImageGrey: cropped image
        """
        image_np_array = self.image_np_array[y_px : (y_px + height_px), x_px : (x_px + width_px)]
        return AVImageGrey(image_np_array)

    def crop_pt(self, left: int, top: int, right: int, bottom: int) -> AVImageGrey:
        """Crop a rectangular portion defined by pixels including pixels of right and bottom.
            Coordinate (0,0) at top left corner.

        Args:
            left (int): start point (left)
            top (int): start point (top)
            right (int): end point (right, inclusive)
            bottom (int): end point (bottom, inclusive)

        Returns:
            AVImageGrey: cropped image
        """
        image_np_array = self.image_np_array[top:bottom, left:right]
        return AVImageGrey(image_np_array)

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

    image1 = image_in.crop_px(10, 0, 350, 600)
    print("width :", image1.width)
    print("height:", image1.height)

    image2 = image_in.crop_pt(199, 99, 401, 501)
    print("width :", image2.width)
    print("height:", image2.height)

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
