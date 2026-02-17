"""Letter alignment and positioning utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Union

from ave.common import AlignX, CenterRef

if TYPE_CHECKING:
    from ave.letter import AvLetter


###############################################################################
# AvLetterAlignment
###############################################################################


class AvLetterAlignment:
    """Static methods for letter alignment and positioning."""

    @staticmethod
    def align_to_center(
        letter: Union[AvLetter, Sequence[AvLetter]],
        x_center: Optional[float] = None,
        y_center: Optional[float] = None,
        reference: CenterRef = CenterRef.LETTER_BOX,
    ) -> None:
        """Align letter position so its reference point center is at the specified coordinates.

        Modifies the letter's xpos and/or ypos to position the letter so that its
        reference point center aligns with the given x_center and/or y_center.
        Accepts a single letter or a sequence of letters.

        Args:
            letter: A single letter or a sequence of letters to align.
            x_center: Target x-coordinate for the letter's center. If None, x-alignment is skipped.
            y_center: Target y-coordinate for the letter's center. If None, y-alignment is skipped.
            reference: Which reference point to use for centering (LETTER_BOX, BOUNDING_BOX, or CENTROID).
        """
        # Check if letter is a sequence (but not a string)
        if isinstance(letter, (list, tuple)) or hasattr(letter, "__iter__"):
            # Handle sequence of letters
            for single_letter in letter:
                AvLetterAlignment.align_to_center(single_letter, x_center, y_center, reference)
        else:
            # Handle single letter
            if reference == CenterRef.CENTROID:
                # For centroid, get current coordinates directly
                current_center_x, current_center_y = letter.centroid()
            else:
                # For box-based references, get the appropriate box
                if reference == CenterRef.BOUNDING_BOX:
                    box = letter.bounding_box
                else:  # LETTER_BOX
                    box = letter.letter_box

                current_center_x = box.xmin + box.width / 2
                current_center_y = box.ymin + box.height / 2

            if x_center is not None:
                # Calculate delta needed to move to target center
                delta_x = x_center - current_center_x
                # Apply delta to letter position
                letter.xpos = letter.xpos + delta_x

            if y_center is not None:
                # Calculate delta needed to move to target center
                delta_y = y_center - current_center_y
                # Apply delta to letter position
                letter.ypos = letter.ypos + delta_y

    @staticmethod
    def align_to_x_border(
        letter: Union[AvLetter, Sequence[AvLetter]],
        xpos_border: float,
        alignment: AlignX = AlignX.LEFT,
    ) -> None:
        """Align letter position to a specified border.

        Modifies the letter's xpos to align its left or right side with the specified border.
        Accepts a single letter or a sequence of letters.

        Args:
            letter: A single letter or a sequence of letters to align.
            xpos_border: The x-coordinate of the border to align to.
            alignment: Which side to align (AlignX.LEFT or AlignX.RIGHT).
        """
        # Check if letter is a sequence (but not a string)
        if isinstance(letter, (list, tuple)) or hasattr(letter, "__iter__"):
            # Handle sequence of letters
            for single_letter in letter:
                AvLetterAlignment.align_to_x_border(single_letter, xpos_border, alignment)
        else:
            # Handle single letter
            bounding_box = letter.bounding_box

            if alignment == AlignX.LEFT:
                # Align left side to border
                delta_x = xpos_border - bounding_box.xmin
            elif alignment == AlignX.RIGHT:
                # Align right side to border
                delta_x = xpos_border - bounding_box.xmax
            else:
                raise ValueError(f"Invalid alignment: {alignment}. Must be AlignX.LEFT or AlignX.RIGHT.")

            # Apply delta to letter position
            letter.xpos = letter.xpos + delta_x
