"""
This script demonstrates how to use fontTools to print the bounding box, width,
ascent, descent, line gap, and units per em of a given character in a font.
"""

from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont


def print_font_metrics(font: TTFont) -> None:
    """
    Prints the ascent, descent, line gap, units per em, ...
    of a given font.

    Args:
        font (TTFont): The font object.
    """
    # Extract the font information from the different tables
    ascender_hhea = font["hhea"].ascender
    ascender_os2 = font["OS/2"].sTypoAscender
    descender_hhea = font["hhea"].descender
    descender_os2 = font["OS/2"].sTypoDescender
    line_gap_hhea = font["hhea"].lineGap
    line_gap_os2 = font["OS/2"].sTypoLineGap
    units_per_em = font["head"].unitsPerEm
    x_height = font["OS/2"].sxHeight
    cap_height = font["OS/2"].sCapHeight

    # Print the font metrics
    print("Font metrics:")
    print("    ascender_hhea  :", ascender_hhea)
    print("    ascender_os2   :", ascender_os2)
    print("    descender_hhea :", descender_hhea)
    print("    descender_os2  :", descender_os2)
    print("    line_gap_hhea  :", line_gap_hhea)
    print("    line_gap_os2   :", line_gap_os2)
    print("    units_per_em   :", units_per_em)
    print("    x_height       :", x_height)
    print("    cap_height     :", cap_height)


def print_glyph_metrics(font: TTFont, char: str) -> None:
    """
    Prints the glyph name, bounding box and width
    of a given character in a font.

    Args:
        font (TTFont): The font object.
        char (str): The character to print the metrics for.
    """
    # Get the glyph for the character
    glyph_name = font.getBestCmap()[ord(char)]
    glyph = font.getGlyphSet()[glyph_name]

    # Create a BoundsPen object to calculate the glyph's bounding box
    bounds_pen = BoundsPen(font.getGlyphSet())
    glyph.draw(bounds_pen)

    # Print the glyph metrics
    print(f'Glyph metrics for character "{char}":')
    print("    glyph name     :", glyph_name)
    print("    bounding box   :", bounds_pen.bounds)
    print("    width          :", glyph.width)


def main():
    """Main function to demonstrate font metrics printing using fontTools.

    This function loads a variable font, displays its available axes and metrics,
    and prints glyph metrics for various characters including German umlauts.
    """

    FONT_FILENAME = "fonts/Grandstander[wght].ttf"  # pylint: disable=invalid-name
    FONT_FILENAME = "fonts/NotoSansMono[wdth,wght].ttf"  # pylint: disable=invalid-name
    FONT_FILENAME = "fonts/Recursive[CASL,CRSV,MONO,slnt,wght].ttf"  # pylint: disable=invalid-name
    FONT_FILENAME = (  # pylint: disable=invalid-name
        "fonts/RobotoFlex[GRAD,XOPQ,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght].ttf"
    )
    # FONT_FILENAME = "fonts/RobotoMono[wght].ttf"

    ttfont = TTFont(FONT_FILENAME)
    font_family = ttfont["name"].getDebugName(1)

    print("--------------------------------------")
    print(f"Available axes of {font_family}:")
    for axis in ttfont["fvar"].axes:
        print(
            f"  - {axis.axisTag}: "
            + f"{axis.minValue:7.1f} to {axis.maxValue:7.1f}, "
            + f"default: {axis.defaultValue:7.1f} "
        )

    print_font_metrics(ttfont)

    print("--------------------------------------")
    # NON-ASCII EXCEPTION: German umlauts needed for comprehensive font testing
    print_glyph_metrics(ttfont, "ä")
    print_glyph_metrics(ttfont, "ö")
    print_glyph_metrics(ttfont, "ü")
    print_glyph_metrics(ttfont, "Ä")
    print_glyph_metrics(ttfont, "Ö")
    print_glyph_metrics(ttfont, "Ü")
    print_glyph_metrics(ttfont, ".")
    print_glyph_metrics(ttfont, " ")
    print_glyph_metrics(ttfont, "/")
    print_glyph_metrics(ttfont, "\\")
    print_glyph_metrics(ttfont, "-")
    print_glyph_metrics(ttfont, "_")

    print("--------------------------------------")
    print("Standard default (wght=400):")
    # NON-ASCII EXCEPTION: German umlaut needed for font testing
    print_glyph_metrics(ttfont, "Ä")
    print_font_metrics(ttfont)

    # print()
    # print("Example wght=400:")
    # varfont = instancer.instantiateVariableFont(ttfont, {"wght": 400})
    # NON-ASCII EXCEPTION: German umlaut needed for font testing
    # print_glyph_metrics(varfont, "Ä")
    # print_font_metrics(ttfont)

    # print()
    # print("Example wght=200:")
    # varfont = instancer.instantiateVariableFont(ttfont, {"wght": 200})
    # NON-ASCII EXCEPTION: German umlaut needed for font testing
    # print_glyph_metrics(varfont, "Ä")
    # print_font_metrics(ttfont)

    # print()
    # print("Example wght=700:")
    # varfont = instancer.instantiateVariableFont(ttfont, {"wght": 700})
    # NON-ASCII EXCEPTION: German umlaut needed for font testing
    # print_glyph_metrics(varfont, "Ä")
    # print_font_metrics(ttfont)

    print("--------------------------------------")


if __name__ == "__main__":
    main()
