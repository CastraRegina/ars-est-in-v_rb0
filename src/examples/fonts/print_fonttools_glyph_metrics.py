"""
This script demonstrates how to use fontTools to print the bounding box, width,
ascent, descent, line gap, and units per em of a given character in a font.
"""

from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from fontTools.pens.boundsPen import BoundsPen


def print_glyph_metrics(font: TTFont, char: str) -> None:
    """
    Prints the bounding box, width, ascent, descent, line gap, and units per em
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

    # Calculate the width, ascent, descent, line gap, and units per em
    width = glyph.width
    ascent = font['hhea'].ascent
    descent = font['hhea'].descent
    line_gap = font['hhea'].lineGap
    units_per_em = font['head'].unitsPerEm

    # Print the glyph metrics
    print(f'Glyph metrics for character "{char}":')
    print("  glyph name:", glyph_name)
    print("  Bounding box:", bounds_pen.bounds)
    print("  Width:", width)
    print("  Ascent:", ascent)
    print("  Descent:", descent)
    print("  Line gap:", line_gap)
    print("  Units per em:", units_per_em)


if __name__ == '__main__':

    FONT_FILENAME = "fonts/Grandstander-VariableFont_wght.ttf"
    FONT_FILENAME = "fonts/NotoSansMono-VariableFont_wdth,wght.ttf"
    FONT_FILENAME = "fonts/Recursive-VariableFont_CASL,CRSV,MONO,slnt,wght.ttf"
    FONT_FILENAME = "fonts/RobotoFlex-VariableFont_GRAD,XTRA," + \
                    "YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght.ttf"
    # FONT_FILENAME = "fonts/RobotoMono-VariableFont_wght.ttf"

    # Load the font file
    ttfont = TTFont(FONT_FILENAME)
    font_family = ttfont['name'].getDebugName(1)

    # Print the metrics for some characters:
    print_glyph_metrics(ttfont, 'ä')
    print_glyph_metrics(ttfont, 'ö')
    print_glyph_metrics(ttfont, 'ü')
    print_glyph_metrics(ttfont, 'Ä')
    print_glyph_metrics(ttfont, 'Ö')
    print_glyph_metrics(ttfont, 'Ü')
    print_glyph_metrics(ttfont, '.')
    print_glyph_metrics(ttfont, ' ')
    print_glyph_metrics(ttfont, '/')
    print_glyph_metrics(ttfont, '\\')
    print_glyph_metrics(ttfont, '-')
    print_glyph_metrics(ttfont, '_')

    print("--------------------------------------")
    print(f'Available axes of {font_family}:')
    for axis in ttfont['fvar'].axes:
        print(f'  - {axis.axisTag}: ' +
              f'{axis.minValue:7.1f} to {axis.maxValue:7.1f}, ' +
              f'default: {axis.defaultValue:7.1f} ')

    print("--------------------------------------")
    print("Standard default (wght=400):")
    print_glyph_metrics(ttfont, 'Ä')

    print("Example wght=400:")
    varfont = instancer.instantiateVariableFont(ttfont, {"wght": 400})
    print_glyph_metrics(varfont, 'Ä')

    print("Example wght=200:")
    varfont = instancer.instantiateVariableFont(ttfont, {"wght": 200})
    print_glyph_metrics(varfont, 'Ä')

    print("Example wght=700:")
    varfont = instancer.instantiateVariableFont(ttfont, {"wght": 700})
    print_glyph_metrics(varfont, 'Ä')

    print("--------------------------------------")
