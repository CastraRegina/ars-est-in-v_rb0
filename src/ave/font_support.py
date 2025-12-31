"""Font properties and supporting utilities for OpenType and SVG fonts."""

from __future__ import annotations

from dataclasses import dataclass

from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont


###############################################################################
# AvFontProperties
###############################################################################
@dataclass
class AvFontProperties:
    """
    Represents the properties of a font.

    The properties are as follows:

    - `ascender`: The highest y-coordinate above the baseline (mostly positive value).
    - `descender`: The lowest y-coordinate below the baseline (usually negative value).
    - `line_gap`: Additional spacing between lines.
    - `x_height`: Height of lowercase 'x'.
    - `cap_height`: Height of uppercase 'H'.
    - `dash_thickness`: Height/thickness of dash '-' character.
    - `units_per_em`: Units per em.
    - `family_name`: Font family name.
    - `subfamily_name`: Style name (Regular, Bold, etc.).
    - `full_name`: Full font name.
    - `version`: Font version string.
    - `license_description`: License text.
    - `font_statistics`: Detailed font statistics including kerning and substitutions.
    - `line_height`: Computed line height of the font (ascender - descender + line_gap).
    """

    ascender: float
    descender: float
    line_gap: float

    x_height: float
    cap_height: float
    dash_thickness: float
    units_per_em: float

    family_name: str
    subfamily_name: str
    full_name: str
    version: str
    license_description: str
    font_statistics: str

    @property
    def line_height(self) -> float:
        """Computed line height of the font (ascender - descender + line_gap)."""
        return self.ascender - self.descender + self.line_gap

    def __init__(
        self,
        ascender: float = 0,
        descender: float = 0,
        line_gap: float = 0,
        x_height: float = 0,
        cap_height: float = 0,
        dash_thickness: float = 0,
        units_per_em: float = 1000,
        family_name: str = "",
        subfamily_name: str = "",
        full_name: str = "",
        version: str = "",
        license_description: str = "",
        font_statistics: str = "",
    ) -> None:
        self.ascender = ascender
        self.descender = descender
        self.line_gap = line_gap
        self.x_height = x_height
        self.cap_height = cap_height
        self.dash_thickness = dash_thickness
        self.units_per_em = units_per_em
        self.family_name = family_name
        self.subfamily_name = subfamily_name
        self.full_name = full_name
        self.version = version
        self.license_description = license_description
        self.font_statistics = font_statistics

    @classmethod
    def from_dict(cls, data: dict) -> AvFontProperties:
        """Create an AvFontProperties instance from a dictionary."""
        return cls(
            ascender=data.get("ascender", 0),
            descender=data.get("descender", 0),
            line_gap=data.get("line_gap", 0),
            x_height=data.get("x_height", 0),
            cap_height=data.get("cap_height", 0),
            dash_thickness=data.get("dash_thickness", 0),
            units_per_em=data.get("units_per_em", 1000),
            family_name=data.get("family_name", ""),
            subfamily_name=data.get("subfamily_name", ""),
            full_name=data.get("full_name", ""),
            version=data.get("version", ""),
            license_description=data.get("license_description", ""),
            font_statistics=data.get("font_statistics", ""),
        )

    def to_dict(self) -> dict:
        """Convert the AvFontProperties instance to a dictionary."""
        return {
            "ascender": self.ascender,
            "descender": self.descender,
            "line_gap": self.line_gap,
            "x_height": self.x_height,
            "cap_height": self.cap_height,
            "dash_thickness": self.dash_thickness,
            "units_per_em": self.units_per_em,
            "family_name": self.family_name,
            "subfamily_name": self.subfamily_name,
            "full_name": self.full_name,
            "version": self.version,
            "license_description": self.license_description,
            "font_statistics": self.font_statistics,
        }

    @classmethod
    def _glyph_visual_height(cls, font: TTFont, char: str) -> float:
        """Return yMax - yMin of the glyph for `char`, or 0.0 if missing/empty."""
        cmap = font.getBestCmap()
        if not cmap or ord(char) not in cmap:
            return 0.0

        glyph_name = cmap[ord(char)]
        glyph_set = font.getGlyphSet()

        pen = BoundsPen(glyph_set)
        try:
            glyph_set[glyph_name].draw(pen)
        except KeyError:
            return 0.0

        if pen.bounds is None:
            return 0.0

        _, y_min, _, y_max = pen.bounds
        return max(0.0, float(y_max - y_min))

    @classmethod
    def _get_name_safe(cls, name_table, name_id: int) -> str:
        """
        Extract a name string with multiple fallbacks.
        This is the most robust way to read name records from real-world fonts.
        """
        # 1. Fast path - getDebugName handles most cases correctly
        name = name_table.getDebugName(name_id)
        if name is not None:
            return name

        # 2. Try Windows Unicode English (platform 3, encoding 1, lang 0x409)
        record = name_table.getName(name_id, 3, 1, 0x409)
        if record is not None:
            try:
                return record.toUnicode()
            except (UnicodeDecodeError, ValueError):
                pass

        # 3. Try any Windows record
        record = name_table.getName(name_id, 3, 1)
        if record is not None:
            try:
                return record.toUnicode()
            except (UnicodeDecodeError, ValueError):
                pass

        # 4. Last resort: iterate all records manually
        for record in name_table.names:
            if record.nameID == name_id:
                try:
                    return record.toUnicode()
                except (UnicodeDecodeError, ValueError, AttributeError):
                    continue

        return ""

    @classmethod
    def from_ttfont(cls, ttfont: TTFont) -> AvFontProperties:
        """
        Create AvFontProperties from a fontTools TTFont object.
        Works with static and variable fonts (as long as the correct variation is active).
        """
        hhea = ttfont["hhea"]
        head = ttfont["head"]
        name_table = ttfont["name"]

        # Generate font statistics using the static method
        stats = cls.get_font_statistics(ttfont)

        return cls(
            ascender=float(hhea.ascender),  # type: ignore
            descender=float(hhea.descender),  # type: ignore
            line_gap=float(hhea.lineGap),  # type: ignore
            # line_height is automatically computed via @computed_field
            x_height=cls._glyph_visual_height(ttfont, "x"),
            cap_height=cls._glyph_visual_height(ttfont, "H"),
            dash_thickness=cls._glyph_visual_height(ttfont, "-"),
            units_per_em=float(head.unitsPerEm),  # type: ignore
            family_name=cls._get_name_safe(name_table, 1),
            subfamily_name=cls._get_name_safe(name_table, 2),
            full_name=cls._get_name_safe(name_table, 4),
            version=cls._get_name_safe(name_table, 5),
            license_description=cls._get_name_safe(name_table, 13),
            font_statistics=stats,
        )

    def info_string(self, show_stats: bool = False) -> str:
        """
        Return a string containing information about the font properties.
        The string is formatted for display in a text box or similar.
        """

        info_string = (
            "-----Font Information:-----\n"
            f"ascender:      {self.ascender:>5.0f} (max distance above baseline = highest y-coord, positive value)\n"
            f"descender:     {self.descender:>5.0f} (max distance below baseline = lowest y-coord, negative value)\n"
            f"line_gap:      {self.line_gap:>5.0f} (additional spacing between lines of text)\n"
            f"line_height:   {self.line_height:>5.0f} (ascender - descender + line_gap)\n"
            f"x_height:      {self.x_height:>5.0f} (height of lowercase 'x')\n"
            f"cap_height:    {self.cap_height:>5.0f} (height of uppercase 'H')\n"
            f"dash_thickness:{self.dash_thickness:>5.0f} (height/thickness of dash '-')\n"
            f"units_per_em:  {self.units_per_em:>5.0f} (number of units per EM)\n"
            f"family_name:         {self.family_name}\n"
            f"subfamily_name:      {self.subfamily_name}\n"
            f"full_name:           {self.full_name}\n"
            f"version:             {self.version}\n"
            f"license_description: {self.license_description}\n"
        )
        if show_stats:
            info_string += f"font_statistics:     {self.font_statistics}\n"
        return info_string

    def __repr__(self) -> str:
        return f"AvFontProperties({self.family_name} {self.subfamily_name}, {self.units_per_em}upem)"

    @staticmethod
    def get_font_statistics(ttfont: TTFont) -> str:
        """Analyze font and return complete statistics as a string."""
        output = []

        # === General Font Statistics ===
        output.append("\n=== General Font Statistics ===")

        # Basic font info
        font_family = ttfont["name"].getDebugName(1)
        font_style = ttfont["name"].getDebugName(2)
        full_name = ttfont["name"].getDebugName(4)
        version = ttfont["name"].getDebugName(5)

        output.append(f"Font Family: {font_family}")
        output.append(f"Style: {font_style}")
        output.append(f"Full Name: {full_name}")
        output.append(f"Version: {version}")

        # Font metrics
        head = ttfont["head"]
        hhea = ttfont["hhea"]
        os2 = ttfont["OS/2"]

        output.append("\nMetrics:")
        output.append(f"  Units per EM: {head.unitsPerEm}")
        output.append(f"  Ascender: {hhea.ascender}")
        output.append(f"  Descender: {hhea.descender}")
        output.append(f"  Line Gap: {hhea.lineGap}")
        output.append(f"  x-height: {os2.sxHeight}")
        output.append(f"  Cap height: {os2.sCapHeight}")

        # Glyph count
        glyph_count = len(ttfont.getGlyphOrder())
        output.append("\nTotal glyphs: " + str(glyph_count))

        # Character coverage
        cmap = ttfont.getBestCmap()
        output.append("Characters mapped: " + str(len(cmap)))

        # Count by script/category
        latin_count = sum(1 for code in cmap if 0x0000 <= code <= 0x007F)
        latin_ext_count = sum(1 for code in cmap if 0x0080 <= code <= 0x00FF)
        latin1_count = sum(1 for code in cmap if 0x0100 <= code <= 0x017F)
        german_chars = sum(1 for code in cmap if code in [ord(c) for c in "ÄÖÜäöüß"])

        output.append("\nCharacter coverage:")
        output.append("  ASCII (0x00-0x7F): " + str(latin_count))
        output.append("  Latin-1 Supplement (0x80-0xFF): " + str(latin_ext_count))
        output.append("  Latin Extended-A (0x100-0x17F): " + str(latin1_count))
        output.append("  German umlauts (ÄÖÜäöüß): " + str(german_chars))

        # Variable font info
        if "fvar" in ttfont:
            output.append("\nVariable Font Axes:")
            for axis in ttfont["fvar"].axes:
                output.append(
                    f"  {axis.axisTag}: {axis.minValue:.1f} to {axis.maxValue:.1f} (default: {axis.defaultValue:.1f})"
                )
        else:
            output.append("\nNot a variable font")

        # Tables present
        output.append("\nFont tables present:")
        for tag in sorted(ttfont.keys()):
            table_info = ""
            if tag == "GPOS":
                table_info = " (kerning/positioning)"
            elif tag == "GSUB":
                table_info = " (substitutions/ligatures)"
            elif tag == "kern":
                table_info = " (legacy kerning)"
            output.append("  " + tag + table_info)

        # === Kerning Statistics ===
        output.append("\n=== Kerning Statistics ===")

        if "GPOS" not in ttfont:
            output.append("  No GPOS table - no kerning information")
        else:
            gpos = ttfont["GPOS"]
            total_pairs = 0
            unique_left_glyphs = set()
            unique_right_glyphs = set()
            kerning_values = []

            # Count by lookup type
            format1_pairs = 0
            has_class_based = False

            if hasattr(gpos.table, "LookupList") and gpos.table.LookupList:
                for lookup_idx, lookup in enumerate(gpos.table.LookupList.Lookup):
                    if lookup.LookupType == 2:  # PairPos lookup
                        output.append(f"\n  Lookup #{lookup_idx}:")

                        for subtable_idx, subtable in enumerate(lookup.SubTable):
                            output.append(f"    Subtable #{subtable_idx} - Format {subtable.Format}")

                            if subtable.Format == 1 and hasattr(subtable, "PairSet"):
                                # Individual glyph pairs
                                for pair_set_idx, pair_set in enumerate(subtable.PairSet):
                                    first_glyph = subtable.Coverage.glyphs[pair_set_idx]
                                    unique_left_glyphs.add(first_glyph)

                                    for pair_value_record in pair_set.PairValueRecord:
                                        total_pairs += 1
                                        format1_pairs += 1
                                        unique_right_glyphs.add(pair_value_record.SecondGlyph)

                                        if pair_value_record.Value1:
                                            kerning_values.append(pair_value_record.Value1.XAdvance)

                            elif subtable.Format == 2:
                                # Class-based kerning
                                has_class_based = True
                                output.append("Class-based kerning (complex to count individual pairs)")

            # Print statistics
            output.append("\nTotal kerning pairs:")
            output.append(str(total_pairs))
            output.append("  Format 1 (individual pairs): " + str(format1_pairs))
            output.append("  Format 2 (class-based): " + ("Yes" if has_class_based else "No"))
            output.append("Unique left glyphs: " + str(len(unique_left_glyphs)))
            output.append("Unique right glyphs: " + str(len(unique_right_glyphs)))

            if kerning_values:
                output.append("\nKerning value statistics (in font units):")
                output.append("  Minimum: " + str(min(kerning_values)))
                output.append("  Maximum: " + str(max(kerning_values)))
                output.append("  Average: " + str(sum(kerning_values) / len(kerning_values)))

                # Count negative vs positive
                negative_count = sum(1 for v in kerning_values if v < 0)
                positive_count = sum(1 for v in kerning_values if v > 0)
                zero_count = sum(1 for v in kerning_values if v == 0)

                output.append(
                    "  Negative values (closer): "
                    + str(negative_count)
                    + " ("
                    + str(negative_count / len(kerning_values) * 100)
                    + "%)"
                )
                output.append(
                    "  Positive values (farther): "
                    + str(positive_count)
                    + " ("
                    + str(positive_count / len(kerning_values) * 100)
                    + "%)"
                )
                output.append(
                    "  Zero values: " + str(zero_count) + " (" + str(zero_count / len(kerning_values) * 100) + "%)"
                )

        # === Extreme Kerning Examples ===
        output.append("\n=== Extreme Kerning Examples ===")
        if "GPOS" in ttfont:
            gpos = ttfont["GPOS"]
            extreme_pairs = []

            if hasattr(gpos.table, "LookupList") and gpos.table.LookupList:
                for lookup in gpos.table.LookupList.Lookup:
                    if lookup.LookupType == 2:
                        for subtable in lookup.SubTable:
                            if subtable.Format == 1 and hasattr(subtable, "PairSet"):
                                for pair_set_idx, pair_set in enumerate(subtable.PairSet):
                                    first_glyph = subtable.Coverage.glyphs[pair_set_idx]
                                    for pair in pair_set.PairValueRecord:
                                        if pair.Value1:
                                            extreme_pairs.append((first_glyph, pair.SecondGlyph, pair.Value1.XAdvance))

            # Sort by kerning value
            extreme_pairs.sort(key=lambda x: x[2])

            output.append("Most negative (closest together):")
            for left, right, value in extreme_pairs[:5]:
                output.append(f"  {left} + {right}: {value}")

            output.append("\nMost positive (farther apart):")
            for left, right, value in extreme_pairs[-5:]:
                output.append(f"  {left} + {right}: {value}")

        # === GSUB Statistics ===
        output.append("\n=== GSUB (Substitutions/Ligatures) Statistics ===")

        if "GSUB" not in ttfont:
            output.append("  No GSUB table - no substitution information")
        else:
            gsub = ttfont["GSUB"]
            total_substitutions = 0
            ligature_count = 0
            single_substitutions = 0
            multiple_substitutions = 0
            alternate_substitutions = 0
            ligature_substitutions = 0

            # Get reverse cmap for glyph name to character mapping
            cmap = ttfont.getBestCmap()
            reverse_cmap = {v: k for k, v in cmap.items()}

            def glyph_to_char(glyph_name: str) -> str:
                """Convert glyph name to actual character if possible."""
                if glyph_name in reverse_cmap:
                    try:
                        return chr(reverse_cmap[glyph_name])
                    except ValueError:
                        return "?"
                return "?"

            # Count different types of substitutions
            if hasattr(gsub.table, "LookupList") and gsub.table.LookupList:
                output.append(f"\n  Total lookups: {len(gsub.table.LookupList.Lookup)}")

                for lookup_idx, lookup in enumerate(gsub.table.LookupList.Lookup):
                    lookup_type = lookup.LookupType
                    type_name = {
                        1: "Single",
                        2: "Multiple",
                        3: "Alternate",
                        4: "Ligature",
                        5: "Context",
                        6: "Chaining Context",
                        7: "Extension",
                        8: "Reverse Chaining Context",
                    }.get(lookup_type, f"Type {lookup_type}")

                    output.append(f"\n  Lookup #{lookup_idx} - {type_name} Substitution:")

                    for subtable_idx, subtable in enumerate(lookup.SubTable):
                        sub_count = 0

                        if lookup_type == 1:  # Single substitution
                            if hasattr(subtable, "mapping"):
                                sub_count = len(subtable.mapping)
                                single_substitutions += sub_count
                                output.append(f"    Subtable #{subtable_idx}: {sub_count} single substitutions")
                                # Show all substitutions
                                for from_glyph, to_glyph in subtable.mapping.items():
                                    from_char = glyph_to_char(from_glyph)
                                    to_char = glyph_to_char(to_glyph)
                                    output.append(f"      {from_glyph} ({from_char}) -> {to_glyph} ({to_char})")

                        elif lookup_type == 2:  # Multiple substitution
                            if hasattr(subtable, "mapping"):
                                sub_count = len(subtable.mapping)
                                multiple_substitutions += sub_count
                                output.append(f"    Subtable #{subtable_idx}: {sub_count} multiple substitutions")
                                for from_glyph, to_glyphs in subtable.mapping.items():
                                    from_char = glyph_to_char(from_glyph)
                                    output.append(f"      {from_glyph} ({from_char}) -> {' + '.join(to_glyphs)}")

                        elif lookup_type == 3:  # Alternate substitution
                            if hasattr(subtable, "alternates"):
                                sub_count = len(subtable.alternates)
                                alternate_substitutions += sub_count
                                output.append(f"    Subtable #{subtable_idx}: {sub_count} alternate substitutions")
                                for from_glyph, alts in subtable.alternates.items():
                                    from_char = glyph_to_char(from_glyph)
                                    output.append(f"      {from_glyph} ({from_char}) -> {' | '.join(alts)}")

                        elif lookup_type == 4:  # Ligature substitution
                            if hasattr(subtable, "LigatureSet"):
                                for lig_set in subtable.LigatureSet:
                                    set_count = len(lig_set.Ligature)
                                    sub_count += set_count
                                    ligature_substitutions += set_count
                                    if set_count > 0:
                                        first_char = glyph_to_char(lig_set.FirstGlyph)
                                        output.append(
                                            f"    First glyph {lig_set.FirstGlyph}"
                                            f" ({first_char}): {set_count} ligatures"
                                        )
                                        for ligature in lig_set.Ligature:
                                            lig_char = glyph_to_char(ligature.LigGlyph)
                                            components = lig_set.FirstGlyph + " + " + " + ".join(ligature.Component)
                                            output.append(f"      {ligature.LigGlyph} ({lig_char}) = {components}")
                                ligature_count += sub_count

                        total_substitutions += sub_count

            # Print summary
            output.append(f"\nTotal substitutions: {total_substitutions}")
            output.append(f"  Single substitutions: {single_substitutions}")
            output.append(f"  Multiple substitutions: {multiple_substitutions}")
            output.append(f"  Alternate substitutions: {alternate_substitutions}")
            output.append(f"  Ligature substitutions: {ligature_substitutions}")

            if ligature_count == 0:
                output.append("\n  No ligatures found in this font")
            else:
                output.append(f"\n  Total ligatures: {ligature_count}")

        return "\n".join(output)
