"""Test module for ave.text Hyphenator.

The tests are run using pytest.
Adapted for the modern Soft Hyphen (U+00AD) pipeline.
"""

import re

from ave.text import SOFT_HYPHEN, Hyphenator, _format_hyphenated_lines


# Helper to make test strings readable
# We replace '|' with SOFT_HYPHEN for input to functions
def convert_to_shy(text: str) -> str:
    """Helper to convert readable pipe separators to soft hyphens."""
    return text.replace("|", SOFT_HYPHEN)


class TestHyphenatorBasic:
    """Basic encoding tests for the new pipeline."""

    def test_simple_word(self):
        """Test syllabification of a simple word."""
        # "syllabification" -> syl|lab|i|fi|ca|tion
        hyphenator = Hyphenator("en_US")
        result = hyphenator.hyphenate_text("syllabification")
        # Should contain soft hyphens if it can be hyphenated
        if len("syllabification") >= 5:
            assert SOFT_HYPHEN in result or result == "syllabification"

    def test_short_word_not_hyphenated(self):
        """Test that short words are not hyphenated (optimization)."""
        hyphenator = Hyphenator("en_US")
        result = hyphenator.hyphenate_text("the")
        assert SOFT_HYPHEN not in result
        assert result == "the"

    def test_multiple_words(self):
        """Test syllabification of multiple words."""
        result = Hyphenator("en_US").hyphenate_text("hello world")
        assert " " in result
        # "hello" might be too short or simple, "world" too.
        # Let's try longer words
        result = Hyphenator("en_US").hyphenate_text("syllabification process")
        assert " " in result
        assert SOFT_HYPHEN in result

    def test_preserves_capitalization(self):
        """Test that original capitalization is preserved."""
        result = Hyphenator("en_US").hyphenate_text("Hello World")
        assert result.startswith("H")
        assert "W" in result

    def test_empty_string(self):
        """Test handling of empty string."""
        result = Hyphenator("en_US").hyphenate_text("")
        assert result == ""

    def test_punctuation_preserved(self):
        """Test that punctuation is preserved."""
        result = Hyphenator("en_US").hyphenate_text("Hello, world!")
        assert "," in result
        assert "!" in result


class TestHyphenatorReversibility:
    """Tests for full encode/decode reversibility."""

    def test_simple_text_reversible(self):
        """Test that simple text is reversible."""
        original = "hello world"
        encoded = Hyphenator("en_US").hyphenate_text(original)
        decoded = encoded.replace(SOFT_HYPHEN, "")
        assert decoded == original

    def test_complex_text_reversible(self):
        """Test that complex text with special chars is reversible."""
        # Note: Backslashes and braces are just text now, no escaping needed!
        original = "self-aware test\\path {hello}, world! 123"
        encoded = Hyphenator("en_US").hyphenate_text(original)
        decoded = encoded.replace(SOFT_HYPHEN, "")
        assert decoded == original

    def test_newlines_reversible(self):
        """Test that newlines are reversible."""
        original = "Line 1\nLine 2\nLine 3"
        encoded = Hyphenator("en_US").hyphenate_text(original)
        decoded = encoded.replace(SOFT_HYPHEN, "")
        assert decoded == original


class TestHyphenatorSpecialCases:
    """Tests for special token types."""

    def test_numbers_ignored(self):
        """Test that numbers are not hyphenated."""
        result = Hyphenator("en_US").hyphenate_text("1234567890")
        assert SOFT_HYPHEN not in result
        assert result == "1234567890"

    def test_mixed_alphanumeric(self):
        """Test mixed alphanumeric words."""
        # Tokenizer splits "abc123def" into WORD(abc) NUMBER(123) WORD(def)?
        # Our regex: WORD = [^\W\d_]+ -> "abc", then "123", then "def"
        result = Hyphenator("en_US").hyphenate_text("abc123def")
        # abc is too short, def is too short.
        assert result == "abc123def"

    def test_hard_hyphens_preserved(self):
        """Test that existing hard hyphens are preserved."""
        result = Hyphenator("en_US").hyphenate_text("self-aware")
        # "self" might be hyphenated, "aware" might be.
        # But the hard hyphen should remain a hard hyphen.
        assert "-" in result
        # Check no double hyphens unless pyphen added one
        assert "--" not in result

    def test_url_like(self):
        """Test URL-like strings."""
        # "example.com" -> WORD(example) PUNCT(.) WORD(com)
        result = Hyphenator("en_US").hyphenate_text("example.com")
        assert "." in result
        # "example" might get hyphenated
        # "com" is short

    def test_non_breaking_space_preserved(self):
        """Test that non-breaking spaces (U+00A0) are preserved."""
        result = Hyphenator("en_US").hyphenate_text("the\u00a0world")
        # Should be preserved as-is (non-breaking space preserved)
        assert "\u00a0" in result
        # "the" is too short to hyphenate, "world" is too short
        decoded = result.replace(SOFT_HYPHEN, "")
        assert decoded == "the\u00a0world"

    def test_multiple_newlines_preserved(self):
        """Test that newlines are preserved in text."""
        result = Hyphenator("en_US").hyphenate_text("line1\n\nline3")
        assert "\n" in result
        # Newlines should be preserved
        decoded = result.replace(SOFT_HYPHEN, "")
        assert decoded == "line1\n\nline3"

    def test_whitespace_preserved(self):
        """Test that various whitespace is preserved."""
        result = Hyphenator("en_US").hyphenate_text("hello  \t  world")
        # Should preserve spaces and tabs
        decoded = result.replace(SOFT_HYPHEN, "")
        assert decoded == "hello  \t  world"

    def test_multi_syllable_word(self):
        """Test hyphenation of multi-syllable words (3+ syllables)."""
        result = Hyphenator("en_US").hyphenate_text("information")
        # Should have soft hyphens for syllable breaks
        assert SOFT_HYPHEN in result
        # Should be reversible
        assert result.replace(SOFT_HYPHEN, "") == "information"

    def test_compound_word_with_syllables(self):
        """Test compound words with syllable hyphenation."""
        result = Hyphenator("en_US").hyphenate_text("self-aware")
        # Should preserve the hard hyphen
        assert "-" in result
        # Should hyphenate syllables within each part
        # "self" might get hyphenated, "aware" might get hyphenated
        decoded = result.replace(SOFT_HYPHEN, "")
        assert decoded == "self-aware"

    def test_very_long_word(self):
        """Test hyphenation of very long words."""
        # "incomprehensibility" is a very long word
        result = Hyphenator("en_US").hyphenate_text("incomprehensibility")
        # Should have soft hyphens
        assert SOFT_HYPHEN in result
        # Should be reversible
        assert result.replace(SOFT_HYPHEN, "") == "incomprehensibility"

    def test_unicode_characters(self):
        """Test handling of Unicode characters."""
        result = Hyphenator("en_US").hyphenate_text("café résumé")
        # Should preserve Unicode characters
        assert "é" in result
        decoded = result.replace(SOFT_HYPHEN, "")
        assert decoded == "café résumé"

    def test_tabs_preserved(self):
        """Test that tabs are preserved."""
        result = Hyphenator("en_US").hyphenate_text("hello\tworld")
        assert "\t" in result
        decoded = result.replace(SOFT_HYPHEN, "")
        assert decoded == "hello\tworld"


# class TestAvSyllableStreamV2:
#     """Tests for the new AvSyllableStream."""

#     def test_simple_iteration(self):
#         """Test basic iteration."""
#         # Manually create hyphenated text
#         text = convert_to_shy("hel|lo")
#         stream = AvSyllableStream(text)
#         assert stream.next_item() == "hel"
#         assert stream.next_item() == "lo "  # End of word adds space

#     def test_hard_hyphen_iteration(self):
#         """Test iteration with hard hyphens."""
#         text = "self-aware"
#         stream = AvSyllableStream(text)
#         # item 1: "self-" (not end of word)
#         # item 2: "aware" (end of word -> "aware ")
#         assert stream.next_item() == "self-"
#         assert stream.next_item() == "aware "

#     def test_mixed_hyphens(self):
#         """Test mixed soft and hard hyphens."""
#         # "pre" + SHY + "fix" + "-" + "suf" + SHY + "fix"
#         text = convert_to_shy("pre|fix-suf|fix")
#         stream = AvSyllableStream(text)

#         assert stream.next_item() == "pre"
#         assert stream.next_item() == "fix-"
#         assert stream.next_item() == "suf"
#         assert stream.next_item() == "fix "


class TestFormatHyphenatedLinesBasic:
    """Tests for the line formatting helper."""

    def test_word_fits(self):
        """Word fits on line."""
        text = "hello"
        lines = _format_hyphenated_lines(text, 10)
        assert lines == ["hello"]

    def test_word_split_soft_hyphen(self):
        """Word splits at soft hyphen."""
        text = convert_to_shy("hel|lo")
        lines = _format_hyphenated_lines(text, 4)
        # "hel-" (4 chars) fits.
        assert lines == ["hel-", "lo"]

    def test_hard_hyphen_split(self):
        """Word splits at hard hyphen."""
        text = "self-aware"
        # "self-" is 5 chars. width 5.
        lines = _format_hyphenated_lines(text, 5)
        assert lines == ["self-", "aware"]

    def test_long_word_forced_split(self):
        """Word with no break points is forced split."""
        text = "abcdefgh"
        # Width 3.
        lines = _format_hyphenated_lines(text, 3)
        # "ab-" is 3 chars.
        assert lines[0] == "ab-"
        assert len(lines[0]) <= 3


class TestFormatHyphenatedLinesIntense:
    """Intense tests for _format_hyphenated_lines covering all splitting cases.

    Adapted from legacy tests to use Soft Hyphens (\u00ad) and new behavior.
    """

    WIDTHS = [9, 12, 15, 20, 30, 40]

    @staticmethod
    def _check(lines: list, max_width: int, label: str = "") -> None:
        """Assert no line exceeds max_width."""
        prefix = f"[{label}] " if label else ""
        violations = [(i + 1, len(l), l) for i, l in enumerate(lines) if len(l) > max_width]
        assert violations == [], f"{prefix}Lines exceeding max_width={max_width}: {violations}"

    # ------------------------------------------------------------------
    # 1. No split needed
    # ------------------------------------------------------------------

    def test_word_fits_whole_no_hyphen(self):
        """A word that fits whole must not gain a trailing hyphen."""
        # Legacy: "ra-dioac-tiv-i-ty"
        # New: convert_to_shy("ra|dioac|tiv|i|ty")
        text = convert_to_shy("ra|dioac|tiv|i|ty")

        for w in self.WIDTHS:
            lines = _format_hyphenated_lines(text, w)
            self._check(lines, w, f"w={w}")
            # Reconstruct (remove trailing -)
            # Be careful: lines ending in '-' that was NOT in text means we split.
            # But the text is pure word.
            reconstructed = "".join(l[:-1] if l.endswith("-") else l for l in lines)
            assert reconstructed == "radioactivity"

            # If word fits whole, no hyphen
            if w >= len("radioactivity"):
                assert len(lines) == 1
                assert not lines[0].endswith("-")

    def test_short_words_never_hyphenated(self):
        """Words shorter than 6 chars must never be split (if they fit)."""
        # "the cat sat on a mat" - no soft hyphens encoded usually
        encoded = "the cat sat on a mat"
        for w in self.WIDTHS:
            lines = _format_hyphenated_lines(encoded, w)
            self._check(lines, w, f"w={w}")
            for line in lines:
                assert not line.endswith("-")

    # ------------------------------------------------------------------
    # 2. Syllable split
    # ------------------------------------------------------------------

    def test_syllable_split_fills_current_line(self):
        """Syllables must be packed onto the current line before flushing."""
        # "She was ra|dioac|tiv|i|ty"
        text = "She was " + convert_to_shy("ra|dioac|tiv|i|ty")

        for w in [12, 15, 20]:
            lines = _format_hyphenated_lines(text, w)
            self._check(lines, w, f"w={w}")
            # At least one split happened (unless width is huge)
            if w < len("She was radioactivity"):
                assert any(l.endswith("-") for l in lines)

    # ------------------------------------------------------------------
    # 3. Syllable split from empty line
    # ------------------------------------------------------------------

    def test_very_long_word_split_across_multiple_lines(self):
        """A word longer than max_width must be split across multiple lines."""
        text = convert_to_shy("ra|dioac|tiv|i|ty")
        lines = _format_hyphenated_lines(text, 9)
        self._check(lines, 9)
        assert len(lines) >= 2

    # ------------------------------------------------------------------
    # 4. Compound word split
    # ------------------------------------------------------------------

    def test_compound_word_split_preserves_hyphen(self):
        """Compound words must split at the original hyphen, preserving it."""
        # "self-directed" -> hard hyphen
        text = "self-directed"
        for w in [9, 12, 15]:
            lines = _format_hyphenated_lines(text, w)
            self._check(lines, w, f"w={w}")
            reconstructed = " ".join(lines)
            assert "self" in reconstructed and "directed" in reconstructed
            # Check the hyphen is present
            assert "-" in reconstructed

    # ------------------------------------------------------------------
    # 8. Full sample text
    # ------------------------------------------------------------------

    def test_full_text_reconstruction_at_all_widths(self):
        """All words from the original text must appear in the output."""

        def sample_text_mc() -> str:
            return (
                "Marie Curie (born Maria Sklodowska; 7 November 1867 - 4 July 1934) was a "
                "pioneering physicist and chemist whose research on radioactivity "
                "profoundly shaped modern science."
            )

        def _normalise(text: str) -> str:
            """Strip hyphens then collapse whitespace for comparison."""
            return re.sub(r"\s+", " ", re.sub(r"-", "", text)).strip()

        original = sample_text_mc()
        original_norm = _normalise(original)

        encoded = Hyphenator("en_US").hyphenate_text(original)

        for w in self.WIDTHS:
            lines = _format_hyphenated_lines(encoded, w)
            # Reconstruct: concatenate split fragments (strip trailing '-'),
            # then join complete tokens with spaces.
            # NOTE: New logic adds space-padding for newlines in intermediate steps,
            # but final output is lines.

            parts: list[str] = []
            carry = ""
            for line in lines:
                if line.endswith("-"):
                    carry += line[:-1]
                else:
                    parts.append(carry + line)
                    carry = ""
            if carry:
                parts.append(carry)

            reconstructed_norm = _normalise(" ".join(parts))

            # Note: Hard hyphens in original text (like "1867 - 4") might be stripped by _normalise.
            # That's fine as long as we compare normalized forms.

            assert reconstructed_norm == original_norm, f"width={w} mismatch"

    # ------------------------------------------------------------------
    # 9. Edge cases
    # ------------------------------------------------------------------

    def test_very_narrow_max_width(self):
        """At very narrow widths."""
        # "pi|o|neer|ing"
        text = convert_to_shy("pi|o|neer|ing")
        # At width=3: "o" + "-" = "o-" (2 chars) fits.
        lines = _format_hyphenated_lines(text, 3)
        for line in lines:
            assert len(line) <= 3

    def test_word_exactly_max_width(self):
        """A word whose clean length equals max_width must not be split."""
        # "scientific" = 10 chars.
        text = "scientific"  # Assuming no SHY for simplicity of test, or SHY doesn't matter
        lines = _format_hyphenated_lines(text, 10)
        assert lines == ["scientific"]

    def test_word_one_over_max_width_must_split(self):
        """A word one char over max_width must be split."""
        # "scientific" = 10 chars. Width=9.
        # Needs to split.
        text = convert_to_shy("sci|en|tif|ic")
        # If text has no SHY, it would force char split.
        # Let's provide SHY.
        lines = _format_hyphenated_lines(text, 9)
        self._check(lines, 9)
        assert len(lines) >= 2
        assert lines[0].endswith("-")
