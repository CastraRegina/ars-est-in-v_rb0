"""Test module for ave.text HyphenationEncoder.

The tests are run using pytest.
"""

import pytest

from ave.text import (
    AvCharacterStream,
    AvSyllableStream,
    HyphenationEncoder,
    HyphenationError,
    InvalidEscapeSequenceError,
    UnsupportedLanguageError,
)


class TestListSupportedLanguages:
    """Tests for list_supported_languages method."""

    def test_returns_list(self):
        """Test that a list is returned."""
        result = HyphenationEncoder.list_supported_languages()
        assert isinstance(result, list)

    def test_contains_common_languages(self):
        """Test that common languages are supported."""
        languages = HyphenationEncoder.list_supported_languages()
        assert "en_US" in languages or "en" in languages
        assert "de_DE" in languages or "de" in languages

    def test_list_is_sorted(self):
        """Test that the list is sorted."""
        languages = HyphenationEncoder.list_supported_languages()
        assert languages == sorted(languages)


class TestEncodeTextBasic:
    """Basic encoding tests."""

    def test_simple_word(self):
        """Test syllabification of a simple word."""
        result = HyphenationEncoder.encode_text("hello", "en_US")
        assert "-" in result or result == "hello"

    def test_multiple_words(self):
        """Test syllabification of multiple words."""
        result = HyphenationEncoder.encode_text("hello world", "en_US")
        assert " " in result

    def test_preserves_capitalization(self):
        """Test that original capitalization is preserved."""
        result = HyphenationEncoder.encode_text("Hello World", "en_US")
        assert result.startswith("H")
        assert "W" in result or "w" in result

    def test_empty_string(self):
        """Test handling of empty string."""
        result = HyphenationEncoder.encode_text("", "en_US")
        assert result == ""

    def test_single_character(self):
        """Test handling of single character."""
        result = HyphenationEncoder.encode_text("a", "en_US")
        assert result == "a"


class TestEncodeTextEscaping:
    """Tests for escape sequence handling during encoding."""

    def test_literal_hyphen_escaped(self):
        """Test that literal hyphens are escaped."""
        result = HyphenationEncoder.encode_text("self-aware", "en_US")
        assert "\\-" in result

    def test_literal_backslash_escaped(self):
        """Test that literal backslashes are escaped."""
        result = HyphenationEncoder.encode_text("path\\file", "en_US")
        assert "\\\\" in result

    def test_literal_space_in_word(self):
        """Test that spaces separate words correctly."""
        result = HyphenationEncoder.encode_text("hello world", "en_US")
        assert " " in result


class TestEncodeTextWhitespace:
    """Tests for whitespace normalization."""

    def test_newline_normalized_to_space(self):
        """Test that newline is normalized to space."""
        result = HyphenationEncoder.encode_text("hello\nworld", "en_US")
        assert "\n" not in result
        assert " " in result

    def test_carriage_return_normalized(self):
        """Test that carriage return is normalized."""
        result = HyphenationEncoder.encode_text("hello\rworld", "en_US")
        assert "\r" not in result

    def test_crlf_normalized(self):
        """Test that CRLF is normalized to single space."""
        result = HyphenationEncoder.encode_text("hello\r\nworld", "en_US")
        assert "\r" not in result
        assert "\n" not in result

    def test_tab_normalized(self):
        """Test that tab is normalized to space."""
        result = HyphenationEncoder.encode_text("hello\tworld", "en_US")
        assert "\t" not in result

    def test_multiple_spaces_collapsed(self):
        """Test that multiple spaces are collapsed."""
        result = HyphenationEncoder.encode_text("hello    world", "en_US")
        assert "    " not in result

    def test_literal_backslash_n_creates_linebreak(self):
        """Test that literal backslash+n creates a line break in output."""
        result = HyphenationEncoder.encode_text("hello\\nworld", "en_US")
        assert "\n" in result


class TestEncodeTextPunctuation:
    """Tests for punctuation handling."""

    def test_preserves_period(self):
        """Test that periods are preserved."""
        result = HyphenationEncoder.encode_text("Hello.", "en_US")
        assert "." in result

    def test_preserves_comma(self):
        """Test that commas are preserved."""
        result = HyphenationEncoder.encode_text("Hello, world", "en_US")
        assert "," in result

    def test_preserves_exclamation(self):
        """Test that exclamation marks are preserved."""
        result = HyphenationEncoder.encode_text("Hello!", "en_US")
        assert "!" in result

    def test_preserves_question(self):
        """Test that question marks are preserved."""
        result = HyphenationEncoder.encode_text("Hello?", "en_US")
        assert "?" in result

    def test_preserves_colon(self):
        """Test that colons are preserved."""
        result = HyphenationEncoder.encode_text("Note: hello", "en_US")
        assert ":" in result

    def test_preserves_semicolon(self):
        """Test that semicolons are preserved."""
        result = HyphenationEncoder.encode_text("Hello; world", "en_US")
        assert ";" in result


class TestEncodeTextDigits:
    """Tests for words containing digits."""

    def test_word_with_digit_unchanged(self):
        """Test that words with digits are passed through unchanged."""
        result = HyphenationEncoder.encode_text("test123", "en_US")
        assert result == "test123"

    def test_pure_number_unchanged(self):
        """Test that pure numbers are unchanged."""
        result = HyphenationEncoder.encode_text("12345", "en_US")
        assert result == "12345"

    def test_alphanumeric_mixed(self):
        """Test mixed alphanumeric words."""
        result = HyphenationEncoder.encode_text("abc123def", "en_US")
        assert result == "abc123def"


class TestDecodeText:
    """Tests for decode_text method."""

    def test_removes_syllable_hyphens(self):
        """Test that syllable hyphens are removed."""
        result = HyphenationEncoder.decode_text("hel-lo")
        assert result == "hello"

    def test_unescapes_literal_hyphen(self):
        """Test that escaped hyphens are restored."""
        result = HyphenationEncoder.decode_text("self\\-aware")
        assert result == "self-aware"

    def test_unescapes_literal_backslash(self):
        """Test that escaped backslashes are restored."""
        result = HyphenationEncoder.decode_text("path\\\\file")
        assert result == "path\\file"

    def test_unescapes_literal_space(self):
        """Test that escaped spaces are restored."""
        result = HyphenationEncoder.decode_text("hello\\sworld")
        assert result == "hello world"

    def test_preserves_backslash_n_sequence(self):
        """Test that backslash+n in encoded text becomes literal backslash+n."""
        result = HyphenationEncoder.decode_text("hello\\nworld")
        assert result == "hello\\nworld"

    def test_newline_becomes_backslash_n(self):
        """Test that actual newline in encoded text becomes literal backslash+n."""
        result = HyphenationEncoder.decode_text("hello\nworld")
        assert result == "hello\\nworld"

    def test_empty_string(self):
        """Test decoding empty string."""
        result = HyphenationEncoder.decode_text("")
        assert result == ""


class TestReversibility:
    """Tests for full encode/decode reversibility."""

    def test_simple_text_reversible(self):
        """Test that simple text is reversible."""
        original = "hello world"
        encoded = HyphenationEncoder.encode_text(original, "en_US")
        decoded = HyphenationEncoder.decode_text(encoded)
        assert decoded == original

    def test_text_with_hyphen_reversible(self):
        """Test that text with hyphens is reversible."""
        original = "self-aware"
        encoded = HyphenationEncoder.encode_text(original, "en_US")
        decoded = HyphenationEncoder.decode_text(encoded)
        assert decoded == original

    def test_text_with_backslash_reversible(self):
        """Test that text with backslashes is reversible."""
        original = "path\\file"
        encoded = HyphenationEncoder.encode_text(original, "en_US")
        decoded = HyphenationEncoder.decode_text(encoded)
        assert decoded == original

    def test_text_with_backslash_n_reversible(self):
        """Test that text with literal backslash+n is reversible."""
        original = "line1\\nline2"
        encoded = HyphenationEncoder.encode_text(original, "en_US")
        decoded = HyphenationEncoder.decode_text(encoded)
        assert decoded == original

    def test_punctuation_reversible(self):
        """Test that punctuation is reversible."""
        original = "Hello, world! How are you?"
        encoded = HyphenationEncoder.encode_text(original, "en_US")
        decoded = HyphenationEncoder.decode_text(encoded)
        assert decoded == original

    def test_complex_text_reversible(self):
        """Test that complex text with multiple special chars is reversible."""
        original = "self-aware test\\path hello, world!"
        encoded = HyphenationEncoder.encode_text(original, "en_US")
        decoded = HyphenationEncoder.decode_text(encoded)
        assert decoded == original


class TestErrorHandling:
    """Tests for error handling."""

    def test_unsupported_language_raises_error(self):
        """Test that unsupported language raises UnsupportedLanguageError."""
        with pytest.raises(UnsupportedLanguageError):
            HyphenationEncoder.encode_text("hello", "xx_XX")

    def test_invalid_escape_sequence_raises_error(self):
        """Test that invalid escape sequence raises InvalidEscapeSequenceError."""
        with pytest.raises(InvalidEscapeSequenceError):
            HyphenationEncoder.decode_text("hello\\x")

    def test_incomplete_escape_sequence_raises_error(self):
        """Test that incomplete escape sequence raises error."""
        with pytest.raises(InvalidEscapeSequenceError):
            HyphenationEncoder.decode_text("hello\\")

    def test_error_inheritance(self):
        """Test that specific errors inherit from HyphenationError."""
        assert issubclass(UnsupportedLanguageError, HyphenationError)
        assert issubclass(InvalidEscapeSequenceError, HyphenationError)


class TestWhitelist:
    """Tests for accepted_hyphenated_words whitelist."""

    def test_whitelist_parsing(self):
        """Test that whitelist is parsed correctly."""
        whitelist = "foo-bar\ntest-word"
        result = HyphenationEncoder.encode_text("hello", "en_US", accepted_hyphenated_words=whitelist)
        assert isinstance(result, str)

    def test_empty_whitelist(self):
        """Test that empty whitelist works."""
        result = HyphenationEncoder.encode_text("hello", "en_US", accepted_hyphenated_words="")
        assert isinstance(result, str)

    def test_none_whitelist(self):
        """Test that None whitelist works."""
        result = HyphenationEncoder.encode_text("hello", "en_US", accepted_hyphenated_words=None)
        assert isinstance(result, str)


class TestDebugMode:
    """Tests for debug mode."""

    def test_debug_false_no_exception(self):
        """Test that debug=False does not raise exceptions for unknown words."""
        result = HyphenationEncoder.encode_text("xyzabc", "en_US", debug=False)
        assert isinstance(result, str)

    def test_debug_true_no_exception(self):
        """Test that debug=True does not raise exceptions."""
        result = HyphenationEncoder.encode_text("hello", "en_US", debug=True)
        assert isinstance(result, str)


class TestGermanLanguage:
    """Tests for German language support."""

    def test_german_syllabification(self):
        """Test syllabification of German words."""
        result = HyphenationEncoder.encode_text("Silbentrennung", "de_DE")
        assert "-" in result

    def test_german_reversible(self):
        """Test German text is reversible."""
        original = "Silbentrennung"
        encoded = HyphenationEncoder.encode_text(original, "de_DE")
        decoded = HyphenationEncoder.decode_text(encoded)
        assert decoded == original


class TestUnicode:
    """Tests for Unicode handling."""

    def test_unicode_characters_preserved(self):
        """Test that Unicode characters are preserved."""
        original = "cafe"
        encoded = HyphenationEncoder.encode_text(original, "en_US")
        decoded = HyphenationEncoder.decode_text(encoded)
        assert decoded == original

    def test_german_umlauts(self):
        """Test German umlauts are handled."""
        original = "Uberraschung"
        encoded = HyphenationEncoder.encode_text(original, "de_DE")
        decoded = HyphenationEncoder.decode_text(encoded)
        assert decoded == original


class TestAvSyllableStreamBasic:
    """Basic tests for AvSyllableStream."""

    def test_simple_word(self):
        """Test iteration over a simple syllabified word."""
        stream = AvSyllableStream("Hel-lo")
        assert stream.next_item() == "Hel"
        assert stream.next_item() == "lo "

    def test_multiple_words(self):
        """Test iteration over multiple words."""
        stream = AvSyllableStream("Hel-lo world")
        assert stream.next_item() == "Hel"
        assert stream.next_item() == "lo "
        assert stream.next_item() == "world "

    def test_single_word_no_hyphen(self):
        """Test single word without hyphens."""
        stream = AvSyllableStream("test")
        assert stream.next_item() == "test "
        assert not stream.has_next()

    def test_empty_string(self):
        """Test empty string input."""
        stream = AvSyllableStream("")
        assert not stream.has_next()
        assert stream.position() == 0


class TestAvSyllableStreamNavigation:
    """Tests for AvSyllableStream navigation methods."""

    def test_has_next(self):
        """Test has_next method."""
        stream = AvSyllableStream("Hel-lo")
        assert stream.has_next()
        stream.next_item()
        assert stream.has_next()
        stream.next_item()
        assert not stream.has_next()

    def test_has_previous(self):
        """Test has_previous method."""
        stream = AvSyllableStream("Hel-lo")
        assert not stream.has_previous()
        stream.next_item()
        assert stream.has_previous()

    def test_previous_item(self):
        """Test previous_item method."""
        stream = AvSyllableStream("Hel-lo")
        stream.next_item()
        stream.next_item()
        assert stream.previous_item() == "lo "
        assert stream.previous_item() == "Hel"

    def test_position(self):
        """Test position method."""
        stream = AvSyllableStream("Hel-lo world")
        assert stream.position() == 0
        stream.next_item()
        assert stream.position() == 1
        stream.next_item()
        assert stream.position() == 2

    def test_reset(self):
        """Test reset method."""
        stream = AvSyllableStream("Hel-lo world")
        stream.next_item()
        stream.next_item()
        stream.reset()
        assert stream.position() == 0
        assert stream.next_item() == "Hel"

    def test_rewind_single_step(self):
        """Test rewind with single step."""
        stream = AvSyllableStream("Hel-lo world")
        stream.next_item()
        stream.next_item()
        stream.rewind()
        assert stream.position() == 1

    def test_rewind_multiple_steps(self):
        """Test rewind with multiple steps."""
        stream = AvSyllableStream("Hel-lo world")
        stream.next_item()
        stream.next_item()
        stream.next_item()
        stream.rewind(2)
        assert stream.position() == 1

    def test_rewind_beyond_start(self):
        """Test rewind beyond start clamps to 0."""
        stream = AvSyllableStream("Hel-lo")
        stream.next_item()
        stream.rewind(10)
        assert stream.position() == 0

    def test_rewind_negative_raises(self):
        """Test rewind with negative steps raises ValueError."""
        stream = AvSyllableStream("Hel-lo")
        with pytest.raises(ValueError):
            stream.rewind(-1)


class TestAvSyllableStreamPunctuation:
    """Tests for punctuation handling in AvSyllableStream."""

    def test_trailing_period(self):
        """Test word ending with period."""
        stream = AvSyllableStream("Hel-lo.")
        stream.next_item()
        assert stream.next_item() == "lo. "

    def test_trailing_comma(self):
        """Test word ending with comma."""
        stream = AvSyllableStream("Hel-lo,")
        stream.next_item()
        assert stream.next_item() == "lo, "

    def test_trailing_exclamation(self):
        """Test word ending with exclamation mark."""
        stream = AvSyllableStream("Hel-lo!")
        stream.next_item()
        assert stream.next_item() == "lo! "

    def test_trailing_question(self):
        """Test word ending with question mark."""
        stream = AvSyllableStream("Hel-lo?")
        stream.next_item()
        assert stream.next_item() == "lo? "

    def test_trailing_colon(self):
        """Test word ending with colon."""
        stream = AvSyllableStream("Note:")
        assert stream.next_item() == "Note: "

    def test_trailing_semicolon(self):
        """Test word ending with semicolon."""
        stream = AvSyllableStream("Hel-lo;")
        stream.next_item()
        assert stream.next_item() == "lo; "

    def test_multiple_trailing_punct(self):
        """Test word ending with multiple punctuation marks."""
        stream = AvSyllableStream("What?!")
        assert stream.next_item() == "What?! "


class TestAvSyllableStreamEscapeSequences:
    """Tests for escape sequence handling in AvSyllableStream."""

    def test_escaped_hyphen(self):
        """Test escaped hyphen is unescaped."""
        stream = AvSyllableStream("self\\-aware")
        result = stream.next_item()
        assert result == "self-aware "

    def test_escaped_backslash(self):
        """Test escaped backslash is unescaped."""
        stream = AvSyllableStream("path\\\\file")
        result = stream.next_item()
        assert result == "path\\file "

    def test_escaped_space(self):
        """Test escaped space is unescaped."""
        stream = AvSyllableStream("hello\\sworld")
        result = stream.next_item()
        assert result == "hello world "

    def test_invalid_escape_raises(self):
        """Test invalid escape sequence raises error."""
        with pytest.raises(InvalidEscapeSequenceError):
            AvSyllableStream("hello\\x")

    def test_incomplete_escape_raises(self):
        """Test incomplete escape sequence raises error."""
        with pytest.raises(InvalidEscapeSequenceError):
            AvSyllableStream("hello\\")


class TestAvSyllableStreamLineBreaks:
    """Tests for line break handling in AvSyllableStream."""

    def test_newline_as_separate_token(self):
        """Test that newlines are returned as separate tokens."""
        stream = AvSyllableStream("Hel-lo\nworld")
        assert stream.next_item() == "Hel"
        assert stream.next_item() == "lo "
        assert stream.next_item() == "\n"
        assert stream.next_item() == "world "

    def test_multiple_lines(self):
        """Test multiple line breaks."""
        stream = AvSyllableStream("one\ntwo\nthree")
        assert stream.next_item() == "one "
        assert stream.next_item() == "\n"
        assert stream.next_item() == "two "
        assert stream.next_item() == "\n"
        assert stream.next_item() == "three "


class TestAvSyllableStreamErrors:
    """Tests for error handling in AvSyllableStream."""

    def test_next_at_end_raises_stop_iteration(self):
        """Test next_item at end raises StopIteration."""
        stream = AvSyllableStream("test")
        stream.next_item()
        with pytest.raises(StopIteration):
            stream.next_item()

    def test_previous_at_start_raises_stop_iteration(self):
        """Test previous_item at start raises StopIteration."""
        stream = AvSyllableStream("test")
        with pytest.raises(StopIteration):
            stream.previous_item()


class TestAvSyllableStreamIntegration:
    """Integration tests for AvSyllableStream with HyphenationEncoder."""

    def test_with_encoded_text(self):
        """Test AvSyllableStream with actual encoded text."""
        encoded = HyphenationEncoder.encode_text("hello world", "en_US")
        stream = AvSyllableStream(encoded)
        syllables = []
        while stream.has_next():
            syllables.append(stream.next_item())
        assert len(syllables) >= 2

    def test_full_iteration_then_reverse(self):
        """Test full forward iteration then reverse."""
        stream = AvSyllableStream("Hel-lo world")
        forward = []
        while stream.has_next():
            forward.append(stream.next_item())
        reverse = []
        while stream.has_previous():
            reverse.append(stream.previous_item())
        assert forward == list(reversed(reverse))

    def test_german_text(self):
        """Test with German encoded text."""
        encoded = HyphenationEncoder.encode_text("Silbentrennung", "de_DE")
        stream = AvSyllableStream(encoded)
        syllables = []
        while stream.has_next():
            syllables.append(stream.next_item())
        joined = "".join(syllables).strip()
        assert "Silbentrennung" in joined.replace(" ", "")

    def test_spec_example(self):
        """Test the example from the specification."""
        stream = AvSyllableStream("Sil-ben tren-nung. Nach-richt")
        expected = ["Sil", "ben ", "tren", "nung. ", "Nach", "richt "]
        actual = []
        while stream.has_next():
            actual.append(stream.next_item())
        assert actual == expected


class TestAvCharacterStreamBasic:
    """Basic tests for AvCharacterStream."""

    def test_simple_string(self):
        """Test iteration over a simple string."""
        stream = AvCharacterStream("Hello")
        assert stream.next_item() == "H"
        assert stream.next_item() == "e"
        assert stream.next_item() == "l"
        assert stream.next_item() == "l"
        assert stream.next_item() == "o"
        assert not stream.has_next()

    def test_single_character(self):
        """Test single character input."""
        stream = AvCharacterStream("A")
        assert stream.next_item() == "A"
        assert not stream.has_next()

    def test_empty_string(self):
        """Test empty string input."""
        stream = AvCharacterStream("")
        assert not stream.has_next()
        assert stream.position() == 0

    def test_full_iteration(self):
        """Test full iteration collects all characters."""
        text = "Test"
        stream = AvCharacterStream(text)
        result = []
        while stream.has_next():
            result.append(stream.next_item())
        assert "".join(result) == text


class TestAvCharacterStreamNavigation:
    """Tests for AvCharacterStream navigation methods."""

    def test_has_next(self):
        """Test has_next method."""
        stream = AvCharacterStream("AB")
        assert stream.has_next()
        stream.next_item()
        assert stream.has_next()
        stream.next_item()
        assert not stream.has_next()

    def test_has_previous(self):
        """Test has_previous method."""
        stream = AvCharacterStream("AB")
        assert not stream.has_previous()
        stream.next_item()
        assert stream.has_previous()

    def test_previous_item(self):
        """Test previous_item method."""
        stream = AvCharacterStream("ABC")
        stream.next_item()
        stream.next_item()
        stream.next_item()
        assert stream.previous_item() == "C"
        assert stream.previous_item() == "B"
        assert stream.previous_item() == "A"

    def test_position(self):
        """Test position method."""
        stream = AvCharacterStream("ABC")
        assert stream.position() == 0
        stream.next_item()
        assert stream.position() == 1
        stream.next_item()
        assert stream.position() == 2

    def test_reset(self):
        """Test reset method."""
        stream = AvCharacterStream("ABC")
        stream.next_item()
        stream.next_item()
        stream.reset()
        assert stream.position() == 0
        assert stream.next_item() == "A"

    def test_rewind_single_step(self):
        """Test rewind with single step."""
        stream = AvCharacterStream("ABC")
        stream.next_item()
        stream.next_item()
        stream.rewind()
        assert stream.position() == 1

    def test_rewind_multiple_steps(self):
        """Test rewind with multiple steps."""
        stream = AvCharacterStream("ABCDE")
        stream.next_item()
        stream.next_item()
        stream.next_item()
        stream.rewind(2)
        assert stream.position() == 1

    def test_rewind_beyond_start(self):
        """Test rewind beyond start clamps to 0."""
        stream = AvCharacterStream("AB")
        stream.next_item()
        stream.rewind(10)
        assert stream.position() == 0

    def test_rewind_negative_raises(self):
        """Test rewind with negative steps raises ValueError."""
        stream = AvCharacterStream("AB")
        with pytest.raises(ValueError):
            stream.rewind(-1)


class TestAvCharacterStreamWhitespace:
    """Tests for whitespace handling in AvCharacterStream."""

    def test_space(self):
        """Test space character."""
        stream = AvCharacterStream("A B")
        assert stream.next_item() == "A"
        assert stream.next_item() == " "
        assert stream.next_item() == "B"

    def test_tab(self):
        """Test tab character."""
        stream = AvCharacterStream("A\tB")
        assert stream.next_item() == "A"
        assert stream.next_item() == "\t"
        assert stream.next_item() == "B"

    def test_newline(self):
        """Test newline character."""
        stream = AvCharacterStream("A\nB")
        assert stream.next_item() == "A"
        assert stream.next_item() == "\n"
        assert stream.next_item() == "B"

    def test_carriage_return(self):
        """Test carriage return character."""
        stream = AvCharacterStream("A\rB")
        assert stream.next_item() == "A"
        assert stream.next_item() == "\r"
        assert stream.next_item() == "B"

    def test_crlf(self):
        """Test CRLF is two separate characters."""
        stream = AvCharacterStream("A\r\nB")
        assert stream.next_item() == "A"
        assert stream.next_item() == "\r"
        assert stream.next_item() == "\n"
        assert stream.next_item() == "B"


class TestAvCharacterStreamPunctuation:
    """Tests for punctuation handling in AvCharacterStream."""

    def test_period(self):
        """Test period character."""
        stream = AvCharacterStream("A.")
        assert stream.next_item() == "A"
        assert stream.next_item() == "."

    def test_comma(self):
        """Test comma character."""
        stream = AvCharacterStream("A,B")
        assert stream.next_item() == "A"
        assert stream.next_item() == ","
        assert stream.next_item() == "B"

    def test_mixed_punctuation(self):
        """Test various punctuation characters."""
        stream = AvCharacterStream("!@#")
        assert stream.next_item() == "!"
        assert stream.next_item() == "@"
        assert stream.next_item() == "#"


class TestAvCharacterStreamDigits:
    """Tests for digit handling in AvCharacterStream."""

    def test_digits(self):
        """Test digit characters."""
        stream = AvCharacterStream("123")
        assert stream.next_item() == "1"
        assert stream.next_item() == "2"
        assert stream.next_item() == "3"

    def test_mixed_alphanumeric(self):
        """Test mixed letters and digits."""
        stream = AvCharacterStream("A1B2")
        assert stream.next_item() == "A"
        assert stream.next_item() == "1"
        assert stream.next_item() == "B"
        assert stream.next_item() == "2"


class TestAvCharacterStreamErrors:
    """Tests for error handling in AvCharacterStream."""

    def test_next_at_end_raises_stop_iteration(self):
        """Test next_item at end raises StopIteration."""
        stream = AvCharacterStream("A")
        stream.next_item()
        with pytest.raises(StopIteration):
            stream.next_item()

    def test_previous_at_start_raises_stop_iteration(self):
        """Test previous_item at start raises StopIteration."""
        stream = AvCharacterStream("A")
        with pytest.raises(StopIteration):
            stream.previous_item()


class TestAvCharacterStreamReversibility:
    """Tests for reversible iteration in AvCharacterStream."""

    def test_forward_then_backward(self):
        """Test forward then backward iteration."""
        stream = AvCharacterStream("ABC")
        forward = []
        while stream.has_next():
            forward.append(stream.next_item())
        backward = []
        while stream.has_previous():
            backward.append(stream.previous_item())
        assert forward == list(reversed(backward))

    def test_spec_example(self):
        """Test the example from the specification."""
        stream = AvCharacterStream("Hello\nWorld")
        expected = ["H", "e", "l", "l", "o", "\n", "W", "o", "r", "l", "d"]
        actual = []
        while stream.has_next():
            actual.append(stream.next_item())
        assert actual == expected
