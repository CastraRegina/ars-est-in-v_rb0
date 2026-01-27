"""Reversible syllabification (hyphenation) module for arbitrary text.

This module provides the HyphenationEncoder class that converts text into a
meta-text representation where syllables are separated by '-' characters.
The transformation is fully reversible.

Uses pyphen as the authoritative syllabification engine and hunspell
as an optional validation/warning layer.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List

import hunspell  # pylint: disable=c-extension-no-member
import pyphen


@dataclass
class _EncodingContext:
    """Internal context for encoding operations."""

    pyphen_dict: pyphen.Pyphen
    hunspell_dict: hunspell.HunSpell | None
    whitelist: set[str]
    debug: bool
    warnings: list[str] = field(default_factory=list)


logger = logging.getLogger(__name__)


class HyphenationError(Exception):
    """Base exception for hyphenation-related errors."""


class UnsupportedLanguageError(HyphenationError):
    """Raised when an unsupported language is requested."""


class InvalidEscapeSequenceError(HyphenationError):
    """Raised when an invalid escape sequence is encountered."""


class HyphenationEncoder:
    """Encoder for reversible syllabification of arbitrary text.

    All methods are static. The class converts text into a meta-text
    representation where syllables are separated by '-' characters.

    Example:
        >>> HyphenationEncoder.encode_text("Hello world", "en_US")
        'Hel-lo world'
        >>> HyphenationEncoder.decode_text("Hel-lo world")
        'Hello world'
    """

    _ESCAPE_MAP: dict[str, str] = {
        "\\": "\\\\",
        "-": "\\-",
        " ": "\\s",
    }

    _UNESCAPE_MAP: dict[str, str] = {
        "\\\\": "\\",
        "\\-": "-",
        "\\s": " ",
    }

    _hunspell_cache: dict[str, hunspell.HunSpell | None] = {}

    @staticmethod
    def list_supported_languages() -> list[str]:
        """Return a list of languages supported by pyphen.

        Returns:
            A sorted list of language codes supported by pyphen.
        """
        return sorted(pyphen.LANGUAGES.keys())

    @staticmethod
    def _get_pyphen_dict(language: str) -> pyphen.Pyphen:
        """Get a pyphen dictionary for the given language.

        Args:
            language: Language code (e.g., 'en_US', 'de_DE').

        Returns:
            A pyphen.Pyphen instance for the language.

        Raises:
            UnsupportedLanguageError: If the language is not supported.
        """
        if language not in pyphen.LANGUAGES:
            supported = ", ".join(sorted(pyphen.LANGUAGES.keys())[:10])
            raise UnsupportedLanguageError(
                f"Language '{language}' is not supported by pyphen. " f"Supported languages include: {supported}..."
            )
        return pyphen.Pyphen(lang=language)

    @staticmethod
    def _get_hunspell_dict(language: str) -> hunspell.HunSpell | None:
        """Get a hunspell dictionary for the given language.

        Args:
            language: Language code (e.g., 'en_US', 'de_DE').

        Returns:
            A hunspell HunSpell instance, or None if unavailable.
        """
        if language in HyphenationEncoder._hunspell_cache:
            return HyphenationEncoder._hunspell_cache[language]

        try:
            lang_code = language.split("_")[0]
            # Try common hunspell dictionary paths
            paths = [
                f"/usr/share/hunspell/{lang_code}.dic",
                f"/usr/share/myspell/{lang_code}.dic",
                f"/usr/share/hunspell/{language}.dic",
            ]

            for dic_path in paths:
                aff_path = dic_path.replace(".dic", ".aff")
                try:
                    dictionary = hunspell.HunSpell(dic_path, aff_path)  # pylint: disable=c-extension-no-member
                    HyphenationEncoder._hunspell_cache[language] = dictionary
                    return dictionary
                except (OSError, FileNotFoundError, RuntimeError):
                    continue

            # If no dictionary found, return None
            HyphenationEncoder._hunspell_cache[language] = None
            return None
        except Exception:  # pylint: disable=broad-except
            HyphenationEncoder._hunspell_cache[language] = None
            return None

    @staticmethod
    def _escape_text(text: str) -> str:
        """Escape special characters in text before syllabification.

        Escaping order matters: backslash first, then dash, then space.

        Args:
            text: The input text to escape.

        Returns:
            The escaped text.
        """
        result = text.replace("\\", "\\\\")
        result = result.replace("-", "\\-")
        result = result.replace(" ", "\\s")
        return result

    @staticmethod
    def _unescape_text(text: str) -> str:
        """Unescape special characters in text after decoding.

        Args:
            text: The escaped text to unescape.

        Returns:
            The unescaped text.

        Raises:
            InvalidEscapeSequenceError: If an invalid escape sequence is found.
        """
        result: list[str] = []
        i = 0
        while i < len(text):
            if text[i] == "\\":
                if i + 1 >= len(text):
                    raise InvalidEscapeSequenceError("Incomplete escape sequence at end of text")
                next_char = text[i + 1]
                if next_char == "\\":
                    result.append("\\")
                elif next_char == "-":
                    result.append("-")
                elif next_char == "s":
                    result.append(" ")
                elif next_char == "n":
                    result.append("\n")
                else:
                    raise InvalidEscapeSequenceError(f"Invalid escape '\\{next_char}' at pos {i}")
                i += 2
            else:
                result.append(text[i])
                i += 1
        return "".join(result)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text.

        Rules:
        - ASCII Line Feed (U+000A) -> space
        - Carriage return -> space
        - Tab -> space
        - Multiple consecutive spaces -> single space

        The literal sequence backslash+n is preserved as a line break marker.

        Args:
            text: The input text.

        Returns:
            Text with normalized whitespace.
        """
        result = text.replace("\r\n", " ")
        result = result.replace("\r", " ")
        result = result.replace("\n", " ")
        result = result.replace("\t", " ")
        result = re.sub(r" +", " ", result)
        return result

    @staticmethod
    def _is_alphabetic_word(word: str) -> bool:
        """Check if a word is purely alphabetic (no digits).

        Args:
            word: The word to check.

        Returns:
            True if the word contains only alphabetic characters.
        """
        return word.isalpha()

    @staticmethod
    def _contains_digit(word: str) -> bool:
        """Check if a word contains any digit.

        Args:
            word: The word to check.

        Returns:
            True if the word contains at least one digit.
        """
        return any(c.isdigit() for c in word)

    @staticmethod
    def _parse_whitelist(whitelist_str: str | None) -> set[str]:
        """Parse the whitelist string into a set of hyphenated words.

        Args:
            whitelist_str: Newline-separated list of hyphenated words.

        Returns:
            A set of whitelisted hyphenated words.
        """
        if not whitelist_str:
            return set()
        return {line.strip() for line in whitelist_str.strip().split("\n") if line.strip()}

    @staticmethod
    def _syllabify_word(word: str, ctx: _EncodingContext) -> str:
        """Syllabify a single word using pyphen.

        Args:
            word: The word to syllabify.
            ctx: The encoding context containing dictionaries and settings.

        Returns:
            The syllabified word with '-' separators.
        """
        if not word:
            return word

        if HyphenationEncoder._contains_digit(word):
            if ctx.debug:
                ctx.warnings.append(f"Unverified (contains digit): '{word}'")
            return word

        if not HyphenationEncoder._is_alphabetic_word(word):
            return word

        hyphenated = ctx.pyphen_dict.inserted(word)

        if ctx.debug and ctx.hunspell_dict is not None:
            if hyphenated in ctx.whitelist:
                pass
            elif not ctx.hunspell_dict.spell(word):
                ctx.warnings.append(f"Unverified (not in dictionary): '{word}'")

        return hyphenated

    @staticmethod
    def _process_token(token: str, ctx: _EncodingContext) -> str:
        """Process a single token, handling punctuation and words.

        Args:
            token: The token to process.
            ctx: The encoding context containing dictionaries and settings.

        Returns:
            The processed token.
        """
        if not token:
            return token

        leading_punct: list[str] = []
        trailing_punct: list[str] = []
        word = token

        while word and not word[0].isalnum():
            if word.startswith("\\"):
                if len(word) >= 2:
                    leading_punct.append(word[:2])
                    word = word[2:]
                else:
                    break
            else:
                leading_punct.append(word[0])
                word = word[1:]

        while word and not word[-1].isalnum():
            if len(word) >= 2 and word[-2] == "\\":
                trailing_punct.insert(0, word[-2:])
                word = word[:-2]
            else:
                trailing_punct.insert(0, word[-1])
                word = word[:-1]

        if word:
            word = HyphenationEncoder._syllabify_word(word, ctx)

        return "".join(leading_punct) + word + "".join(trailing_punct)

    @staticmethod
    def encode_text(
        text: str,
        language: str,
        accepted_hyphenated_words: str | None = None,
        debug: bool = False,
    ) -> str:
        """Encode text with syllable separators.

        Converts text into a meta-text representation where syllables
        are separated by '-' characters. The transformation is fully
        reversible using decode_text().

        Args:
            text: The input text to encode.
            language: Language code (e.g., 'en_US', 'de_DE').
            accepted_hyphenated_words: Optional newline-separated list of
                whitelisted hyphenated words that suppress warnings.
            debug: If True, warnings are logged. If False, warnings are silent.

        Returns:
            The encoded text with syllable separators.

        Raises:
            UnsupportedLanguageError: If the language is not supported.
        """
        ctx = _EncodingContext(
            pyphen_dict=HyphenationEncoder._get_pyphen_dict(language),
            hunspell_dict=(HyphenationEncoder._get_hunspell_dict(language) if debug else None),
            whitelist=HyphenationEncoder._parse_whitelist(accepted_hyphenated_words),
            debug=debug,
        )

        normalized = HyphenationEncoder._normalize_whitespace(text)
        line_parts = normalized.split("\\n")
        encoded_lines: list[str] = []

        for line_part in line_parts:
            escaped = HyphenationEncoder._escape_text(line_part)
            tokens = escaped.split("\\s")
            encoded_tokens = [HyphenationEncoder._process_token(token, ctx) for token in tokens]
            encoded_lines.append(" ".join(encoded_tokens))

        result = "\n".join(encoded_lines)

        if debug and ctx.warnings:
            for warning in ctx.warnings:
                logger.warning(warning)

        return result

    @staticmethod
    def decode_text(meta_text: str) -> str:
        """Decode syllabified text back to original form.

        Reverses the encoding performed by encode_text(), removing
        syllable separators and unescaping special characters.

        Args:
            meta_text: The encoded text with syllable separators.

        Returns:
            The decoded original text.

        Raises:
            InvalidEscapeSequenceError: If an invalid escape sequence is found.
        """
        result: list[str] = []
        i = 0
        while i < len(meta_text):
            if meta_text[i] == "\\":
                if i + 1 < len(meta_text):
                    next_char = meta_text[i + 1]
                    if next_char == "\\":
                        result.append("\\")
                        i += 2
                    elif next_char == "-":
                        result.append("-")
                        i += 2
                    elif next_char == "s":
                        result.append(" ")
                        i += 2
                    elif next_char == "n":
                        result.append("\\n")
                        i += 2
                    else:
                        raise InvalidEscapeSequenceError(f"Invalid escape '\\{next_char}' at pos {i}")
                else:
                    raise InvalidEscapeSequenceError("Incomplete escape sequence at end of text")
            elif meta_text[i] == "-":
                i += 1
            elif meta_text[i] == "\n":
                result.append("\\n")
                i += 1
            else:
                result.append(meta_text[i])
                i += 1

        return "".join(result)


class StreamBase(ABC):
    """Abstract base class for stateful stream providers.

    This class provides common navigation functionality for streams that
    provide items one by one. Subclasses must implement _initialize_items
    to populate the internal items list and _format_item to format items.

    Example:
        >>> class MyStream(StreamBase):
        ...     def _initialize_items(self, input_data):
        ...         self._items = list(input_data)
        ...     def _format_item(self, index):
        ...         return self._items[index]
    """

    def __init__(self, input_data: Any) -> None:
        """Initialize the stream with input data.

        Args:
            input_data: The input data to be processed by the subclass.
        """
        self._items: List[Any] = []
        self._cursor: int = 0
        self._initialize_items(input_data)

    @abstractmethod
    def _initialize_items(self, input_data: Any) -> None:
        """Initialize the internal items list from input data.

        Args:
            input_data: The input data to process.
        """

    @abstractmethod
    def _format_item(self, index: int) -> str:
        """Format an item at the given index for output.

        Args:
            index: Index of the item to format.

        Returns:
            The formatted string representation of the item.
        """

    def next_item(self) -> str:
        """Return the next item and advance the cursor.

        Returns:
            The next formatted item.

        Raises:
            StopIteration: If there are no more items.
        """
        if not self.has_next():
            raise StopIteration("No more items")

        item = self._format_item(self._cursor)
        self._cursor += 1
        return item

    def previous_item(self) -> str:
        """Return the previous item and move the cursor backward.

        Returns:
            The previous formatted item.

        Raises:
            StopIteration: If at the start (no previous item).
        """
        if not self.has_previous():
            raise StopIteration("No previous item")

        self._cursor -= 1
        return self._format_item(self._cursor)

    def rewind(self, steps: int = 1) -> None:
        """Move the cursor backward by the specified number of steps.

        Args:
            steps: Number of positions to move backward. Defaults to 1.

        Raises:
            ValueError: If steps is negative.
        """
        if steps < 0:
            raise ValueError("steps must be non-negative")
        self._cursor = max(0, self._cursor - steps)

    def has_next(self) -> bool:
        """Check if there is a next item.

        Returns:
            True if there is a next item, False otherwise.
        """
        return self._cursor < len(self._items)

    def has_previous(self) -> bool:
        """Check if there is a previous item.

        Returns:
            True if there is a previous item, False otherwise.
        """
        return self._cursor > 0

    def reset(self) -> None:
        """Reset the cursor to the start."""
        self._cursor = 0

    def position(self) -> int:
        """Return the current cursor position.

        Returns:
            The current cursor position (0-indexed).
        """
        return self._cursor


@dataclass
class _SyllableUnit:
    """Internal representation of a syllable unit.

    Attributes:
        text: The syllable text without trailing '-'.
        ends_word: True if this syllable ends a word.
        trailing_punct: Punctuation chars (.,;:!?) at word end, else empty.
    """

    text: str
    ends_word: bool
    trailing_punct: str


class SyllableStream(StreamBase):
    """Stateful syllable provider for encoded text from HyphenationEncoder.

    This class takes the output string from HyphenationEncoder.encode_text()
    and provides syllables one by one through navigation methods. The iteration
    is reversible and deterministic.

    Example:
        >>> stream = SyllableStream("Hel-lo world")
        >>> stream.next_item()
        'Hel'
        >>> stream.next_item()
        'lo '
        >>> stream.next_item()
        'world '
        >>> stream.previous_item()
        'world '
        >>> stream.reset()
        >>> stream.position()
        0
    """

    _TRAILING_PUNCT_CHARS: str = ".,;:!?"

    def _initialize_items(self, input_data: str) -> None:
        """Initialize syllable units from encoded text.

        Args:
            input_data: String produced by HyphenationEncoder.encode_text().

        Raises:
            InvalidEscapeSequenceError: If an invalid escape sequence is found.
        """
        self._units: list[_SyllableUnit] = []
        self._parse(input_data)
        self._items = self._units

    def _format_item(self, index: int) -> str:
        """Format a syllable unit for output.

        Args:
            index: Index of the syllable unit.

        Returns:
            Formatted syllable string.
        """
        unit = self._units[index]

        if unit.text == "\n":
            return "\n"

        if unit.ends_word:
            return unit.text + unit.trailing_punct + " "
        return unit.text

    def _unescape_syllable(self, text: str) -> str:
        """Unescape special characters in a syllable.

        Args:
            text: The escaped syllable text.

        Returns:
            The unescaped syllable text.

        Raises:
            InvalidEscapeSequenceError: If an invalid escape sequence is found.
        """
        result: list[str] = []
        i = 0
        while i < len(text):
            if text[i] == "\\":
                if i + 1 >= len(text):
                    raise InvalidEscapeSequenceError("Incomplete escape sequence at end of syllable")
                next_char = text[i + 1]
                if next_char == "\\":
                    result.append("\\")
                elif next_char == "-":
                    result.append("-")
                elif next_char == "s":
                    result.append(" ")
                elif next_char == "n":
                    result.append("\n")
                else:
                    raise InvalidEscapeSequenceError(f"Invalid escape '\\{next_char}' at position {i}")
                i += 2
            else:
                result.append(text[i])
                i += 1
        return "".join(result)

    def _extract_trailing_punct(self, text: str) -> tuple[str, str]:
        """Extract trailing punctuation from text.

        Args:
            text: The text to process.

        Returns:
            A tuple of (text_without_punct, trailing_punct).
        """
        trailing = []
        while text and text[-1] in self._TRAILING_PUNCT_CHARS:
            trailing.insert(0, text[-1])
            text = text[:-1]
        return text, "".join(trailing)

    def _parse(self, encoded_text: str) -> None:
        """Parse encoded text into syllable units.

        Args:
            encoded_text: The encoded text to parse.
        """
        if not encoded_text:
            return

        lines = encoded_text.split("\n")

        for line_idx, line in enumerate(lines):
            if line_idx > 0:
                self._units.append(_SyllableUnit(text="\n", ends_word=True, trailing_punct=""))

            words = line.split(" ")

            for word in words:
                if not word:
                    continue

                syllables = self._split_word_into_syllables(word)

                for syl_idx, syllable in enumerate(syllables):
                    is_last_syllable = syl_idx == len(syllables) - 1

                    if is_last_syllable:
                        text, punct = self._extract_trailing_punct(syllable)
                        unescaped = self._unescape_syllable(text)
                        self._units.append(
                            _SyllableUnit(
                                text=unescaped,
                                ends_word=True,
                                trailing_punct=punct,
                            )
                        )
                    else:
                        unescaped = self._unescape_syllable(syllable)
                        self._units.append(
                            _SyllableUnit(
                                text=unescaped,
                                ends_word=False,
                                trailing_punct="",
                            )
                        )

    def _split_word_into_syllables(self, word: str) -> list[str]:
        """Split a word into syllables, respecting escape sequences.

        Args:
            word: The word to split.

        Returns:
            A list of syllables.
        """
        syllables: list[str] = []
        current: list[str] = []
        i = 0

        while i < len(word):
            if word[i] == "\\":
                if i + 1 < len(word):
                    current.append(word[i])
                    current.append(word[i + 1])
                    i += 2
                else:
                    current.append(word[i])
                    i += 1
            elif word[i] == "-":
                if current:
                    syllables.append("".join(current))
                    current = []
                i += 1
            else:
                current.append(word[i])
                i += 1

        if current:
            syllables.append("".join(current))

        return syllables if syllables else [word]


class LetterStream(StreamBase):
    """Stateful letter provider for arbitrary input strings.

    This class takes any arbitrary string and provides characters one by one
    through navigation methods. All characters are treated equally, including
    spaces, tabs, line breaks, punctuation, digits, and letters.

    Example:
        >>> stream = LetterStream("Hello")
        >>> stream.next_item()
        'H'
        >>> stream.next_item()
        'e'
        >>> stream.previous_item()
        'e'
        >>> stream.reset()
        >>> stream.position()
        0
    """

    def _initialize_items(self, input_data: str) -> None:
        """Initialize character items from input text.

        Args:
            input_data: Any arbitrary string.
        """
        self._items = list(input_data)

    def _format_item(self, index: int) -> str:
        """Format a character at the given index for output.

        Args:
            index: Index of the character.

        Returns:
            The character at the given index.
        """
        return self._items[index]
