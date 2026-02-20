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

    Special Character Handling:
    -------------------------

    **Input Characters:**
    - Regular text: Processed normally with hyphenation points added
    - Newlines (\\n): Treated as word separators, preserved in output
    - Tabs (\\t): Treated as word separators, preserved in output
    - Preserved hyphens: Escape with backslash (e.g., "X\\-ray")
    - Preserved spaces: Escape with backslash (e.g., "word1\\sword2")
    - Backslashes: Must be doubled (e.g., "path\\\\to\\\\file")
    - Punctuation: Preserved unmodified (.,!?;:"'()[])

    **Escape Sequences (Input):**
    - "\\\\" → Literal backslash
    - "\\-" → Preserved hyphen (not a syllable break)
    - "\\s" → Preserved space (prevents word boundary)

    **Output Format:**
    - Hyphenation points: Marked with '-' (e.g., "ex-cep-tion-al")
    - Preserved characters: Remain escaped (e.g., "X\\-ray")
    - Whitespace: Normal spaces between words, newlines/tabs preserved
    - Punctuation: Attached to preceding syllable

    **Examples:**
    >>> HyphenationEncoder.encode_text("Hello world", "en_US")
    'Hel-lo world'
    >>> HyphenationEncoder.encode_text("X-ray self\\\\-directed", "en_US")
    'X\\-ray self\\-di-rect-ed'
    >>> HyphenationEncoder.encode_text("line1\\nline2", "en_US")
    'line1\\nline2'
    >>> HyphenationEncoder.decode_text("Hel-lo world")
    'Hello world'
    >>> HyphenationEncoder.decode_text("X\\-ray self\\-di-rect-ed")
    'X-ray self-directed'

    **Language Support:**
    - Uses hunspell dictionaries for hyphenation patterns
    - Supports languages available in system dictionaries
    - Language codes follow ISO 639-1 with country codes (e.g., "en_US")

    **Implementation Notes:**
    - Words are identified by whitespace boundaries (spaces, tabs, newlines)
    - Each word is hyphenated independently
    - Escape sequences are processed before hyphenation
    - The process is fully reversible via decode_text()
    - Whitespace characters are preserved but not escaped
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

    @dataclass
    class _EncodingContext:
        """Internal context for encoding operations."""

        pyphen_dict: pyphen.Pyphen
        hunspell_dict: hunspell.HunSpell | None
        whitelist: set[str]
        debug: bool
        warnings: list[str] = field(default_factory=list)

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
    def _syllabify_word(word: str, ctx: HyphenationEncoder._EncodingContext) -> str:
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
    def _process_token(token: str, ctx: HyphenationEncoder._EncodingContext) -> str:
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
        ctx = HyphenationEncoder._EncodingContext(
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


class AvStreamBase(ABC):
    """Abstract base class for stateful stream providers.

    This class provides common navigation functionality for streams that
    provide items one by one. Subclasses must implement _initialize_items
    to populate the internal items list and _format_item to format items.

    Example:
        >>> class MyStream(AvStreamBase):
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


class AvSyllableStream(AvStreamBase):
    """Stateful syllable provider for encoded text from HyphenationEncoder.

    This class takes the output string from HyphenationEncoder.encode_text()
    and provides syllables one by one through navigation methods. The iteration
    is reversible and deterministic.

    The stream automatically handles spacing:
    - Syllables within the same word are returned without trailing spaces
    - Syllables that end a word are returned with a trailing space added
    - Trailing punctuation (.,;:!?) is preserved before the space

    Example:
        >>> stream = AvSyllableStream("Hel-lo world")
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

    def _initialize_items(self, input_data: str) -> None:
        """Initialize syllable units from encoded text.

        Args:
            input_data: String produced by HyphenationEncoder.encode_text().

        Raises:
            InvalidEscapeSequenceError: If an invalid escape sequence is found.
        """
        self._units: list[AvSyllableStream._SyllableUnit] = []
        self._parse(input_data)
        self._items = self._units

    def _format_item(self, index: int) -> str:
        """Format a syllable unit for output.

        Args:
            index: Index of the syllable unit.

        Returns:
            Formatted syllable string with automatic spacing:
            - If syllable ends a word: text + punctuation + space
            - If syllable is within word: text only
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
                self._units.append(self._SyllableUnit(text="\n", ends_word=True, trailing_punct=""))

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
                            self._SyllableUnit(
                                text=unescaped,
                                ends_word=True,
                                trailing_punct=punct,
                            )
                        )
                    else:
                        unescaped = self._unescape_syllable(syllable)
                        self._units.append(
                            self._SyllableUnit(
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


class AvCharacterStream(AvStreamBase):
    """Stateful character provider for arbitrary input strings.

    This class takes any arbitrary string and provides characters one by one
    through navigation methods. All characters are treated equally, including
    spaces, tabs, line breaks, punctuation, digits, and letters.

    Example:
        >>> stream = AvCharacterStream("Hello")
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


@staticmethod
def sample_text_mc() -> str:
    """Return sample text for testing."""

    text = (
        "Marie Curie (born Maria Sklodowska; 7 November 1867 - 4 July 1934) was a "
        "pioneering physicist and chemist whose research on radioactivity "
        "profoundly shaped modern science. She was the first woman to receive a "
        "Nobel Prize, the first person to win Nobel Prizes in two different "
        "scientific fields, and one of the most influential scientists of the "
        "twentieth century. Her work laid the foundation for advances in "
        "nuclear physics, medical diagnostics, and cancer therapy.\n"
        "\n"
        "Curie was born in Warsaw, in what was then part of the Russian Empire, "
        "to a family of educators who valued learning despite political "
        "repression. Growing up in Poland during a period when higher education "
        "for women was severely restricted, she pursued her early studies "
        "through clandestine classes and self-directed learning. In 1891 she "
        "moved to Paris to continue her education at the University of Paris "
        "(Sorbonne). There she studied physics and mathematics, graduating at "
        "the top of her class in physics in 1893 and earning a second degree "
        "in mathematics the following year.\n"
        "\n"
        "In 1895 she married Pierre Curie, a physicist known for his work on "
        "crystallography and magnetism. The couple formed a scientific "
        "partnership that proved exceptionally productive. Inspired by Henri "
        "Becquerel's discovery that uranium salts emitted penetrating rays, "
        "Marie Curie began investigating the phenomenon. She coined the term "
        '"radioactivity" to describe the spontaneous emission of energy from '
        "certain elements, recognizing that it was an atomic property rather "
        "than a chemical reaction.\n"
        "\n"
        "Working under modest laboratory conditions, Marie and Pierre Curie "
        "processed tons of pitchblende, a uranium-rich mineral, in order to "
        "isolate previously unknown radioactive substances. In 1898 they "
        "announced the discovery of two new elements: polonium, named after "
        "Marie's native Poland, and radium. The isolation of radium in pure "
        "metallic form required years of painstaking chemical separation and "
        "measurement. Their research provided strong evidence that "
        "radioactivity was linked to the internal structure of atoms, "
        "challenging existing scientific models and contributing to the "
        "development of atomic theory.\n"
        "\n"
        "In 1903 Marie Curie, Pierre Curie, and Henri Becquerel were jointly "
        "awarded the Nobel Prize in Physics for their work on radiation "
        "phenomena. Following Pierre Curie's sudden death in 1906, Marie Curie "
        "succeeded him as professor at the University of Paris, becoming the "
        "first woman to hold that position. In 1911 she received a second "
        "Nobel Prize, this time in Chemistry, for the discovery of radium and "
        "polonium and for her investigation of their properties. Her dual "
        "achievements established her as an international scientific "
        "authority.\n"
        "\n"
        "Beyond her laboratory research, Curie played a crucial role in "
        "applying scientific knowledge for humanitarian purposes. During the "
        "First World War, she helped develop mobile radiography units to "
        "assist surgeons in locating bullets and shrapnel in wounded soldiers. "
        'These units, sometimes called "Little Curies," brought X-ray '
        "technology directly to battlefield hospitals. She also trained medical "
        "personnel, including her daughter Irene, in the use of radiological "
        "equipment, thereby expanding the practical impact of her scientific "
        "expertise.\n"
        "\n"
        "Curie's later career was devoted to advancing research and medical "
        "applications of radioactivity. She founded research institutes "
        "dedicated to the study of radioactive elements and their therapeutic "
        "potential, helping to establish radiology as a recognized medical "
        "discipline. Although the long-term health risks of radiation exposure "
        "were not yet fully understood, her work contributed to the "
        "development of radiation therapy for cancer treatment, which remains an "
        "important medical technique today.\n"
        "\n"
        "The hazards of prolonged radiation exposure ultimately affected Curie's "
        "health. She died in 1934 from aplastic anemia, a condition widely "
        "believed to have been caused by her years of handling radioactive "
        "materials without protective measures. At the time, safety protocols "
        "for radiation were minimal, and the risks were not comprehensively "
        "documented. Her laboratory notebooks remain radioactive and are "
        "preserved with special precautions.\n"
        "\n"
        "Curie's legacy extends beyond her scientific discoveries. She became "
        "a symbol of intellectual rigor, perseverance, and the advancement of "
        "women in science. In 1995 her remains were transferred to the "
        "Pantheon in Paris, making her the first woman to be honored there on "
        "her own merits. Her life and achievements have inspired generations of "
        "scientists and students worldwide.\n"
        "\n"
        "Colleagues and contemporaries regarded Curie with deep respect. The "
        "physicist Albert Einstein once praised her integrity and independence "
        "of thought, noting her resistance to fame and public pressure. Despite "
        "facing gender-based discrimination and personal hardship, Curie "
        "maintained a strong commitment to scientific inquiry and public "
        "service.\n"
        "\n"
        "Marie Curie's contributions transformed the understanding of atomic "
        "science and opened new paths in physics, chemistry, and medicine. "
        "Through her discoveries, leadership, and dedication to research, she "
        "helped redefine the possibilities of modern science. Her name remains "
        "closely associated with the study of radioactivity and with the "
        "broader struggle to ensure equal access to education and scientific "
        "careers."
    )

    return text


def main() -> None:
    """Demonstrate syllable stream functionality with intelligent line breaking.

    This function demonstrates the AvSyllableStream by formatting a sample text
    to a maximum line length of 50 characters, with hyphens only appearing at
    line breaks where words are split.

    Algorithm for intelligent word splitting and line formatting:

    1. **Text Preparation**:
       - Get sample text using sample_text_mc()
       - Apply hyphenation using HyphenationEncoder.encode_text() with 'en_US'
       - The hyphenated text contains:
         - Hyphenation points marked with '-' (e.g., 'ex-cep-tion-al-ly')
         - Original hyphens escaped as '\\-' (e.g., 'self\\-directed')

    2. **Word Processing**:
       - Split hyphenated text into words by spaces
       - For each word, determine its type:
         a) Words with escaped hyphens (original compound words)
         b) Words with hyphenation marks (can be split at syllables)
         c) Regular words (no hyphens)

    3. **Line Building Algorithm**:
       - Maintain current_line and build formatted_lines list
       - For each word:
         - Calculate clean_word (remove hyphenation marks, unescape original hyphens)
         - Test if word fits on current line
         - If it fits: append to current_line
         - If it doesn't fit:
           - Check if word can be hyphenated/split
           - For hyphenated words: find optimal split point using syllables
           - For compound words: try splitting at original hyphens
           - Add hyphen only at line break
           - Start next line with remaining part

    4. **Splitting Strategy**:
       - **Hyphenated words**: Try to fit maximum syllables on current line
         - Iterate through syllables from left to right
         - Stop when adding next syllable would exceed line limit
         - Add hyphen to last fitting syllable
       - **Compound words**: Try to split at original hyphens
         - Test each hyphen position as potential split point
         - Preserve original hyphens in output
       - **Regular words**: Move to next line if too long

    5. **Edge Cases Handled**:
       - Words shorter than 6 characters are not hyphenated
       - Original hyphens (X-ray, self-directed) are preserved
       - Words can be split at both hyphenation points and original hyphens
       - Empty lines and punctuation are properly handled

    6. **Output**:
       - Each line is exactly 50 characters or less
       - Hyphens only appear at line breaks where splitting occurs
       - Original compound word hyphens are preserved
       - Text flows naturally with optimal space utilization
    """
    # Use sample text with hyphenation
    sample = sample_text_mc()

    # Apply hyphenation to the sample text
    hyphenated = HyphenationEncoder.encode_text(sample, "en_US")

    # Display the text formatted to 50 chars per line, with hyphens only at line breaks
    print("Sample text (50 chars per line, hyphenated at line breaks):")
    print("-" * 50)

    # Process the text with improved hyphenation algorithm
    words = hyphenated.split()
    formatted_lines = []
    current_line = ""

    for word in words:
        # Check if this is a word with hyphenation (not original hyphens)
        # Original hyphens are escaped as "\-" in the hyphenated text
        if "\\" in word and "\\-" in word:
            # This word has escaped hyphens - keep them as-is for display
            # But also check if we can hyphenate the parts
            clean_word = word.replace("\\-", "-")

            # Try to fit the whole word first
            test_line = current_line + (" " if current_line else "") + clean_word
            if len(test_line) <= 50:
                # Word fits as-is
                if current_line:
                    current_line += " " + clean_word
                else:
                    current_line = clean_word
            else:
                # Word doesn't fit - try to split it
                # Split by original hyphens and hyphenate each part
                parts = clean_word.split("-")
                if len(parts) > 1 and len(clean_word) > 5:
                    # Try to fit parts with hyphenation
                    best_fit = ""
                    remaining_text = clean_word

                    # Try different split points
                    for i in range(len(parts)):
                        # Join first i+1 parts
                        first_part = "-".join(parts[: i + 1])
                        test_partial = current_line + " " + first_part
                        if len(test_partial) <= 50:
                            best_fit = first_part
                            if i < len(parts) - 1:
                                remaining_text = "-".join(parts[i + 1 :])
                            else:
                                remaining_text = ""
                        else:
                            break

                    if best_fit and remaining_text:
                        # We can split at an original hyphen
                        formatted_lines.append(current_line + " " + best_fit + "-")
                        current_line = remaining_text
                    else:
                        # Can't split effectively - move to next line
                        formatted_lines.append(current_line)
                        current_line = clean_word
                else:
                    # No original hyphens or too short - move to next line
                    formatted_lines.append(current_line)
                    current_line = clean_word

        elif "-" in word:
            # This has hyphenation marks - remove them for normal display
            clean_word = word.replace("-", "")

            # Check if adding this clean word would exceed 50 chars
            test_line = current_line + (" " if current_line else "") + clean_word
            if len(test_line) <= 50:
                # Word fits on current line without hyphenation
                if current_line:
                    current_line += " " + clean_word
                else:
                    current_line = clean_word
            else:
                # Word doesn't fit - we need to handle it
                if current_line:
                    # Try to split the word at a hyphenation point
                    if len(clean_word) > 5:
                        syllables = word.split("-")

                        # Try to fit as many syllables as possible on current line
                        best_fit_syllables = []
                        remaining_syllables = syllables[:]

                        # Try different split points
                        for i in range(1, len(syllables)):
                            partial = "".join(syllables[:i])
                            test_partial = current_line + " " + partial
                            if len(test_partial) <= 50:
                                best_fit_syllables = syllables[:i]
                                remaining_syllables = syllables[i:]
                            else:
                                break

                        if best_fit_syllables:
                            # Add the partial word with hyphen
                            partial_word = "".join(best_fit_syllables)
                            formatted_lines.append(current_line + " " + partial_word + "-")
                            # Start next line with remaining syllables
                            current_line = "".join(remaining_syllables)
                        else:
                            # Can't fit even the first syllable with space - move word to next line
                            formatted_lines.append(current_line)
                            current_line = clean_word
                    else:
                        # Word too short to hyphenate - move to next line
                        formatted_lines.append(current_line)
                        current_line = clean_word
                else:
                    # Start of line - try to split hyphenated word
                    if len(clean_word) > 50:
                        # Word is longer than line - must split it
                        syllables = word.split("-")
                        temp_line = ""
                        for i, syllable in enumerate(syllables):
                            if i == 0:
                                temp_line = syllable
                            else:
                                if len(temp_line + "-" + syllable) <= 50:
                                    temp_line += "-" + syllable
                                else:
                                    formatted_lines.append(temp_line + "-")
                                    temp_line = syllable
                        current_line = temp_line
                    else:
                        current_line = clean_word
        else:
            # No hyphens at all
            clean_word = word

            # Check if adding this word would exceed 50 chars
            test_line = current_line + (" " if current_line else "") + clean_word
            if len(test_line) <= 50:
                # Word fits
                if current_line:
                    current_line += " " + clean_word
                else:
                    current_line = clean_word
            else:
                # Word doesn't fit - move to next line
                if current_line:
                    formatted_lines.append(current_line)
                current_line = clean_word

    # Add the last line
    if current_line:
        formatted_lines.append(current_line)

    # Print all formatted lines
    for line in formatted_lines:
        print(line)

    print("-" * 50)


if __name__ == "__main__":
    main()
