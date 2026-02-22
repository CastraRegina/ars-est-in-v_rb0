"""
Modern text processing pipeline for reversible syllabification using Unicode Soft Hyphens.

This module implements a tokenization-based approach to text hyphenation.
It parses text into a stream of tokens (words, whitespace, punctuation),
applies hyphenation only to valid word tokens using pyphen, and reassembles
the text using Unicode Soft Hyphens (U+00AD) as break markers.

This approach is:
1.  **Standard-compliant**: Uses U+00AD (SHY) instead of custom escape sequences.
2.  **Robust**: Accurately handles mixed content, punctuation, and whitespace.
3.  **Reversible**: De-hyphenation is simply removing U+00AD characters.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Iterator, List, Optional

import hunspell  # pylint: disable=c-extension-no-member
import pyphen

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

SOFT_HYPHEN = "\u00ad"


class TokenType(Enum):
    """Types of text tokens."""

    WORD = auto()
    NUMBER = auto()
    PUNCTUATION = auto()
    WHITESPACE = auto()
    OTHER = auto()


@dataclass
class Token:
    """A span of text with a semantic type."""

    type: TokenType
    text: str


class HyphenationError(Exception):
    """Base exception for hyphenation errors."""


class UnsupportedLanguageError(HyphenationError):
    """Raised when a requested language is not supported."""


class InvalidEscapeSequenceError(HyphenationError):
    """Raised when an invalid escape sequence is encountered (Legacy compatibility)."""


# -------------------------------------------------------------------------
# Tokenizer
# -------------------------------------------------------------------------


class Tokenizer:
    """Regex-based tokenizer for text processing."""

    # pylint: disable=too-few-public-methods

    # Regex patterns
    # Note: We use a simple definition of 'word' as alphabetic characters.
    # We explicitly exclude digits and underscores from words.

    _PATTERNS = [
        (TokenType.WORD, r"[^\W\d_]+"),  # Letters only (Unicode aware)
        (TokenType.NUMBER, r"\d+(?:[\.,]\d+)*"),
        (TokenType.WHITESPACE, r"\s+"),
        (TokenType.PUNCTUATION, r"[^\w\s]"),  # Symbols/Punctuation
        (TokenType.OTHER, r"."),  # Fallback
    ]

    _REGEX = re.compile(
        "|".join(f"(?P<{t.name}>{p})" for t, p in _PATTERNS),
        re.UNICODE | re.DOTALL,
    )

    @staticmethod
    def tokenize(text: str) -> Iterator[Token]:
        """Yield tokens from the input text.

        Args:
            text: The string to tokenize.

        Yields:
            Token objects.
        """
        for match in Tokenizer._REGEX.finditer(text):
            kind = match.lastgroup
            value = match.group()
            if kind:
                yield Token(TokenType[kind], value)


# -------------------------------------------------------------------------
# Hyphenator
# -------------------------------------------------------------------------


class Hyphenator:
    """Stateful helper that inserts soft hyphens via pyphen dictionaries.

    Example:
        >>> hyphenator = Hyphenator("en_US")
        >>> input_text = "self-aware\\n hello\\u00A0information world"
        >>> hyphenated = hyphenator.hyphenate_text(input_text)
        >>> print(repr(hyphenated))
        'self-aware\\n hel\xadlo\\xa0in\xadfor\xadma\xadtion world'
        >>>
        >>> # Explanation:
        >>> # - "self-aware": Compound word, hyphen preserved, syllables hyphenated
        >>> # - "\\n": Newline preserved as-is
        >>> # - "\\u00A0": Non-breaking space (U+00A0) preserved as-is
        >>> # - "hello": Hyphenated into syllables with soft hyphens (U+00AD)
        >>> # - "information": Multi-syllable word, hyphenated at syllable boundaries
        >>> # - "world": Too short, not hyphenated
    """

    def __init__(self, language: str = "en_US") -> None:
        """Initialize the hyphenator for a specific language.

        Args:
            language: BCP 47 language code (e.g., 'en_US').

        Raises:
            UnsupportedLanguageError: If the language is not supported by pyphen.
        """
        self.language = language
        try:
            self._pyphen = pyphen.Pyphen(lang=language)
        except Exception as e:
            if language not in pyphen.LANGUAGES:
                supported = ", ".join(sorted(pyphen.LANGUAGES.keys())[:10])
                raise UnsupportedLanguageError(
                    f"Language '{language}' is not supported. Supported: {supported}..."
                ) from e
            # If it failed for another reason, re-raise
            raise e

        self._hunspell: Optional[hunspell.HunSpell] = None

    @property
    def pyphen_dict(self) -> pyphen.Pyphen:
        """Access the underlying pyphen dictionary."""
        return self._pyphen

    def hyphenate_word(self, word: str) -> str:
        """Hyphenate a single word using soft hyphens.

        Args:
            word: A single word token (no whitespace/punctuation).

        Returns:
            The word with U+00AD inserted at break points.

        Special Cases:
            - Non-breaking spaces (U+00A0): Preserved as-is by the tokenizer's
                WHITESPACE pattern and never reach this method.
            - Regular hyphens in compound words (e.g., "self-aware"): Protected
                during hyphenation to preserve the original compound word structure.
                The method temporarily replaces existing hyphens with a marker,
                applies syllable hyphenation, then restores the original hyphens.
        """
        # Optimization: Don't hyphenate short words
        if len(word) < 5:
            return word

        # Protect existing hyphens in compound words (e.g., "self-aware")
        # Use zero-width space as a temporary marker
        PROTECTED_HYPHEN = "\u200b"  # pylint: disable=invalid-name
        protected = word.replace("-", PROTECTED_HYPHEN)

        hyphenated = self._pyphen.inserted(protected)
        if hyphenated == protected:
            return word

        # Replace the hard hyphen inserted by pyphen with soft hyphen
        hyphenated = hyphenated.replace("-", SOFT_HYPHEN)

        # Restore original hyphens from compound words
        hyphenated = hyphenated.replace(PROTECTED_HYPHEN, "-")

        return hyphenated

    def hyphenate_text(self, text: str) -> str:
        """Hyphenate arbitrary text.

        Parses text into tokens, hyphenates words, and reassembles.

        Args:
            text: The input text.

        Returns:
            Text with Soft Hyphens inserted.

        Special Cases Handled:
            - Non-breaking spaces (U+00A0): Preserved as-is by the tokenizer's
                WHITESPACE pattern and never hyphenated.
            - Regular hyphens in compound words (e.g., "self-aware"): Protected
                during hyphenation to preserve the original compound word structure.
                The hyphen between word parts is preserved, while syllable breaks
                within each word part use soft hyphens (U+00AD).
            - Punctuation and symbols: Preserved as-is without hyphenation.
            - Numbers: Not hyphenated.
            - Short words (< 5 chars): Not hyphenated (optimization).
        """
        chunks = []
        for token in Tokenizer.tokenize(text):
            if token.type == TokenType.WORD:
                chunks.append(self.hyphenate_word(token.text))
            else:
                chunks.append(token.text)
        return "".join(chunks)


# -------------------------------------------------------------------------
# Stream Classes (Updated for Soft Hyphen)
# -------------------------------------------------------------------------


class AvStreamBase(ABC):
    """Abstract base class for stateful stream providers."""

    def __init__(self, input_data: Any) -> None:
        self._items: List[Any] = []
        self._cursor: int = 0
        self._initialize_items(input_data)

    @abstractmethod
    def _initialize_items(self, input_data: Any) -> None:
        """Initialize items from input."""

    @abstractmethod
    def _format_item(self, index: int) -> str:
        """Format item at index."""

    def next_item(self) -> str:
        """Get the next item in the stream."""
        if not self.has_next():
            raise StopIteration("No more items")
        item = self._format_item(self._cursor)
        self._cursor += 1
        return item

    def previous_item(self) -> str:
        """Get the previous item in the stream."""
        if not self.has_previous():
            raise StopIteration("No previous item")
        self._cursor -= 1
        return self._format_item(self._cursor)

    def rewind(self, steps: int = 1) -> None:
        """Rewind the cursor by a number of steps."""
        if steps < 0:
            raise ValueError("steps must be non-negative")
        self._cursor = max(0, self._cursor - steps)

    def has_next(self) -> bool:
        """Check if there are more items to consume."""
        return self._cursor < len(self._items)

    def has_previous(self) -> bool:
        """Check if there are items to rewind to."""
        return self._cursor > 0

    def reset(self) -> None:
        """Reset the cursor to the beginning."""
        self._cursor = 0

    def position(self) -> int:
        """Get the current cursor position."""
        return self._cursor


# class AvSyllableStream(AvStreamBase):
#     """Syllable provider for Soft-Hyphenated text.

#     Parses text containing U+00AD and yields syllables.
#     """

#     @dataclass
#     class _SyllableUnit:
#         text: str
#         ends_word: bool
#         trailing_punct: str

#     def _initialize_items(self, input_data: str) -> None:
#         self._units: list[AvSyllableStream._SyllableUnit] = []
#         self._parse(input_data)
#         self._items = self._units

#     def _format_item(self, index: int) -> str:
#         unit = self._units[index]
#         if unit.text == "\n":
#             return "\n"
#         if unit.ends_word:
#             return unit.text + unit.trailing_punct + " "
#         return unit.text

#     def _parse(self, text: str) -> None:
#         if not text:
#             return

#         lines = text.split("\n")
#         for i, line in enumerate(lines):
#             if i > 0:
#                 self._units.append(self._SyllableUnit("\n", True, ""))

#             if not line:
#                 continue

#             words = line.split(" ")
#             for word_idx, word in enumerate(words):
#                 if not word and word_idx < len(words) - 1:
#                     continue
#                 if not word:
#                     continue

#                 self._process_chunk(word)

#     def _process_chunk(self, chunk: str) -> None:
#         """Process a chunk (word + punct + hyphens)."""
#         core, trailing_punct = self._extract_trailing_punct(chunk)

#         # Split on SHY or Hard Hyphen.
#         parts = re.split(f"({SOFT_HYPHEN}|-)", core)

#         syllables = []
#         current = ""

#         for part in parts:
#             if part == "-":
#                 # Hard hyphen attaches to PREVIOUS syllable
#                 if current:
#                     current += "-"
#                     syllables.append(current)
#                     current = ""
#                 elif syllables:
#                     syllables[-1] += "-"
#                 else:
#                     # Hyphen at start?
#                     current += "-"
#             elif part == SOFT_HYPHEN:
#                 # Soft hyphen is a break.
#                 if current:
#                     syllables.append(current)
#                     current = ""
#             else:
#                 if part:
#                     current += part

#         if current:
#             syllables.append(current)

#         if not syllables:
#             if trailing_punct:
#                 self._units.append(self._SyllableUnit("", True, trailing_punct))
#             return

#         for idx, syl in enumerate(syllables):
#             is_last = idx == len(syllables) - 1
#             self._units.append(
#                 self._SyllableUnit(
#                     text=syl,
#                     ends_word=is_last,
#                     trailing_punct=trailing_punct if is_last else "",
#                 )
#             )

#     def _extract_trailing_punct(self, text: str) -> tuple[str, str]:
#         punct_chars = ".,;:!?"
#         trailing = []
#         while text and text[-1] in punct_chars:
#             trailing.insert(0, text[-1])
#             text = text[:-1]
#         return text, "".join(trailing)


class AvCharacterStream(AvStreamBase):
    """Character provider."""

    def _initialize_items(self, input_data: str) -> None:
        self._items = list(input_data)

    def _format_item(self, index: int) -> str:
        return self._items[index]


# -------------------------------------------------------------------------
# Formatting / Line Breaking Helpers
# -------------------------------------------------------------------------


def _format_hyphenated_lines(text: str, max_width: int) -> list[str]:
    """Format text into lines respecting max_width and soft hyphens.

    Args:
        text: Text containing U+00AD soft hyphens.
        max_width: Max line length.

    Returns:
        List of formatted lines.

    Special Cases Handled:
        - Newlines: A newline character (\\n) ends the current line and starts a new one.
            Multiple consecutive newlines are collapsed into a single line break; empty lines
            are not preserved in the output.
        - Soft hyphens (U+00AD): Used as syllable break points for line splitting.
        - Regular hyphens in compound words: Preserved as-is.
        - Non-breaking spaces (U+00A0): Treated as regular whitespace.
    """
    lines = []
    current_line = ""

    # Replace newlines with space-padding to treat as word separator
    chunks = text.replace("\n", " \n ").split(" ")

    for word in chunks:
        if not word:
            continue
        if word == "\n":
            if current_line:
                lines.append(current_line)
                current_line = ""
            continue

        # Parse word into segments ONCE
        segments = []
        curr_seg = ""
        for ch in word:
            if ch == SOFT_HYPHEN:
                if curr_seg:
                    segments.append((curr_seg, True))  # Needs added hyphen
                    curr_seg = ""
            elif ch == "-":
                curr_seg += "-"
                segments.append((curr_seg, False))  # Hyphen already there
                curr_seg = ""
            else:
                curr_seg += ch
        if curr_seg:
            segments.append((curr_seg, False))

        if not segments:
            continue

        # Process segments
        seg_idx = 0
        while seg_idx < len(segments):
            # Try to fit remaining segments onto current_line
            sep = " " if current_line else ""
            base_len = len(current_line) + len(sep)

            best_k = -1
            current_seq_text_len = 0

            # Find the longest sequence of segments we can fit
            for k in range(seg_idx, len(segments)):
                seg_text, needs_hyphen = segments[k]
                current_seq_text_len += len(seg_text)

                is_last_in_word = k == len(segments) - 1
                trailing = "-" if (needs_hyphen and not is_last_in_word) else ""

                total_candidate_len = base_len + current_seq_text_len + len(trailing)

                if total_candidate_len <= max_width:
                    best_k = k
                else:
                    break

            if best_k >= seg_idx:
                # We can fit segments[seg_idx ... best_k]
                chunk_str = ""
                for k in range(seg_idx, best_k + 1):
                    seg_text, needs_hyphen = segments[k]
                    chunk_str += seg_text
                    if k == best_k and needs_hyphen and k != len(segments) - 1:
                        chunk_str += "-"

                current_line += sep + chunk_str
                seg_idx = best_k + 1

                # If we haven't finished the word, we MUST split line here
                if seg_idx < len(segments):
                    lines.append(current_line)
                    current_line = ""
            else:
                # Cannot fit even the first segment (segments[seg_idx]) on current line.
                if current_line:
                    lines.append(current_line)
                    current_line = ""
                    continue  # Retry with empty line

                # Line is empty, but segment doesn't fit. Force char split.
                huge_seg_text, huge_seg_hyphen = segments[seg_idx]

                idx_char = 0
                while idx_char < len(huge_seg_text):
                    limit = max_width - 1  # Reserve space for hyphen
                    if limit < 1:  # pylint: disable=consider-using-max-builtin
                        limit = 1

                    chunk = huge_seg_text[idx_char : idx_char + limit]

                    next_idx = idx_char + limit

                    if next_idx < len(huge_seg_text):
                        lines.append(chunk + "-")
                    else:
                        # Last chunk of this segment.
                        # Check if we need a trailing hyphen for the segment itself
                        is_last_in_word = seg_idx == len(segments) - 1
                        trailing = "-" if (huge_seg_hyphen and not is_last_in_word) else ""

                        # Since we forced split, current_line is empty.
                        # Does this chunk+trailing fit?
                        # It should, because chunk <= max_width-1.
                        # But let's be safe.
                        if len(chunk) + len(trailing) <= max_width:
                            current_line = chunk + trailing
                        else:
                            # Extremely rare case: max_width=1, chunk=1, trailing=1 -> 2.
                            # We must emit line and overflow/wrap?
                            # For width=1, "a-" takes 2 chars. Impossible.
                            # We just emit it.
                            current_line = chunk + trailing

                    idx_char = next_idx

                seg_idx += 1

    if current_line:
        lines.append(current_line)

    return lines


@staticmethod
def sample_text_mc() -> str:
    """Return sample text for testing."""

    text = (
        "M.\u00a0C. (born M.\u00a0S.; 7 November 1867 - 4 July 1934) was a "
        "pioneering physicist and chemist whose research on radioactivity "
        "profoundly shaped modern science. She was the first woman to receive a "
        "Nobel Prize, the first person to win Nobel Prizes in two different "
        "scientific fields, and one of the most influential scientists of the "
        "twentieth century. Her work laid the foundation for advances in "
        "nuclear physics, medical diagnostics, and cancer therapy.\n"
        "\n"
        "C. was born in Warsaw, in what was then part of the Russian Empire, "
        "to a family of educators who valued learning despite political "
        "repression. Growing up in Poland during a period when higher education "
        "for women was severely restricted, she pursued her early studies "
        "through clandestine classes and self-directed learning. In 1891 she "
        "moved to Paris to continue her education at the University of Paris "
        "(Sorbonne). There she studied physics and mathematics, graduating at "
        "the top of her class in physics in 1893 and earning a second degree "
        "in mathematics the following year.\n"
        "\n"
        "In 1895 she married P.\u00a0C., a physicist known for his work on "
        "crystallography and magnetism. The couple formed a scientific "
        "partnership that proved exceptionally productive. Inspired by Henri "
        "Becquerel's discovery that uranium salts emitted penetrating rays, "
        "M.\u00a0C. began investigating the phenomenon. She coined the term "
        '"radioactivity" to describe the spontaneous emission of energy from '
        "certain elements, recognizing that it was an atomic property rather "
        "than a chemical reaction.\n"
        "\n"
        "Working under modest laboratory conditions, M. and P.\u00a0C. "
        "processed tons of pitchblende, a uranium-rich mineral, in order to "
        "isolate previously unknown radioactive substances. In 1898 they "
        "announced the discovery of two new elements: polonium, named after "
        "M.'s native Poland, and radium. The isolation of radium in pure "
        "metallic form required years of painstaking chemical separation and "
        "measurement. Their research provided strong evidence that "
        "radioactivity was linked to the internal structure of atoms, "
        "challenging existing scientific models and contributing to the "
        "development of atomic theory.\n"
        "\n"
        "In 1903 M.\u00a0C., P.\u00a0C., and Henri Becquerel were jointly "
        "awarded the Nobel Prize in Physics for their work on radiation "
        "phenomena. Following P.\u00a0C.'s sudden death in 1906, M.\u00a0C. "
        "succeeded him as professor at the University of Paris, becoming the "
        "first woman to hold that position. In 1911 she received a second "
        "Nobel Prize, this time in Chemistry, for the discovery of radium and "
        "polonium and for her investigation of their properties. Her dual "
        "achievements established her as an international scientific "
        "authority.\n"
        "\n"
        "Beyond her laboratory research, C. played a crucial role in "
        "applying scientific knowledge for humanitarian purposes. During the "
        "First World War, she helped develop mobile radiography units to "
        "assist surgeons in locating bullets and shrapnel in wounded soldiers. "
        'These units, sometimes called "Little C.s," brought X-ray '
        "technology directly to battlefield hospitals. She also trained medical "
        "personnel, including her daughter Irene, in the use of radiological "
        "equipment, thereby expanding the practical impact of her scientific "
        "expertise.\n"
        "\n"
        "C.'s later career was devoted to advancing research and medical "
        "applications of radioactivity. She founded research institutes "
        "dedicated to the study of radioactive elements and their therapeutic "
        "potential, helping to establish radiology as a recognized medical "
        "discipline. Although the long-term health risks of radiation exposure "
        "were not yet fully understood, her work contributed to the "
        "development of radiation therapy for cancer treatment, which remains an "
        "important medical technique today.\n"
        "\n"
        "The hazards of prolonged radiation exposure ultimately affected C.'s "
        "health. She died in 1934 from aplastic anemia, a condition widely "
        "believed to have been caused by her years of handling radioactive "
        "materials without protective measures. At the time, safety protocols "
        "for radiation were minimal, and the risks were not comprehensively "
        "documented. Her laboratory notebooks remain radioactive and are "
        "preserved with special precautions.\n"
        "\n"
        "C.'s legacy extends beyond her scientific discoveries. She became "
        "a symbol of intellectual rigor, perseverance, and the advancement of "
        "women in science. In 1995 her remains were transferred to the "
        "Pantheon in Paris, making her the first woman to be honored there on "
        "her own merits. Her life and achievements have inspired generations of "
        "scientists and students worldwide.\n"
        "\n"
        "Colleagues and contemporaries regarded C. with deep respect. The "
        "physicist Albert Einstein once praised her integrity and independence "
        "of thought, noting her resistance to fame and public pressure. Despite "
        "facing gender-based discrimination and personal hardship, C. "
        "maintained a strong commitment to scientific inquiry and public "
        "service.\n"
        "\n"
        "M.\u00a0C.'s contributions transformed the understanding of atomic "
        "science and opened new paths in physics, chemistry, and medicine. "
        "Through her discoveries, leadership, and dedication to research, she "
        "helped redefine the possibilities of modern science. Her name remains "
        "closely associated with the study of radioactivity and with the "
        "broader struggle to ensure equal access to education and scientific "
        "careers. [Text generated by AI]"
    )

    return text


def main() -> None:
    """Demonstrate syllable-based text formatting with 30 characters per line."""
    # Get the sample text
    sample = sample_text_mc()

    # Hyphenate the text using Hyphenator
    hyphenator = Hyphenator("en_US")
    hyphenated = hyphenator.hyphenate_text(sample)

    # Format the text with maximum 30 characters per line
    max_width = 17
    formatted_lines = _format_hyphenated_lines(hyphenated, max_width)

    # Print the result
    print("Sample text formatted with 15 characters per line:")
    print("-" * 50)
    for i, line in enumerate(formatted_lines):
        # Find first syllable of the first word by hyphenating it
        if line:
            first_word = line.split()[0]  # Get first word
            # Hyphenate the first word to find its syllable structure
            hyphenated_first = hyphenator.hyphenate_text(first_word)
            # Get the first syllable (split on soft hyphens or regular hyphens)
            if SOFT_HYPHEN in hyphenated_first:
                first_syllable = hyphenated_first.split(SOFT_HYPHEN, maxsplit=1)[0]
            elif "-" in hyphenated_first:
                first_syllable = hyphenated_first.split("-", maxsplit=1)[0]
            else:
                first_syllable = first_word
        else:
            first_syllable = ""
        first_syllable_len = len(first_syllable)

        # Check if next line's first syllable could fit on current line
        marker = ""
        if i < len(formatted_lines) - 1:
            next_line = formatted_lines[i + 1]
            if next_line:
                next_first_word = next_line.split()[0]
                next_hyphenated = hyphenator.hyphenate_text(next_first_word)
                if SOFT_HYPHEN in next_hyphenated:
                    next_first_syllable = next_hyphenated.split(SOFT_HYPHEN, maxsplit=1)[0]
                elif "-" in next_hyphenated:
                    next_first_syllable = next_hyphenated.split("-", maxsplit=1)[0]
                else:
                    next_first_syllable = next_first_word
                next_first_syllable_len = len(next_first_syllable)
                # Check if current line + space + next first syllable fits
                combined_len = len(line) + 1 + next_first_syllable_len
                if combined_len <= max_width:
                    marker = f"<-FIT(+{max_width - combined_len})"

        line_width = max_width + 2
        print(f"[{len(line):3}] [{first_syllable_len:2}] {line:_<{line_width}} {marker}")
    print("-" * 50)


if __name__ == "__main__":
    main()
