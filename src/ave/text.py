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
# Stream Classes
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
        return self._cursor < self.count()

    def has_previous(self) -> bool:
        """Check if there are items to rewind to."""
        return self._cursor > 0

    def reset(self) -> None:
        """Reset the cursor to the beginning."""
        self._cursor = 0

    def position(self) -> int:
        """Get the current cursor position."""
        return self._cursor

    def set_position(self, position: int) -> None:
        """Set the cursor to a specific position."""
        self._cursor = max(0, min(position, self.count()))

    def count(self) -> int:
        """Get the total number of items in the stream."""
        return len(self._items)

    def item_at(self, position: int) -> str:
        """Get item at specific position without consuming."""
        if 0 <= position < len(self._items):
            return self._items[position]
        return ""

    def next_item_variants(self, char_length: int) -> list[tuple[int, str]]:
        """Get variants of next items with their cursor positions.

        Returns a list of (new_cursor_position, variant) tuples where:
        - First tuple: position after consuming next item, and the item
        - Second tuple: position after consuming next two items, and the concatenated items
        - Third tuple: position after consuming next three items, and the concatenated items
        - And so on, until the concatenated string exceeds char_length

        The cursor position is NOT changed by this method.

        Args:
            char_length: Maximum character length for the variants

        Returns:
            List of (new_cursor_position, variant_string) tuples
        """
        variants = []
        current_pos = self._cursor
        accumulated = ""

        while self.has_next() and len(accumulated) <= char_length:
            item = self._format_item(self._cursor)
            accumulated += item
            if len(accumulated) <= char_length:
                new_pos = current_pos + len(variants) + 1
                variants.append((new_pos, accumulated))
                self._cursor += 1
            else:
                break

        # Restore cursor position
        self._cursor = current_pos
        return variants


class AvCharacterStream(AvStreamBase):
    """Character provider."""

    def _initialize_items(self, input_data: str) -> None:
        self._items = list(input_data)

    def _format_item(self, index: int) -> str:
        return self._items[index]


class AvSyllableStream(AvStreamBase):
    """Syllable provider for hyphenated text.

    Takes text and a Hyphenator, splits text into syllables that can be
    consumed as stream items.
    """

    def __init__(self, text: str, hyphenator: Hyphenator):
        """Initialize with text and hyphenator.

        Args:
            text: The input text to split into syllables
            hyphenator: Hyphenator instance to use for syllable detection
        """
        self._text = text
        self._hyphenator = hyphenator
        super().__init__(text)  # Pass text to parent for initialization

    def _initialize_items(self, input_data: str) -> None:
        """Initialize with syllables split from text."""
        # Split text into syllables using the hyphenator
        self._items = self._split_text_into_syllables(self._text)

    def _split_text_into_syllables(self, text: str) -> list[str]:
        """Split text into syllable items.

        Each item is one of:
        - A syllable (pure text, no hyphens or spaces)
        - A space character ' '
        - A newline character '\n'
        - Punctuation
        - A hard hyphen '-'

        Note: Soft hyphens are NOT included in items. They are only
        used internally to determine syllable boundaries.
        """
        items = []

        # Process character by character to maintain exact structure
        i = 0
        while i < len(text):
            char = text[i]

            if char.isspace():
                # Whitespace
                items.append(char)
                i += 1
            elif not char.isalnum() and char != "-":
                # Punctuation (except hyphen)
                items.append(char)
                i += 1
            elif char.isalnum() or char == "-":
                # Start of a word or contains hyphen
                # Find the full word
                word_start = i
                while i < len(text) and (text[i].isalnum() or text[i] == "-"):
                    i += 1
                word = text[word_start:i]

                # Skip if it's just a hyphen
                if word == "-":
                    items.append("-")
                    continue

                # Hyphenate the word (clean of hyphens first)
                clean_word = word.replace("-", "")
                if clean_word:
                    # If word contains hyphens, treat each part separately
                    if "-" in word:
                        parts = word.split("-")
                        for part_idx, part in enumerate(parts):
                            if part:
                                # Hyphenate this part
                                part_hyphenated = self._hyphenator.hyphenate_text(part)
                                part_syllables = part_hyphenated.split(SOFT_HYPHEN)
                                for syllable in part_syllables:
                                    if syllable:
                                        items.append(syllable)
                            # Add hyphen between parts (except after last)
                            if part_idx < len(parts) - 1:
                                items.append("-")
                    else:
                        # Regular word without hyphens
                        hyphenated = self._hyphenator.hyphenate_text(clean_word)
                        syllables = hyphenated.split(SOFT_HYPHEN)
                        for syllable in syllables:
                            if syllable:
                                items.append(syllable)

        return items

    def _format_item(self, index: int) -> str:
        """Format a syllable item for output.

        Best practice: Each item should be ready to output as-is.
        The stream items are structured so that concatenation produces
        the original text exactly as input.
        """
        return self._items[index]

    _CLOSING_PUNCT: set[str] = set(".,;:!?")
    _OPENING_BRACKETS: set[str] = set("([{")
    _CLOSING_BRACKETS: set[str] = set(")]}")
    _WHITESPACE: set[str] = set(" \t\r\f\v")

    def _is_forbidden_start(self, ch: str) -> bool:
        """Check if character is forbidden at fragment start."""
        return ch in self._WHITESPACE or ch in self._CLOSING_PUNCT or ch in self._CLOSING_BRACKETS or ch == "-"

    def _is_forbidden_end(self, ch: str) -> bool:
        """Check if character is forbidden at fragment end."""
        return ch in self._OPENING_BRACKETS

    def _skip_leading_ws(self, pos: int) -> int:
        """Advance pos past non-newline whitespace."""
        items = self._items
        n = len(items)
        while pos < n and items[pos].isspace() and items[pos] != "\n":
            pos += 1
        return pos

    def next_item_variants(self, char_length: int) -> list[tuple[int, str]]:
        """
        Generate candidate text fragments for the next justified line segment.

        This function produces all valid substrings starting at the current cursor
        position that may be placed into the next output line whose target width is
        `char_length`. The returned variants must respect word-processing and
        typography rules required for fully justified text composition.

        The function is intended to be used by a justification engine that selects
        the best variant and later distributes inter-word spacing so that the line
        exactly fills the target width.

        ------------------------------------------------------------------------
        FUNCTIONAL GUARANTEES
        ------------------------------------------------------------------------

        The returned variants must satisfy:

        1. **Whitespace correctness**
        • The fragment must never start or end with whitespace.
        • Internal whitespace must remain unchanged (space collapsing is handled
            by the justification phase).
        • Preserved spaces (including non-breaking spaces and soft hyphens)
            must not appear at fragment boundaries.
        • Line breaks are not allowed before non-breaking spaces (U+00A0).

        2. **Valid line break boundaries**
        • A fragment must not start with punctuation that visually belongs to the
            previous line (e.g. ", . ; : ! ? ) ] }").
        • A fragment must not start with closing brackets or hyphens.
        • A fragment must not end with opening brackets ("( [ {").

        3. **Word integrity**
        • The fragment must represent a valid word or partial word.
        • When breaking mid-word, a hyphen is added if needed.
        • Words may contain hyphenation points indicated by soft hyphens (U+00AD).
        • Soft hyphens (U+00AD) are reserved spaces and must not be followed
            by additional hyphens (prevents "word--" with soft hyphen).
        • Possessive apostrophe-s ('s) must stay attached to the preceding word.

        4. **Structural preservation**
        • Existing newlines represent hard breaks and must not be crossed unless
            explicitly allowed.
        • Non-breaking sequences (e.g. protected tokens) must never be split.

        5. **Length constraint**
        • The visual length of the fragment must be ≤ `char_length`.
        • The fragment length is measured before justification spacing expansion.

        ------------------------------------------------------------------------
        TYPOGRAPHIC REQUIREMENTS FOR JUSTIFIED TEXT
        ------------------------------------------------------------------------

        These variants enable the justification engine to later apply rules such as:

        • Lines (except the final line of a paragraph) should visually align with
        both margins by distributing extra space between words.
        • Spacing expansion must remain within acceptable limits to avoid visual
        gaps ("rivers") and uneven density.
        • Hyphenation should be used when necessary to avoid excessive spacing and
        provide additional break opportunities.
        • Very short trailing words, widows, and orphans should be avoided where
        possible.
        • Non-breaking constructs (numbers with units, names, abbreviations) must
        remain intact.

        ------------------------------------------------------------------------
        RETURNS
        ------------------------------------------------------------------------

        A list of (new_cursor_position, fragment_string) tuples representing all
        valid next-line candidates, sorted by fragment length ascending (shortest first).
        Each variant is one syllable longer than the previous, providing maximum
        flexibility for the justification engine.
        """
        if char_length <= 0:
            return []

        items = self._items
        n = len(items)
        start = self._cursor

        if start >= n:
            return []

        # Skip leading whitespace
        while start < n and items[start].isspace() and items[start] != "\n":
            start += 1

        if start >= n:
            return []

        # Build variants by adding one item at a time
        variants: list[tuple[int, str]] = []
        seen_fragments: set[str] = set()  # Deduplicate by fragment text
        fragment_parts: list[str] = []
        fragment_len = 0
        i = start
        in_word = False  # Track if we're inside a word

        while i < n:
            item = items[i]

            # Stop at newline (hard break)
            if item == "\n":
                break

            # Check if adding this item would exceed char_length
            item_len = len(item)
            if fragment_len + item_len > char_length:
                break

            # Classify current item
            is_syllable = item.isalnum() or (len(item) > 1 and item[0].isalnum())
            is_whitespace = item.isspace()
            is_hard_hyphen = item == "-"

            # Update word tracking
            if is_syllable:
                in_word = True
            elif is_whitespace or is_hard_hyphen or item in self._CLOSING_PUNCT:
                in_word = False

            # Add item to fragment
            fragment_parts.append(item)
            fragment_len += item_len
            i += 1

            # Skip creating variant if we just added whitespace
            # (would create duplicate after stripping)
            if is_whitespace:
                continue

            # Determine if we need a hyphen at this break point
            # Add hyphen if: we're in a word AND next item continues the word
            # (either a syllable or a hard hyphen in a compound word)
            next_item = items[i] if i < n else None
            next_is_syllable = next_item is not None and (
                next_item.isalnum() or (len(next_item) > 1 and next_item[0].isalnum())
            )
            next_is_compound_hyphen = next_item == "-"
            needs_hyphen = in_word and (next_is_syllable or next_is_compound_hyphen)

            # Build the fragment
            fragment = "".join(fragment_parts).strip()

            if not fragment:
                continue

            # Skip if fragment starts with forbidden character
            if self._is_forbidden_start(fragment[0]):
                continue

            # Skip if fragment ends with forbidden character (opening bracket),
            # whitespace, or soft hyphen (preserved space)
            if self._is_forbidden_end(fragment[-1]) or fragment[-1].isspace() or fragment[-1] == "\u00ad":
                continue

            # Add hyphen if breaking mid-word (but not if already ends with hyphen)
            # Also don't add hyphen after soft hyphens (U+00AD) which are reserved spaces
            if needs_hyphen and not fragment.endswith("-") and not fragment.endswith("\u00ad"):
                # Check if hyphen would exceed length
                if fragment_len + 1 > char_length:
                    continue
                display_fragment = fragment + "-"
            else:
                display_fragment = fragment

            # Check if next item is a non-breaking space (should not be broken before)
            if i < n and items[i] == "\u00a0":  # Non-breaking space
                continue

            # Calculate next cursor position (skip whitespace)
            next_cursor = i
            while next_cursor < n and items[next_cursor].isspace() and items[next_cursor] != "\n":
                next_cursor += 1

            # Check for orphaned punctuation at next position
            if next_cursor < n:
                next_start = items[next_cursor]
                if next_start in self._CLOSING_PUNCT or next_start in self._CLOSING_BRACKETS:
                    continue

                # Check for possessive 's - should stay attached to previous word
                if next_start == "'" and next_cursor + 1 < n and items[next_cursor + 1] == "s":
                    # print(f"DEBUG2: Skipping {display_fragment} before 's", flush=True)
                    continue

            # Skip duplicates (same fragment text)
            if display_fragment in seen_fragments:
                continue
            seen_fragments.add(display_fragment)

            variants.append((next_cursor, display_fragment))

        return variants


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


def main_test_hyphenator(max_width: int = 15) -> None:
    """Demonstrate syllable-based text formatting with 30 characters per line."""
    # Get the sample text
    sample = sample_text_mc()

    # Hyphenate the text using Hyphenator
    hyphenator = Hyphenator("en_US")
    hyphenated = hyphenator.hyphenate_text(sample)

    # Format the text with maximum 30 characters per line
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


def main_test_avsyllablestream(max_width: int = 15) -> None:
    """Demonstrate AvSyllableStream producing same output as main_test_Hyphenator."""
    print("Sample text formatted with 15 characters per line using AvSyllableStream:")
    print("-" * 50)

    # Get the sample text and create Hyphenator
    sample = sample_text_mc()
    hyphenator = Hyphenator("en_US")
    stream = AvSyllableStream(sample, hyphenator)

    formatted_lines: list[str] = []

    # Simple algorithm: pick the longest variant that fits for each line
    # Variants are sorted shortest-first, so the last one is the longest
    while stream.has_next():
        variants = stream.next_item_variants(max_width)

        if not variants:
            # Skip position (e.g., at newline)
            stream.set_position(min(stream.position() + 1, stream.count()))
            continue

        # Pick the longest variant (last in list)
        new_pos, variant = variants[-1]
        formatted_lines.append(variant)
        stream.set_position(new_pos)

    # Print the result with analysis (same as main_test_Hyphenator)
    for i, line in enumerate(formatted_lines):
        if line:
            first_word = line.split()[0]
            hyphenated_first = hyphenator.hyphenate_text(first_word)
            if SOFT_HYPHEN in hyphenated_first:
                first_syllable = hyphenated_first.split(SOFT_HYPHEN, maxsplit=1)[0]
            elif "-" in hyphenated_first:
                first_syllable = hyphenated_first.split("-", maxsplit=1)[0]
            else:
                first_syllable = first_word
        else:
            first_syllable = ""
        first_syllable_len = len(first_syllable)

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
                # Add 1 for hyphen if the word has more syllables after the first
                needs_hyphen = next_first_syllable != next_first_word
                next_first_syllable_len = len(next_first_syllable) + (1 if needs_hyphen else 0)
                combined_len = len(line) + 1 + next_first_syllable_len
                if combined_len <= max_width:
                    marker = f"<-FIT(+{max_width - combined_len})>"

        print(f"[{len(line):3}] [{first_syllable_len:2}] {line:_<{max_width}} {marker}")
    print("-" * 50)


def main_test_syllable_next_item_variants() -> None:
    """Demonstrate next_item_variants functionality with syllable stream."""
    print("Demonstrating next_item_variants with AvSyllableStream:")
    print("-" * 60)

    # Create a syllable stream with comprehensive test input
    # Covers all rules and special cases:
    # - Regular words and spaces
    # - Punctuation (closing: .,;:!? and opening: ([{)
    # - Hard hyphens
    # - Non-breaking spaces
    # - Newlines (hard breaks)
    # - Words that can be hyphenated
    # - Edge cases (single chars, orphaned punctuation)

    sample = (
        "Word1 word2-hyphenated word3. Punctuation! (opening brackets) "
        "closing brackets]. Non\u00adbreaking\u00adspaces and soft-hyphens.\n"
        "Newline creates hard break. Verylongwordthatneedshyphenation "
        "short orphaned punctuation... () function() end ( multitopic ). END"
    )
    hyphenator = Hyphenator("en_US")
    stream = AvSyllableStream(sample, hyphenator)

    # First, show all items with positions for reference
    print("Stream items (position: item):")
    for i in range(stream.count()):
        item = stream.item_at(i)
        print(f"  {i:2}: {repr(item)}")
    print()

    # Test at specific interesting positions with different char_length values
    test_cases = [
        # (position, char_length, description)
        (0, 20, "Start: regular words"),
        (0, 8, "Start: short limit"),
        (2, 15, "At hyphenated word (word2)"),
        (3, 15, "At hard hyphen"),
        (4, 15, "In hyphenated word (hy)"),
        (9, 15, "At word before punctuation"),
        (10, 20, "At closing punctuation (.)"),
        (12, 20, "At word that needs hyphenation (Punc)"),
        (15, 20, "After hyphenated word (tion)"),
        (16, 20, "At exclamation mark"),
        (18, 25, "At opening bracket"),
        (24, 20, "At closing bracket"),
        (34, 30, "At non-breaking spaces (Non)"),
        (45, 20, "At soft hyphen"),
        (46, 15, "After soft hyphen"),
        (50, 20, "At newline"),
        (51, 20, "After newline (hard break)"),
        (52, 15, "At very long word (needs hyphenation)"),
        (54, 10, "Short limit at long word"),
        (57, 20, "At orphaned punctuation"),
    ]

    for pos, char_length, desc in test_cases:
        if pos < stream.count():
            stream.reset()
            # Move to test position
            for _ in range(pos):
                if stream.has_next():
                    stream.next_item()

            # Get variants
            variants = stream.next_item_variants(char_length)

            print(f"\n{desc}")
            print(f"Position {pos} (item: '{stream.item_at(pos)}'), " f"max_length={char_length}:")
            if variants:
                for i, (new_pos, variant) in enumerate(variants, 1):
                    print(f"  Variant {i:2}: '{variant}' (length: {len(variant)}, new_pos: {new_pos})")
            else:
                print("  No variants (empty or at end)")
        else:
            print(f"\n{desc}")
            print(f"Position {pos} is beyond stream length")

    print()

    # Test with empty stream
    empty_stream = AvSyllableStream("", hyphenator)
    empty_variants = empty_stream.next_item_variants(5)
    print(f"Empty stream variants: {empty_variants}")
    print("-" * 60)

    # Get all variants from the entire sample
    stream.reset()
    all_variants = []
    while stream.has_next():
        variants = stream.next_item_variants(1000)  # Use a large limit to get all possible variants
        if variants:
            all_variants.extend(variants)
            # Move to the position of the longest variant to avoid overlap
            stream.set_position(variants[-1][0])
        else:
            stream.set_position(stream.position() + 1)

    print("\nAll variants from the entire sample:")
    print(sample)
    for i, (pos, variant) in enumerate(all_variants, 1):
        print(f"  {i:3}: '{variant}' (pos: {pos})")
    print(f"Total variants: {len(all_variants)}")


def main_test_character_variants() -> None:
    """Demonstrate next_item_variants with AvCharacterStream."""
    print("Demonstrating next_item_variants with AvCharacterStream:")
    print("-" * 60)

    # Create a character stream with sample text
    sample = "Hello, world!"
    stream = AvCharacterStream(sample)

    # Test at different positions
    test_positions = [0, 5, 7, 13]
    char_length = 10

    for pos in test_positions:
        if pos < len(sample):
            stream.reset()
            # Move to test position
            for _ in range(pos):
                if stream.has_next():
                    stream.next_item()

            # Get variants
            variants = stream.next_item_variants(char_length)

            print(f"Position {pos} (char: '{sample[pos] if pos < len(sample) else ''}'), max_length={char_length}:")
            for i, (new_pos, variant) in enumerate(variants, 1):
                print(f"  Variant {i:2}: '{variant}' (length: {len(variant)}, new_pos: {new_pos})")

            # Demonstrate jumping to a variant position
            # Note: Variants are sorted shortest-first
            if variants:
                chosen_variant = 2  # Choose the second variant (second shortest)
                if chosen_variant <= len(variants):
                    target_pos, _ = variants[chosen_variant - 1]
                    stream.reset()
                    # Move to original position
                    for _ in range(pos):
                        if stream.has_next():
                            stream.next_item()
                    # Move cursor to the position after the consumed items
                    stream.set_position(target_pos)
                    print(f"  -> Jumped to variant {chosen_variant}, cursor now at position {stream.position()}")
            print()

    # Test with empty stream
    empty_stream = AvCharacterStream("")
    empty_variants = empty_stream.next_item_variants(5)
    print(f"Empty stream variants: {empty_variants}")
    print("-" * 60)


if __name__ == "__main__":
    # print("main_test_Hyphenator()")
    # main_test_hyphenator(20)
    # print("-" * 50)
    # print("main_test_AvSyllableStream()")
    # main_test_avsyllablestream(20)
    # print("main_test_syllable_next_item_variants()")
    # main_test_syllable_next_item_variants()
    print("main_test_character_variants()")
    main_test_character_variants()
