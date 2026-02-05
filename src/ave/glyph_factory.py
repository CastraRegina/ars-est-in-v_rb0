"""Composition-based glyph factory system.

This module provides a simplified, composition-based approach to glyph factories
that separates concerns into distinct components: sources, caches, and transformers.
"""

from __future__ import annotations

import gzip
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from fontTools.ttLib import TTFont

from ave.font_support import AvFontProperties
from ave.glyph import AvGlyph

###############################################################################
# Core Protocols
###############################################################################


class GlyphSource(Protocol):
    """Protocol for glyph sources.

    A glyph source is responsible for providing glyphs and font properties.
    Sources can load from TTFont, files, or other factories.
    """

    def get_glyph(self, character: str) -> AvGlyph:
        """Get a glyph for the specified character.

        Args:
            character: The character to get a glyph for.

        Returns:
            AvGlyph instance for the character.

        Raises:
            KeyError: If the character is not available.
        """

    def get_font_properties(self) -> AvFontProperties:
        """Get font properties from this source.

        Returns:
            AvFontProperties for the font.
        """


###############################################################################
# Cache Interface
###############################################################################


class GlyphCache(ABC):
    """Abstract interface for glyph caching.

    A cache stores glyphs that have been loaded or generated to avoid
    repeated expensive operations.
    """

    @abstractmethod
    def get(self, key: str) -> Optional[AvGlyph]:
        """Get a glyph from the cache.

        Args:
            key: The cache key (typically the character).

        Returns:
            The cached glyph, or None if not in cache.
        """

    @abstractmethod
    def put(self, key: str, glyph: AvGlyph) -> None:
        """Store a glyph in the cache.

        Args:
            key: The cache key (typically the character).
            glyph: The glyph to cache.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the cache to persistent storage.

        Args:
            path: Path to save the cache to.
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the cache from persistent storage.

        Args:
            path: Path to load the cache from.
        """

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert cache to dictionary representation.

        Returns:
            Dictionary containing cache data.
        """

    @abstractmethod
    def from_dict(self, data: dict) -> None:
        """Load cache from dictionary representation.

        Args:
            data: Dictionary containing cache data.
        """


###############################################################################
# Transformer Interface
###############################################################################


class GlyphTransformer(ABC):  # pylint: disable=too-few-public-methods
    """Abstract interface for glyph transformations.

    A transformer modifies glyphs in some way (e.g., polygonization,
    simplification, coordinate transformation).
    """

    @abstractmethod
    def transform(self, glyph: AvGlyph) -> AvGlyph:
        """Transform a glyph.

        Args:
            glyph: The glyph to transform.

        Returns:
            A new transformed glyph.
        """
        raise NotImplementedError("Subclasses must implement transform")


###############################################################################
# Cache Implementations
###############################################################################


@dataclass
class MemoryGlyphCache(GlyphCache):
    """In-memory glyph cache.

    Stores glyphs in a dictionary for fast access during runtime.
    Does not persist across program runs.
    """

    def __init__(self):
        """Initialize an empty memory cache."""
        self._cache: Dict[str, AvGlyph] = {}

    def get(self, key: str) -> Optional[AvGlyph]:
        """Get a glyph from the memory cache."""
        return self._cache.get(key)

    def put(self, key: str, glyph: AvGlyph) -> None:
        """Store a glyph in the memory cache."""
        self._cache[key] = glyph

    def save(self, path: str) -> None:
        """Save cache to a compressed JSON file.

        Args:
            path: Path to save the cache file.
        """
        data = self.to_dict()
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(target_path, "wt", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> None:
        """Load cache from a compressed JSON file.

        Args:
            path: Path to the cache file.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is invalid.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Cache file not found: {path}")

        try:
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        except (gzip.BadGzipFile, OSError):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in cache file {path}: {e}") from e

        self.from_dict(data)

    def to_dict(self) -> dict:
        """Convert cache to dictionary.

        Returns:
            Dictionary with glyphs serialized.
        """
        glyphs_dict = {char: glyph.to_dict() for char, glyph in self._cache.items()}
        return {
            "format_version": 1,
            "type": "MemoryGlyphCache",
            "characters": "".join(sorted(glyphs_dict.keys())),
            "glyphs": glyphs_dict,
        }

    def from_dict(self, data: dict) -> None:
        """Load cache from dictionary.

        Args:
            data: Dictionary containing cache data.
        """
        self._cache.clear()
        glyph_entries = data.get("glyphs", {})

        for character, glyph_data in glyph_entries.items():
            try:
                self._cache[character] = AvGlyph.from_dict(glyph_data)
            except (ValueError, KeyError, TypeError) as e:
                print(f"Warning: Failed to load glyph for '{character}': {e}")


@dataclass
class PersistentGlyphCache(GlyphCache):
    """Cache that persists to disk.

    Combines in-memory caching with automatic persistence to a file.
    """

    _file_path: str
    _memory_cache: MemoryGlyphCache

    def __init__(self, file_path: str, auto_load: bool = True):
        """Initialize a persistent cache.

        Args:
            file_path: Path to the cache file.
            auto_load: If True, attempt to load existing cache on init.
        """
        self._file_path = file_path
        self._memory_cache = MemoryGlyphCache()

        if auto_load and Path(file_path).exists():
            try:
                self.load(file_path)
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not load cache from {file_path}: {e}")

    def get(self, key: str) -> Optional[AvGlyph]:
        """Get a glyph from the cache."""
        return self._memory_cache.get(key)

    def put(self, key: str, glyph: AvGlyph) -> None:
        """Store a glyph in the cache."""
        self._memory_cache.put(key, glyph)

    def save(self, path: Optional[str] = None) -> None:
        """Save cache to file.

        Args:
            path: Optional path override. Uses the default path if None.
        """
        save_path = path or self._file_path
        self._memory_cache.save(save_path)

    def load(self, path: Optional[str] = None) -> None:
        """Load cache from file.

        Args:
            path: Optional path override. Uses the default path if None.
        """
        load_path = path or self._file_path
        self._memory_cache.load(load_path)

    def to_dict(self) -> dict:
        """Convert cache to dictionary."""
        return self._memory_cache.to_dict()

    def from_dict(self, data: dict) -> None:
        """Load cache from dictionary."""
        self._memory_cache.from_dict(data)


###############################################################################
# Source Implementations
###############################################################################


@dataclass
class TTFontGlyphSource:
    """Glyph source that loads from a TTFont.

    Extracts glyphs directly from TrueType/OpenType font files.
    """

    _ttfont: TTFont

    def __init__(self, ttfont: TTFont):
        """Initialize with a TTFont instance.

        Args:
            ttfont: The TTFont to load glyphs from.
        """
        self._ttfont = ttfont

    def get_glyph(self, character: str) -> AvGlyph:
        """Get a glyph from the TTFont.

        Args:
            character: The character to get a glyph for.

        Returns:
            AvGlyph instance loaded from the font.

        Raises:
            KeyError: If the character is not in the font.
        """
        return AvGlyph.from_ttfont_character(self._ttfont, character)

    def get_font_properties(self) -> AvFontProperties:
        """Get font properties from the TTFont.

        Returns:
            AvFontProperties extracted from the font.
        """
        return AvFontProperties.from_ttfont(self._ttfont)

    @property
    def ttfont(self) -> TTFont:
        """Get the underlying TTFont instance."""
        return self._ttfont


@dataclass
class FallbackGlyphSource:
    """Glyph source that tries primary then secondary sources.

    Attempts to get glyphs from the primary source first. If that fails,
    falls back to the secondary source.
    """

    _primary: GlyphSource
    _secondary: GlyphSource
    _font_props_from_secondary: bool

    def __init__(
        self,
        primary: GlyphSource,
        secondary: GlyphSource,
        font_props_from_secondary: bool = True,
    ):
        """Initialize with primary and secondary sources.

        Args:
            primary: The primary source to try first.
            secondary: The secondary source to use as fallback.
            font_props_from_secondary: If True, get font properties from secondary.
        """
        self._primary = primary
        self._secondary = secondary
        self._font_props_from_secondary = font_props_from_secondary

    def get_glyph(self, character: str) -> AvGlyph:
        """Get a glyph, trying primary then secondary source.

        Args:
            character: The character to get a glyph for.

        Returns:
            AvGlyph from one of the sources.

        Raises:
            KeyError: If neither source has the character.
        """
        try:
            return self._primary.get_glyph(character)
        except KeyError:
            return self._secondary.get_glyph(character)

    def get_font_properties(self) -> AvFontProperties:
        """Get font properties from configured source.

        Returns:
            AvFontProperties from primary or secondary based on configuration.
        """
        if self._font_props_from_secondary:
            return self._secondary.get_font_properties()
        return self._primary.get_font_properties()


@dataclass
class CacheOnlySource:
    """Glyph source that only provides cached glyphs and font properties.

    This source cannot generate new glyphs - it only serves what's in the cache.
    Useful when loading from a cache file without a TTFont.
    """

    _font_properties: AvFontProperties

    def __init__(self, font_properties: AvFontProperties):
        """Initialize with font properties.

        Args:
            font_properties: Font properties to provide.
        """
        self._font_properties = font_properties

    def get_glyph(self, character: str) -> AvGlyph:
        """Raise KeyError - this source cannot generate glyphs.

        Args:
            character: The requested character.

        Raises:
            KeyError: Always, as this source only works with cached glyphs.
        """
        raise KeyError(
            f"Character '{character}' not in cache. "
            "CacheOnlySource cannot generate new glyphs. "
            "Provide a TTFontGlyphSource to enable glyph generation."
        )

    def get_font_properties(self) -> AvFontProperties:
        """Get font properties.

        Returns:
            The stored font properties.
        """
        return self._font_properties


@dataclass
class FactoryGlyphSource:
    """Adapter that wraps another factory as a glyph source.

    Allows using any object with get_glyph() and get_font_properties()
    methods as a source.
    """

    _factory: Any

    def __init__(self, factory: Any):
        """Initialize with a factory.

        Args:
            factory: Any object with get_glyph() and get_font_properties() methods.
        """
        self._factory = factory

    def get_glyph(self, character: str) -> AvGlyph:
        """Get a glyph from the wrapped factory."""
        return self._factory.get_glyph(character)

    def get_font_properties(self) -> AvFontProperties:
        """Get font properties from the wrapped factory."""
        return self._factory.get_font_properties()


###############################################################################
# Transformer Implementations
###############################################################################


@dataclass
class PolygonizeTransformer(GlyphTransformer):
    """Transformer that polygonizes glyph paths.

    Converts curved paths into line segments for algorithms that work
    better with polygons.
    """

    _steps: int

    def __init__(self, steps: int = 50):
        """Initialize the polygonizer.

        Args:
            steps: Number of line segments per curve. 0 means no polygonization.
        """
        self._steps = steps

    def transform(self, glyph: AvGlyph) -> AvGlyph:
        """Polygonize the glyph's path.

        Args:
            glyph: The glyph to polygonize.

        Returns:
            New glyph with polygonized path.
        """
        if self._steps == 0:
            return glyph

        polygonized_path = glyph.path.polygonize(self._steps)
        return AvGlyph(
            character=glyph.character,
            advance_width=glyph.width(),
            path=polygonized_path,
        )

    @property
    def steps(self) -> int:
        """Get the number of polygonization steps."""
        return self._steps


###############################################################################
# Unified Factory
###############################################################################


@dataclass
class AvGlyphFactory:
    """Unified glyph factory using composition.

    Combines a source, optional cache, and optional transformer to create
    a flexible glyph production pipeline.

    The factory operates in this order:
    1. Check cache (if present)
    2. Get from source
    3. Apply transformation (if present)
    4. Store in cache (if present)
    5. Return glyph
    """

    source: GlyphSource
    cache: Optional[GlyphCache] = None
    transformer: Optional[GlyphTransformer] = None

    def get_glyph(self, character: str) -> AvGlyph:
        """Get a glyph for the specified character.

        The factory follows this pipeline:
        1. Check cache first if available
        2. If not cached, get from source
        3. Apply transformation if configured
        4. Cache the result if cache is available
        5. Return the glyph

        Args:
            character: The character to get a glyph for.

        Returns:
            AvGlyph instance for the character.

        Raises:
            KeyError: If the character is not available in the source.
        """
        # 1. Check cache first
        if self.cache:
            cached = self.cache.get(character)
            if cached:
                return cached

        # 2. Get from source
        # pylint: disable=assignment-from-no-return
        glyph = self.source.get_glyph(character)

        # 3. Apply transformation
        if self.transformer:
            glyph = self.transformer.transform(glyph)

        # 4. Cache result
        if self.cache:
            self.cache.put(character, glyph)

        return glyph

    def get_font_properties(self) -> AvFontProperties:
        """Get font properties from the source.

        Returns:
            AvFontProperties from the underlying source.
        """
        return self.source.get_font_properties()

    @property
    def glyphs(self) -> Dict[str, AvGlyph]:
        """Get cached glyphs if cache is available.

        Returns:
            Dictionary of cached glyphs, or empty dict if no cache.
        """
        if self.cache and isinstance(self.cache, (MemoryGlyphCache, PersistentGlyphCache)):
            if isinstance(self.cache, PersistentGlyphCache):
                return self.cache._memory_cache._cache  # pylint: disable=protected-access
            return self.cache._cache  # pylint: disable=protected-access
        return {}

    def to_cache_dict(self) -> dict:
        """Create a dictionary representation of cached glyphs.

        Returns:
            Dictionary with glyphs and font properties.
        """
        if not self.cache:
            return {
                "format_version": 1,
                "type": "AvGlyphFactory",
                "characters": "",
                "glyphs": {},
                "font_properties": self.get_font_properties().to_dict(),
            }

        cache_dict = self.cache.to_dict()
        cache_dict["font_properties"] = self.get_font_properties().to_dict()
        cache_dict["type"] = "AvGlyphFactory"
        return cache_dict

    @classmethod
    def from_cache_dict(cls, data: dict, source: Optional[GlyphSource] = None) -> "AvGlyphFactory":
        """Create a factory from cache data.

        Args:
            data: Dictionary with cached glyphs.
            source: The source to use for uncached glyphs. If None, creates a cache-only source.

        Returns:
            AvGlyphFactory with loaded cache.
        """
        cache = MemoryGlyphCache()
        cache.from_dict(data)

        # If no source provided, create cache-only source from font properties
        if source is None:
            font_props_dict = data.get("font_properties")
            if not font_props_dict:
                raise ValueError("Cache data missing font_properties and no source provided")
            font_props = AvFontProperties.from_dict(font_props_dict)
            source = CacheOnlySource(font_props)

        return cls(source=source, cache=cache)

    def save_to_file(self, file_path: str) -> None:
        """Save factory cache to file.

        Args:
            file_path: Path to save the cache to.
        """
        # Use to_cache_dict to ensure font_properties are included
        data = self.to_cache_dict()

        target_path = Path(file_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(target_path, "wt", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: str, source: Optional[GlyphSource] = None) -> "AvGlyphFactory":
        """Load factory from cache file.

        Args:
            file_path: Path to the cache file.
            source: Optional source to use for uncached glyphs. If None, creates a
                cache-only source that will raise KeyError for uncached characters.

        Returns:
            AvGlyphFactory with loaded cache.

        Example:
            # Load cache-only (will error on uncached glyphs)
            factory = AvGlyphFactory.load_from_file("cache.json.zip")

            # Load with TTFont fallback for uncached glyphs
            from fontTools.ttLib import TTFont
            ttfont = TTFont("font.ttf")
            factory = AvGlyphFactory.load_from_file(
                "cache.json.zip",
                source=TTFontGlyphSource(ttfont)
            )
        """
        # Load the cache file to get the data
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Cache file not found: {file_path}")

        try:
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        except (gzip.BadGzipFile, OSError):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        # Check if this is a factory cache (with font_properties) or raw cache
        if "font_properties" in data:
            # This is a factory cache - use from_cache_dict
            return cls.from_cache_dict(data, source)

        # This is a raw cache - need font_properties from source or error
        if source is None:
            raise ValueError(
                f"Cache file {file_path} missing font_properties. "
                "Cannot create cache-only factory without font properties."
            )
        # Load as raw cache into a MemoryGlyphCache
        cache = MemoryGlyphCache()
        cache.from_dict(data)
        return cls(source=source, cache=cache)

    @staticmethod
    def create_from_ttfont(
        ttfont: TTFont,
        cache_path: Optional[str] = None,
        polygonize_steps: int = 0,
    ) -> "AvGlyphFactory":
        """Create a cached TTFont factory with optional polygonization.

        Args:
            ttfont: The TTFont to load glyphs from.
            cache_path: Optional path for persistent cache.
            polygonize_steps: Number of steps for polygonization (0 = disabled).

        Returns:
            AvGlyphFactory configured for common use case.
        """
        source = TTFontGlyphSource(ttfont)

        cache: Optional[GlyphCache] = None
        if cache_path:
            cache = PersistentGlyphCache(cache_path, auto_load=False)
        else:
            cache = MemoryGlyphCache()

        transformer: Optional[GlyphTransformer] = None
        if polygonize_steps > 0:
            transformer = PolygonizeTransformer(polygonize_steps)

        return AvGlyphFactory(source=source, cache=cache, transformer=transformer)

    @staticmethod
    def create_with_polygonizer(
        source: GlyphSource,
        steps: int = 50,
    ) -> "AvGlyphFactory":
        """Create a factory that polygonizes glyphs from a source.

        Args:
            source: The source to get glyphs from.
            steps: Number of polygonization steps.

        Returns:
            AvGlyphFactory with polygonization transformer.
        """
        return AvGlyphFactory(
            source=source,
            transformer=PolygonizeTransformer(steps),
        )

    @staticmethod
    def create_with_fallback(
        primary: GlyphSource,
        secondary: GlyphSource,
        font_props_from_secondary: bool = True,
    ) -> "AvGlyphFactory":
        """Create a factory with primary and fallback sources.

        Args:
            primary: Primary source to try first.
            secondary: Secondary source as fallback.
            font_props_from_secondary: Use secondary for font properties.

        Returns:
            AvGlyphFactory with fallback capability.
        """
        fallback_source = FallbackGlyphSource(primary, secondary, font_props_from_secondary)
        return AvGlyphFactory(source=fallback_source)
