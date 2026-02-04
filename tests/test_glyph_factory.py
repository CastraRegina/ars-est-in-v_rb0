"""Comprehensive tests for the new composition-based glyph factory system."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from fontTools.ttLib import TTFont

from ave.font_support import AvFontProperties
from ave.glyph import AvGlyph
from ave.glyph_factory import (
    AvGlyphFactory,
    FactoryGlyphSource,
    FallbackGlyphSource,
    MemoryGlyphCache,
    PersistentGlyphCache,
    PolygonizeTransformer,
    TTFontGlyphSource,
)
from ave.path import AvPath


class TestMemoryGlyphCache:
    """Test the memory-based glyph cache."""

    def test_empty_cache(self):
        """Test that empty cache returns None."""
        cache = MemoryGlyphCache()
        assert cache.get("A") is None

    def test_put_and_get(self):
        """Test storing and retrieving glyphs."""
        cache = MemoryGlyphCache()

        # Create a simple glyph
        points = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 0.0]], dtype=np.float64)
        path = AvPath(points, ["M", "L"])
        glyph = AvGlyph("A", 100.0, path)

        cache.put("A", glyph)
        retrieved = cache.get("A")

        assert retrieved is not None
        assert retrieved.character == "A"
        assert retrieved.width() == 100.0

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        cache = MemoryGlyphCache()

        # Add some glyphs
        points = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 0.0]], dtype=np.float64)
        path = AvPath(points, ["M", "L"])
        glyph_a = AvGlyph("A", 100.0, path)
        glyph_b = AvGlyph("B", 120.0, path)

        cache.put("A", glyph_a)
        cache.put("B", glyph_b)

        # Serialize
        data = cache.to_dict()
        assert data["type"] == "MemoryGlyphCache"
        assert "A" in data["characters"]
        assert "B" in data["characters"]

        # Deserialize into new cache
        new_cache = MemoryGlyphCache()
        new_cache.from_dict(data)

        assert new_cache.get("A") is not None
        assert new_cache.get("B") is not None
        assert new_cache.get("A").character == "A"

    def test_save_and_load(self):
        """Test saving and loading from file."""
        cache = MemoryGlyphCache()

        points = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 0.0]], dtype=np.float64)
        path = AvPath(points, ["M", "L"])
        glyph = AvGlyph("X", 95.0, path)
        cache.put("X", glyph)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json.gz", delete=False) as f:
            temp_path = f.name

        try:
            cache.save(temp_path)

            # Load into new cache
            new_cache = MemoryGlyphCache()
            new_cache.load(temp_path)

            retrieved = new_cache.get("X")
            assert retrieved is not None
            assert retrieved.character == "X"
            assert retrieved.width() == 95.0
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestPersistentGlyphCache:
    """Test the persistent glyph cache."""

    def test_auto_load(self):
        """Test automatic loading on initialization."""
        cache = MemoryGlyphCache()

        points = np.array([[0.0, 0.0, 0.0], [50.0, 50.0, 0.0]], dtype=np.float64)
        path = AvPath(points, ["M", "L"])
        glyph = AvGlyph("Z", 80.0, path)
        cache.put("Z", glyph)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json.gz", delete=False) as f:
            temp_path = f.name

        try:
            cache.save(temp_path)

            # Create persistent cache with auto-load
            persistent = PersistentGlyphCache(temp_path, auto_load=True)

            retrieved = persistent.get("Z")
            assert retrieved is not None
            assert retrieved.character == "Z"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_no_auto_load(self):
        """Test that auto_load=False doesn't load existing file."""
        cache = MemoryGlyphCache()

        points = np.array([[0.0, 0.0, 0.0], [50.0, 50.0, 0.0]], dtype=np.float64)
        path = AvPath(points, ["M", "L"])
        glyph = AvGlyph("Y", 75.0, path)
        cache.put("Y", glyph)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json.gz", delete=False) as f:
            temp_path = f.name

        try:
            cache.save(temp_path)

            # Create persistent cache without auto-load
            persistent = PersistentGlyphCache(temp_path, auto_load=False)

            # Should not have the glyph
            assert persistent.get("Y") is None
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestTTFontGlyphSource:
    """Test the TTFont glyph source."""

    @pytest.fixture
    def ttfont(self):
        """Load a test font."""
        return TTFont("fonts/RobotoFlex[GRAD,XOPQ,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght].ttf")

    def test_get_glyph(self, ttfont):
        """Test getting a glyph from TTFont."""
        source = TTFontGlyphSource(ttfont)
        glyph = source.get_glyph("A")

        assert glyph.character == "A"
        assert glyph.width() > 0
        assert len(glyph.path.points) > 0

    def test_get_font_properties(self, ttfont):
        """Test getting font properties."""
        source = TTFontGlyphSource(ttfont)
        props = source.get_font_properties()

        assert props.units_per_em > 0
        assert props.ascender > 0

    def test_invalid_character(self, ttfont):
        """Test that invalid characters raise KeyError."""
        source = TTFontGlyphSource(ttfont)

        with pytest.raises(KeyError):
            source.get_glyph("\u9999")


class TestFallbackGlyphSource:
    """Test the fallback glyph source."""

    @pytest.fixture
    def ttfont(self):
        """Load a test font."""
        return TTFont("fonts/RobotoFlex[GRAD,XOPQ,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght].ttf")

    def test_primary_source_success(self, ttfont):
        """Test that primary source is used when available."""
        primary = TTFontGlyphSource(ttfont)
        secondary = TTFontGlyphSource(ttfont)

        fallback = FallbackGlyphSource(primary, secondary)
        glyph = fallback.get_glyph("A")

        assert glyph.character == "A"

    def test_fallback_to_secondary(self, ttfont):
        """Test fallback to secondary source."""

        # Create a mock primary that raises KeyError
        class MockPrimary:
            """Mock glyph source that always fails to get glyphs."""

            def get_glyph(self, character: str):
                """Raise KeyError to test fallback behavior."""
                raise KeyError(f"Character {character} not found")

            def get_font_properties(self):
                """Return mock font properties."""
                return AvFontProperties(units_per_em=1000)

        primary = MockPrimary()
        secondary = TTFontGlyphSource(ttfont)

        fallback = FallbackGlyphSource(primary, secondary)
        glyph = fallback.get_glyph("A")

        assert glyph.character == "A"

    def test_font_properties_from_secondary(self, ttfont):
        """Test getting font properties from secondary."""
        primary = TTFontGlyphSource(ttfont)
        secondary = TTFontGlyphSource(ttfont)

        fallback = FallbackGlyphSource(primary, secondary, font_props_from_secondary=True)
        props = fallback.get_font_properties()

        assert props.units_per_em > 0


class TestPolygonizeTransformer:
    """Test the polygonization transformer."""

    @pytest.fixture
    def ttfont(self):
        """Load a test font."""
        return TTFont("fonts/RobotoFlex[GRAD,XOPQ,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght].ttf")

    def test_polygonize_glyph(self, ttfont):
        """Test that curves are converted to lines."""
        source = TTFontGlyphSource(ttfont)
        glyph = source.get_glyph("S")  # S has curves

        # Check if original has curves
        has_curves = glyph.path.has_curves

        transformer = PolygonizeTransformer(steps=10)
        polygonized = transformer.transform(glyph)

        if has_curves:
            # Polygonized path should have no curves
            assert not polygonized.path.has_curves

        # Should have same character
        assert polygonized.character == "S"

    def test_zero_steps_no_transform(self, ttfont):
        """Test that zero steps returns original glyph."""
        source = TTFontGlyphSource(ttfont)
        glyph = source.get_glyph("A")

        transformer = PolygonizeTransformer(steps=0)
        result = transformer.transform(glyph)

        # Should return same glyph
        assert result is glyph


class TestAvGlyphFactory:
    """Test the unified glyph factory."""

    @pytest.fixture
    def ttfont(self):
        """Load a test font."""
        return TTFont("fonts/RobotoFlex[GRAD,XOPQ,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght].ttf")

    def test_basic_factory(self, ttfont):
        """Test basic factory without cache or transformer."""
        source = TTFontGlyphSource(ttfont)
        factory = AvGlyphFactory(source=source)

        glyph = factory.get_glyph("A")
        assert glyph.character == "A"

    def test_factory_with_cache(self, ttfont):
        """Test factory with caching."""
        source = TTFontGlyphSource(ttfont)
        cache = MemoryGlyphCache()
        factory = AvGlyphFactory(source=source, cache=cache)

        # First call should cache
        glyph1 = factory.get_glyph("B")
        # Second call should use cache
        glyph2 = factory.get_glyph("B")

        assert glyph1.character == "B"
        assert glyph2.character == "B"
        # Should be same object from cache
        assert cache.get("B") is not None

    def test_factory_with_transformer(self, ttfont):
        """Test factory with transformer."""
        source = TTFontGlyphSource(ttfont)
        transformer = PolygonizeTransformer(steps=10)
        factory = AvGlyphFactory(source=source, transformer=transformer)

        glyph = factory.get_glyph("S")

        # Should be polygonized if original had curves
        assert glyph.character == "S"

    def test_factory_with_cache_and_transformer(self, ttfont):
        """Test factory with both cache and transformer."""
        source = TTFontGlyphSource(ttfont)
        cache = MemoryGlyphCache()
        transformer = PolygonizeTransformer(steps=10)
        factory = AvGlyphFactory(source=source, cache=cache, transformer=transformer)

        glyph1 = factory.get_glyph("C")
        glyph2 = factory.get_glyph("C")

        # Both should be same (cached)
        assert glyph1.character == "C"
        assert glyph2.character == "C"

    def test_get_font_properties(self, ttfont):
        """Test getting font properties."""
        source = TTFontGlyphSource(ttfont)
        factory = AvGlyphFactory(source=source)

        props = factory.get_font_properties()
        assert props.units_per_em > 0

    def test_to_cache_dict(self, ttfont):
        """Test serialization to dict."""
        source = TTFontGlyphSource(ttfont)
        cache = MemoryGlyphCache()
        factory = AvGlyphFactory(source=source, cache=cache)

        # Load some glyphs
        factory.get_glyph("A")
        factory.get_glyph("B")

        data = factory.to_cache_dict()
        assert "glyphs" in data
        assert "font_properties" in data
        assert "A" in data["glyphs"]
        assert "B" in data["glyphs"]

    def test_from_cache_dict(self, ttfont):
        """Test loading from cache dict."""
        source = TTFontGlyphSource(ttfont)
        cache = MemoryGlyphCache()
        factory = AvGlyphFactory(source=source, cache=cache)

        # Load and cache some glyphs
        factory.get_glyph("X")
        factory.get_glyph("Y")

        # Serialize
        data = factory.to_cache_dict()

        # Create new factory from dict
        new_source = TTFontGlyphSource(ttfont)
        new_factory = AvGlyphFactory.from_cache_dict(data, new_source)

        # Should have cached glyphs
        glyph_x = new_factory.get_glyph("X")
        glyph_y = new_factory.get_glyph("Y")

        assert glyph_x.character == "X"
        assert glyph_y.character == "Y"

    def test_save_and_load_from_file(self, ttfont):
        """Test saving and loading from file."""
        source = TTFontGlyphSource(ttfont)
        cache = MemoryGlyphCache()
        factory = AvGlyphFactory(source=source, cache=cache)

        # Cache some glyphs
        factory.get_glyph("M")
        factory.get_glyph("N")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json.gz", delete=False) as f:
            temp_path = f.name

        try:
            factory.save_to_file(temp_path)

            # Load into new factory
            new_source = TTFontGlyphSource(ttfont)
            new_factory = AvGlyphFactory.load_from_file(temp_path, new_source)

            # Should have cached glyphs
            glyph_m = new_factory.get_glyph("M")
            glyph_n = new_factory.get_glyph("N")

            assert glyph_m.character == "M"
            assert glyph_n.character == "N"
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestConvenienceFunctions:
    """Test convenience factory creation functions."""

    @pytest.fixture
    def ttfont(self):
        """Load a test font."""
        return TTFont("fonts/RobotoFlex[GRAD,XOPQ,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght].ttf")

    def test_create_cached_ttfont_factory(self, ttfont):
        """Test creating a cached TTFont factory."""
        factory = AvGlyphFactory.create_from_ttfont(ttfont)

        glyph = factory.get_glyph("A")
        assert glyph.character == "A"

    def test_create_cached_ttfont_factory_with_polygonization(self, ttfont):
        """Test creating factory with polygonization."""
        factory = AvGlyphFactory.create_from_ttfont(ttfont, polygonize_steps=10)

        glyph = factory.get_glyph("S")
        assert glyph.character == "S"

    def test_create_cached_ttfont_factory_with_file(self, ttfont):
        """Test creating factory with persistent cache."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json.gz", delete=False) as f:
            temp_path = f.name

        try:
            factory = AvGlyphFactory.create_from_ttfont(ttfont, cache_path=temp_path)
            factory.get_glyph("D")
            factory.save_to_file(temp_path)

            # Load new factory from same file
            factory2 = AvGlyphFactory.create_from_ttfont(ttfont, cache_path=temp_path)
            glyph = factory2.get_glyph("D")
            assert glyph.character == "D"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_create_polygonizing_factory(self, ttfont):
        """Test creating polygonizing factory."""
        source = TTFontGlyphSource(ttfont)
        factory = AvGlyphFactory.create_with_polygonizer(source, steps=20)

        glyph = factory.get_glyph("O")
        assert glyph.character == "O"

    def test_create_dual_source_factory(self, ttfont):
        """Test creating dual source factory."""
        primary = TTFontGlyphSource(ttfont)
        secondary = TTFontGlyphSource(ttfont)

        factory = AvGlyphFactory.create_with_fallback(primary, secondary)
        glyph = factory.get_glyph("P")
        assert glyph.character == "P"


class TestFactoryGlyphSource:
    """Test the factory adapter source."""

    @pytest.fixture
    def ttfont(self):
        """Load a test font."""
        return TTFont("fonts/RobotoFlex[GRAD,XOPQ,XTRA,YOPQ,YTAS,YTDE,YTFI,YTLC,YTUC,opsz,slnt,wdth,wght].ttf")

    def test_wrap_factory_as_source(self, ttfont):
        """Test wrapping a factory as a source."""
        # Create a factory
        inner_factory = AvGlyphFactory.create_from_ttfont(ttfont)

        # Wrap it as a source
        source = FactoryGlyphSource(inner_factory)

        # Use it in another factory
        outer_factory = AvGlyphFactory(source=source, cache=MemoryGlyphCache())

        glyph = outer_factory.get_glyph("R")
        assert glyph.character == "R"
