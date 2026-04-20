"""Tests for tile-refiner."""

import pytest
from tile_refiner import Tile, TileRefiner, compute_tfidf


def test_tile_creation():
    """Test creating a tile."""
    tile = Tile(content="Hello world", source="test")
    assert tile.content == "Hello world"
    assert tile.source == "test"
    assert tile.compute_hash()


def test_tile_hash_consistency():
    """Test hash consistency."""
    tile1 = Tile(content="Same content")
    tile2 = Tile(content="Same content")
    assert tile1.compute_hash() == tile2.compute_hash()


def test_tokenize():
    """Test tokenization."""
    from tile_refiner.refiner import tokenize
    words = tokenize("Hello, World! This is a test.")
    assert "hello" in words
    assert "world" in words
    assert "test" in words


def test_deduplicate():
    """Test deduplication."""
    refiner = TileRefiner()
    tiles = [
        Tile(content="Unique one"),
        Tile(content="Duplicate"),
        Tile(content="Duplicate"),
        Tile(content="Unique two"),
    ]
    unique = refiner.deduplicate(tiles)
    assert len(unique) == 3
    contents = [t.content for t in unique]
    assert "Duplicate" in contents
    assert contents.count("Duplicate") == 1


def test_refiner_basic():
    """Test basic refinement."""
    refiner = TileRefiner(min_confidence=0.01)  # Lower threshold for tests
    tiles = [
        Tile(content="Python is a great programming language for developers"),
        Tile(content="JavaScript frameworks are popular in web development"),
    ]
    artifacts = refiner.refine(tiles)
    assert len(artifacts) >= 1
    assert all(a.confidence > 0 for a in artifacts)


def test_refiner_with_query():
    """Test refinement with query boosting."""
    refiner = TileRefiner(min_confidence=0.01)
    tiles = [
        Tile(content="Python programming tutorial and code examples"),
        Tile(content="Cooking recipes and tips for food preparation"),
    ]
    artifacts = refiner.refine(tiles, query="python code")
    assert len(artifacts) >= 1


def test_confidence_boosting():
    """Test confidence boosting."""
    refiner = TileRefiner()
    base = 0.5
    boosted = refiner.boost_confidence(base, keyword_matches=3, total_keywords=5)
    assert boosted > base
    assert boosted <= 1.0


def test_compute_tfidf():
    """Test TF-IDF computation."""
    from tile_refiner.refiner import compute_idf, tokenize

    docs = [
        "python is great",
        "python programming",
        "java is also good"
    ]
    tokens_list = [tokenize(d) for d in docs]
    idf = compute_idf(tokens_list)

    # "python" appears in 2/3 docs, should have lower IDF than "great"
    doc_tokens = tokenize("python is great")
    tfidf = compute_tfidf(doc_tokens, idf)

    assert "python" in tfidf
    assert "great" in tfidf
    # IDF for rarer terms should be higher
    assert idf.get("great", 0) > idf.get("python", 0)


def test_extract_keywords():
    """Test keyword extraction."""
    from tile_refiner.refiner import extract_keywords, compute_idf, tokenize

    docs = ["machine learning algorithms", "deep learning networks"]
    tokens_list = [tokenize(d) for d in docs]
    idf = compute_idf(tokens_list)

    keywords = extract_keywords(docs[0], idf)
    assert len(keywords) <= 5
    assert all(isinstance(k, tuple) and len(k) == 2 for k in keywords)


def test_min_confidence_filter():
    """Test minimum confidence filtering."""
    refiner = TileRefiner(min_confidence=0.9)
    tiles = [
        Tile(content="x"),  # Will have low confidence
        Tile(content="machine learning algorithms are powerful tools"),
    ]
    artifacts = refiner.refine(tiles)
    # Only high-confidence artifacts should pass
    assert all(a.confidence >= 0.9 for a in artifacts)


def test_corpus_tracking():
    """Test corpus document tracking."""
    refiner = TileRefiner(min_confidence=0.01)
    assert refiner.get_corpus_size() == 0

    tiles = [Tile(content="doc1"), Tile(content="doc2")]
    refiner.refine(tiles)
    assert refiner.get_corpus_size() == 2

    vocab = refiner.get_vocabulary()
    assert len(vocab) > 0


def test_artifact_structure():
    """Test artifact data structure."""
    refiner = TileRefiner(min_confidence=0.01)
    tiles = [Tile(content="test content with meaningful text", source="source1")]
    artifacts = refiner.refine(tiles)

    assert len(artifacts) >= 1
    artifact = artifacts[0]
    assert artifact.content == "test content with meaningful text"
    assert artifact.sources == ["source1"]
    assert artifact.artifact_hash
    assert isinstance(artifact.keywords, list)
    assert 0 <= artifact.confidence <= 1.0
