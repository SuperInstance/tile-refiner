"""Tests for tile-refiner — raw tiles to structured artifacts."""
from tile_refiner import Tile, TileRefiner, compute_tfidf

def test_tile_create():
    t = Tile(content="Rust is fast", source="research", tile_type="text")
    assert t.content == "Rust is fast"
    print("PASS: tile create")

def test_refiner_processes():
    r = TileRefiner(min_confidence=0.1)
    r.add_document("Python is interpreted and has a GIL")
    r.add_document("Python is great for AI and ML")
    r.add_document("Python is interpreted and easy to learn")
    tiles = [Tile(content="Python is interpreted", source="docs")]
    artifacts = r.refine(tiles)
    assert isinstance(artifacts, list)
    print(f"PASS: refiner → {len(artifacts)} artifacts")

def test_tfidf():
    doc = ["rust", "language", "is", "fast"]
    idf = {"rust": 0.5, "language": 0.3, "fast": 0.7}
    scores = compute_tfidf(doc, idf)
    assert isinstance(scores, dict)
    print(f"PASS: tfidf → {len(scores)} scores")

def test_dedup():
    r = TileRefiner()
    tiles = [
        Tile(content="same content here", source="a"),
        Tile(content="same content here", source="b"),
        Tile(content="different content", source="c"),
    ]
    deduped = r.deduplicate(tiles)
    assert len(deduped) < len(tiles)
    print(f"PASS: dedup → {len(tiles)} → {len(deduped)}")

def test_vocabulary():
    r = TileRefiner()
    r.add_document("neural networks learn from data")
    r.add_document("transformers are neural architectures")
    vocab = r.get_vocabulary()
    assert len(vocab) > 0
    print(f"PASS: vocabulary → {len(vocab)} terms")

if __name__ == "__main__":
    test_tile_create()
    test_refiner_processes()
    test_tfidf()
    test_dedup()
    test_vocabulary()
    print("\nAll 5 pass. Raw tiles become diamonds.")
