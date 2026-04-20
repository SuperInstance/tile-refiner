"""Tile refiner implementation."""

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter


@dataclass
class Tile:
    """A raw tile of content."""

    content: str
    source: str = ""
    tile_type: str = "text"
    metadata: dict = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute content hash for deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


def tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase and extract words."""
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    return words


def compute_idf(documents: List[List[str]]) -> Dict[str, float]:
    """Compute inverse document frequency for terms."""
    idf = {}
    n = len(documents)
    if n == 0:
        return {}

    for doc in documents:
        unique_terms = set(doc)
        for term in unique_terms:
            idf[term] = idf.get(term, 0) + 1

    for term, count in idf.items():
        idf[term] = math.log(n / count)

    return idf


def compute_tfidf(document: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """Compute TF-IDF scores for a document."""
    tf = Counter(document)
    max_tf = max(tf.values()) if tf else 1

    tfidf = {}
    for term, count in tf.items():
        normalized_tf = 0.5 + 0.5 * (count / max_tf)
        tfidf[term] = normalized_tf * idf.get(term, 0)

    return tfidf


def extract_keywords(
    document: str,
    idf: Dict[str, float],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """Extract top-k keywords from document."""
    tokens = tokenize(document)
    if not tokens:
        return []
    tfidf = compute_tfidf(tokens, idf)
    sorted_terms = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
    return sorted_terms[:top_k]


@dataclass
class Artifact:
    """A refined artifact with structured data."""

    content: str
    keywords: List[Tuple[str, float]]
    confidence: float
    sources: List[str] = field(default_factory=list)
    artifact_hash: str = ""


class TileRefiner:
    """Refine raw tiles into structured artifacts."""

    def __init__(self, min_confidence: float = 0.1):
        self.min_confidence = min_confidence
        self._seen_hashes: Set[str] = set()
        self._documents: List[List[str]] = []
        self._idf: Dict[str, float] = {}

    def add_document(self, text: str) -> None:
        """Add document to IDF corpus."""
        tokens = tokenize(text)
        self._documents.append(tokens)
        # Recompute IDF
        self._idf = compute_idf(self._documents)

    def deduplicate(self, tiles: List[Tile]) -> List[Tile]:
        """Remove duplicate tiles based on content hash."""
        unique_tiles = []
        for tile in tiles:
            tile_hash = tile.compute_hash()
            if tile_hash not in self._seen_hashes:
                self._seen_hashes.add(tile_hash)
                unique_tiles.append(tile)
        return unique_tiles

    def boost_confidence(
        self,
        base_confidence: float,
        keyword_matches: int,
        total_keywords: int
    ) -> float:
        """Boost confidence based on keyword relevance."""
        if total_keywords == 0:
            return base_confidence

        match_ratio = keyword_matches / total_keywords
        boost = 1.0 + (match_ratio * 0.5)  # Up to 1.5x boost
        return min(1.0, base_confidence * boost)

    def refine(
        self,
        tiles: List[Tile],
        query: Optional[str] = None
    ) -> List[Artifact]:
        """Refine tiles into artifacts."""
        # Deduplicate
        unique_tiles = self.deduplicate(tiles)

        if not unique_tiles:
            return []

        # Add to corpus for IDF
        for tile in unique_tiles:
            self.add_document(tile.content)

        # Extract keywords and compute confidence
        artifacts = []
        query_keywords = set(tokenize(query)) if query else set()

        for tile in unique_tiles:
            keywords = extract_keywords(tile.content, self._idf, top_k=5)

            # Base confidence from keyword scores
            if keywords:
                # Use max TF-IDF score as base, normalized by content length
                max_score = max(score for _, score in keywords)
                content_length_factor = min(1.0, len(tile.content) / 50.0)
                base_conf = max_score * content_length_factor
            else:
                # Fallback: give some confidence based on content length
                base_conf = min(0.5, len(tile.content) / 100.0)

            # Boost based on query relevance
            if query_keywords:
                keyword_terms = set(k for k, _ in keywords)
                matches = len(keyword_terms & query_keywords)
                confidence = self.boost_confidence(base_conf, matches, len(query_keywords))
            else:
                confidence = base_conf

            if confidence >= self.min_confidence:
                artifact = Artifact(
                    content=tile.content,
                    keywords=keywords,
                    confidence=confidence,
                    sources=[tile.source] if tile.source else [],
                    artifact_hash=tile.compute_hash()
                )
                artifacts.append(artifact)

        # Sort by confidence
        artifacts.sort(key=lambda a: a.confidence, reverse=True)
        return artifacts

    def get_corpus_size(self) -> int:
        """Get number of documents in corpus."""
        return len(self._documents)

    def get_vocabulary(self) -> Set[str]:
        """Get all unique terms in corpus."""
        vocab = set()
        for doc in self._documents:
            vocab.update(doc)
        return vocab
