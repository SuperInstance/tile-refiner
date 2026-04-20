"""
Core tile processing and artifact generation.

Tiles are raw data chunks that get refined into structured artifacts
with metadata and lineage information.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Tile:
    """A raw data tile awaiting refinement."""

    raw_data: dict[str, Any]
    tile_type: str
    source: str
    tile_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.tile_id is None:
            content = f"{self.tile_type}:{self.source}:{self.timestamp}:{json.dumps(self.raw_data, sort_keys=True)}"
            self.tile_id = hashlib.sha256(content.encode()).hexdigest()[:16]

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to this tile."""
        self.metadata[key] = value


@dataclass
class ArtifactSchema:
    """Schema definition for structured artifacts."""

    name: str
    fields: dict[str, str]
    required_fields: set[str]
    schema_id: Optional[str] = None

    def __post_init__(self):
        if self.schema_id is None:
            content = f"{self.name}:{json.dumps(sorted(self.fields.items()))}"
            self.schema_id = hashlib.sha256(content.encode()).hexdigest()[:12]

    def validate(self, data: dict[str, Any]) -> bool:
        """Validate data against this schema."""
        for field in self.required_fields:
            if field not in data:
                return False
        return True


@dataclass
class Artifact:
    """A refined, structured artifact."""

    schema_name: str
    data: dict[str, Any]
    artifact_id: str
    source_tile_id: str
    created_at: float = field(default_factory=time.time)
    lineage: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert artifact to dictionary."""
        return {
            "schema_name": self.schema_name,
            "data": self.data,
            "artifact_id": self.artifact_id,
            "source_tile_id": self.source_tile_id,
            "created_at": self.created_at,
            "lineage": self.lineage
        }

    def add_lineage(self, tile_id: str) -> None:
        """Add a tile ID to the lineage chain."""
        if tile_id not in self.lineage:
            self.lineage.append(tile_id)


class TileRefiner:
    """Refines raw tiles into structured artifacts."""

    def __init__(self):
        self._schemas: dict[str, ArtifactSchema] = {}
        self._artifacts: dict[str, Artifact] = {}
        self._tiles_processed: int = 0

    def register_schema(self, schema: ArtifactSchema) -> None:
        """Register a schema for artifact generation."""
        self._schemas[schema.name] = schema

    def refine_tile(self, tile: Tile, schema_name: str) -> Optional[Artifact]:
        """Refine a raw tile into a structured artifact."""
        if schema_name not in self._schemas:
            raise ValueError(f"Unknown schema: {schema_name}")

        schema = self._schemas[schema_name]

        # Extract and structure data
        refined_data = self._extract_structured_data(tile, schema)

        # Validate against schema
        if not schema.validate(refined_data):
            return None

        # Create artifact
        artifact_id = hashlib.sha256(
            f"{tile.tile_id}:{schema_name}:{json.dumps(refined_data, sort_keys=True)}".encode()
        ).hexdigest()[:16]

        artifact = Artifact(
            schema_name=schema_name,
            data=refined_data,
            artifact_id=artifact_id,
            source_tile_id=tile.tile_id,
            lineage=[tile.tile_id]
        )

        self._artifacts[artifact_id] = artifact
        self._tiles_processed += 1

        return artifact

    def _extract_structured_data(self, tile: Tile, schema: ArtifactSchema) -> dict[str, Any]:
        """Extract structured data from raw tile based on schema."""
        result = {}

        # Extract fields that exist in raw data
        for field_name, field_type in schema.fields.items():
            if field_name in tile.raw_data:
                result[field_name] = tile.raw_data[field_name]

        # Add metadata fields
        result["_source"] = tile.source
        result["_tile_type"] = tile.tile_type
        result["_timestamp"] = tile.timestamp

        return result

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Retrieve an artifact by ID."""
        return self._artifacts.get(artifact_id)

    def get_artifacts_by_schema(self, schema_name: str) -> list[Artifact]:
        """Get all artifacts of a specific schema type."""
        return [
            a for a in self._artifacts.values()
            if a.schema_name == schema_name
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get refiner statistics."""
        return {
            "tiles_processed": self._tiles_processed,
            "artifacts_created": len(self._artifacts),
            "schemas_registered": len(self._schemas)
        }

    def refine_batch(self, tiles: list[Tile], schema_name: str) -> list[Artifact]:
        """Refine multiple tiles in batch."""
        artifacts = []
        for tile in tiles:
            artifact = self.refine_tile(tile, schema_name)
            if artifact:
                artifacts.append(artifact)
        return artifacts
