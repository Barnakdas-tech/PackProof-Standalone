"""Validated models shared across layers.

The boundary every external value crosses on its way into the engine: registry JSON,
`package.json` fields, command-line references, and the record of a bytes-on-disk download.
A type here has already been vetted, so layers above this one read fields instead of
re-checking strings, and layers below it never see a PackProof-specific type.

Import direction matters. `registry/`, including its metadata and artifact clients, may
import from here; nothing in this package may import from them, which is what lets
`analyzer/` reason about a package without knowing which registry it came from.
"""

from packproof.schemas.artifacts import ArtifactVerification, PackageArtifact
from packproof.schemas.models import (
    PackageDistribution,
    PackageRef,
    PackageRepository,
    PackageVersionMetadata,
)

__all__ = [
    "ArtifactVerification",
    "PackageArtifact",
    "PackageDistribution",
    "PackageRef",
    "PackageRepository",
    "PackageVersionMetadata",
]
