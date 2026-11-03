"""Registry-facing layers: what a reference means, and how its metadata is retrieved.

This is where npm's shape lives — `@scope/name@version`, dist-tags, the registry host —
so that nothing above it has to know which registry a package came from. The models these
functions return come from `schemas/` for exactly that reason: a PyPI fetcher added later
belongs in a sibling package, while `analyzer/` keeps reading the same `PackageRef`.

Two halves, one boundary. Parsing is pure text work and opens nothing; the two clients in
`npm_fetcher.py` and `npm_download.py` are the only code in PackProof that asks the registry
for anything, and they ask only when a caller calls them. Importing this package reads no
environment, opens no socket and touches no filesystem — which is what lets `analyzer/` and
the CLI import a reference parser without acquiring a network dependency they cannot mock
away.

Metadata and bytes are separated on purpose. `npm_fetcher.py` records `dist.tarball` as *text
about* an artifact and never requests it; `npm_download.py` is the only module that asks for
those bytes, and it asks for a target `download_policy.py` has already approved — one host,
HTTPS, no redirect, no environment proxy, and never over the configured size ceiling. It
writes the verified tarball into a workspace and stops there: nothing in this package
unpacks an archive, runs a lifecycle script, or imports a byte of what it downloaded.
"""

from packproof.registry.artifact_store import ArtifactStore, Finalization
from packproof.registry.errors import (
    ArtifactCollisionError,
    ArtifactError,
    ArtifactIntegrityError,
    ArtifactInternalError,
    ArtifactSettlementError,
    ArtifactStreamTruncatedError,
    ArtifactTooLargeError,
    ArtifactWriteError,
    PackageNotFoundError,
    RegistryError,
    RegistryHTTPError,
    RegistryInvalidResponseError,
    RegistryMetadataError,
    RegistryNetworkError,
    RegistryNotJSONError,
    RegistryRedirectError,
    RegistryResponseTooLargeError,
    RegistryTimeoutError,
    UnexpectedDownloadOriginError,
    UnsupportedDownloadTargetError,
    VersionNotFoundError,
    WorkspaceDurabilityError,
    WorkspaceEntryInvalidError,
    WorkspaceOutOfSpaceError,
)
from packproof.registry.npm import InvalidPackageReferenceError, parse_npm_reference
from packproof.registry.npm_download import NpmArtifactClient
from packproof.registry.npm_fetcher import NPM_REGISTRY_HOST, NpmRegistryClient, npm_package_url

__all__ = [
    "NPM_REGISTRY_HOST",
    "ArtifactCollisionError",
    "ArtifactError",
    "ArtifactIntegrityError",
    "ArtifactInternalError",
    "ArtifactSettlementError",
    "ArtifactStore",
    "ArtifactStreamTruncatedError",
    "ArtifactTooLargeError",
    "ArtifactWriteError",
    "Finalization",
    "InvalidPackageReferenceError",
    "NpmArtifactClient",
    "NpmRegistryClient",
    "PackageNotFoundError",
    "RegistryError",
    "RegistryHTTPError",
    "RegistryInvalidResponseError",
    "RegistryMetadataError",
    "RegistryNetworkError",
    "RegistryNotJSONError",
    "RegistryRedirectError",
    "RegistryResponseTooLargeError",
    "RegistryTimeoutError",
    "UnexpectedDownloadOriginError",
    "UnsupportedDownloadTargetError",
    "VersionNotFoundError",
    "WorkspaceDurabilityError",
    "WorkspaceEntryInvalidError",
    "WorkspaceOutOfSpaceError",
    "npm_package_url",
    "parse_npm_reference",
]
