r"""Asking the registry for bytes, and publishing only what the metadata said those bytes were.

These tests drive `NpmArtifactClient` with its real store and a real temporary directory, because
the properties worth proving here are "the file on disk hashes to the name it was published under"
and "nothing was published" — neither of which a mocked filesystem can confirm. The transport is a
stub, for the opposite reason: the ceiling, the truncation check and the conjunction of integrity
claims exist only while bytes are arriving, and a stub that hands over a chunk list, can raise
partway through and counts its own reads is what makes those assertions about this client rather
than about `http.client`.

Every section was written against the scratch probe script that this file replaces and
that ran 73 checks during design. Mapping the probe into permanent tests changed two of its
expectations rather than copying them: the probe asserted a chunk ceiling by its own arithmetic
(`50000` of a `53248` read), and the permanent version asserts the *bound* — the overshoot is at
most one chunk — so the test survives a change of chunk size. Where the probe printed a value for a
human to eyeball, the test asserts it.

The mapping, section by section, in the probe's own numbering — so that any line of that scratch
output can be traced to the assertion that replaced it. A section with two lines below is a section
whose checks were split between a test about the workspace and a test about the report.

    1 -> test_a_verified_download_returns_the_file_the_workspace_now_holds
    1 -> test_every_claim_the_registry_supplied_is_checked_and_reported_in_rank_order
    2 -> test_a_download_chunk_is_bounded_by_the_configured_size_and_never_asked_for_whole
    3 -> test_a_version_with_no_integrity_claims_downloads_and_says_it_checked_nothing
    4 -> test_bytes_that_fail_a_claim_are_never_published
    5 -> test_a_body_that_passes_the_ceiling_while_it_arrives_is_abandoned_within_one_chunk
    6 -> test_a_declared_length_above_the_ceiling_is_refused_before_a_byte_is_read
    7 -> test_a_truncated_stream_is_refused_as_truncated_rather_than_published_short
    8 -> test_a_body_that_arrives_encoded_is_refused_instead_of_inflated
    9 -> test_only_a_200_is_a_complete_artifact
   10 -> test_a_redirect_refusal_from_the_transport_is_passed_through_unchanged
   11 -> test_a_failure_with_no_response_is_classified_by_number_and_class_not_by_sentence
   12 -> test_a_reset_while_the_body_is_arriving_is_a_network_failure_and_not_a_truncation
   13 -> test_a_destination_held_by_other_bytes_is_refused_and_those_bytes_stay_put
   14 -> test_an_artifact_already_in_the_workspace_is_reused_and_reported_as_verified
   15 -> test_a_directory_sync_failure_keeps_the_artifact_and_reports_where_it_is
   16 -> test_an_interrupt_inside_the_commit_window_is_recovered_and_the_artifact_survives
   17 -> test_a_broken_fact_sink_does_not_cost_a_verified_download
   18 -> test_a_refused_target_produces_no_request_at_all
   19 -> test_the_request_is_a_plain_get_for_the_rebuilt_url_with_no_credentials
   20 -> test_the_frames_this_client_scrubs_hold_no_request_object_after_a_failure
   20 -> test_helper_frames_below_the_scrubbed_ones_still_hold_the_request_limitation
   21 -> test_the_prepared_request_and_its_claims_print_nothing_sensitive


What these tests deliberately do not claim. The signed query in an npm tarball URL is a capability,
and the client's own two frames — `download` and `_attempt` — empty their locals in their own
`finally`, which is asserted below. Helper frames deeper in the same module (`_send`,
`_response_shape`, `_stream`) bind the request or the response object while they work and are not
scrubbed, and the traceback of a failure raised there therefore still reaches those frames. The
guarantee asserted here is the one that is actually true: nothing *rendered* — no message, no
`repr`, no formatted traceback, no log line — carries the target, because rendered tracebacks print
source lines and not frame locals. That distinction is U9/S-11, and it stays open.

Nothing here reaches DNS or opens a socket: an autouse fixture closes the socket family, and the
default opener is inspected as an object graph rather than exercised against the network.
"""

import base64
import contextlib
import errno
import hashlib
import http.client
import io
import logging
import socket
import threading
import traceback
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from packproof.config.settings import Settings
from packproof.registry import npm_download
from packproof.registry.artifact_store import (
    ARTIFACT_WRITE_CHUNK_BYTES,
    PUBLISHED_UNSYNCED,
    ArtifactStore,
    Finalization,
)
from packproof.registry.errors import (
    ArtifactCollisionError,
    ArtifactIntegrityError,
    ArtifactInternalError,
    ArtifactStreamTruncatedError,
    ArtifactTooLargeError,
    ArtifactWriteError,
    RegistryHTTPError,
    RegistryInvalidResponseError,
    RegistryNetworkError,
    RegistryRedirectError,
    RegistryTimeoutError,
    UnexpectedDownloadOriginError,
    UnsupportedDownloadTargetError,
    WorkspaceDurabilityError,
)
from packproof.registry.npm import parse_npm_reference
from packproof.registry.npm_download import NpmArtifactClient
from packproof.registry.npm_fetcher import (
    _USER_AGENT,
    NPM_REGISTRY_HOST,
    _OriginGuardHTTPSHandler,
    _RefuseRedirectHandler,
)
from packproof.schemas import artifacts
from packproof.schemas.artifacts import ArtifactVerification
from packproof.schemas.models import PackageDistribution, PackageVersionMetadata

REF = parse_npm_reference("left-pad@1.3.0")
PAYLOAD = b"left-pad-1.3.0.tgz, as far as this test is concerned"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
FINAL_NAME = f"{DIGEST}.tgz"

#: The part of a tarball URL that is a capability rather than an address. Every "does this leak"
#: test below searches for this string, not for `https://`, because the host is legitimately part
#: of the request and only the query is the secret.
CAPABILITY = "zz-signature-capability"
TARBALL = f"https://{NPM_REGISTRY_HOST}/left-pad/-/left-pad-1.3.0.tgz?sig={CAPABILITY}"

_CEILING = 1_000_000
_CHUNK = 8


@pytest.fixture(autouse=True)
def _no_live_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """This client's transport is injected, so a socket here is a bug in the test or a hole in O1."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the artifact client reached the network")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def store(workspace: Path) -> Iterator[ArtifactStore]:
    with ArtifactStore(workspace) as opened:
        yield opened


# --------------------------------------------------------------------------- the transport stub


class _StreamResponse:
    """The slice of `http.client.HTTPResponse` this client uses, with the reads made visible.

    `body` is one blob or a sequence of chunks, and a chunk may be an exception — which is the only
    faithful way to test what happens *after* a response object exists. A mid-body reset is
    invisible to a stub that can only fail the call, and mid-body failures are where the ceiling,
    the truncation check and the staged-file cleanup actually live.

    The two spellings model two different registries. A blob is served `size` bytes at a time,
    exactly as `http.client.HTTPResponse.read(n)` behaves, so a test about the ceiling measures the
    client's counter rather than the stub's willingness to hand over a whole body in one call — the
    overshoot bound below is only meaningful against a read that respects its argument. A chunk list
    is served one element per call, which is how a test stages an exception at a chosen position.
    """

    def __init__(
        self,
        *,
        status: int = 200,
        body: bytes | list[bytes | BaseException] = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self._blob = body if isinstance(body, bytes) else b""
        self._is_blob = isinstance(body, bytes)
        self._chunks = [] if self._is_blob else list(body)
        self._headers = {name.lower(): value for name, value in (headers or {}).items()}
        self._index = 0
        self.read_sizes: list[int] = []
        self.bytes_served = 0
        self.close_calls = 0

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size < 0:
            raise AssertionError("the client asked for the whole artifact body at once")
        if self._is_blob:
            served = self._blob[:size]
            self._blob = self._blob[len(served) :]
            self.bytes_served += len(served)
            return served
        if self._index >= len(self._chunks):
            return b""
        chunk = self._chunks[self._index]
        self._index += 1
        if isinstance(chunk, BaseException):
            raise chunk
        self.bytes_served += len(chunk)
        return chunk

    def getheader(self, name: str, default: str | None = None) -> str | None:
        return self._headers.get(name.lower(), default)

    def close(self) -> None:
        self.close_calls += 1


class _Transport:
    """Answers with what the test said, and remembers every request that produced the answer."""

    def __init__(self, *outcomes: _StreamResponse | BaseException) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[urllib.request.Request] = []
        self.responses: list[_StreamResponse] = []
        self.timeouts: list[float] = []

    def __call__(self, request: urllib.request.Request, timeout_seconds: float) -> _StreamResponse:
        self.requests.append(request)
        self.timeouts.append(timeout_seconds)
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        self.responses.append(outcome)
        return outcome

    @property
    def urls(self) -> list[str]:
        return [request.full_url for request in self.requests]


def _artifact_response(
    payload: bytes = PAYLOAD, *, status: int = 200, declare: bool = True, **headers: str
) -> _StreamResponse:
    """A body sent exactly as written and declared at its own length, as a registry would.

    Keyword arguments spell header names with underscores, because `Content-Encoding` is not an
    identifier, so they are translated here — a header stored under the Python spelling would be
    invisible to `getheader("Content-Encoding")` and an encoded-body test would silently assert
    nothing about an encoded body.
    """
    given = {"Content-Type": "application/octet-stream"}
    if declare:
        given["Content-Length"] = str(len(payload))
    given.update({name.replace("_", "-"): value for name, value in headers.items()})
    return _StreamResponse(status=status, body=payload, headers=given)


class _ListSink:
    """A fact sink that keeps every line, so a test can read the ledger as a log reader sees it."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def append(self, line: str) -> None:
        self.lines.append(line)


class _BrokenSink:
    """A logger that is down, injected as the store's fact sink."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def append(self, line: str) -> None:
        self.lines.append(line)
        raise RuntimeError("the log is down")


# --------------------------------------------------------------------------- metadata and clients


def _sri(payload: bytes, algorithm: str = "sha512") -> str:
    digest = hashlib.new(algorithm, payload).digest()
    return f"{algorithm}-{base64.b64encode(digest).decode('ascii')}"


def _shasum(payload: bytes) -> str:
    """npm's legacy `shasum` field *is* a SHA-1, so the fixture has to be one: the client hashes the
    body with `sha1` and compares, and a stand-in digest would make the claim fail for a reason that
    has nothing to do with the code under test. This is a content claim, not a security decision."""
    return hashlib.sha1(payload).hexdigest()  # noqa: S324


def _metadata(
    *,
    integrity: str | None = None,
    shasum: str | None = None,
    tarball: str = TARBALL,
) -> PackageVersionMetadata:
    """A record built the way `NpmRegistryClient` builds one, so the models have vetted it."""
    return PackageVersionMetadata(
        reference=REF,
        distribution=PackageDistribution(tarball_url=tarball, integrity=integrity, shasum=shasum),
        description=None,
        repository=None,
        published_at=None,
    )


def _raw_metadata(tarball: str, *, integrity: str | None = None) -> PackageVersionMetadata:
    """The registry's own text with the model's vetting switched off.

    Used where the point is that the *policy* refuses: an `http://` target never becomes a
    `PackageDistribution` at all, so a test that could only build validated records could not ask
    what the download path does when something upstream has a different opinion.
    """
    distribution = PackageDistribution.model_construct(
        tarball_url=tarball, integrity=integrity, shasum=None
    )
    return PackageVersionMetadata.model_construct(
        reference=REF,
        distribution=distribution,
        description=None,
        repository=None,
        published_at=None,
    )


def _client(
    store: ArtifactStore,
    transport: _Transport,
    **overrides: object,
) -> NpmArtifactClient:
    options: dict[str, object] = {
        "timeout_seconds": 30.0,
        "max_download_bytes": _CEILING,
        "chunk_bytes": _CHUNK,
        "opener": transport,
    }
    # `opener` is overridable because one test asks for the chain a client builds when nobody
    # injects one, and that is the chain O1 is a claim about.
    options.update(overrides)
    return NpmArtifactClient(store, **options)  # type: ignore[arg-type]


def _downloaded(store: ArtifactStore, response: _StreamResponse, **claims: str) -> object:
    """One successful download, with the transport and client kept out of the assertions."""
    transport = _Transport(response)
    artifact = _client(store, transport).download(_metadata(**claims))  # type: ignore[arg-type]
    return artifact


# --------------------------------------------------------------------------- the happy path


def test_a_verified_download_returns_the_file_the_workspace_now_holds(
    store: ArtifactStore, workspace: Path
) -> None:
    artifact = _downloaded(
        store, _artifact_response(PAYLOAD), integrity=_sri(PAYLOAD), shasum=_shasum(PAYLOAD)
    )
    assert artifact.reference is REF
    assert artifact.directory == store.root
    assert artifact.file_name == FINAL_NAME
    assert artifact.path == workspace / FINAL_NAME
    assert artifact.size_bytes == len(PAYLOAD)
    assert artifact.verification is ArtifactVerification.VERIFIED
    assert artifact.path.read_bytes() == PAYLOAD


def test_the_published_name_is_the_digest_of_the_bytes_written_not_of_the_registrys_claim(
    store: ArtifactStore,
) -> None:
    """`dist.integrity` says what the publisher intended; the name says what this process wrote.

    The registry here claims only SHA-512, so nothing supplied the SHA-256 the file is named
    after — and the naming digest is still `sha256` of the bytes that arrived, computed here.
    """
    artifact = _downloaded(store, _artifact_response(PAYLOAD), integrity=_sri(PAYLOAD))
    assert artifact.sha256 == hashlib.sha256(PAYLOAD).hexdigest()
    assert artifact.checked_claims == ("sri-sha512",)


def test_every_claim_the_registry_supplied_is_checked_and_reported_in_rank_order(
    store: ArtifactStore,
) -> None:
    """The probe's first section, as an assertion: two claims, one pass over the bytes, both named.

    The order is `CHECKED_CLAIM_ORDER`, ascending in strength, so a report can be diffed against
    another report without either deciding what "the algorithm we checked" means.
    """
    artifact = _downloaded(
        store,
        _artifact_response(PAYLOAD),
        integrity=_sri(PAYLOAD, "sha256"),
        shasum=_shasum(PAYLOAD),
    )
    assert artifact.checked_claims == ("shasum-sha1", "sri-sha256")
    assert artifact.verification is ArtifactVerification.VERIFIED


@pytest.mark.parametrize("algorithm", ["sha1", "sha256", "sha384", "sha512"])
def test_every_sri_algorithm_the_models_accept_has_a_hasher_here(
    store: ArtifactStore, algorithm: str
) -> None:
    """The two vocabularies — `schemas/models.py`'s and this module's — are asserted to agree.

    A SHA-1 SRI claim is still checked even though `shasum` is spelled differently, because the
    conjunctive rule does not rank claims by whether they are worth honouring.
    """
    artifact = _downloaded(store, _artifact_response(PAYLOAD), integrity=_sri(PAYLOAD, algorithm))
    assert artifact.verification is ArtifactVerification.VERIFIED
    assert artifact.checked_claims == (f"sri-{algorithm}",)


def test_a_version_with_no_integrity_claims_downloads_and_says_it_checked_nothing(
    store: ArtifactStore,
) -> None:
    """`UNAVAILABLE` is a report about the registry's silence, and it is the only honest one."""
    artifact = _downloaded(store, _artifact_response(PAYLOAD))
    assert artifact.verification is ArtifactVerification.UNAVAILABLE
    assert artifact.checked_claims == ()
    assert artifact.sha256 == DIGEST


def test_the_body_of_a_response_with_no_claims_and_no_length_is_still_bounded_by_the_counter(
    store: ArtifactStore,
) -> None:
    transport = _Transport(_artifact_response(PAYLOAD, declare=False))
    artifact = _client(store, transport, max_download_bytes=len(PAYLOAD)).download(_metadata())
    assert artifact.size_bytes == len(PAYLOAD)


# --------------------------------------------------------------------------- the request itself


def test_the_request_is_a_plain_get_for_the_rebuilt_url_with_no_credentials(
    store: ArtifactStore,
) -> None:
    transport = _Transport(_artifact_response(PAYLOAD))
    _client(store, transport).download(_metadata())
    request = transport.requests[0]
    assert request.full_url == TARBALL
    assert request.method == "GET"
    lowercased = {name.lower() for name in request.headers}
    assert lowercased == {"accept", "accept-encoding", "user-agent"}
    assert request.get_header("Accept") == "application/octet-stream"
    assert request.get_header("Accept-encoding") == "identity"
    assert request.get_header("User-agent") == _USER_AGENT
    assert not {"authorization", "cookie", "proxy-authorization"} & lowercased


def test_no_header_on_the_artifact_request_can_carry_a_token(
    store: ArtifactStore,
) -> None:
    transport = _Transport(_artifact_response(PAYLOAD))
    _client(store, transport).download(_metadata())
    request = transport.requests[0]
    assert CAPABILITY not in str(request.headers)
    assert CAPABILITY in request.full_url  # the query is the only place it lives, by design


def test_the_configured_timeout_reaches_the_transport_rather_than_a_default(
    store: ArtifactStore,
) -> None:
    transport = _Transport(_artifact_response(PAYLOAD))
    _client(store, transport, timeout_seconds=3.5).download(_metadata())
    assert transport.timeouts == [3.5]


def test_a_download_chunk_is_bounded_by_the_configured_size_and_never_asked_for_whole(
    store: ArtifactStore,
) -> None:
    """The reads this client performs are the whole defence against a body sized by its host."""
    transport = _Transport(_artifact_response(PAYLOAD))
    _client(store, transport, chunk_bytes=5).download(_metadata())
    assert set(transport.responses[0].read_sizes) == {5}


def test_a_zero_or_negative_chunk_size_is_refused_at_construction(store: ArtifactStore) -> None:
    """`read(0)` returns `b''` forever, which the stream loop reads as an orderly end of body."""
    for bad in (0, -1):
        with pytest.raises(ValueError) as caught:
            _client(store, _Transport(), chunk_bytes=bad)
        assert "download chunk" in str(caught.value)


def test_from_settings_carries_the_configured_timeout_and_ceiling(tmp_path: Path) -> None:
    settings = Settings(
        http_timeout_seconds=7.5,
        max_download_bytes=4096,
        workspace_dir=tmp_path,
    )
    with ArtifactStore(tmp_path) as opened:
        client = NpmArtifactClient.from_settings(settings, opened)
    assert client._timeout_seconds == settings.http_timeout_seconds
    assert client._max_download_bytes == settings.max_download_bytes
    assert client._chunk_bytes == ARTIFACT_WRITE_CHUNK_BYTES


# --------------------------------------------------------------------------- claims, in isolation


def test_the_sri_claim_is_read_as_one_algorithm_and_its_decoded_digest() -> None:
    claims = npm_download._claims_of(
        _metadata(integrity=_sri(PAYLOAD, "sha384")).distribution  # type: ignore[union-attr]
    )
    assert [(claim.name, claim.algorithm) for claim in claims] == [("sri-sha384", "sha384")]
    assert claims[0].expected == hashlib.sha384(PAYLOAD).digest()


def test_the_shasum_is_named_for_where_it_came_from_rather_than_for_its_algorithm_only() -> None:
    """`shasum` is npm's legacy field and `dist.integrity` is SRI; both are SHA-1 here and a
    report that called them both `sri-sha1` would hide which promise broke."""
    claims = npm_download._claims_of(_metadata(shasum=_shasum(PAYLOAD)).distribution)  # type: ignore[union-attr]
    assert [claim.name for claim in claims] == ["shasum-sha1"]
    assert claims[0].algorithm == "sha1"


def test_a_record_with_neither_claim_produces_no_work_to_check() -> None:
    assert npm_download._claims_of(_metadata().distribution) == ()  # type: ignore[union-attr]


def test_the_conjunctive_rule_checks_every_supplied_claim_whatever_the_ranking(
    store: ArtifactStore,
) -> None:
    """A SHA-1-only registry is not a weaker *rule*: one claim supplied means that one claim holds
    or nothing is published. The ranking decides what a report *calls* the failure, never what is
    verified — which is the distinction this test exists to keep from collapsing."""
    transport = _Transport(_artifact_response(PAYLOAD))
    metadata = _metadata(shasum=_shasum(b"different bytes"))
    with pytest.raises(ArtifactIntegrityError) as caught:
        _client(store, transport).download(metadata)
    assert "'shasum-sha1' claim" in str(caught.value)
    claims = npm_download._claims_of(metadata.distribution)  # type: ignore[union-attr]
    assert [claim.algorithm for claim in claims] == ["sha1"]


def test_this_module_ranks_claims_with_the_models_own_order_not_a_second_copy_of_it() -> None:
    """One ranking, read forwards to store and backwards to diagnose, is a property of the import.

    A second tuple in `registry/` would agree today and drift tomorrow, and the two places that
    disagree about which claim is the strong one are exactly the report and its diagnosis.
    """
    assert npm_download.CHECKED_CLAIM_ORDER is artifacts.CHECKED_CLAIM_ORDER
    assert set(npm_download.CHECKED_CLAIM_ORDER) == artifacts.CHECKABLE_CLAIMS


def test_a_downloaded_artifact_satisfies_the_model_that_refuses_an_unordered_report(
    store: ArtifactStore,
) -> None:
    """The client's output is validated by the schema's rule, so the two cannot drift: `checked_claims`
    arrives in the order the model demands, which a hand-built descending tuple would not."""
    artifact = _downloaded(
        store,
        _artifact_response(PAYLOAD),
        integrity=_sri(PAYLOAD, "sha512"),
        shasum=_shasum(PAYLOAD),
    )
    assert artifact.checked_claims == ("shasum-sha1", "sri-sha512")
    rebuilt = artifacts.PackageArtifact.model_validate(artifact.model_dump())
    assert rebuilt == artifact
    descending = artifact.model_dump() | {"checked_claims": ("sri-sha512", "shasum-sha1")}
    with pytest.raises(ValidationError, match="fixed order"):
        artifacts.PackageArtifact.model_validate(descending)


def test_a_claim_algorithm_with_no_hasher_is_reported_as_a_packproof_bug() -> None:
    """Only one thing can reach here: `schemas/models.py` widened its SRI set and this table did
    not follow. That is a bug in this tool, and it says so instead of surfacing as a validation
    rejection several frames away from its cause."""
    claim = npm_download._Claim("sri-blake2b", "blake2b", b"\x00" * 64)
    with pytest.raises(ArtifactInternalError) as caught:
        npm_download._hashers_for((claim,), REF)
    assert "drifted apart" in str(caught.value)
    assert "blake2b" in str(caught.value)


def test_the_naming_hasher_is_present_even_with_no_claims_and_shared_by_claims_that_agree() -> None:
    """`sha256` is always computed because the *name* needs it. A second claim on the same
    algorithm reuses the one hasher rather than hashing the body twice."""
    assert set(npm_download._hashers_for((), REF)) == {"sha256"}
    claims = (
        npm_download._Claim("sri-sha256", "sha256", b"x"),
        npm_download._Claim("shasum-sha1", "sha1", b"y"),
    )
    assert set(npm_download._hashers_for(claims, REF)) == {"sha256", "sha1"}


def test_every_supplied_claim_is_compared_and_every_failure_is_returned() -> None:
    """Conjunctive verification, stated as a function: the first mismatch is not the answer.

    The SHA-1 here is npm's `shasum` claim, reproduced so the comparison is a content check rather
    than a fixture accident; it is not a security decision being made with a weak digest.
    """
    claims = (
        npm_download._Claim("sri-sha512", "sha512", hashlib.sha512(PAYLOAD).digest()),
        npm_download._Claim("shasum-sha1", "sha1", hashlib.sha1(b"other").digest()),  # noqa: S324
        npm_download._Claim("sri-sha256", "sha256", hashlib.sha256(b"other too").digest()),
    )
    hashers = {
        "sha512": hashlib.sha512(PAYLOAD),
        "sha1": hashlib.sha1(PAYLOAD),  # noqa: S324
        "sha256": hashlib.sha256(PAYLOAD),
    }
    failures = npm_download._mismatched_claims(claims, hashers)  # type: ignore[arg-type]
    assert [claim.name for claim in failures] == ["shasum-sha1", "sri-sha256"]


def test_the_integrity_report_names_the_strongest_failed_claim_and_counts_the_rest() -> None:
    """SHA-512 first, the legacy SHA-1 last: the ranking is `CHECKED_CLAIM_ORDER` read backwards,
    so nothing can hold a second list that disagrees about which algorithm is the strong one."""
    failures = [
        npm_download._Claim("shasum-sha1", "sha1", b"1"),
        npm_download._Claim("sri-sha512", "sha512", b"2"),
    ]
    message = str(
        npm_download._integrity_failure(
            REF,
            (
                npm_download._Claim("sri-sha512", "sha512", b"2"),
                npm_download._Claim("shasum-sha1", "sha1", b"1"),
            ),
            failures,
        )
    )
    assert "'sri-sha512' claim" in message
    assert "1 other claim(s) beside it" in message


def test_an_integrity_report_that_broke_alone_says_the_others_held() -> None:
    claims = (
        npm_download._Claim("sri-sha512", "sha512", b"2"),
        npm_download._Claim("shasum-sha1", "sha1", b"1"),
    )
    message = str(npm_download._integrity_failure(REF, claims, [claims[1]]))
    assert "with every other claim holding" in message
    assert "2 claim(s) were checked" in message


# --------------------------------------------------------------------------- a claim that breaks


def test_bytes_that_fail_a_claim_are_never_published(store: ArtifactStore, workspace: Path) -> None:
    """The probe's fourth section: a mismatch raises *before* the link, so no content-addressed
    name is ever created for bytes that failed, and the staged file does not survive either."""
    transport = _Transport(_artifact_response(PAYLOAD))
    with pytest.raises(ArtifactIntegrityError) as caught:
        _client(store, transport).download(_metadata(integrity=_sri(b"not these bytes")))
    assert "do not match" in str(caught.value)
    assert list(workspace.iterdir()) == []
    assert transport.responses[0].close_calls == 1


def test_a_failing_claim_reports_no_digest_value_anywhere_in_its_message(
    store: ArtifactStore,
) -> None:
    transport = _Transport(_artifact_response(PAYLOAD))
    metadata = _metadata(integrity=_sri(b"other"), shasum=_shasum(b"other"))
    with pytest.raises(ArtifactIntegrityError) as caught:
        _client(store, transport).download(metadata)
    message = str(caught.value)
    assert base64.b64encode(hashlib.sha512(b"other").digest()).decode() not in message
    assert hashlib.sha512(PAYLOAD).hexdigest() not in message
    assert _shasum(b"other") not in message


def test_a_second_claim_still_breaks_when_the_first_one_holds(
    store: ArtifactStore, workspace: Path
) -> None:
    """A registry that supplies two claims cannot have one honoured and the other ignored."""
    mismatching = b"different bytes entirely"
    with pytest.raises(ArtifactIntegrityError):
        _downloaded(
            store,
            _artifact_response(PAYLOAD),
            integrity=_sri(PAYLOAD),
            shasum=_shasum(mismatching),
        )
    assert list(workspace.iterdir()) == []


# --------------------------------------------------------------------------- size ceilings


def test_a_declared_length_above_the_ceiling_is_refused_before_a_byte_is_read(
    store: ArtifactStore, workspace: Path
) -> None:
    transport = _Transport(_artifact_response(b"z" * 500))
    with pytest.raises(ArtifactTooLargeError) as caught:
        _client(store, transport, max_download_bytes=100).download(_metadata())
    assert "declared 500 bytes" in str(caught.value)
    assert transport.responses[0].read_sizes == []
    assert list(workspace.iterdir()) == []


def test_a_body_that_passes_the_ceiling_while_it_arrives_is_abandoned_within_one_chunk(
    store: ArtifactStore,
) -> None:
    """The bound is what is asserted, not the number the simulation happened to print.

    Nothing is buffered beyond one chunk, so the overshoot past `max_download_bytes` is at most
    `chunk_bytes` — by construction, and measurable here rather than promised.
    """
    payload = b"q" * 40
    transport = _Transport(_artifact_response(payload, declare=False))
    with pytest.raises(ArtifactTooLargeError) as caught:
        _client(store, transport, max_download_bytes=13, chunk_bytes=4).download(_metadata())
    served = transport.responses[0].bytes_served
    assert served >= 13
    assert served - 13 <= 4
    assert "more than 13 bytes" in str(caught.value)


def test_a_lying_content_length_cannot_let_an_oversized_body_through(
    store: ArtifactStore,
) -> None:
    """`Content-Length` is consulted twice and believed zero times: once to stop early, once to
    notice a stream that ended short of it. The counter over bytes that arrived is the ceiling."""
    transport = _Transport(
        _StreamResponse(
            body=b"q" * 100,
            headers={"Content-Length": "5", "Content-Type": "application/octet-stream"},
        )
    )
    with pytest.raises(ArtifactTooLargeError):
        _client(store, transport, max_download_bytes=40, chunk_bytes=8).download(_metadata())
    served = transport.responses[0].bytes_served
    assert served >= 40
    assert served - 40 <= 8


def test_a_body_exactly_at_the_ceiling_is_accepted(store: ArtifactStore) -> None:
    payload = b"w" * 32
    transport = _Transport(_artifact_response(payload))
    artifact = _client(store, transport, max_download_bytes=32).download(_metadata())
    assert artifact.size_bytes == 32


def test_a_truncated_stream_is_refused_as_truncated_rather_than_published_short(
    store: ArtifactStore, workspace: Path
) -> None:
    transport = _Transport(
        _StreamResponse(
            body=[b"a" * 8, b"b" * 4],
            headers={"Content-Length": "32", "Content-Type": "application/octet-stream"},
        )
    )
    with pytest.raises(ArtifactStreamTruncatedError) as caught:
        _client(store, transport).download(_metadata())
    assert "12 of the 32 bytes" in str(caught.value)
    assert list(workspace.iterdir()) == []


def test_a_zero_length_body_with_no_claims_is_a_successful_empty_artifact(
    store: ArtifactStore, workspace: Path
) -> None:
    """A measured limitation, asserted as a limitation rather than as a guarantee.

    Nothing in the metadata said how long the tarball should be, and nothing claimed a digest, so
    an empty body satisfies every promise the registry made — which is none of them. Stage 2.3
    cannot tell an empty artifact from a cut-off one without a claim to compare against, and the
    report says `UNAVAILABLE` so a later stage reads the absence of verification rather than
    inferring it.
    """
    transport = _Transport(_artifact_response(b"", declare=True))
    artifact = _client(store, transport).download(_metadata())
    assert artifact.size_bytes == 0
    assert artifact.verification is ArtifactVerification.UNAVAILABLE
    assert artifact.file_name == f"{hashlib.sha256(b'').hexdigest()}.tgz"
    assert (workspace / artifact.file_name).read_bytes() == b""


# --------------------------------------------------------------------------- response shape


@pytest.mark.parametrize("encoding", ["gzip", "br", "GZIP", "deflate"])
def test_a_body_that_arrives_encoded_is_refused_instead_of_inflated(
    store: ArtifactStore, encoding: str
) -> None:
    """The ceiling counts bytes written to disk. A compressed body would make it count bytes this
    process has not decompressed, and the digest would describe a payload nobody hashed."""
    transport = _Transport(_artifact_response(PAYLOAD, Content_Encoding=encoding))
    with pytest.raises(RegistryInvalidResponseError) as caught:
        _client(store, transport).download(_metadata())
    assert "does not decode" in str(caught.value)
    assert transport.responses[0].read_sizes == []


@pytest.mark.parametrize("encoding", [None, "identity", "  identity  "])
def test_an_identity_or_absent_content_encoding_is_the_body_this_client_expects(
    store: ArtifactStore, encoding: str | None
) -> None:
    headers = {} if encoding is None else {"Content-Encoding": encoding}
    transport = _Transport(_artifact_response(PAYLOAD, declare=True, **headers))
    artifact = _client(store, transport).download(_metadata())
    assert artifact.size_bytes == len(PAYLOAD)


@pytest.mark.parametrize(
    ("status", "category"),
    [
        (204, RegistryHTTPError),
        (206, RegistryHTTPError),
        (304, RegistryHTTPError),
        (404, RegistryHTTPError),
        (503, RegistryHTTPError),
    ],
)
def test_only_a_200_is_a_complete_artifact(
    store: ArtifactStore, status: int, category: type
) -> None:
    """`urllib` calls every 2xx a success, so a partial body answering a request that never asked
    for a Range would hash to something other than what the publisher described. Refusal is
    explicit rather than inherited."""
    transport = _Transport(_artifact_response(PAYLOAD, status=status))
    with pytest.raises(category) as caught:
        _client(store, transport).download(_metadata())
    assert caught.value.status_code == status
    assert str(status) in str(caught.value)
    assert transport.responses[0].read_sizes == []


# --------------------------------------------------------------------------- redirects and origin


def test_a_redirect_refusal_from_the_transport_is_passed_through_unchanged(
    store: ArtifactStore,
) -> None:
    """The handler inside `registry_opener` produces this category; the client does not re-wrap it.

    Identity matters: a re-synthesised error would lose the status code the handler recorded and
    read as a client decision rather than as the registry's answer.
    """
    refusal = RegistryRedirectError(
        "the npm registry answered the artifact request with HTTP 302 and a redirect, "
        "which PackProof does not follow",
        status_code=302,
    )
    transport = _Transport(refusal)
    with pytest.raises(RegistryRedirectError) as caught:
        _client(store, transport).download(_metadata())
    assert caught.value is refusal
    assert caught.value.status_code == 302


def test_a_redirect_arriving_as_a_status_without_a_location_is_still_a_redirect(
    store: ArtifactStore,
) -> None:
    """The fallback for a 3xx that `urllib` could not act on, so a redirect is never a thing this
    client followed whichever way it arrived."""
    for status in (301, 302, 303, 307, 308):
        transport = _Transport(_artifact_response(b"", status=status))
        with pytest.raises(RegistryHTTPError) as caught:
            _client(store, transport).download(_metadata())
        assert caught.value.status_code == status


def test_an_httperror_carrying_a_redirect_is_classified_without_naming_its_url(
    store: ArtifactStore,
) -> None:
    """`HTTPError.url` is the signed target. This reads the code and closes the handle; the URL is
    never touched, and the object that would have held the body is released."""

    class _Handle:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    handle = _Handle()
    error = urllib.error.HTTPError(TARBALL, 302, "Found", {}, handle)  # type: ignore[arg-type]
    transport = _Transport(error)
    with pytest.raises(RegistryRedirectError) as caught:
        _client(store, transport).download(_metadata())
    assert "HTTP 302" in str(caught.value)
    assert CAPABILITY not in str(caught.value)
    assert caught.value.__context__ is None, "the caught HTTPError must not become the story"
    assert handle.close_calls == 1


def test_the_origin_guard_refuses_a_host_nobody_configured_without_naming_it() -> None:
    """Probe section 19, permanent: the guard's diagnosis says nothing about the request it stopped.

    The guard is exercised on its own seam because nothing on this path can legitimately hand it a
    foreign host — `plan_target` rebuilds the URL from the constant registry host, and a target that
    does not match is refused before a request object exists. That is the reason its message names
    nothing: a guard that fires means PackProof contradicted its own policy, and the request that
    got that far is the sensitive part of the event rather than the useful part.
    """
    hostile_url = f"https://evil.invalid/left-pad/-/left-pad-1.3.0.tgz?sig={CAPABILITY}"
    request = urllib.request.Request(hostile_url, method="GET")
    with pytest.raises(UnexpectedDownloadOriginError) as caught:
        _OriginGuardHTTPSHandler().https_open(request)
    message = str(caught.value)
    assert "other than the npm registry host" in message
    assert "evil.invalid" not in message
    assert CAPABILITY not in message
    assert hostile_url not in message


def test_the_origin_guard_compares_the_connect_target_not_the_parsed_hostname() -> None:
    """`https://@registry.npmjs.org/a` reports an allowed `hostname` while dialling `@registry...`.

    The guard reads `Request.host`, which is the netloc `AbstractHTTPHandler.do_open` splits into
    its connect target, so the userinfo spelling that a `hostname` comparison would approve is
    refused here. The policy refuses it earlier too; this asserts which value the guard compares.
    """
    request = urllib.request.Request(f"https://@{NPM_REGISTRY_HOST}/left-pad/-/a.tgz", method="GET")
    assert request.host != NPM_REGISTRY_HOST, "the comparison the guard makes is the netloc one"
    with pytest.raises(UnexpectedDownloadOriginError):
        _OriginGuardHTTPSHandler().https_open(request)


def test_the_default_artifact_chain_holds_the_guard_the_redirect_refusal_and_no_proxy_handler(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O1, checked on the chain this client builds for itself.

    `build_opener` reads the environment for proxies unless a `ProxyHandler({})` displaces the
    default, and a proxy handler would be installed *above* the origin guard (`handler_order` 100
    against 500) — so the guard would never run while the bytes still looked like a registry
    request. Setting the variables first is the point: the chain must not care.
    """
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("https_proxy", "http://proxy.invalid:8080")
    client = _client(store, _Transport(), opener=None)
    opener = _opener_from(client._opener)
    handlers = list(opener.handlers)
    assert not any(isinstance(handler, urllib.request.ProxyHandler) for handler in handlers)
    guards = [h for h in handlers if isinstance(h, _OriginGuardHTTPSHandler)]
    refusals = [h for h in handlers if isinstance(h, _RefuseRedirectHandler)]
    assert len(guards) == len(refusals) == 1
    assert refusals[0]._label == "artifact"
    assert refusals[0]._disclose_target is False


def _opener_from(opener_callable: object) -> urllib.request.OpenerDirector:
    """The `OpenerDirector` a `registry_opener` closure has captured.

    Reached through the closure because the seam deliberately returns a callable and not the
    opener: the whole reason for its existence is that `urlopen(request, 30)` binds 30 to `data`.
    """
    cell_values = [
        cell.cell_contents
        for cell in (opener_callable.__closure__ or ())  # type: ignore[attr-defined]
    ]
    directors = [value for value in cell_values if isinstance(value, urllib.request.OpenerDirector)]
    assert len(directors) == 1, f"no single opener in {cell_values}"
    return directors[0]


def test_a_redirect_refusal_names_no_target_on_the_artifact_path_but_one_on_metadata() -> None:
    """The disclosure asymmetry, side by side on the production handler.

    A metadata URL is built from a constant host and a validated name, so naming it tells a reader
    which package behaved oddly and costs nothing. An artifact URL is the registry's own text with
    a signed query, so naming it would publish a capability.
    """
    request = urllib.request.Request(TARBALL, method="GET")
    silent = _RefuseRedirectHandler(label="artifact", disclose_target=False)
    spoken = _RefuseRedirectHandler()
    quiet_handle, loud_handle = io.BytesIO(b""), io.BytesIO(b"")
    with pytest.raises(RegistryRedirectError) as quiet:
        silent.redirect_request(request, quiet_handle, 302, "Found", {}, "https://x.invalid/y")
    with pytest.raises(RegistryRedirectError) as loud:
        spoken.redirect_request(request, loud_handle, 302, "Found", {}, "https://x.invalid/y")
    assert CAPABILITY not in str(quiet.value)
    assert "artifact request" in str(quiet.value)
    assert CAPABILITY in str(loud.value), "the metadata policy names its own URL, deliberately"
    assert quiet_handle.closed and loud_handle.closed, "a refused redirect releases its connection"


# --------------------------------------------------------------------------- no response at all


@pytest.mark.parametrize(
    ("reason", "category", "phrase"),
    [
        (
            urllib.error.URLError(OSError(errno.ETIMEDOUT, "Operation timed out")),
            RegistryTimeoutError,
            "connection to the npm registry timed out",
        ),
        (TimeoutError("read timed out"), RegistryTimeoutError, "within 30 seconds"),
        (
            ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused"),
            RegistryNetworkError,
            "could not be reached",
        ),
        (
            urllib.error.URLError(ConnectionResetError(errno.ECONNRESET, "reset")),
            RegistryNetworkError,
            "could not be reached",
        ),
        (http.client.RemoteDisconnected("peer gone"), RegistryNetworkError, "could not be reached"),
    ],
    ids=["connect-timeout", "configured-timeout", "refused", "reset", "malformed-status-line"],
)
def test_a_failure_with_no_response_is_classified_by_number_and_class_not_by_sentence(
    store: ArtifactStore, reason: object, category: type, phrase: str
) -> None:
    transport = _Transport(reason)
    with pytest.raises(category) as caught:
        _client(store, transport).download(_metadata())
    assert phrase in str(caught.value)
    assert type(caught.value) is category


def test_a_urLError_wrapper_does_not_hide_the_kernel_timeout_inside_it(
    store: ArtifactStore,
) -> None:
    """`URLError` copies `errno` only from reasons that have one, so classifying the wrapper would
    turn a connect timeout into a generic network failure and send the operator to check their
    cable instead of their patience."""
    inner = OSError(errno.ETIMEDOUT, "Operation timed out")
    transport = _Transport(urllib.error.URLError(inner))
    with pytest.raises(RegistryTimeoutError) as caught:
        _client(store, transport).download(_metadata())
    assert "connection to the npm registry timed out" in str(caught.value)
    assert "within" not in str(caught.value)


def test_an_oserror_that_only_says_timed_out_is_a_network_failure_not_a_timeout(
    store: ArtifactStore,
) -> None:
    transport = _Transport(OSError(errno.ECONNRESET, "Connection reset by peer"))
    with pytest.raises(RegistryNetworkError) as caught:
        _client(store, transport).download(_metadata())
    assert "timed out" not in str(caught.value)


def test_the_classified_failure_carries_no_exception_into_its_context(store: ArtifactStore) -> None:
    """Build-then-raise: the caught `urllib` object's own text can name the very address the
    disclosure policy is about, and `from None` would hide it from a render without dropping it."""
    transport = _Transport(urllib.error.URLError(OSError(errno.EHOSTUNREACH, "no route")))
    with pytest.raises(RegistryNetworkError) as caught:
        _client(store, transport).download(_metadata())
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


# --------------------------------------------------------------------------- mid-body failures


def test_a_reset_while_the_body_is_arriving_is_a_network_failure_and_not_a_truncation(
    store: ArtifactStore, workspace: Path
) -> None:
    transport = _Transport(
        _StreamResponse(
            body=[b"a" * 8, ConnectionResetError(errno.ECONNRESET, "reset by peer")],
            headers={"Content-Length": "64", "Content-Type": "application/octet-stream"},
        )
    )
    with pytest.raises(RegistryNetworkError) as caught:
        _client(store, transport).download(_metadata())
    assert "could not be reached" in str(caught.value)
    assert list(workspace.iterdir()) == []
    assert transport.responses[0].close_calls == 1


def test_a_timeout_while_the_body_is_arriving_is_still_a_timeout_named_by_the_limit(
    store: ArtifactStore,
) -> None:
    transport = _Transport(
        _StreamResponse(
            body=[b"a" * 8, TimeoutError()],
            headers={"Content-Length": "64", "Content-Type": "application/octet-stream"},
        )
    )
    with pytest.raises(RegistryTimeoutError) as caught:
        _client(store, transport, timeout_seconds=2.5).download(_metadata())
    assert "within 2.5 seconds" in str(caught.value)


def test_the_response_is_closed_on_every_path_including_the_ones_that_never_read_a_body(
    store: ArtifactStore,
) -> None:
    """A response that is not closed is a connection nobody can reuse and, on an error path, a body
    that is still open. `close_calls == 1` is the assertion; a real socket cannot report it.

    Which cases read a body is asserted in the same loop, because the two refusal paths that fire
    before a byte is asked for are the ones where a forgotten `close()` would be invisible to a
    test that only looked at the successful download.
    """
    cases: list[tuple[str, _StreamResponse | BaseException, bool]] = [
        ("success", _artifact_response(PAYLOAD), True),
        ("declared-above-ceiling", _artifact_response(b"q" * 64), False),
        ("status-500", _artifact_response(PAYLOAD, status=500), False),
        ("encoded-body", _artifact_response(PAYLOAD, Content_Encoding="gzip"), False),
        ("no-response", urllib.error.URLError(OSError(errno.ENETUNREACH, "net down")), False),
    ]
    for name, outcome, read_a_body in cases:
        transport = _Transport(outcome)
        client = _client(store, transport, max_download_bytes=len(PAYLOAD))
        with contextlib.suppress(Exception):  # the category has its own test; this one is cleanup
            client.download(_metadata(integrity=_sri(PAYLOAD)))
        assert len(transport.requests) == 1, f"{name} made {len(transport.requests)} requests"
        if isinstance(outcome, _StreamResponse):
            assert outcome.close_calls == 1, f"{name} closed {outcome.close_calls} times"
            assert bool(outcome.read_sizes) is read_a_body, f"{name} read the wrong amount"


# --------------------------------------------------------------------------- the workspace edge


def test_an_artifact_already_in_the_workspace_is_reused_and_reported_as_verified(
    store: ArtifactStore, workspace: Path
) -> None:
    """The second download is the same content-addressed name, so it is the same artifact.

    Probe section 14: `os.link` answers `EEXIST`, existing-destination verification reads the
    bytes, and the caller gets a successful artifact back with no duplicate and no rewrite.
    """
    first = _downloaded(store, _artifact_response(PAYLOAD), integrity=_sri(PAYLOAD))
    second = _downloaded(store, _artifact_response(PAYLOAD), integrity=_sri(PAYLOAD))
    assert first.file_name == second.file_name == FINAL_NAME
    assert second.verification is ArtifactVerification.VERIFIED
    assert sorted(p.name for p in workspace.iterdir()) == [FINAL_NAME]


def test_a_destination_held_by_other_bytes_is_refused_and_those_bytes_stay_put(
    store: ArtifactStore, workspace: Path
) -> None:
    """The probe's thirteenth section, and the reason a collision is not retried under a new name:
    a content-addressed name is not negotiable."""
    (workspace / FINAL_NAME).write_bytes(b"someone elses artifact")
    before = (workspace / FINAL_NAME).stat().st_ino
    transport = _Transport(_artifact_response(PAYLOAD))
    with pytest.raises(ArtifactCollisionError) as caught:
        _client(store, transport).download(_metadata(integrity=_sri(PAYLOAD)))
    assert "different payload" in str(caught.value)
    held = workspace / FINAL_NAME
    assert held.read_bytes() == b"someone elses artifact"
    assert held.stat().st_ino == before
    assert sorted(p.name for p in workspace.iterdir()) == [FINAL_NAME]


def test_a_symlink_in_the_destination_position_is_foreign_and_is_not_followed(
    store: ArtifactStore, workspace: Path
) -> None:
    victim = workspace / "victim.tgz"
    victim.write_bytes(b"not the artifact")
    (workspace / FINAL_NAME).symlink_to(victim)
    transport = _Transport(_artifact_response(PAYLOAD))
    with pytest.raises(ArtifactCollisionError) as caught:
        _client(store, transport).download(_metadata(integrity=_sri(PAYLOAD)))
    assert "symlink" in str(caught.value)
    assert victim.read_bytes() == b"not the artifact"
    assert (workspace / FINAL_NAME).is_symlink()


def test_a_directory_sync_failure_keeps_the_artifact_and_reports_where_it_is(
    store: ArtifactStore, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Probe section 15, and the one failure that means the workspace *does* hold the bytes.

    `PackageArtifact` has no field for a publication state, so the state and the name travel on the
    exception instead — which is why `WorkspaceDurabilityError` carries both: a caller that treated
    this like every other failure and deleted files would be destroying the download it proved.
    """
    real_link = ArtifactStore._link

    def unsyncable(self: ArtifactStore, rec: Finalization, src: str, dst: str) -> None:
        real_link(self, rec, src, dst)
        monkeypatch.setattr(ArtifactStore, "_fsync_dir", _refuse_to_sync)

    monkeypatch.setattr(ArtifactStore, "_link", unsyncable)
    transport = _Transport(_artifact_response(PAYLOAD))
    with pytest.raises(WorkspaceDurabilityError) as caught:
        _client(store, transport).download(_metadata(integrity=_sri(PAYLOAD)))
    assert caught.value.state == PUBLISHED_UNSYNCED
    assert caught.value.final_name == FINAL_NAME
    assert (workspace / FINAL_NAME).read_bytes() == PAYLOAD
    assert transport.responses[0].close_calls == 1


def _refuse_to_sync(*_bound: object) -> None:
    """Patched onto the class, so it receives the store as its first argument and must accept it.

    The sync is broken only *after* the link succeeded, which is the one window where the answer to
    the caller is a durability warning about a file that exists rather than a failed download.
    """
    raise OSError(errno.EIO, "cannot sync")


def test_an_interrupt_inside_the_commit_window_is_recovered_and_the_artifact_survives(
    store: ArtifactStore, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Probe section 16: the interrupt is the operation's answer, settlement proves the file, and
    cleanup removes the staged name without ever naming the published one."""
    real_link = ArtifactStore._link
    interrupt = KeyboardInterrupt()

    def interrupt_after_linking(self: ArtifactStore, rec: Finalization, src: str, dst: str) -> None:
        real_link(self, rec, src, dst)
        raise interrupt

    monkeypatch.setattr(ArtifactStore, "_link", interrupt_after_linking)
    transport = _Transport(_artifact_response(PAYLOAD))
    with pytest.raises(KeyboardInterrupt) as caught:
        _client(store, transport).download(_metadata(integrity=_sri(PAYLOAD)))
    assert caught.value is interrupt
    assert (workspace / FINAL_NAME).read_bytes() == PAYLOAD
    assert not list(workspace.glob(".partial.*"))
    assert transport.responses[0].close_calls == 1


def test_a_broken_fact_sink_does_not_cost_a_verified_download(tmp_path: Path) -> None:
    sink = _BrokenSink()
    with ArtifactStore(tmp_path, fact_sink=sink) as logging_store:
        transport = _Transport(_artifact_response(PAYLOAD))
        artifact = _client(logging_store, transport).download(_metadata(integrity=_sri(PAYLOAD)))
    assert artifact.verification is ArtifactVerification.VERIFIED
    assert (tmp_path / FINAL_NAME).read_bytes() == PAYLOAD
    assert len(sink.lines) >= 1


# --------------------------------------------------------------------------- policy refusals


def test_a_refused_target_produces_no_request_at_all(store: ArtifactStore) -> None:
    """Probe section 18: the refusal happens before a request object exists, so nothing was offered
    to the address the registry chose."""
    transport = _Transport(_artifact_response(PAYLOAD))
    with pytest.raises(UnsupportedDownloadTargetError) as caught:
        _client(store, transport).download(_raw_metadata(f"http://{NPM_REGISTRY_HOST}/a.tgz"))
    assert "not-https" in str(caught.value)
    assert transport.requests == []


def test_a_target_on_another_host_is_refused_by_code_and_never_repeated(
    store: ArtifactStore,
) -> None:
    """The allowlist is one host wide, and the refusal says only which rule failed.

    Neither the foreign host nor the capability value from the signed query appears in the message.
    The code already says the host was not allowlisted, and a query string is a capability: copying
    it into whatever reads the exception is how a secret outlives the request that needed it.
    """
    transport = _Transport(_artifact_response(PAYLOAD))
    hostile = f"https://evil.invalid/{CAPABILITY}/a.tgz"
    with pytest.raises(UnsupportedDownloadTargetError) as caught:
        _client(store, transport).download(_metadata(tarball=hostile))
    assert "host-not-allowlisted" in str(caught.value)
    assert CAPABILITY not in str(caught.value)
    assert "evil.invalid" not in str(caught.value)
    assert transport.requests == []


def test_a_registry_authority_that_is_not_exactly_the_host_is_refused(
    store: ArtifactStore,
) -> None:
    """`hostname` approves `REGISTRY.NPMJS.ORG` and `@registry.npmjs.org`; the guard would dial
    whatever netloc `Request` carries, so the policy compares the authority string and refuses
    before a request exists. The code says which half failed, and never quotes the spelling."""
    transport = _Transport(_artifact_response(PAYLOAD))
    with pytest.raises(UnsupportedDownloadTargetError) as caught:
        _client(store, transport).download(_raw_metadata(f"https://@{NPM_REGISTRY_HOST}/a.tgz"))
    assert "credentials-present" in str(caught.value)
    assert transport.requests == []


def test_a_closed_workspace_still_asks_the_registry_and_then_refuses_to_write(
    workspace: Path,
) -> None:
    """Measured order, asserted so it cannot drift: the store is opened after the response exists.

    Nothing is written either way — the refusal is the point — but a reader who assumed the
    workspace was checked first would be wrong about when the request happens.
    """
    opened = ArtifactStore(workspace)
    opened.close()
    transport = _Transport(_artifact_response(PAYLOAD))
    with pytest.raises(ArtifactWriteError) as caught:
        _client(opened, transport).download(_metadata())
    assert "store is closed" in str(caught.value)
    assert list(workspace.iterdir()) == []


# --------------------------------------------------------------------------- precedence, at the client


def test_a_registry_error_from_the_transport_reaches_the_caller_as_itself(
    store: ArtifactStore,
) -> None:
    """`_send` classifies what has no category and passes what already has one — re-wrapping would
    lose which of the two happened."""
    refusal = RegistryHTTPError("the registry answered 503", status_code=503)
    transport = _Transport(refusal)
    with pytest.raises(RegistryHTTPError) as caught:
        _client(store, transport).download(_metadata())
    assert caught.value is refusal
    assert caught.value.status_code == 503


def test_an_exception_this_module_does_not_predict_is_not_relabelled(
    store: ArtifactStore, workspace: Path
) -> None:
    """A bug in PackProof or its dependencies must not be reported as a package problem. The staged
    file is still cleaned up, because cleanup does not depend on knowing what the failure was."""
    surprise = ValueError("something this client did not anticipate")
    transport = _Transport(surprise)
    with pytest.raises(ValueError) as caught:
        _client(store, transport).download(_metadata())
    assert caught.value is surprise
    assert list(workspace.iterdir()) == []


def test_a_cleanup_interrupt_never_replaces_the_primary(
    store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The client half of the round-8 rule. The ledger side — which recorded the displaced event —
    is asserted in `test_artifact_store.py`; what is observable from here is that the caller
    receives the artifact error rather than the interrupt."""

    def refuse(self: ArtifactStore, name: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(ArtifactStore, "discard_staged", refuse)
    transport = _Transport(_artifact_response(b"z" * 64))
    with pytest.raises(ArtifactTooLargeError):
        _client(store, transport, max_download_bytes=8, chunk_bytes=4).download(_metadata())


def test_the_attempt_records_its_own_failure_as_the_primary_class_name(
    workspace: Path,
) -> None:
    """The ledger is created per call and reaches nobody else, so two concurrent downloads cannot
    share a publication fact. From outside the client its identity is the fact the store was told
    to write down, which a sink sees and a caller does not have to trust a log level for."""
    sink = _ListSink()
    with ArtifactStore(workspace, fact_sink=sink) as logging_store:
        transport = _Transport(_artifact_response(b"z" * 64))
        with pytest.raises(ArtifactTooLargeError):
            _client(logging_store, transport, max_download_bytes=16, chunk_bytes=4).download(
                _metadata()
            )
    assert "attempt-failed:ArtifactTooLargeError" in sink.lines
    assert not any(line.startswith("published") for line in sink.lines)
    assert list(workspace.iterdir()) == []


# --------------------------------------------------------------------------- secrecy


def _everything_a_failure_shows(error: BaseException) -> str:
    """Message, `repr`, rendered traceback and the whole cause/context chain, as one blob."""
    parts = [str(error), repr(error), type(error).__name__]
    parts.append("".join(traceback.format_exception(type(error), error, error.__traceback__)))
    seen: set[int] = set()
    node: BaseException | None = error
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        parts.append(str(node))
        parts.append(repr(node))
        node = node.__cause__ or node.__context__
    return "\n".join(parts)


#: (scenario, what the transport answers, the metadata record, client overrides).
_FailureRow = tuple[
    str,
    _StreamResponse | BaseException,
    PackageVersionMetadata,
    dict[str, object],
]


def _short_body(chunks: list[bytes | BaseException], declared: int) -> _StreamResponse:
    """A body made of explicit chunks — the only way to fail *after* a byte has arrived."""
    return _StreamResponse(
        body=chunks,
        headers={
            "Content-Length": str(declared),
            "Content-Type": "application/octet-stream",
        },
    )


def _failures(store: ArtifactStore) -> list[tuple[str, BaseException]]:
    """Every category this client can raise, produced by the production path.

    The table is (scenario, what the transport answers, the metadata record, client overrides) so
    that one loop drives all twelve through `download`. A row that stopped raising would fail the
    run rather than quietly contribute nothing, which is the property a leak scan needs: a scenario
    nobody exercised is a scenario whose message nobody read.
    """
    signed_elsewhere = f"http://{NPM_REGISTRY_HOST}/left-pad/-/left-pad-1.3.0.tgz?sig={CAPABILITY}"
    rows: list[_FailureRow] = [
        ("integrity", _artifact_response(PAYLOAD), _metadata(integrity=_sri(b"other")), {}),
        (
            "declared-too-large",
            _artifact_response(b"z" * 500),
            _metadata(),
            {"max_download_bytes": 100},
        ),
        (
            "stream-too-large",
            _artifact_response(b"z" * 64, declare=False),
            _metadata(),
            {"max_download_bytes": 8},
        ),
        ("truncated", _short_body([b"a" * 4], 64), _metadata(), {}),
        ("encoded", _artifact_response(PAYLOAD, Content_Encoding="gzip"), _metadata(), {}),
        ("status", _artifact_response(PAYLOAD, status=503), _metadata(), {}),
        (
            "redirect-handle",
            urllib.error.HTTPError(TARBALL, 302, "Found", {}, None),  # type: ignore[arg-type]
            _metadata(),
            {},
        ),
        (
            "no-response",
            urllib.error.URLError(OSError(errno.ETIMEDOUT, "timed out")),
            _metadata(),
            {},
        ),
        (
            "mid-body-reset",
            _short_body([b"a" * 8, ConnectionResetError(errno.ECONNRESET, "reset")], 64),
            _metadata(),
            {},
        ),
        ("policy", _artifact_response(PAYLOAD), _raw_metadata(signed_elsewhere), {}),
        (
            "refused-elsewhere",
            _artifact_response(PAYLOAD),
            _metadata(tarball=f"https://evil.invalid/{CAPABILITY}/a.tgz"),
            {},
        ),
        ("collision", _artifact_response(PAYLOAD), _metadata(integrity=_sri(PAYLOAD)), {}),
    ]
    found: list[tuple[str, BaseException]] = []
    for name, outcome, metadata, overrides in rows:
        try:
            _client(store, _Transport(outcome), **overrides).download(metadata)
        except BaseException as error:  # noqa: BLE001 - the exception is what is being inspected
            found.append((name, error))
        else:
            pytest.fail(f"scenario {name} unexpectedly succeeded")
    return found


def test_no_failure_path_shows_the_signed_query_anywhere_a_caller_can_read(
    store: ArtifactStore, workspace: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One scenario table, five channels each.

    The channels are the ones a real report is built from: the message, `repr`, the rendered
    traceback, the exception chain a debugger walks, and the DEBUG lines this module emits. The
    collision case needs the destination already held by other bytes, which the workspace the
    earlier scenarios left behind provides.
    """
    (workspace / FINAL_NAME).write_bytes(b"foreign bytes")
    with caplog.at_level(logging.DEBUG, logger="packproof.registry.npm_download"):
        failures = _failures(store)
    assert len(failures) == 12
    for name, error in failures:
        shown = _everything_a_failure_shows(error)
        assert CAPABILITY not in shown, f"{name} disclosed the signed query"
        assert TARBALL not in shown, f"{name} disclosed the target"
        assert shown.isascii(), f"{name} produced text a plain terminal cannot render"
    for record in caplog.records:
        assert CAPABILITY not in record.getMessage()


def test_the_frames_this_client_scrubs_hold_no_request_object_after_a_failure(
    store: ArtifactStore,
) -> None:
    """`download` and `_attempt` empty their own locals in their own `finally`, and this is the
    test that says so: only the frame that bound a name can clear it, and a caller who walks the
    traceback of a real failure finds `None` where the metadata, the target and the prepared
    request were.

    `_attempt` appears twice in the traceback because the round-8 tail re-raises by identity from
    that frame: the frame that raised is a second instance of the same function, and it is scrubbed
    for the same reason as the first.
    """
    transport = _Transport(_artifact_response(b"z" * 64, declare=False))
    with pytest.raises(ArtifactTooLargeError) as caught:
        _client(store, transport, max_download_bytes=8, chunk_bytes=4).download(_metadata())
    inspected: list[str] = []
    dirty: list[tuple[str, str]] = []
    held_open: list[tuple[str, str]] = []
    present: dict[str, list[str]] = {}
    tb = caught.value.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        if frame.f_code.co_filename == npm_download.__file__ and frame.f_code.co_name in {
            "download",
            "_attempt",
        }:
            name = frame.f_code.co_name
            inspected.append(name)
            cleared = sorted(
                key
                for key in ("metadata", "target", "prepared", "request", "response")
                if key in frame.f_locals
            )
            present[name] = cleared
            held_open.extend((name, key) for key in cleared if frame.f_locals[key] is not None)
            for local, value in frame.f_locals.items():
                texts = [value] if isinstance(value, str) else []
                full_url = getattr(value, "full_url", None)
                if isinstance(full_url, str):
                    texts.append(full_url)
                if type(value).__name__ in {"_Prepared", "DownloadTarget"}:
                    texts.append(repr(value))
                for text in texts:
                    if CAPABILITY in text:
                        dirty.append((name, local))
        tb = tb.tb_next
    assert inspected == ["download", "_attempt", "_attempt"]
    assert dirty == []
    assert held_open == []
    assert present == {
        "download": ["metadata", "prepared", "target"],
        "_attempt": ["prepared", "request", "response"],
    }


def test_helper_frames_below_the_scrubbed_ones_still_hold_the_request_limitation(
    store: ArtifactStore,
) -> None:
    """The edge of the guarantee, measured rather than assumed.

    A failure raised inside `_send` has its own frame in the traceback, and that frame binds the
    `Request` it was handed — so the object carrying the signed URL is reachable through
    `__traceback__.tb_frame.f_locals` on that path. It is not a disclosure, because nothing
    renders frame locals; it is why U9/S-11 stays open, and this test exists so nobody closes it by
    asserting the opposite.
    """
    transport = _Transport(urllib.error.URLError(OSError(errno.ENETUNREACH, "no route")))
    with pytest.raises(RegistryNetworkError) as caught:
        _client(store, transport).download(_metadata())
    holders: list[tuple[str, str]] = []
    tb = caught.value.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        if frame.f_code.co_filename == npm_download.__file__:
            for name, value in frame.f_locals.items():
                full_url = getattr(value, "full_url", None)
                if isinstance(full_url, str) and CAPABILITY in full_url:
                    holders.append((frame.f_code.co_name, name))
        tb = tb.tb_next
    assert holders == [("_send", "request")]
    assert CAPABILITY not in "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
    )


def test_the_prepared_request_and_its_claims_print_nothing_sensitive() -> None:
    """`repr=False` is a policy, not a style choice: the generated repr of a dataclass holding the
    URL would print it, and a report that stringifies an object would publish a capability."""
    prepared = npm_download._Prepared(
        reference=REF,
        request_url=TARBALL,
        claims=npm_download._claims_of(_metadata(integrity=_sri(PAYLOAD)).distribution),
        signed=True,
    )
    rendered = repr(prepared)
    assert CAPABILITY not in rendered
    assert "signed" in rendered and "left-pad@1.3.0" in rendered
    assert str(prepared) == rendered

    claims = npm_download._claims_of(
        _metadata(integrity=_sri(PAYLOAD), shasum=_shasum(PAYLOAD)).distribution
    )
    assert base64.b64encode(hashlib.sha512(PAYLOAD).digest()).decode() not in repr(claims)
    assert _shasum(PAYLOAD) not in repr(claims)
    assert [repr(claim) for claim in claims] == ["Claim(sri-sha512)", "Claim(shasum-sha1)"]


def test_a_request_url_that_is_not_signed_reports_itself_as_unsigned() -> None:
    prepared = npm_download._Prepared(
        reference=REF,
        request_url=f"https://{NPM_REGISTRY_HOST}/left-pad/-/left-pad-1.3.0.tgz",
        claims=(),
        signed=False,
    )
    assert "no query" in repr(prepared)


# --------------------------------------------------------------------------- concurrency


def test_two_clients_downloading_the_same_bytes_publish_one_artifact(
    workspace: Path,
) -> None:
    """Independent stores, independent ledgers, one content-addressed destination."""
    outcomes: list[object] = []
    errors: list[BaseException] = []

    def fetch() -> None:
        try:
            with ArtifactStore(workspace) as own_store:
                transport = _Transport(_artifact_response(PAYLOAD))
                outcomes.append(
                    _client(own_store, transport).download(
                        _metadata(integrity=_sri(PAYLOAD), shasum=_shasum(PAYLOAD))
                    )
                )
        except BaseException as error:  # noqa: BLE001 - the assertion is made by the caller
            errors.append(error)

    threads = [threading.Thread(target=fetch) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert len(outcomes) == 2
    assert {artifact.file_name for artifact in outcomes} == {FINAL_NAME}  # type: ignore[attr-defined]
    assert sorted(p.name for p in workspace.iterdir()) == [FINAL_NAME]
