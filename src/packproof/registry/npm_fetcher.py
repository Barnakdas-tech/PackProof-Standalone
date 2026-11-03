"""npm registry metadata retrieval, and the transport policy every registry request shares.

`docs/architecture.md` puts registry communication in `registry/` so nothing above it has
to know which registry a package came from, and puts external values into typed models in
`schemas/`. This module is where those two rules meet: it builds one request, reads a
bounded number of bytes back, and hands them to `PackageVersionMetadata`, so a caller gets
either a validated model or a category from `registry/errors.py` — never a dictionary.

`registry_opener()` is the other half of what this module owns. The rules it installs — one
host, no redirect following, no environment proxy — describe a transport, not a document type,
and `npm_download.py` asks the same host for bytes that cost far more to get wrong. One place
builds that chain so both clients answer to the same one.

What the metadata client deliberately does not do, in order of how easy each is to get wrong:

- It does not fetch a tarball. `dist.tarball` is recorded as text and this client never
  requests it, so the ceiling that applies here is a ceiling on JSON.
- It does not follow redirects. A `Location` value is a URL supplied *inside* a response,
  and the handler that would have followed one is replaced by one that refuses.
- It does not decompress. The request asks for `identity`, and a response that arrives
  encoded anyway is refused rather than decoded, so the byte limit counts exactly the bytes
  the parser is handed and no compressed body can inflate past it.
- It does not print a traceback, a response body or a registry-supplied string beyond a
  quoted, length-capped fragment. Status codes, byte counts and field names carry the
  diagnosis.

The transport is injectable because the size limit, the truncation check and the chunk loop
only exist while bytes are arriving: a test that patches `json.loads` proves nothing about a
stream that never ran. `fetch_version_metadata` asks for the whole package document and
selects the version out of it, which is what lets "this package does not exist" and "this
version was never published" be told apart from one request instead of two.
"""

import contextlib
import errno
import http.client
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from pydantic import ValidationError

from packproof import __version__
from packproof.config.settings import Settings
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
    ascii_text,
    quote_value,
)
from packproof.schemas.models import (
    PackageDistribution,
    PackageRef,
    PackageRepository,
    PackageVersionMetadata,
)

logger = logging.getLogger(__name__)

#: The registry this client speaks to. A constant rather than an argument: Stage 2.2's scope
#: is the official npm registry, and an operator-selectable host would be an
#: operator-selectable source for the provenance claims every later stage is built on.
NPM_REGISTRY_HOST = "registry.npmjs.org"

#: How many bytes to take from the stream at a time. The response is buffered until it is
#: parsed — a JSON parser needs the whole document — so this number decides only how much
#: overshoot the ceiling can be exceeded by, and it is small so that overshoot is small.
_METADATA_READ_CHUNK_BYTES = 64 * 1024

_USER_AGENT = f"packproof/{__version__}"
_ACCEPTED_MEDIA_TYPE = "application/json"

#: Registry keys per `PackageDistribution` field, in the order a report reads them.
_DISTRIBUTION_FIELDS = (
    ("tarball", "tarball_url"),
    ("integrity", "integrity"),
    ("shasum", "shasum"),
)
_REPOSITORY_FIELDS = (("url", "url"), ("type", "kind"), ("directory", "directory"))


class MetadataResponse(Protocol):
    """The slice of an HTTP response this client uses, and the contract an injected transport must meet.

    `http.client.HTTPResponse` already satisfies it, so the real transport needs no
    adapter. `status` is the numeric response code, `read(size)` returns up to `size` bytes
    and an empty bytes at the end of the body, and `close()` releases the connection.
    """

    status: int

    def read(self, size: int = -1) -> bytes: ...

    def getheader(self, name: str, default: str | None = None) -> str | None: ...

    def close(self) -> None: ...


MetadataOpener = Callable[[urllib.request.Request, float], MetadataResponse]


def npm_package_url(fullname: str) -> str:
    """The registry URL that describes `fullname`, assembled one URL component at a time.

    `quote(..., safe="@")` leaves the scope sigil readable and percent-encodes the `/`
    between scope and name, which is how npm's HTTP API documents a scoped package:
    `@babel/core` is fetched as `@babel%2Fcore`, one path segment. A literal
    `/@babel/core` is two segments, and the registry answers that with a redirect it does
    not even need to explain.

    Nothing is concatenated here. Scheme and host are constants that `urlunsplit` writes
    into their own fields, so a name containing `//`, `?`, `#` or `@` cannot move the request
    to another host or add a query. That property matters more than the encoding does: the
    reference has already passed through `schemas/`, but this is the function that decides
    where the bytes come from.

    The version is not part of the URL at all — it is selected out of the returned document
    — so the `+` that build metadata carries never has to survive a query encoding, which is
    the hazard `PackageRef` warns about in its own docstring.
    """
    encoded_name = urllib.parse.quote(fullname, safe="@")
    return urllib.parse.urlunsplit(("https", NPM_REGISTRY_HOST, f"/{encoded_name}", "", ""))


class _RefuseRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Turns any redirect from the registry into a refusal instead of a second request.

    The redirect target is deliberately not named in the message. It is a URL supplied
    inside a response, and printing it invites whoever reads the error to paste it into a
    browser or a `curl` — which is the exact request this class exists to stop.

    Two requests reach this handler and only one of them is safe to name. A metadata URL is
    assembled by `npm_package_url` from a constant host and a validated package name, so
    printing it tells the reader which package the registry behaved oddly about and costs
    nothing. An artifact URL is the registry's own text, and npm serves tarballs with
    time-limited signed queries: printing it would publish a capability. Hence `label` and
    `disclose_target` — and hence the defaults, which are the metadata policy that Stage 2.2
    was reviewed and tested with. The artifact opener asks for the non-disclosing shape by
    hand rather than inheriting a flag whose permissive spelling is the shorter one.
    """

    def __init__(self, *, label: str = "metadata", disclose_target: bool = True) -> None:
        super().__init__()
        self._label = label
        self._disclose_target = disclose_target

    def _refusal(self, req: urllib.request.Request, code: int) -> RegistryRedirectError:
        """The one message this handler produces, built from the two arguments it was given.

        `ascii_text` is what makes the disclosing shape safe: `req.full_url` is a string this
        module built, but `ascii()` guarantees a name that happened to carry non-ASCII cannot
        make the log line unprintable.
        """
        if self._disclose_target:
            return RegistryRedirectError(
                f"the npm registry answered the {self._label} request for "
                f"{ascii_text(req.full_url)} with HTTP {code} and a redirect, "
                "which PackProof does not follow",
                status_code=code,
            )
        return RegistryRedirectError(
            f"the npm registry answered the {self._label} request with HTTP {code} "
            "and a redirect, which PackProof does not follow",
            status_code=code,
        )

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request:
        """Refuse the redirect, releasing the redirect response first.

        `HTTPRedirectHandler.http_error_302` calls this to ask whether it may follow, and only
        *after* the answer comes back does it run its own `fp.read(); fp.close()`. Raising here is
        the point of the class, so that cleanup is never reached: without the call below, every
        refused redirect strands a connection. `close()` without the read is deliberate — draining
        the body is what `urllib` does to enable connection reuse, and a redirect response's body is
        untrusted text this client has no reason to buffer.

        `newurl` is the registry's own answer, and it is not repeated here for the same reason the
        handler exists: a printed target is a target someone pastes into a browser.
        """
        try:
            _close_handle(fp)
            raise self._refusal(req, code)
        finally:
            # `req` carries the URL, `newurl` carries the registry's own answer and `headers`
            # carries the `Location` behind it. Rebinding them empties the one frame on this path
            # that this module owns, once the refusal has been built — and a linter reading it as
            # unused assignment is reading it correctly, because naming nothing is the job here.
            req = newurl = headers = fp = None  # noqa: F841


#: The guard's whole diagnosis, with nothing about the request in it. The guard fires precisely
#: when a request PackProof should never have built reached the transport, which makes that
#: request's address the sensitive part of the event rather than the useful part.
_ORIGIN_REFUSAL_MESSAGE = (
    "PackProof refused to connect: the request reached its transport addressed to a host other "
    "than the npm registry host, which is the only host PackProof connects to"
)


class _OriginGuardHTTPSHandler(urllib.request.HTTPSHandler):
    """The last check before a socket: this transport connects to one host, and only to it.

    Subclassing `HTTPSHandler` is not a style preference, it is the mechanism. `build_opener`
    assembles its default set by skipping any default class that one of the passed handlers
    `isinstance`-matches, so a subclass displaces the stock handler and becomes the sole
    `https` entry in the chain. A sibling `BaseHandler` with an `https_open` method would be
    installed *after* the stock `HTTPSHandler` — `bisect` sorts by `handler_order` (500 for
    both), so the order passed to `build_opener` does not save it — and its check would never
    run, while every test still passed.

    `req.host` is the compared value rather than `urlsplit(req.full_url).hostname` because the
    two are measurably not the same string. `hostname` lowercases and strips userinfo; the
    netloc `Request` carries, and `AbstractHTTPHandler.do_open` splits into the connect target,
    is verbatim. `https://@registry.npmjs.org/a.tgz` therefore reports an allowed `hostname`
    while dialing `@registry.npmjs.org`, and only the netloc comparison catches it.

    The scrub is why this frame exists as one method and not a delegating helper: urllib calls
    this function, so this is the frame that binds `req`, and a caller's `finally` provably
    cannot empty a child frame's locals. Stdlib frames above it still hold the request — the
    boundary of the guarantee, measured and documented, is PackProof's own frames.
    """

    def https_open(self, req: urllib.request.Request) -> Any:
        """Connect only to the registry host; refuse the request without naming it."""
        try:
            if req.host != NPM_REGISTRY_HOST:
                raise UnexpectedDownloadOriginError(_ORIGIN_REFUSAL_MESSAGE)
            return super().https_open(req)
        finally:
            req = None


def registry_opener(*, label: str, disclose_target: bool) -> MetadataOpener:
    """An opener for one host over HTTPS: redirect refusal, origin guard, no environment proxies.

    Built when a client is constructed, never at import: `build_opener` reads proxy settings
    from the environment, and a module that touched the environment on import would make
    `packproof.registry` unimportable in a process that has not finished configuring.

    `urllib.request.ProxyHandler({})` is the reason the environment cannot rewrite this
    request. A `ProxyHandler` has `handler_order` 100 and `HTTPSHandler` 500, so a proxy
    discovered in `HTTPS_PROXY` is installed as the chain's `https_open` and the origin guard
    below it is never consulted — and the bytes still look like a request to the registry while
    a third party chooses where they go. Passing an empty mapping works in two steps:
    `build_opener` skips any default class one of the arguments `isinstance`-matches, which
    keeps out the default handler that reads the environment; and the empty instance itself
    contributes no `https_open`, so `OpenerDirector.add_handler` never inserts it — measured,
    the finished chain holds no `ProxyHandler` at all. Arbitrary proxies are not a supported
    configuration in this stage, which is also what makes the guard's host comparison
    meaningful.

    The wrapper exists for one reason: `urlopen(request, 30)` binds 30 to `data`, not to
    `timeout`, and a request with data becomes a POST. The seam both clients use takes the
    timeout as its second argument, so the keyword binding is decided here, once, instead of
    at every call site.
    """
    opener = urllib.request.build_opener(
        _RefuseRedirectHandler(label=label, disclose_target=disclose_target),
        _OriginGuardHTTPSHandler(),
        urllib.request.ProxyHandler({}),
    )

    def open_request(request: urllib.request.Request, timeout_seconds: float) -> MetadataResponse:
        return opener.open(request, timeout=timeout_seconds)

    return open_request


def _default_opener() -> MetadataOpener:
    """The metadata client's opener, with the metadata disclosure policy.

    Exists as named because Stage 2.2 reviewed and tested this exact spelling: metadata
    requests name their own URL in a refusal, since `npm_package_url` built it from a
    constant host and a validated name.
    """
    return registry_opener(label="metadata", disclose_target=True)


class NpmRegistryClient:
    """Retrieves the registry's metadata for one exact, already-validated package reference.

    The limits arrive as arguments instead of being read from the environment here: a client
    that quietly invented its own timeout would be a client whose timeout nobody configured,
    and `from_settings()` is the one line that connects this class to `PACKPROOF_*`.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float,
        max_metadata_bytes: int,
        opener: MetadataOpener | None = None,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._max_metadata_bytes = max_metadata_bytes
        self._opener: MetadataOpener = _default_opener() if opener is None else opener

    @classmethod
    def from_settings(cls, settings: Settings) -> "NpmRegistryClient":
        """A client using the configured per-request timeout and metadata ceiling."""
        return cls(
            timeout_seconds=settings.http_timeout_seconds,
            max_metadata_bytes=settings.max_metadata_bytes,
        )

    def fetch_version_metadata(self, ref: PackageRef) -> PackageVersionMetadata:
        """Ask the npm registry about `ref.fullname` and return the record for `ref.version`.

        Args:
            ref: A reference that has already been through `schemas/`. This method does not
                parse text into a `PackageRef`; a name that reached here came from a
                validated reference or from nothing at all.

        Raises:
            PackageNotFoundError: The registry has no package by that name.
            VersionNotFoundError: The package exists; this version of it is not in its document.
            RegistryNetworkError: No response was received.
            RegistryTimeoutError: The registry did not answer within the configured timeout.
            RegistryHTTPError: The registry answered with a status that is not a document.
            RegistryRedirectError: The registry answered with a redirect.
            RegistryResponseTooLargeError: The body passed the ceiling while being read.
            RegistryNotJSONError: The body was not JSON.
            RegistryMetadataError: The JSON was not a usable record of this package.
            UnexpectedDownloadOriginError: The origin guard in `registry_opener()` fired, which
                means a request addressed somewhere other than the registry reached the
                transport. It is not a `RegistryError` and is deliberately not wrapped: this is
                PackProof contradicting its own policy, not the registry answering badly.
        """
        request = urllib.request.Request(  # noqa: S310  (bandit flags URL opens; the URL is
            # built by `npm_package_url()` from a constant scheme and host, and the only
            # variable in it is a package name that has already been through `schemas/` —
            # there is no path, query or authority in this string that anything supplied.)
            npm_package_url(ref.fullname),
            headers={
                "Accept": _ACCEPTED_MEDIA_TYPE,
                "Accept-Encoding": "identity",
                "User-Agent": _USER_AGENT,
            },
            method="GET",
        )
        logger.debug("npm metadata request for %s@%s", ref.fullname, ref.version)

        response = self._open(request, ref)
        try:
            document = self._read_document(response, ref)
        finally:
            response.close()

        metadata = _version_metadata(document, ref)
        logger.debug("npm metadata accepted for %s@%s", ref.fullname, ref.version)
        return metadata

    def _open(self, request: urllib.request.Request, ref: PackageRef) -> MetadataResponse:
        """Send the request and translate everything that can go wrong before a body exists.

        This is also where an error response's handle is released. `fetch_version_metadata`'s
        `finally` can only close a response that was returned to it, and a non-2xx arrives as an
        `HTTPError` that was raised instead — a handle `urllib` leaves open, one connection per
        refused reference for the garbage collector to find. Redirects are refused further down,
        inside the opener, and release their handle there.
        """
        try:
            return self._opener(request, self._timeout_seconds)
        except RegistryError:
            # `_RefuseRedirectHandler` raises from inside the transport while the response is
            # being processed. These categories are already the answer — re-wrapping one would
            # lose the status code it carries — and none of them is an `OSError`, so this
            # clause cannot swallow a real connection failure.
            raise
        except urllib.error.HTTPError as exc:
            _close_handle(exc.fp)
            raise _status_error(ref, exc) from exc
        except urllib.error.URLError as exc:
            raise self._no_response_error(ref, exc.reason) from exc
        except (TimeoutError, http.client.HTTPException, OSError) as exc:
            raise self._no_response_error(ref, exc) from exc

    def _read_document(self, response: MetadataResponse, ref: PackageRef) -> object:
        """Check the response line and the headers this client trusts nothing from, then parse."""
        if response.status != 200:
            raise RegistryHTTPError(
                f"the npm registry answered the metadata request for {ref.fullname} "
                f"with HTTP {response.status}",
                status_code=response.status,
            )

        encoding = response.getheader("Content-Encoding")
        if encoding is not None and encoding.strip().lower() not in ("", "identity"):
            raise RegistryInvalidResponseError(
                f"the npm registry sent metadata for {ref.fullname} encoded as "
                f"{quote_value(encoding)}, which PackProof does not decode"
            )

        declared = _declared_length(response)
        if declared is not None and declared > self._max_metadata_bytes:
            raise RegistryResponseTooLargeError(
                f"the npm registry declared {declared} bytes of metadata for {ref.fullname}, "
                f"above the {self._max_metadata_bytes} byte ceiling PackProof reads"
            )

        body = self._read_bounded(response, ref, declared=declared)
        logger.debug(
            "npm metadata received for %s: %s bytes",
            ref.fullname,
            len(body),
        )
        try:
            return json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
            raise RegistryNotJSONError(
                f"the npm registry did not send readable JSON for {ref.fullname} "
                f"({len(body)} bytes read, {type(exc).__name__} while parsing; the body is "
                "not included)"
            ) from exc

    def _read_bounded(
        self, response: MetadataResponse, ref: PackageRef, *, declared: int | None
    ) -> bytes:
        """Read the body a chunk at a time, refusing the moment the ceiling is passed.

        `Content-Length` is used twice and believed zero times: a declared size above the
        ceiling stops the request before any body is read, and a body that arrives shorter
        than declared is refused as truncated. Neither check is what bounds the read — the
        counter in this loop is, so a registry that declares 200 bytes and streams 200
        megabytes is stopped at the same place, and one that sends no header at all is
        stopped too.

        The buffer peaks at the ceiling plus one chunk: the check happens after a chunk is
        appended, because a stream does not tell you it intends to stop.
        """
        collected = bytearray()
        while True:
            try:
                chunk = response.read(_METADATA_READ_CHUNK_BYTES)
            except TimeoutError as exc:
                raise self._timeout_error(ref) from exc
            except (http.client.HTTPException, OSError) as exc:
                raise _network_error(ref, exc) from exc

            if not chunk:
                if declared is not None and len(collected) < declared:
                    raise RegistryInvalidResponseError(
                        f"the npm registry ended the metadata response for {ref.fullname} "
                        f"after {len(collected)} of the {declared} bytes it declared"
                    )
                return bytes(collected)

            collected += chunk
            if len(collected) > self._max_metadata_bytes:
                raise RegistryResponseTooLargeError(
                    f"the npm registry sent more than {self._max_metadata_bytes} bytes of "
                    f"metadata for {ref.fullname}, so the response was abandoned"
                )

    def _timeout_error(self, ref: PackageRef) -> RegistryTimeoutError:
        """Names the limit that was crossed, because it is the number an operator can change."""
        return RegistryTimeoutError(
            f"the npm registry did not answer the metadata request for {ref.fullname} "
            f"within {self._timeout_seconds:g} seconds"
        )

    def _no_response_error(self, ref: PackageRef, reason: object) -> RegistryError:
        """Classify a failure that produced no response — by number and class, never by sentence.

        The order is the whole design. `errno.ETIMEDOUT` is read first because Python turns that
        number into a `TimeoutError`, and the two mean different things: the kernel gave up on the
        connection, whereas this process's configured limit expired while waiting on a peer that
        had accepted it. Reporting the configured number for the first would send an operator to
        change a limit that was never the reason, and reporting a connection timeout as a generic
        network failure would hide the one detail that makes it retryable.
        """
        if _is_os_timeout(reason):
            return _connection_timeout_error(ref, reason)
        if isinstance(reason, TimeoutError):
            return self._timeout_error(ref)
        return _network_error(ref, reason)


def _status_error(
    ref: PackageRef, error: urllib.error.HTTPError
) -> RegistryHTTPError | PackageNotFoundError:
    """Map a response status to a category, naming nothing but the number.

    A 404 is turned into `PackageNotFoundError` on the strength of the request this client
    makes: it asks for a package, never for a version, so at this URL a 404 can only mean the
    package is not there. A 401 or 403 is reported as an HTTP error rather than a
    credentials problem, because PackProof sends no credentials and would be lying to say
    otherwise.
    """
    status = error.code
    if status == 404:
        return PackageNotFoundError(
            f"npm has no package named {ref.fullname} (the registry answered HTTP 404)"
        )
    if status in (301, 302, 303, 307, 308):
        return RegistryRedirectError(
            f"the npm registry redirected the metadata request for {ref.fullname} "
            f"(HTTP {status}), which PackProof does not follow",
            status_code=status,
        )
    return RegistryHTTPError(
        f"the npm registry answered the metadata request for {ref.fullname} with HTTP {status}",
        status_code=status,
    )


def _network_error(ref: PackageRef, reason: object) -> RegistryNetworkError:
    """A failure with no response behind it.

    Only the exception's class name is repeated. Its own message can name a host, a proxy, a
    certificate path or a local file — facts about this machine that a registry error has no
    business reporting and that a person reading a PackProof log did not ask for.
    """
    return RegistryNetworkError(
        f"the npm registry could not be reached for {ref.fullname} "
        f"({ascii_text(type(reason).__name__)})"
    )


def _is_os_timeout(reason: object) -> bool:
    """Whether an OS-level failure reports a connection timeout, by number and not by sentence.

    `errno.ETIMEDOUT` is written by the operating system where the socket gave up, so it cannot be
    borrowed from a proxy's or a library's prose: an `OSError` whose `strerror` happens to read
    "timed out" for some other reason stays the network failure it is. The comparison is against
    this platform's constant rather than a literal, because the same condition is 60 on one kernel
    and 110 on another.
    """
    return getattr(reason, "errno", None) == errno.ETIMEDOUT


def _connection_timeout_error(ref: PackageRef, reason: object) -> RegistryTimeoutError:
    """An operating-system connect timeout: the same category, a different sentence.

    Deliberately not `_timeout_error`. This process's configured limit never fired — the kernel set
    its own patience and the peer did not answer it — and "did not answer within 7.5 seconds" would
    send an operator to change a number that was not the reason. Like `_network_error`, only the
    exception's class name crosses, because a connect timeout's own text names the host that did
    not answer.
    """
    return RegistryTimeoutError(
        f"the connection to the npm registry timed out before the metadata request for "
        f"{ref.fullname} could be answered ({ascii_text(type(reason).__name__)})"
    )


def _close_handle(handle: Any) -> None:
    """Release a response handle without reading a byte of it.

    Two call sites, both on a path where the response never reaches
    `fetch_version_metadata`'s `finally`: `urllib` raises an `HTTPError` that carries the
    handle, and its redirect machinery hands the handle to `redirect_request` directly.
    `close()` is the whole interaction — what a registry sends alongside a 404 or a 302 is
    its own text, and it stays in the buffer instead of becoming PackProof's problem. An
    absent or already-closed handle is left alone, which also makes this safe to call twice,
    though nothing here does.

    A `close()` that fails is suppressed. The status code is the diagnosis, and a socket that will
    not release must not replace `HTTP 503` with an `OSError` that says nothing about the registry.
    """
    if handle is None or getattr(handle, "closed", False):
        return
    with contextlib.suppress(OSError):
        handle.close()


def _declared_length(response: MetadataResponse) -> int | None:
    """`Content-Length` as an untrusted hint: a number, or nothing at all.

    A header that is absent, blank, negative, repeated (which `getheader` renders as a
    comma-joined string) or written in any other form yields None, and the caller then
    enforces the ceiling while reading — which is what it does regardless.
    """
    raw = response.getheader("Content-Length")
    if raw is None:
        return None
    text = raw.strip()
    if not text.isascii() or not text.isdigit():
        return None
    return int(text)


def _explain_validation(error: ValidationError) -> str:
    """Reduce a `ValidationError` to field names and reasons that are safe to print.

    `str(error)` is not used: it embeds pydantic's rendering of the offending input, which is
    whatever the registry sent. Each reason is taken from the structured error list and
    escaped, so the message can say `dist.shasum: …` without carrying the value's bytes into
    a terminal that PackProof does not configure.
    """
    reasons = []
    for item in error.errors():
        path = ".".join(ascii_text(part) for part in item["loc"]) or "document"
        reasons.append(f"{path}: {ascii_text(item['msg'])}")
    return "npm's metadata did not validate (" + "; ".join(reasons) + ")"


def _mapping_or_error(value: object, *, label: str, ref: PackageRef) -> dict[str, Any]:
    """A JSON object from the response, or a refusal naming the field that was not one.

    Keys of a decoded JSON object are always strings, which is what lets the callers below
    use `.get()` on the result without checking each key's type first.
    """
    if not isinstance(value, dict):
        raise RegistryMetadataError(
            f"the {ascii_text(label)} in npm's metadata for {ref.fullname} is "
            f"{ascii_text(type(value).__name__)}, not an object"
        )
    return value


def _selected_version(document: object, ref: PackageRef) -> dict[str, Any]:
    """Walk from a registry document to the one entry that describes `ref`.

    Each check here answers a different question, and the two that answer it best are the
    identity checks: a document whose `name` is not the package asked for, and a version
    entry whose own `name` or `version` disagrees with where it is filed. The second is the
    one that matters most, because an entry filed under `1.0.0` that describes `1.0.1` would
    otherwise attach real-looking integrity values to the wrong artifact — and every later
    stage in this project reasons about the artifact this reference names.
    """
    if not isinstance(document, dict):
        raise RegistryMetadataError(
            f"npm's metadata for {ref.fullname} is {ascii_text(type(document).__name__)}, "
            "not a JSON object"
        )

    reported_name = document.get("name")
    if reported_name != ref.fullname:
        raise RegistryMetadataError(
            f"npm's document for {ref.fullname} describes {quote_value(reported_name)} instead"
        )

    versions = _mapping_or_error(document.get("versions"), label="versions", ref=ref)
    if ref.version not in versions:
        raise VersionNotFoundError(
            f"npm has no version {quote_value(ref.version)} of {ref.fullname} "
            f"({len(versions)} published version(s) are listed in that document)"
        )

    entry = versions[ref.version]
    if not isinstance(entry, dict):
        raise RegistryMetadataError(
            f"npm's entry for {ref.fullname}@{ref.version} is "
            f"{ascii_text(type(entry).__name__)}, not an object"
        )

    for key, expected in (("name", ref.fullname), ("version", ref.version)):
        reported = entry.get(key)
        if reported is not None and reported != expected:
            raise RegistryMetadataError(
                f"npm's entry filed under {quote_value(ref.version)} carries a different "
                f"{ascii_text(key)}: {quote_value(reported)}"
            )

    return entry


def _projected(
    model: type[Any], raw: Mapping[str, Any], field_map: tuple[tuple[str, str], ...]
) -> Any:
    """Build *model* from the keys it declares, or raise a registry category.

    Only the mapped keys are copied, so every other field in a response — `signatures`,
    `attestations`, `_npmUser`, a `readme` — stays in the bytes it arrived in. The models
    forbid extra keys, which is what turns a typo in `field_map` into a loud failure here
    instead of a silently missing value later.
    """
    fields: dict[str, Any] = {}
    for registry_key, model_field in field_map:
        if registry_key in raw:
            fields[model_field] = raw[registry_key]

    try:
        return model(**fields)
    except ValidationError as exc:
        raise RegistryMetadataError(_explain_validation(exc)) from exc


def _repository_of(entry: Mapping[str, Any], ref: PackageRef) -> PackageRepository | None:
    """The version's own `repository`, in whichever of npm's two spellings it arrived.

    npm documents `repository` as either an object or a shorthand string, and the string may
    be `"user/repo"`, `"github:user/repo"` or a URL. A shorthand is stored exactly as it was
    written: expanding it into a host PackProof chose would put a claim about where the
    source lives into the evidence the claim is supposed to be about.

    A top-level `repository` is deliberately not consulted. It belongs to the most recent
    publish, and the version being analyzed may have been tagged from anywhere.
    """
    raw = entry.get("repository")
    if raw is None:
        return None
    if isinstance(raw, str):
        return _projected(PackageRepository, {"url": raw}, (("url", "url"),))
    if isinstance(raw, dict):
        return _projected(PackageRepository, raw, _REPOSITORY_FIELDS)
    raise RegistryMetadataError(
        f"npm's repository for {ref.fullname}@{ref.version} is "
        f"{ascii_text(type(raw).__name__)}, not an object or a string"
    )


def _version_metadata(document: object, ref: PackageRef) -> PackageVersionMetadata:
    """Assemble the validated answer for `ref`, or refuse the document.

    Absent and unusable are kept apart all the way through. A field the registry did not send
    becomes None, which a report prints as "the registry did not say"; a field it sent in a
    shape that cannot be read is an error, because reporting it as absent would claim the
    registry said nothing when in fact PackProof could not understand it.
    """
    entry = _selected_version(document, ref)

    raw_dist = _mapping_or_error(entry.get("dist"), label="dist", ref=ref)
    distribution = _projected(PackageDistribution, raw_dist, _DISTRIBUTION_FIELDS)

    raw_time = document.get("time")
    published_at = None
    if raw_time is not None:
        published_at = _mapping_or_error(raw_time, label="time", ref=ref).get(ref.version)

    try:
        return PackageVersionMetadata(
            reference=ref,
            distribution=distribution,
            description=entry.get("description"),
            repository=_repository_of(entry, ref),
            published_at=published_at,
        )
    except ValidationError as exc:
        raise RegistryMetadataError(_explain_validation(exc)) from exc
