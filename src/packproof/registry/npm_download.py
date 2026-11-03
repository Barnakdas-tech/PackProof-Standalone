r"""npm artifact retrieval: one streamed tarball, verified before a single byte is published.

`docs/architecture.md` puts registry communication in `registry/` and external values in
`schemas/`, and Stage 2.3B is where those rules stop being about text: this module asks the
registry for bytes, writes them to a workspace PackProof created, and hands back a
`PackageArtifact` only if what arrived matches what the metadata claimed about it.

The transport policy is not restated here. `registry_opener()` comes from `npm_fetcher.py`, so
the artifact path runs through the same origin guard, the same redirect refusal and the same
"no environment proxy" rule as the metadata path — one chain, two callers. Four of that
module's helpers are imported rather than duplicated here because they encode transport rules
and not document rules: releasing a handle without reading a byte of it, treating
`Content-Length` as a hint that is believed zero times, telling a kernel connect timeout from a
configured one, and the User-Agent this process sends. A second spelling of a security rule is
a second rule to forget.

What this stage deliberately does not do:

- It does not name the tarball URL anywhere a caller can see. That URL is built by
  `download_policy.plan_target` out of the registry's own text, and npm serves tarballs with
  time-limited signed queries, so the query is a capability. Nothing here puts it in an
  exception message, a log line, a `repr()` or a returned object, and the two frames that must
  hold it to do their work empty their own locals in their own `finally`. The boundary of that
  guarantee is PackProof's frames; the standard library's frames above them are documented in
  `docs/architecture.md`, and rendered tracebacks never print frame locals.
- It does not extract, unpack or execute anything. The artifact is a `.tgz` on disk and stays
  one: no archive member name and no URL path component is ever an input to a filesystem name.
- It does not publish what it cannot check. Every claim the registry supplied is compared over
  the bytes as they arrive, and a mismatch raises before the link — so no content-addressed
  name is ever created for bytes that failed a claim, and an artifact reported as `VERIFIED`
  has had all of its claims hold.
- It does not decompress. The request asks for `identity` and a response that arrives encoded
  is refused, so the ceiling counts exactly the bytes written to disk.

Failure handling follows the round-8 precedence policy ported in `artifact_store.py`: one
attempt is one reduction, the first exception event out of the body is the primary and escapes
by identity, and everything cleanup does afterwards is recorded rather than raised. A failure
before publication still discards its staged file, through the same `discard_staged` the
cleanup runner uses, so "the download failed" and "the workspace holds a file nobody told you
about" cannot both be true.
"""

import base64
import hashlib
import hmac
import http.client
import logging
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from packproof.config.settings import Settings
from packproof.registry.artifact_store import (
    ARTIFACT_WRITE_CHUNK_BYTES,
    ArtifactStore,
    Finalization,
    StagedFile,
    annotate_primary,
    finalize_artifact,
    note_ledger,
)
from packproof.registry.download_policy import TargetRefusal, plan_target
from packproof.registry.errors import (
    ArtifactIntegrityError,
    ArtifactInternalError,
    ArtifactStreamTruncatedError,
    ArtifactTooLargeError,
    RegistryError,
    RegistryHTTPError,
    RegistryInvalidResponseError,
    RegistryNetworkError,
    RegistryRedirectError,
    RegistryTimeoutError,
    UnexpectedDownloadOriginError,
    UnsupportedDownloadTargetError,
    errno_text,
    quote_value,
)
from packproof.registry.npm_fetcher import (
    _USER_AGENT,
    MetadataOpener,
    MetadataResponse,
    _close_handle,
    _declared_length,
    _is_os_timeout,
    registry_opener,
)
from packproof.schemas.artifacts import (
    CHECKED_CLAIM_ORDER,
    ArtifactVerification,
    PackageArtifact,
)
from packproof.schemas.models import PackageDistribution, PackageRef, PackageVersionMetadata

logger = logging.getLogger(__name__)

#: What a packed tarball is: bytes, not a document. Nothing parses the body, so the media type
#: exists to stop an intermediary from deciding to hand back something else in its place.
_ACCEPTED_ARTIFACT_MEDIA_TYPE = "application/octet-stream"

#: The algorithm every content-addressed name is built from, whatever else is being hashed.
#: Fixed here rather than taken from the metadata: `dist.integrity` describes what the
#: publisher intended, and this name describes what this process wrote.
_NAME_ALGORITHM = "sha256"

#: The statuses that mean "look elsewhere", and therefore "do not download". `urllib` would
#: follow them; the handler `registry_opener()` installs refuses instead, and this tuple is the
#: fallback for a 3xx that arrives with no `Location` to follow.
_REDIRECT_STATUSES = (301, 302, 303, 307, 308)


class _Hasher(Protocol):
    """What an object from `hashlib` does, spelled once so the tables below have a type.

    Not `hashlib._Hasher`, which is private and has moved between releases.
    """

    def update(self, chunk: bytes) -> None: ...

    def digest(self) -> bytes: ...

    def hexdigest(self) -> str: ...


#: The SRI algorithms PackProof will hash, spelled as `schemas/models.py` spells them.
#: Deliberately a table and not `hashlib.new(algorithm)`: that call accepts `md5`, `blake2b`
#: and `shake_128`, so a registry document naming one of them would have PackProof computing a
#: digest no integrity report can be built on. A lookup makes the vocabulary a fact about this
#: module rather than about whatever OpenSSL the interpreter was built against.
_SRI_HASHERS: dict[str, Callable[[], _Hasher]] = {
    "sha1": hashlib.sha1,
    "sha256": hashlib.sha256,
    "sha384": hashlib.sha384,
    "sha512": hashlib.sha512,
}


def _at(ref: PackageRef) -> str:
    """`fullname@version`, the one identity a message on this path may name.

    Both halves have been through `schemas/`, which rejects non-ASCII and control characters,
    so this is safe to interpolate directly — and it is what lets every artifact message
    describe the package without describing where its bytes came from.
    """
    return f"{ref.fullname}@{ref.version}"


@dataclass(frozen=True, repr=False)
class _Claim:
    """One integrity claim, reduced to a reportable name and the bytes to compare against.

    `expected` is the registry's base64 (or hex) text decoded once, here at the seam, so every
    comparison downstream is between raw digest bytes rather than between two spellings of
    them. `repr=False` because the generated repr would print those bytes: a report showing
    expected-versus-actual digests for a hostile artifact is printing the publisher's chosen
    strings into a terminal, which is the thing `ArtifactIntegrityError` exists not to do.
    """

    name: str
    algorithm: str
    expected: bytes

    def __repr__(self) -> str:
        return f"Claim({self.name})"


@dataclass(frozen=True, repr=False)
class _Prepared:
    """The request the transport may make, and everything a report may say about it.

    Built in `download` while the metadata object is still in scope, out of fields the models
    already validated. `request_url` is the only member that can carry a signed query and the
    only member the transport reads. `repr=False` for the reason `DownloadTarget` gives: a
    leak-free repr is not a property of a class until somebody writes one.
    """

    reference: PackageRef
    request_url: str
    claims: tuple[_Claim, ...]
    signed: bool

    def __repr__(self) -> str:
        query = "signed" if self.signed else "no query"
        return f"Prepared({_at(self.reference)}, {query}, {len(self.claims)} claim(s))"


@dataclass(frozen=True)
class _AttemptOutcome:
    """What a finished attempt knows, in scalars.

    No exception object, no response and nothing registry-supplied: that is what lets the
    returned `PackageArtifact` be assembled from a value that cannot carry a URL, a traceback
    or a response body into the model a report prints.
    """

    digest: str
    final_name: str
    size_bytes: int
    verification: ArtifactVerification
    checked_claims: tuple[str, ...]


def _artifact_request(request_url: str) -> urllib.request.Request:
    """Wrap the approved URL in the one object that will be handed to the transport.

    `noqa: S310` is earned rather than assumed: `request_url` has already been through
    `download_policy.plan_target`, which requires `https`, requires the registry host spelled
    exactly, refuses credentials and a fragment, and rebuilds the string from those approved
    components instead of reusing the registry's text. The guard `registry_opener()` installs
    then checks the same fact again at the point where a socket would open.

    No credentials are sent, in any header, ever: an npm tarball is public, and a request that
    could carry a token would be a request whose failure could describe one.
    """
    return urllib.request.Request(  # noqa: S310
        request_url,
        headers={
            "Accept": _ACCEPTED_ARTIFACT_MEDIA_TYPE,
            "Accept-Encoding": "identity",
            "User-Agent": _USER_AGENT,
        },
        method="GET",
    )


def _claims_of(distribution: PackageDistribution) -> tuple[_Claim, ...]:
    """Every integrity claim the registry supplied for one version, as far as it can be read.

    `integrity` and `shasum` are both optional, and both are parsed here under an assumption the
    models established: `PackageDistribution._is_a_complete_sri_hash` proved the SRI string is
    one `algorithm-base64` pair whose base64 decodes to exactly that algorithm's digest length,
    and `_is_a_lowercase_sha1` proved `shasum` is 40 lowercase hex digits. So neither
    `b64decode` nor `fromhex` below can raise — a value in any other shape never became a
    `PackageDistribution` at all.

    A multi-value SRI list (`sha512-…,sha256-…`) cannot reach here for the same reason: the
    validator partitions on the first `-` and requires the whole remainder to decode as one
    digest of one known algorithm, so a comma is refused at the boundary instead of becoming a
    claim PackProof checked half of.
    """
    claims: list[_Claim] = []
    if distribution.integrity is not None:
        algorithm, _, encoded = distribution.integrity.partition("-")
        claims.append(_Claim(f"sri-{algorithm}", algorithm, base64.b64decode(encoded)))
    if distribution.shasum is not None:
        claims.append(_Claim("shasum-sha1", "sha1", bytes.fromhex(distribution.shasum)))
    return tuple(claims)


def _hashers_for(claims: tuple[_Claim, ...], ref: PackageRef) -> dict[str, _Hasher]:
    """The digest set to compute while streaming: the naming hash, plus one per claim.

    One pass over the bytes feeds every hasher, so checking two claims does not cost two reads
    of the artifact. `sha256` is always present because the content-addressed name needs it even
    when the registry said nothing about integrity at all.
    """
    hashers: dict[str, _Hasher] = {_NAME_ALGORITHM: hashlib.sha256()}
    for claim in claims:
        constructor = _SRI_HASHERS.get(claim.algorithm)
        if constructor is None:
            # Reachable in exactly one way: `schemas/models.py` widens `_SRI_DIGEST_LENGTHS` and
            # this table does not follow. That is a PackProof bug and not a package problem, and
            # it says so here instead of surfacing as a pydantic rejection of `sri-blake2b`
            # several frames away from its cause.
            raise ArtifactInternalError(
                f"the registry published a {quote_value(claim.algorithm)} claim for {_at(ref)} "
                "and PackProof has no hasher for it; the algorithms `schemas/models.py` accepts "
                "as SRI and the algorithms this module computes have drifted apart"
            )
        hashers.setdefault(claim.algorithm, constructor())
    return hashers


def _mismatched_claims(claims: tuple[_Claim, ...], hashers: dict[str, _Hasher]) -> list[_Claim]:
    """Every claim that did not hold. All of them are compared; the first failure is not one.

    Conjunctive verification is the requirement, and comparing all of them before raising is
    what makes the report able to say how many held. `compare_digest` over raw bytes rather than
    `==` over hex text keeps a spelling difference from ever being the reason a claim "failed",
    and it answers a length mismatch with `False` instead of raising on one — which matters
    because the model already guarantees the length, so a claim that somehow arrives with a
    short digest is a mismatch and not a crash.
    """
    failures: list[_Claim] = []
    for claim in claims:
        if not hmac.compare_digest(hashers[claim.algorithm].digest(), claim.expected):
            failures.append(claim)
    return failures


def _integrity_failure(
    ref: PackageRef, claims: tuple[_Claim, ...], failures: list[_Claim]
) -> ArtifactIntegrityError:
    """Name the strongest claim that failed, and how many others did too.

    The ranking is `CHECKED_CLAIM_ORDER` read backwards rather than a second list of names: the
    order a report stores claims in is already ascending in strength, so keeping one ranking
    means nothing can disagree about which algorithm is the strong one. That is where SHA-512
    preference lives — a registry that supplied an SHA-512 claim gets it named first, and the
    SHA-1 shasum a pre-SRI release can offer is named last — and it never changes *which* claims
    are checked, because the conjunctive rule decides that.

    Neither digest appears in the message. `schemas/artifacts.py` refuses an artifact for bytes
    that failed a claim, so this sentence's job is to say which promise broke, not to publish
    the bytes someone chose.
    """
    ranked = sorted(failures, key=lambda claim: CHECKED_CLAIM_ORDER.index(claim.name), reverse=True)
    strongest = ranked[0]
    others = len(ranked) - 1
    beside = (
        f"and {others} other claim(s) beside it" if others else "with every other claim holding"
    )
    return ArtifactIntegrityError(
        f"the bytes PackProof downloaded for {_at(ref)} do not match the "
        f"{quote_value(strongest.name)} claim the registry published for them, {beside}: "
        f"{len(claims)} claim(s) were checked, none is reported as verified, and no digest "
        "value is repeated here"
    )


def _status_failure(ref: PackageRef, error: urllib.error.HTTPError) -> RegistryError:
    """Map an error status to a category, naming the number and nothing else.

    A 3xx reaching here is the shape `urllib` produces when it was handed a redirect it could
    not act on — no `Location`, or one it could not resolve. The ordinary case is refused by
    the handler inside `registry_opener()` before this runs, and becomes the same category, so
    a redirect is never a thing this client followed whichever way it arrived.
    """
    status = error.code
    if status in _REDIRECT_STATUSES:
        return RegistryRedirectError(
            f"the npm registry redirected the artifact request for {_at(ref)} (HTTP {status}), "
            "which PackProof neither follows nor reissues",
            status_code=status,
        )
    return RegistryHTTPError(
        f"the npm registry answered the artifact request for {_at(ref)} with HTTP {status}",
        status_code=status,
    )


def _timeout_failure(ref: PackageRef, timeout_seconds: float) -> RegistryTimeoutError:
    """This process's own limit expired while waiting for the artifact."""
    return RegistryTimeoutError(
        f"the npm registry did not answer the artifact request for {_at(ref)} "
        f"within {timeout_seconds:g} seconds"
    )


def _reason_of(error: BaseException) -> object:
    """`URLError.reason` where there is one, and the error itself otherwise.

    This is the metadata client's rule, restated because it decides a category. `URLError`
    wraps the failure that stopped a response from arriving, and only copies `errno` from
    reasons that have one — so classifying the wrapper turns a kernel connect timeout that
    arrived through `URLError` into a generic network failure, and sends an operator to check
    their cable instead of their patience.
    """
    return error.reason if isinstance(error, urllib.error.URLError) else error


def _no_response_failure(ref: PackageRef, reason: object, timeout_seconds: float) -> RegistryError:
    """A failure with no response behind it, classified by number and class, never by sentence.

    The same three-way split the metadata client makes, for the same measured reasons:
    `errno.ETIMEDOUT` means the kernel gave up on a connection, a `TimeoutError` means the
    configured limit expired against a peer that had accepted it, and anything else is a network
    failure that more patience would not improve. Only the exception's class name crosses into
    the message, because its own text can name a host, a certificate path or — on this path — a
    URL whose query is a capability.
    """
    if _is_os_timeout(reason):
        return RegistryTimeoutError(
            f"the connection to the npm registry timed out before the artifact request for "
            f"{_at(ref)} could be answered ({quote_value(type(reason).__name__)})"
        )
    if isinstance(reason, TimeoutError):
        return _timeout_failure(ref, timeout_seconds)
    return RegistryNetworkError(
        f"the npm registry could not be reached for {_at(ref)} "
        f"({quote_value(type(reason).__name__)})"
    )


def _target_refusal(code: str, ref: PackageRef) -> UnsupportedDownloadTargetError:
    """The one message the policy path produces: a code from a closed vocabulary, no target.

    `code` is interpolated through `quote_value` even though every member of
    `download_policy.TARGET_REFUSAL_CODES` is this project's own ASCII literal, because the
    alternative is a message that trusts a set defined elsewhere to keep containing only that.
    """
    return UnsupportedDownloadTargetError(
        f"PackProof will not download the artifact for {_at(ref)}: the registry's "
        f"`dist.tarball` was refused as {quote_value(code)}, and the target itself is not "
        "repeated here"
    )


class NpmArtifactClient:
    """Downloads one package's tarball into a caller-owned workspace, or refuses to.

    The store is a constructor argument rather than something built here: the workspace belongs
    to the caller, its descriptor is the caller's to close, and a client that opened a directory
    behind the object that owns it would leave a descriptor nobody can release. The
    `Finalization` ledger is created per `download` call and reaches nobody else, which is what
    keeps two concurrent downloads from sharing a publication fact.

    The transport is injectable for the reason the metadata client gives for itself: the
    ceiling, the truncation check, the conjunction of integrity claims and the precedence of
    exceptions all exist only while bytes are arriving, and a test that replaced this whole
    method would prove nothing about any of them.
    """

    def __init__(
        self,
        store: ArtifactStore,
        *,
        timeout_seconds: float,
        max_download_bytes: int,
        opener: MetadataOpener | None = None,
        chunk_bytes: int = ARTIFACT_WRITE_CHUNK_BYTES,
    ) -> None:
        # Injectable, so defaulted-and-checked like the ceiling it shares a job with: a chunk of
        # `0` makes `read(0)` return `b''` forever, which this loop would read as an orderly end
        # of body, and a negative one asks the response for the whole thing at once — the one
        # read this client must never perform.
        if chunk_bytes <= 0:
            raise ValueError(
                f"a download chunk is {chunk_bytes} bytes; the stream loop only stops on an "
                "empty read, so that number has to be one this process can ask for"
            )
        self._store = store
        self._timeout_seconds = timeout_seconds
        self._max_download_bytes = max_download_bytes
        self._chunk_bytes = chunk_bytes
        # Built when a client is constructed, never at import: `registry_opener` reads proxy
        # settings from the environment. `disclose_target=False` is the artifact disclosure
        # policy — a redirect on this path is refused without naming the URL that was redirected.
        self._opener: MetadataOpener = (
            registry_opener(label="artifact", disclose_target=False) if opener is None else opener
        )

    @classmethod
    def from_settings(cls, settings: Settings, store: ArtifactStore) -> "NpmArtifactClient":
        """A client using the configured timeout and download ceiling, publishing into *store*."""
        return cls(
            store,
            timeout_seconds=settings.http_timeout_seconds,
            max_download_bytes=settings.max_download_bytes,
        )

    def download(self, metadata: PackageVersionMetadata) -> PackageArtifact:
        """Fetch the artifact `metadata` describes, verify it, and publish it under its own name.

        Args:
            metadata: The record `NpmRegistryClient` produced for the reference being analyzed.
                Nothing here re-reads the registry: `dist.tarball` is the only address this
                method will consider and `dist.integrity` / `dist.shasum` are the only claims it
                will honour.

        Returns:
            A `PackageArtifact` naming the file in the workspace, its size and digest as this
            process wrote them, and whether the registry's claims were checked or were absent.
            A returned artifact is always on disk: every failure path raises.

        Raises:
            UnsupportedDownloadTargetError: The registry's `dist.tarball` is not an address
                PackProof will connect to. No request was made, so nothing was offered to it.
            UnexpectedDownloadOriginError: The origin guard fired, which means a request
                PackProof should never have built reached its transport.
            RegistryRedirectError: The registry answered with a redirect, which is never followed.
            RegistryHTTPError: Any other status, including 206 Partial Content and 404.
            RegistryNetworkError, RegistryTimeoutError: Nothing came back, in either of the two
                ways that happens.
            RegistryInvalidResponseError: The body arrived encoded, so the ceiling would be
                counting bytes this process has not decompressed.
            ArtifactTooLargeError: The declared or the actual size passed `max_download_bytes`.
            ArtifactStreamTruncatedError: The stream ended short of the length it declared.
            ArtifactIntegrityError: A supplied claim did not hold. Nothing is published.
            ArtifactWriteError, WorkspaceOutOfSpaceError, WorkspaceEntryInvalidError: The
                workspace refused the bytes.
            ArtifactCollisionError: The content-addressed destination is held by something
                PackProof will not replace.
            ArtifactSettlementError: A link was started and its outcome could not be read back.
            WorkspaceDurabilityError: The artifact *is* committed and the directory entry could
                not be synced — a durability warning about the workspace, never a claim that the
                download was lost. See `artifact_store.PUBLISHED_UNSYNCED`.
            ArtifactInternalError: `schemas/` and this module disagree about which SRI
                algorithms exist, which is a bug in PackProof and not a fact about a package.

        Any of these may arrive after the staged file has been discarded: cleanup runs for every
        failure, and no cleanup action on this path names a published file.
        """
        target = plan_target(metadata)
        if isinstance(target, TargetRefusal):
            raise _target_refusal(target.code, metadata.reference)

        ref = metadata.reference
        record = Finalization(_at(ref))
        prepared: _Prepared | None = None
        outcome: _AttemptOutcome | None = None
        try:
            prepared = _Prepared(
                reference=ref,
                request_url=target.request_url,
                claims=_claims_of(metadata.distribution),
                signed=target.signed,
            )
            outcome = self._attempt(prepared, record)
        finally:
            # `metadata` carries `dist.tarball`, `target` carries it again and `prepared` carries
            # the rebuilt request URL. This frame belongs to the traceback of anything the
            # attempt raised, so the names go empty on the way out whether the body finished or
            # not — and a caller who walks these frames finds `None` where a signed URL was.
            metadata = target = prepared = None  # noqa: F841

        # `outcome` is assigned on the only path that reaches here: `_attempt` either returns a
        # completed outcome or raises, so there is no "what if it was None" branch to write, and
        # writing one would put a success path in the code that does not exist.
        return PackageArtifact(
            reference=ref,
            directory=self._store.root,
            file_name=outcome.final_name,
            sha256=outcome.digest,
            size_bytes=outcome.size_bytes,
            verification=outcome.verification,
            checked_claims=outcome.checked_claims,
        )

    def _attempt(self, prepared: _Prepared, record: Finalization) -> _AttemptOutcome:
        """One request, one staged file, one reduction: the design's single-exit attempt.

        Every event leaves this frame the same way — caught into `held`, recorded by class name,
        and re-raised by identity at the tail after the response and the staged file have been
        released. Nothing in the body returns early, so there is exactly one place a primary can
        escape and exactly one place cleanup runs. The tail raises rather than doing it from
        inside the `finally`, because measured P4c says a `raise` from that `finally` would not
        merely take the slot from `held`: it would leave `held` with no `__context__` at all.

        The order inside the body is the order the guarantees depend on: read the shape of the
        response, stage a file under a name this process invented, stream bytes into it while
        hashing, check every claim, and only then name the digest and hand it to publication. A
        failure at any step reaches the same `finally`.
        """
        ref = prepared.reference
        held: BaseException | None = None
        result: _AttemptOutcome | None = None
        request: urllib.request.Request | None = None
        response: MetadataResponse | None = None
        staged: StagedFile | None = None
        try:
            logger.debug("npm artifact request for %s", _at(ref))
            request = _artifact_request(prepared.request_url)
            response = self._send(request, ref)
            declared = self._response_shape(response, ref)
            hashers = _hashers_for(prepared.claims, ref)
            staged = self._store.stage(record)
            received = self._stream(response, staged, hashers, ref, declared)
            failures = _mismatched_claims(prepared.claims, hashers)
            if failures:
                raise _integrity_failure(ref, prepared.claims, failures)
            # Verified, and only now named: the content-addressed name is a fact about the last
            # byte, and a stream that failed a claim never reaches `finalize`.
            record.digest = hashers[_NAME_ALGORITHM].hexdigest()
            staged.seal()
            finalize_artifact(self._store, record)
            supplied = {claim.name for claim in prepared.claims}
            result = _AttemptOutcome(
                digest=record.digest,
                final_name=record.published_as,
                size_bytes=received,
                verification=(
                    ArtifactVerification.VERIFIED
                    if prepared.claims
                    else ArtifactVerification.UNAVAILABLE
                ),
                checked_claims=tuple(name for name in CHECKED_CLAIM_ORDER if name in supplied),
            )
            logger.debug(
                "npm artifact published for %s: %s bytes, %s",
                _at(ref),
                received,
                result.verification.value,
            )
        except BaseException as error:  # re-raised by identity below, never reduced here
            held = error
            # Wrapped, for the measured reason given in `finalize_artifact`: the recording below
            # touches the sink, and an interrupt landing on that call while this clause is still
            # handling `held` would otherwise take the slot and become the operation's story.
            try:
                record.pending_failure = type(error).__name__
                self._store.record_fact("attempt-failed", type(error).__name__, record)
            except BaseException as secondary:  # recorded, never propagated
                name = type(secondary).__name__
                record.interrupts_seen.append(name)
                record.displaced.append(f"in-handler:{name}")
                note_ledger(record, "displaced", f"in-handler:{name}")
        finally:
            try:
                self._release(response, staged, record)
            except BaseException as secondary:  # guarded: this finally must not displace `held`
                name = type(secondary).__name__
                record.interrupts_seen.append(name)
                record.displaced.append(f"in-finally:{name}")
                note_ledger(record, "displaced", f"in-finally:{name}")
                if held is None:
                    raise  # nothing else owns the slot, so the cleanup event is the primary
            request = prepared = response = staged = None  # noqa: F841
        if held is not None:
            annotate_primary(record, held)
            raise held
        return result

    def _send(self, request: urllib.request.Request, ref: PackageRef) -> MetadataResponse:
        """Make the request and translate everything that can go wrong before a body exists.

        Each handler *builds* the failure and the raise happens after the clause, with no
        exception in flight: raising inside one would attach the caught `urllib` exception as
        `__context__`, and on this path that object's own text can name the very address the
        disclosure policy is about. `from None` would hide it from a rendered traceback without
        removing it from `__context__`, which is the difference between redacting a value and
        not holding it.

        Already-categorised exceptions pass through untouched, because a redirect refusal carries
        its status code and the guard's refusal carries the fact that PackProof contradicted its
        own policy — re-wrapping either would lose which of the two happened.
        """
        failure: BaseException | None = None
        try:
            return self._opener(request, self._timeout_seconds)
        except (RegistryError, UnexpectedDownloadOriginError) as already_named:
            failure = already_named
        except urllib.error.HTTPError as error:
            # `error.fp` is the handle `urllib` leaves open when it raises instead of returning;
            # it is released here because the caller's `finally` can only close a response that
            # was returned. `error.url` is the signed target, and is never read by this module.
            _close_handle(error.fp)
            failure = _status_failure(ref, error)
        except (TimeoutError, http.client.HTTPException, OSError) as no_response:
            # `URLError` and `HTTPError` are both `OSError` subclasses, so the two clauses above
            # win first and everything left here produced no response at all. The reason inside a
            # `URLError` is what carries the number this classification reads.
            failure = _no_response_failure(ref, _reason_of(no_response), self._timeout_seconds)
        raise failure

    def _response_shape(self, response: MetadataResponse, ref: PackageRef) -> int | None:
        """The three things a response line may say that this client refuses, then the ceiling.

        `status != 200` rather than a rejection list: `urllib` treats every 2xx as success, and
        a partial body served to a request that never asked for a Range is a stream that will
        hash to something other than what the publisher described. `Content-Length` is consulted
        twice and believed zero times — once to stop before a byte is read, once to recognise a
        stream that ended short of it — while the counter in `_stream` is what actually bounds
        the read.
        """
        if response.status != 200:
            raise RegistryHTTPError(
                f"the npm registry answered the artifact request for {_at(ref)} with "
                f"HTTP {response.status}, which is not a complete artifact",
                status_code=response.status,
            )
        encoding = response.getheader("Content-Encoding")
        if encoding is not None and encoding.strip().lower() not in ("", "identity"):
            raise RegistryInvalidResponseError(
                f"the npm registry sent the artifact for {_at(ref)} encoded as "
                f"{quote_value(encoding)}, which PackProof does not decode"
            )
        declared = _declared_length(response)
        if declared is not None and declared > self._max_download_bytes:
            raise ArtifactTooLargeError(
                f"the npm registry declared {declared} bytes for {_at(ref)}, above the "
                f"{self._max_download_bytes} byte ceiling PackProof downloads, so no body "
                "was read"
            )
        return declared

    def _stream(
        self,
        response: MetadataResponse,
        staged: StagedFile,
        hashers: dict[str, _Hasher],
        ref: PackageRef,
        declared: int | None,
    ) -> int:
        """Read a chunk at a time, write it, feed every hasher from it, and stop at the ceiling.

        Nothing about the body is buffered beyond one chunk, so there is no aggregate for a
        hostile response to grow: the peak is the ceiling plus one chunk, and that overshoot is
        `chunk_bytes` by construction rather than by hope. `staged.write` runs before the ceiling
        check, so the limit is enforced against bytes that reached the file — the only kind of
        number a digest of that file can be compared against.

        There is no deadline across reads. `timeout_seconds` is the socket's per-operation
        patience, which a registry that dribbles one chunk every five seconds for an hour will
        satisfy forever; the ceiling is what stops that response, and it stops it after
        `max_download_bytes` rather than after a wall-clock budget. Stated here because it is a
        limit this stage does not implement.
        """
        received = 0
        while True:
            read_failure: BaseException | None = None
            try:
                chunk = response.read(self._chunk_bytes)
            except TimeoutError:
                read_failure = _timeout_failure(ref, self._timeout_seconds)
            except (http.client.HTTPException, OSError) as error:
                read_failure = _no_response_failure(ref, error, self._timeout_seconds)
            if read_failure is not None:
                raise read_failure

            if not chunk:
                if declared is not None and received < declared:
                    raise ArtifactStreamTruncatedError(
                        f"the npm registry ended the artifact response for {_at(ref)} after "
                        f"{received} of the {declared} bytes it declared"
                    )
                return received

            for hasher in hashers.values():
                hasher.update(chunk)
            staged.write(chunk)
            received += len(chunk)
            if received > self._max_download_bytes:
                raise ArtifactTooLargeError(
                    f"the npm registry sent more than {self._max_download_bytes} bytes for "
                    f"{_at(ref)}, so the stream was abandoned after {received} bytes"
                )

    def _release(
        self, response: MetadataResponse | None, staged: StagedFile | None, record: Finalization
    ) -> None:
        """Let go of the stream, and of the staged file if nothing was published.

        Called from the `finally`, so everything here is either incapable of raising or an event
        the caller records as secondary. `Exception` and not `BaseException`: a close that is
        *interrupted* is a real interrupt, and the guarded clause around this call is what keeps
        it from displacing a primary that already owns the slot.

        The staged discard is `store.discard_staged` — the step the cleanup runner uses — so a
        failure before publication cannot leave a `.partial.*` file that no ledger mentions, and
        the two paths cannot drift into different definitions of "cleaned up". A staged name the
        runner already retired, or already recorded as an orphan, is left alone; nothing on this
        path ever names a published file.
        """
        if response is not None:
            try:
                response.close()
            except Exception as error:  # NOT BaseException: see the docstring
                name = type(error).__name__
                record.cleanup_failures.append(f"close-failed:{name}")
                detail = errno_text(error) if isinstance(error, OSError) else "no detail permitted"
                self._store.record_fact("close-failed", f"{name} {detail}", record)
        if staged is not None:
            staged.close()
        if (
            record.staged_name is not None
            and record.published_as is None
            and not record.orphaned_temp
        ):
            self._store.discard_staged(record)
