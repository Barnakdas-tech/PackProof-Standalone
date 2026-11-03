"""The failure categories a registry call can end in.

Every one of these says something the caller cannot infer from a single generic error:
whether the package does not exist, whether the *version* does not exist, whether the
registry was unreachable at all, and whether it answered but could not be understood. A
"metadata could not be retrieved" exception that covers all four would be honest, and
would send the next person to the wrong place — a typo in a version, a network that is
down and a registry serving garbage call for different reactions.

Nothing here carries a response body, a reason phrase or a URL taken from a response. A
registry's own error text is written by whoever operates that registry, and npm's bodies
name paths and internals; quoting them into a PackProof message would put untrusted bytes
in front of a terminal that PackProof does not configure. Where a message has to name
registry-controlled content — a field that failed validation — it quotes the *field*, and
the value goes through `quote_value()`.

Messages are ASCII. Same rule as `schemas/models.py`: an error about input that could not
be read must not itself fail to be written, so any value a message carries is escaped
before it gets there.

Two families live here, because a registry call has two ways to be wrong. `RegistryError`
and its subclasses describe what the registry *said* — no such package, no such version, an
answer that is not JSON. `ArtifactError` and its subclasses describe what PackProof could
or could not *do* with that answer: a target it will not connect to, bytes that do not
match the digest supplied for them, a workspace that refused a write. The download path
raises from both, and a caller that treats them as one taxonomy loses the distinction that
makes "the registry is down" and "this artifact is not the one described" different
answers to different questions.
"""

import os

#: How much of a registry-supplied value an error message may repeat. Same ceiling as the
#: reference parser's, so one bad field cannot size a PackProof error line either.
_MAX_ECHOED_VALUE = 120


def quote_value(value: object) -> str:
    """`ascii()` of a registry-controlled value, shortened.

    Used where a message has to show *what came back* — a field that failed validation, a
    name that is not the one asked for. `ascii()` rather than `repr()` because `repr()`
    leaves a printable non-ASCII character as it is, and a message carrying one cannot be
    encoded on the ASCII terminal this program does not configure. The length cap is why
    this is a function and not a call site: the value on the other side of a socket chose
    its own size.
    """
    rendered = ascii(value)
    if len(rendered) <= _MAX_ECHOED_VALUE:
        return rendered
    return f"{rendered[:_MAX_ECHOED_VALUE]}... ({len(rendered)} characters)"


def ascii_text(value: object) -> str:
    """`value` as text guaranteed ASCII, without the quotes `ascii()` would add.

    Used for things spliced *into* a sentence — a field name, a `ValidationError` message —
    where quotation marks would read as part of the name. `backslashreplace` escapes control
    characters, printable non-ASCII and the lone surrogates that a plain `encode("ascii")`
    would raise on, so nothing that arrives from a response can reach a terminal as an
    instruction. Real ASCII words pass through unchanged, which is why ordinary messages keep
    their shape.
    """
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


class RegistryError(Exception):
    """Base class for every way a registry call can fail.

    Catch this to catch all of them; the subclasses exist so a caller can react differently
    to "not published" and "could not reach the registry" without reading messages.
    """


class RegistryNetworkError(RegistryError):
    """The registry could not be reached, so no response was received at all.

    DNS failure, connection refused, a reset mid-handshake, a TLS problem. The underlying
    `OSError` is never repeated in the message, because the exception's own text can name a host
    or a file this process could not read.

    Whether it is chained at all is decided per client, and deliberately. The metadata client
    raises `from` the caught error, so `__cause__` carries the `OSError`. The artifact client
    raises outside its `except` clause, leaving both `__cause__` and `__context__` as `None`: on
    that path the caught object's own text carries the signed tarball URL, which is the thing the
    disclosure policy is about. One test pins each shape.
    """


class RegistryTimeoutError(RegistryError):
    """Nothing came back in time, in either of the two ways that happens.

    The first is the configured one: the request went out and `PACKPROOF_HTTP_TIMEOUT_SECONDS`
    expired before an answer arrived. The second is the kernel's: the connection to the registry
    timed out at `errno.ETIMEDOUT` before the request could be answered at all — which is why that
    case names the connection instead of the configured limit. The limit did not fire, and quoting
    it would send an operator to change a number that was not the reason.

    Kept apart from `RegistryNetworkError` because a timeout is the one failure an operator may
    reasonably want to retry with more patience. "Refused", "unreachable" and "reset" are not
    improved by waiting longer, and a class that covered all of them would make retry a guess.
    """


class RegistryHTTPError(RegistryError):
    """The registry answered with a status that is not a successful document.

    Carries the numeric status, which is the only part of the response line PackProof
    repeats: the reason phrase beside it is server-chosen text.
    """

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class RegistryRedirectError(RegistryHTTPError):
    """The registry answered with a redirect, and PackProof does not follow registry redirects.

    A redirect target is a URL supplied inside a registry response. Following it would mean
    PackProof asking a host nobody configured for the bytes it then reports provenance
    about, so the request stops here with the status recorded and the target unread.
    """


class RegistryResponseTooLargeError(RegistryError):
    """The metadata body exceeded `max_metadata_bytes` while it was being read.

    The limit is enforced against bytes actually received, not against a `Content-Length`
    header, so this fires whether the registry declares a huge body, a small one and then
    streams forever, or no size at all.
    """


class RegistryInvalidResponseError(RegistryError):
    """The registry answered, but what came back is not metadata PackProof can vouch for.

    Split into the two things a caller may want to tell apart: text that is not JSON at all
    (a login page, a proxy's error template, an empty body), and JSON that is well-formed
    but does not describe the package that was asked about in a shape the models accept.
    """


class RegistryNotJSONError(RegistryInvalidResponseError):
    """The response body could not be parsed as JSON."""


class RegistryMetadataError(RegistryInvalidResponseError):
    """The JSON was readable but the document was not a usable package record.

    Covers a missing `versions` map, a field of the wrong type, and the two mismatches that
    matter most: a document whose name is not the package asked for, and a version entry
    that describes a different version than the key it is filed under.
    """


class PackageNotFoundError(RegistryError):
    """The registry has no package by that name.

    npm answers this one with a 404, which is also how it answers several other problems,
    so the class means only what it says: the registry said it has no such package.
    """


class VersionNotFoundError(RegistryError):
    """The package exists; this exact version of it does not.

    Decided from the document that was retrieved, so no second request is needed to tell
    the two apart, and the count of published versions comes along to make "which half of
    this reference is wrong" answerable without naming versions a person did not ask for.
    """


def errno_text(error: OSError) -> str:
    """How an `OSError` may be described: its number resolved to the operating system's own words.

    `str(error)` is refused because it interleaves a filename, and on this path the filename is
    whatever the failing call reached — which for a workspace under attack is not the path anyone
    configured. `os.strerror()` returns the system's description of the *number*, so the message
    carries a fact about the failure rather than a path.

    Two numbers are named explicitly rather than left to falsiness: `errno is None` (an `OSError`
    built by hand, which the standard library permits) and `errno == 0` (a real error carrying no
    number). Both are false in Python, and both used to render as "no error happened".
    """
    code = error.errno if isinstance(error.errno, int) else None
    if code is None:
        return f"{type(error).__name__} carrying no errno"
    if code == 0:
        return f"{type(error).__name__} carrying errno 0 (unnamed)"
    try:
        return ascii_text(os.strerror(code))
    except Exception as rendering_error:  # noqa: BLE001 - a refusal must not fail to be worded
        return (
            f"{type(error).__name__} with errno {code} that could not be named "
            f"({type(rendering_error).__name__})"
        )


class ArtifactError(Exception):
    """Base for every way an artifact can fail to reach the workspace as verified bytes.

    Deliberately separate from `RegistryError`. The two answer different questions — "what did the
    registry say" and "what could PackProof do with the answer" — and the download path can raise
    from either family: an `HTTP 503` is a registry answer, a digest mismatch is not. A caller that
    wants every failure from one call catches both.
    """


class UnsupportedDownloadTargetError(ArtifactError):
    """The registry's `dist.tarball` is not an address PackProof will connect to.

    Raised by the policy layer before a request exists, so no connection was attempted and no
    credential could have been offered to it. The message names a refusal code from a closed
    vocabulary — the offending URL is never repeated, because the registry's own text is the
    untrusted input here and a signed query string in it is a capability, not a diagnosis.
    """


class UnexpectedDownloadOriginError(ArtifactError):
    """The transport was about to connect somewhere the policy did not approve.

    This is the guard firing, which means an earlier layer already made a mistake: the policy
    assembles the request from a constant host, so nothing PackProof built should ever reach the
    handler with a different origin. It exists because "the code above me checked" is not a
    control, and because the connection is the last point at which a wrong host is still fixable.

    Neither the offending host nor the URL appears in the message. A guard that prints the target
    it just refused prints a target a reader can paste into a browser.
    """


class ArtifactTooLargeError(ArtifactError):
    """The artifact body passed `max_download_bytes` while it was being read.

    A separate class from `RegistryResponseTooLargeError` rather than a shared one: that limit is
    a ceiling on a document held entirely in memory before a parser sees it, this one is a ceiling
    on bytes streamed to disk, and an operator changing the wrong one of the two would be told
    about the other. The enforcement mechanism is the same — counted against bytes that arrived,
    never against a `Content-Length` claim.
    """


class ArtifactStreamTruncatedError(ArtifactError):
    """The artifact response ended before the length it declared.

    Distinct from a connection reset, which is a network failure: this is a stream that closed
    cleanly and left fewer bytes behind than it promised, which is the shape of a response someone
    cut short rather than a path that broke.
    """


class ArtifactIntegrityError(ArtifactError):
    """The bytes that arrived are not the bytes the metadata described.

    Raised instead of returning an artifact, so "the download finished" can never be read as "the
    artifact matches" by anything downstream. The message names which claim failed to hold and how
    many other claims were checked; it does not carry either digest, because a report that prints
    expected-versus-actual digest bytes for a hostile artifact is printing the attacker's chosen
    strings into the terminal.
    """


class ArtifactWriteError(ArtifactError):
    """The publication attempt failed and nothing was published.

    The one artifact failure that means the workspace is unchanged, which is why it is worth its
    own class: `ArtifactCollisionError` and `WorkspaceDurabilityError` both describe a workspace
    that now holds something.
    """


class WorkspaceOutOfSpaceError(ArtifactWriteError):
    """The filesystem refused the bytes for a space or quota reason.

    Spelled separately because "make the package smaller" and "free space on that volume" are
    different instructions to an operator, and an `ENOSPC` reported as a generic write failure
    sends them checking the wrong thing.
    """


class WorkspaceEntryInvalidError(ArtifactWriteError):
    """The workspace is not something PackProof can safely write into.

    A descriptor-relative operation found the directory it holds open has been replaced, or a name
    it generated is already taken by something it did not create. Both mean the workspace and this
    process no longer agree about what is where, which is a condition to stop on rather than
    recover from by deleting entries.
    """


class ArtifactCollisionError(ArtifactError):
    """The artifact destination is held by something PackProof will not replace.

    Reached only from `errno.EEXIST` on the link, and only after the destination was inspected: a
    directory, a symlink, a socket, an unreadable regular file, or a regular file whose contents
    hash to something else. The bytes this operation streamed are discarded; the entry that was
    already there is left exactly where it was, because PackProof did not create it and therefore
    does not get to decide what happens to it.
    """


class WorkspaceDurabilityError(ArtifactError):
    """The artifact is committed; the workspace directory entry could not be synced.

    Carries `state` and `final_name` because the artifact exists and a later stage has to be told
    where. This is a durability warning about the *directory*, never a claim that the download
    failed: a caller that deleted the artifact on seeing this would be destroying the one thing
    the operation proved it had.
    """

    def __init__(self, message: str, *, state: str, final_name: str) -> None:
        super().__init__(message)
        self.state = state
        self.final_name = final_name


class ArtifactSettlementError(ArtifactError):
    """A link was started and its outcome could not be determined at all.

    Different from `NOTHING_PUBLISHED`, which is a known answer. This means the syscall may have
    created the destination and the read-only reconciliation could not say either way — the stat
    itself failed — so the workspace needs a human or a later sweep rather than PackProof acting
    on a guess.
    """


class ArtifactInternalError(ArtifactError):
    """Something PackProof did not predict happened, and it is not the artifact's fault.

    Wrapping an unexpected exception is how a bug in this tool gets reported as a package problem.
    This class is the honest version of that: it names the exception's class and nothing else, so
    the diagnosis says "PackProof hit something it did not anticipate" without turning the
    exception's own message — which may carry a path or a URL — into a claim about a package.
    """
