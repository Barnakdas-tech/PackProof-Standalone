r"""From an untrusted `dist.tarball` to a request URL PackProof is willing to connect to.

The registry writes this field. `schemas/models.py` already proved it is a well-formed HTTPS
URL — scheme, characters, no fragment, no credentials — because that is what a *record* of
someone else's claim needs, and a foreign host that exists is still worth reporting:
"npm says this version's tarball lives at example.com" is a finding, not a parse error.

Deciding to *download* it is a different question, and answering it with the same check would
make the finding and the capability the same thing. So nothing in this module trusts what the
model accepted, and the difference shows up in three places measured rather than assumed:

* **The parsed host is not the connected host.** `urlsplit("https://REGISTRY.NPMJS.ORG/a").hostname`
  is `registry.npmjs.org` — lowercased, userinfo-stripped — while `Request.host` for the same
  string is `REGISTRY.NPMJS.ORG`, and `https://@registry.npmjs.org/a` parses to the approved
  hostname with a netloc that begins with `@`. An allowlist keyed on `hostname` therefore
  approves both, and the connection goes somewhere else. The authority is approved by exact
  string equality against two spellings, and the request URL is rebuilt from the approved
  components so that what is asserted afterwards is what is dialled.
* **The port is part of the authority.** `Request.host` for `https://registry.npmjs.org:443/a`
  is `registry.npmjs.org:443` — the port rides along in the value the guard compares — and
  `parts.port` *raises* `ValueError` for `:abc` while `http.client` will happily dial `:0`.
  Reading the port is the one parse step that can throw, so it is read explicitly.
* **A control character does not survive parsing.** `urlsplit` removes a tab from the URL it
  was given: `https://host\<TAB>/a.tgz` comes back with the path `//a.tgz`. Checking the parsed
  parts for control characters would find none, so the raw string is checked first.

Refusals are codes from a closed vocabulary, never the URL. A signed npm tarball URL carries a
capability in its query string — whoever holds it can fetch the artifact without being the one
who was granted it — and the registry's chosen text is the untrusted input on this path. The
same reasoning covers success: `DownloadTarget` keeps the query because the request needs it,
and its `__repr__` prints the host and the path only. An accidental `print(target)` in a future
call site then cannot leak a token, which is a stronger property than a rule telling the next
person not to print it.

Nothing here imports a socket, opens a file or writes a log line, and no function raises for a
reason a caller could act on: a decision is returned, and the caller's single raising frame
decides what the operator hears. That is a control, not a style preference. A policy that
raised from inside this frame would put the URL it refused on the traceback it built, and every
frame between here and the caller would be a place the target could be found.
"""

import urllib.parse
from dataclasses import dataclass

from packproof.registry.npm_fetcher import NPM_REGISTRY_HOST
from packproof.schemas.models import PackageRef, PackageVersionMetadata

#: The only hosts this process will fetch package bytes from. Size one, on purpose: npm's own
#: metadata points tarballs at `registry.npmjs.org`, and a second entry would need a second
#: justification. Nothing reads an environment variable or a `Settings` field to widen it —
#: an allowlist an operator can extend at runtime is a prompt-injection target.
ALLOWED_DOWNLOAD_HOSTS = frozenset({NPM_REGISTRY_HOST})

#: The one port that may appear, and only as a redundant spelling of the default.
DEFAULT_HTTPS_PORT = 443

#: Every refusal this module can return. A refusal not in here is a bug in this module, which
#: is why the codes are a set rather than a comment: the tests enumerate it, so adding a code
#: without adding its case fails.
TARGET_REFUSAL_CODES = frozenset(
    {
        "unparseable-url",
        "not-https",
        "non-ascii-target",
        "control-character-target",
        "credentials-present",
        "fragment-present",
        "host-missing",
        "host-not-allowlisted",
        "port-unparseable",
        "port-not-https-default",
        "authority-not-exact",
        "path-missing",
    }
)

#: The authorities `Request.host` can be asked to dial for an approved host: the bare name, and
#: the same name with its default port spelled out. Both rebuild to the bare name.
_APPROVED_AUTHORITIES = frozenset({NPM_REGISTRY_HOST, f"{NPM_REGISTRY_HOST}:{DEFAULT_HTTPS_PORT}"})


@dataclass(frozen=True)
class TargetRefusal:
    """Why a target was refused: a code, and nothing a reader could paste into a browser.

    The field's vocabulary is closed by `TARGET_REFUSAL_CODES`, so this type carries no room to
    put a message together out of the input. A refusal that quoted the URL it refused would be a
    refusal that can be logged — and the log is the one channel this project cannot lock.
    """

    code: str

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, repr=False)
class DownloadTarget:
    """An approved request: the URL to dial, and the safe words to describe it with.

    `request_url` is the only field that can hold a signed query, and it is the only field the
    transport reads. Everything a report or an error message is built from is beside it —
    `host`, `path` and `signed` — so "describe the artifact's target" never requires printing
    the one component that must stay private.

    The dataclass is `repr=False` rather than left to the generated one deliberately: the
    generated `__repr__` lists every field including the URL, so a leak-free `repr` is not a
    property of the class until it is written.
    """

    request_url: str
    host: str
    path: str
    signed: bool
    reference: PackageRef

    def __repr__(self) -> str:
        query = "signed" if self.signed else "no query"
        return (
            f"DownloadTarget(host={self.host!r}, path={self.path!r}, {query}, "
            f"reference={self.reference.name!r}@{self.reference.version!r})"
        )


def plan_target(metadata: PackageVersionMetadata) -> DownloadTarget | TargetRefusal:
    """Turn one validated metadata record into an approved request, or into a code.

    Exactly one of the two is returned, always. The order of the checks is the order that names
    the most specific reason first: an authority carrying `@` is reported as credentials rather
    than as a non-matching string, because that is what the parse found, and the two spellings
    of the same bad URL should tell an operator the same thing every time.

    The path is passed through as the registry wrote it, including forms like `//other/x` and
    `/a/../../passwd`. That is not an oversight: the connect target is fixed by the authority
    check, no proxy is installed to reinterpret the request line, and this stage never turns a
    URL component into a filesystem name — which is the only place a traversal in the path
    could ever become a vulnerability. `DownloadTarget.path` exists so a report can say what
    was asked for; it is not, and must not become, an input to a filename.
    """
    raw = metadata.distribution.tarball_url

    # Checked against the raw string, before parsing: a control character can be silently
    # removed by the parser, so its absence afterwards proves nothing about what was sent.
    if not raw.isascii():
        return TargetRefusal("non-ascii-target")
    if any(character < " " or character == "\x7f" for character in raw):
        return TargetRefusal("control-character-target")
    if "#" in raw:
        # Raw again: an empty fragment is invisible in `parts.fragment` but is still the
        # syntax this stage refuses, and `urlunsplit` would silently drop it.
        return TargetRefusal("fragment-present")

    try:
        parts = urllib.parse.urlsplit(raw)
    except ValueError:
        # `https://[bad/a.tgz` — an unterminated IPv6 literal — dies here, before there are
        # parts to inspect. The registry's text, not a PackProof bug, so it becomes a code.
        return TargetRefusal("unparseable-url")
    try:
        port = parts.port
    except ValueError:
        # `:abc` and `:99999` raise here while `http.client` would dial `:0` without a word,
        # which is why the port is read explicitly rather than left to the authority check.
        return TargetRefusal("port-unparseable")

    if parts.scheme != "https":
        return TargetRefusal("not-https")
    if "@" in parts.netloc:
        return TargetRefusal("credentials-present")
    if parts.hostname is None:
        return TargetRefusal("host-missing")
    if parts.hostname not in ALLOWED_DOWNLOAD_HOSTS:
        return TargetRefusal("host-not-allowlisted")
    if port is not None and port != DEFAULT_HTTPS_PORT:
        return TargetRefusal("port-not-https-default")
    if parts.netloc not in _APPROVED_AUTHORITIES:
        # Reachable even though the hostname matched: case, userinfo, a trailing dot or an
        # empty port all leave `hostname` equal while `Request.host` would not be.
        return TargetRefusal("authority-not-exact")
    if not parts.path.startswith("/") or parts.path == "/":
        return TargetRefusal("path-missing")

    # Rebuilt, not reused. The tuple is the approved components — our scheme, our host, the
    # registry's path and query, and no fragment — which is what makes the equality the guard
    # later asserts on `Request.host` a fact rather than a hope.
    request_url = urllib.parse.urlunsplit(("https", NPM_REGISTRY_HOST, parts.path, parts.query, ""))
    return DownloadTarget(
        request_url=request_url,
        host=NPM_REGISTRY_HOST,
        path=parts.path,
        signed=bool(parts.query),
        reference=metadata.reference,
    )
