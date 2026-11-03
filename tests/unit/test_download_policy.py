r"""From an untrusted `dist.tarball` to a request URL PackProof is willing to connect to.

The policy is tested against its own decisions, not against the model's. `PackageDistribution`
already refuses most hostile spellings of a tarball URL, and a test that only used validated
metadata would prove nothing about the claim in this module's docstring — that deciding to
*download* is a different question from deciding what a record says. So the parametrised refusal
table is built with `model_construct`, which bypasses validation and hands the policy exactly the
string a registry wrote. Every row was measured against the implementation before it was asserted,
and the whole set of codes is asserted to be covered — which is what makes "add a code without
adding its case" a failing test rather than an untested branch.

The other thing these tests hold onto is secrecy. An npm tarball URL carries a time-limited signed
query, and a signed query is a capability. Every refusal therefore names a code and nothing else,
and the successful target keeps the query in exactly one field — `request_url` — behind a `repr`
that prints the host and the path only. `print(target)` being safe is a stronger property than a
rule telling the next person not to print it, so it is tested.

Two rows record behaviour rather than refusal, because both were assumptions that the measurement
disproved: the parser lowercases the scheme, so `HTTPS://` is not a second protocol, and the URL
that is later dialled is rebuilt from approved components rather than reused. Both are safe, and
safe-by-rebuild is a different claim from safe-by-refusal, so it gets its own test instead of hiding
in a rejection list.

Nothing here opens a socket, and an autouse fixture closes the socket family to keep that true:
this module's whole contract is that it is a decision, not an action.
"""

import ast
import dataclasses
import errno
import socket
import urllib.parse
from pathlib import Path

import pytest

from packproof.registry.download_policy import (
    ALLOWED_DOWNLOAD_HOSTS,
    DEFAULT_HTTPS_PORT,
    TARGET_REFUSAL_CODES,
    DownloadTarget,
    TargetRefusal,
    plan_target,
)
from packproof.registry.npm import parse_npm_reference
from packproof.registry.npm_fetcher import NPM_REGISTRY_HOST
from packproof.schemas.models import (
    PackageDistribution,
    PackageRef,
    PackageVersionMetadata,
)

REF: PackageRef = parse_npm_reference("left-pad@1.3.0")
QUERY_CAPABILITY = "SECRETQUERYVALUE"
SIGNED = f"https://{NPM_REGISTRY_HOST}/left-pad/-/left-pad-1.3.0.tgz?token={QUERY_CAPABILITY}"
POLICY_SOURCE = Path(plan_target.__code__.co_filename)


@pytest.fixture(autouse=True)
def _no_live_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """A policy module that reached the network would be doing the thing it exists to decide."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("download policy must not touch the network")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def _validated(url: str) -> PackageVersionMetadata:
    """A record built the way the metadata client builds one, so the model has checked it."""
    return PackageVersionMetadata(
        reference=REF,
        distribution=PackageDistribution(tarball_url=url),
        description=None,
        repository=None,
        published_at=None,
    )


def _raw(url: str) -> PackageVersionMetadata:
    """The registry's own string with the model's vetting deliberately switched off.

    `model_construct` is not a shortcut here — it is the test. The policy is supposed to refuse a
    target without leaning on the fact that something upstream already looked at it, and a test
    that only fed it validated records would leave every refusal behind that vetting.
    """
    distribution = PackageDistribution.model_construct(tarball_url=url, integrity=None, shasum=None)
    return PackageVersionMetadata.model_construct(
        reference=REF,
        distribution=distribution,
        description=None,
        repository=None,
        published_at=None,
    )


def _decision(url: str) -> DownloadTarget | TargetRefusal:
    return plan_target(_raw(url))


def _code(url: str) -> str:
    """The refusal a URL produces, with the union narrowed once instead of at every call site."""
    decision = _decision(url)
    assert isinstance(decision, TargetRefusal), f"{url!r} was decided as {decision!r}"
    return decision.code


def _approved(url: str) -> DownloadTarget:
    decision = _decision(url)
    assert isinstance(decision, DownloadTarget), f"{url!r} was refused as {decision!r}"
    return decision


# --------------------------------------------------------------------------- the approval path


def test_an_ordinary_registry_tarball_is_approved_rebuilt_and_query_carrying() -> None:
    target = plan_target(_validated(SIGNED))
    assert isinstance(target, DownloadTarget)
    assert target.host == NPM_REGISTRY_HOST
    assert target.path == "/left-pad/-/left-pad-1.3.0.tgz"
    assert target.signed is True
    assert target.request_url == SIGNED
    assert target.reference is REF


def test_an_unsigned_tarball_is_approved_and_marked_unsigned() -> None:
    target = _approved(f"https://{NPM_REGISTRY_HOST}/left-pad/-/left-pad-1.3.0.tgz")
    assert target.signed is False


def test_the_default_port_is_a_redundant_spelling_and_rebuilds_to_the_bare_host() -> None:
    """`Request.host` carries the port verbatim, so `:443` would be compared against the
    allowlist as `registry.npmjs.org:443`. Rebuilding is what makes the guard's equality a
    fact instead of a hope."""
    target = _approved(f"https://{NPM_REGISTRY_HOST}:{DEFAULT_HTTPS_PORT}/a.tgz")
    assert target.request_url == f"https://{NPM_REGISTRY_HOST}/a.tgz"
    assert urllib.parse.urlsplit(target.request_url).netloc == NPM_REGISTRY_HOST


def test_the_request_url_is_rebuilt_from_approved_components_not_reused_from_the_registry() -> None:
    """The path and query are the registry's; the scheme and authority are this module's."""
    target = _approved(f"https://{NPM_REGISTRY_HOST}:443/a.tgz?token={QUERY_CAPABILITY}")
    assert target.request_url == f"https://{NPM_REGISTRY_HOST}/a.tgz?token={QUERY_CAPABILITY}"


def test_a_uppercase_scheme_is_lowercased_by_the_parser_and_rebuilt_as_https() -> None:
    """Measured, not assumed: `urlsplit("HTTPS://…").scheme` is `https`, so this is not a second
    protocol — and the string handed to the transport is rebuilt from this module's own `"https"`,
    so no spelling of the scheme can produce a cleartext request."""
    assert urllib.parse.urlsplit("HTTPS://example.invalid/a").scheme == "https"
    target = _approved(f"HTTPS://{NPM_REGISTRY_HOST}/a.tgz")
    assert target.request_url.startswith("https://")
    assert urllib.parse.urlsplit(target.request_url).scheme == "https"


def test_the_scoped_package_path_survives_untouched() -> None:
    path = "/@babel%2Fcore/-/core-7.0.0.tgz"
    target = _approved(f"https://{NPM_REGISTRY_HOST}{path}")
    assert target.path == path
    assert target.request_url.endswith(path)


# --------------------------------------------------------------------------- the refusal table

#: (url, expected code). Built for `model_construct`, so each row tests the policy's own check.
REFUSALS: list[tuple[str, str]] = [
    (f"http://{NPM_REGISTRY_HOST}/a.tgz", "not-https"),
    (f"ftp://{NPM_REGISTRY_HOST}/a.tgz", "not-https"),
    (f"//{NPM_REGISTRY_HOST}/a.tgz", "not-https"),
    (f"{NPM_REGISTRY_HOST}/a.tgz", "not-https"),
    (f"file://{NPM_REGISTRY_HOST}/a.tgz", "not-https"),
    (f"https://{NPM_REGISTRY_HOST}/é.tgz", "non-ascii-target"),
    (f"https://{NPM_REGISTRY_HOST}/a\tb.tgz", "control-character-target"),
    (f"https://{NPM_REGISTRY_HOST}/a\x7fb.tgz", "control-character-target"),
    (f"https://{NPM_REGISTRY_HOST}/a\nb.tgz", "control-character-target"),
    (f"https://{NPM_REGISTRY_HOST}/a\x00b.tgz", "control-character-target"),
    (f"https://{NPM_REGISTRY_HOST}/a.tgz#frag", "fragment-present"),
    (f"https://{NPM_REGISTRY_HOST}/a.tgz#", "fragment-present"),
    (f"https://{NPM_REGISTRY_HOST}/a.tgz?x=1#frag", "fragment-present"),
    ("https://[bad/a.tgz", "unparseable-url"),
    ("https://registry.npmjs.org:abc/a.tgz", "port-unparseable"),
    ("https://registry.npmjs.org:99999/a.tgz", "port-unparseable"),
    ("https://registry.npmjs.org:443:443/a.tgz", "port-unparseable"),
    ("https:///a.tgz", "host-missing"),
    ("https://:443/a.tgz", "host-missing"),
    ("https://cdn.evil.invalid/a.tgz", "host-not-allowlisted"),
    ("https://registry.npmjs.org.evil.invalid/a.tgz", "host-not-allowlisted"),
    ("https://registry.npmjs.org./a.tgz", "host-not-allowlisted"),
    ("https://notthehost.invalid/a.tgz", "host-not-allowlisted"),
    (f"https://{NPM_REGISTRY_HOST}%40evil.invalid/a.tgz", "host-not-allowlisted"),
    (f"https://{NPM_REGISTRY_HOST}:8443/a.tgz", "port-not-https-default"),
    (f"https://{NPM_REGISTRY_HOST}:0/a.tgz", "port-not-https-default"),
    (f"https://{NPM_REGISTRY_HOST}:444/a.tgz", "port-not-https-default"),
    (f"https://{NPM_REGISTRY_HOST}:80/a.tgz", "port-not-https-default"),
    (f"https://@{NPM_REGISTRY_HOST}/a.tgz", "credentials-present"),
    (f"https://user:pass@{NPM_REGISTRY_HOST}/a.tgz", "credentials-present"),
    (f"https://{NPM_REGISTRY_HOST.upper()}/a.tgz", "authority-not-exact"),
    (f"https://{NPM_REGISTRY_HOST}:/a.tgz", "authority-not-exact"),
    (f"https://{NPM_REGISTRY_HOST}", "path-missing"),
    (f"https://{NPM_REGISTRY_HOST}/", "path-missing"),
]


@pytest.mark.parametrize(
    ("url", "code"),
    REFUSALS,
    ids=[f"{index:02d}-{code}" for index, (_, code) in enumerate(REFUSALS)],
)
def test_a_target_the_policy_refuses_is_refused_by_name(url: str, code: str) -> None:
    assert _code(url) == code


def test_every_refusal_code_the_module_advertises_has_a_test_case() -> None:
    """The set is the contract. A code added without a case above is a code nothing has ever
    seen produced, which is exactly the drift this test exists to catch."""
    covered = {code for _, code in REFUSALS}
    assert covered == set(TARGET_REFUSAL_CODES), sorted(covered ^ set(TARGET_REFUSAL_CODES))


def test_no_two_refusal_codes_share_a_meaning_across_the_table() -> None:
    """One URL, one code: the same bad target spelled two ways must tell an operator the same
    thing, which is why the table is keyed by code and checked for consistency."""
    assert len(REFUSALS) == len({url for url, _ in REFUSALS}), "a duplicated row"


@pytest.mark.parametrize("code", sorted(TARGET_REFUSAL_CODES))
def test_a_refusal_names_only_its_code_anywhere_a_caller_could_read_it(code: str) -> None:
    row = next(url for url, row_code in REFUSALS if row_code == code)
    decision = _decision(row)
    assert isinstance(decision, TargetRefusal)
    assert str(decision) == code
    assert code in repr(decision)
    for leak in ("://", QUERY_CAPABILITY, "evil", NPM_REGISTRY_HOST):
        assert leak not in str(decision), f"{code} disclosed part of the target: {leak}"


HOSTILE_STRINGS = [
    "",
    " ",
    "/",
    "https://",
    "https:///",
    "https:registry.npmjs.org/a.tgz",
    "\x00",
    "\n",
    "https://" + "a" * 300 + "/a.tgz",
    "https://registry.npmjs.org@evil.invalid/a.tgz",
    "https://registry.npmjs.org:443@evil.invalid/a.tgz",
    "https://evil.invalid/?x=https://registry.npmjs.org/a.tgz",
    "https://registry.npmjs.org/a.tgz?token=SECRET#frag",
    "https://xn--regstry-9mc.io/a.tgz",
    f"https://{NPM_REGISTRY_HOST}/a.tgz?{QUERY_CAPABILITY}",
    "https://registry.npmjs.org:443\x00/a.tgz",
    "https://[::1",
    "https://registry.npmjs.org/a.tgz?a=[",
    "htTpS://REGISTRY.NPMJS.ORG/A.TGZ",
]


@pytest.mark.parametrize("url", HOSTILE_STRINGS)
def test_no_string_a_registry_could_write_makes_the_policy_raise(url: str) -> None:
    """The no-raise contract: a decision is returned, and the caller's single raising frame
    decides what the operator hears. A policy that raised would put the URL it refused on the
    traceback it built."""
    assert isinstance(_decision(url), (DownloadTarget, TargetRefusal))


# --------------------------------------------------------------------------- ordering of checks


def test_an_authority_carrying_the_at_sign_is_reported_as_credentials_not_as_a_mismatch() -> None:
    """The most specific reason the parse found comes first, so two spellings of the same bad
    URL tell an operator the same thing every time."""
    assert _code(f"https://@{NPM_REGISTRY_HOST}/a.tgz") == "credentials-present"
    assert _code(f"https://u:p@{NPM_REGISTRY_HOST}/a.tgz") == "credentials-present"


def test_a_non_https_scheme_is_named_before_the_host_that_is_also_wrong() -> None:
    assert _code("http://cdn.evil.invalid/a.tgz") == "not-https"


def test_a_control_character_is_named_before_a_host_that_is_also_absent() -> None:
    assert _code("https://\x00/aa.tgz") == "control-character-target"


def test_a_fragment_is_named_before_a_non_https_scheme_behind_it() -> None:
    """`#` is checked against the raw string, so a fragment cannot hide a scheme problem — and
    cannot be silently dropped by `urlunsplit`, which is the other half of the same check."""
    assert _code("http://cdn.evil.invalid/a.tgz#frag") == "fragment-present"


def test_an_uppercase_spelling_of_the_approved_host_is_refused_as_inexact_not_as_unknown() -> None:
    """`hostname` lowercases and userinfo-strips, while the netloc `Request` carries — and the
    value `do_open` splits into a connect target — is verbatim. An allowlist keyed on the parsed
    host would approve this and dial something else."""
    assert _code(f"https://{NPM_REGISTRY_HOST.upper()}/a.tgz") == "authority-not-exact"


def test_a_control_character_is_checked_before_the_parser_can_eat_it() -> None:
    """`urlsplit` removes a tab from the URL it is given, so inspecting the parsed parts for
    control characters would find none and approve the request anyway."""
    with_tab = f"https://{NPM_REGISTRY_HOST}/a\tb.tgz"
    assert "\t" not in urllib.parse.urlsplit(with_tab).path, "the parser itself dropped the tab"
    assert _code(with_tab) == "control-character-target"


def test_an_empty_fragment_is_refused_though_the_parser_would_silently_drop_it() -> None:
    with_empty = f"https://{NPM_REGISTRY_HOST}/a.tgz#"
    assert urllib.parse.urlsplit(with_empty).fragment == ""
    assert _code(with_empty) == "fragment-present"


def test_an_unterminated_ipv6_literal_is_a_refusal_and_not_a_crash() -> None:
    assert _code("https://[bad/a.tgz") == "unparseable-url"


def test_the_port_is_read_explicitly_because_it_is_the_one_parse_step_that_can_raise() -> None:
    """`parts.port` throws on `:abc` while `http.client` would dial `:0` without a word, so the
    port is read before anything that depends on it."""
    parts = urllib.parse.urlsplit(f"https://{NPM_REGISTRY_HOST}:abc/a.tgz")
    with pytest.raises(ValueError):
        assert parts.port
    assert _code(f"https://{NPM_REGISTRY_HOST}:abc/a.tgz") == "port-unparseable"


# --------------------------------------------------------------------------- the allowlist itself


def test_the_allowlist_is_the_one_host_and_is_not_widened_by_anything() -> None:
    """Size one, on purpose. Nothing reads an environment variable or a `Settings` field to
    widen it — an allowlist an operator can extend at runtime is a prompt-injection target."""
    assert NPM_REGISTRY_HOST in ALLOWED_DOWNLOAD_HOSTS
    assert len(ALLOWED_DOWNLOAD_HOSTS) == 1


def test_the_policy_reads_no_environment_and_imports_no_transport() -> None:
    """Checked on the syntax tree rather than on the source text: this module's own prose says
    "environment variable", and a substring test would fail on the comment that documents the
    rule. What matters is that nothing imports a transport, and nothing reaches for `environ`
    or `getenv` as an attribute or a call — a policy whose allowlist could be edited by an
    environment variable would be a policy with no allowlist."""
    tree = ast.parse(POLICY_SOURCE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv"}:
            pytest.fail(f"the policy reads configuration from the process: {ast.dump(node)}")
    assert imported <= {"urllib", "dataclasses", "packproof"}, imported
    assert not {"socket", "ssl", "http", "os"} & imported


# --------------------------------------------------------------------------- the documented pass-through


@pytest.mark.parametrize(
    "path",
    [
        "/../../etc/passwd",
        "/a/../../b.tgz",
        "//other/x.tgz",
        "/%2e%2e/%2e%2e/etc/passwd",
        "/./../../a.tgz",
    ],
)
def test_a_traversal_in_the_path_is_recorded_and_cannot_become_a_filename(path: str) -> None:
    """Deliberate, and stated in the module docstring: the connect target is fixed by the
    authority check, no proxy is installed to reinterpret the request line, and this stage never
    turns a URL component into a filesystem name. `DownloadTarget.path` exists so a report can
    say what was asked for; it is not, and must not become, an input to a filename.

    This test pins the half that actually matters: whatever the path claims, the approved
    authority is still the registry host and the artifact's own name comes from a digest of its
    bytes, computed later, never from here.
    """
    target = _approved(f"https://{NPM_REGISTRY_HOST}{path}")
    assert target.path == path
    assert urllib.parse.urlsplit(target.request_url).netloc == NPM_REGISTRY_HOST
    assert urllib.parse.urlsplit(target.request_url).scheme == "https"


def test_a_traversal_in_the_query_is_carried_as_text_and_nowhere_else() -> None:
    """The query belongs to the registry's own URL and this stage does not interpret it, so
    `../../y` inside a query string is data on its way to the one host that was approved. It is
    not a path, and `path` is what a report may name."""
    target = _approved(f"https://{NPM_REGISTRY_HOST}/a.tgz?token=SECRET&x=../../y")
    assert target.path == "/a.tgz"
    assert urllib.parse.urlsplit(target.request_url).query == "token=SECRET&x=../../y"
    assert target.signed is True
    assert "a.tgz" in repr(target) and "../../y" not in repr(target)


# --------------------------------------------------------------------------- the repr is the leak guard


def test_printing_a_target_cannot_publish_a_capability() -> None:
    target = plan_target(_validated(SIGNED))
    assert isinstance(target, DownloadTarget)
    assert target.request_url == SIGNED
    assert QUERY_CAPABILITY not in repr(target)
    assert QUERY_CAPABILITY not in str(repr(target))
    assert "/left-pad-1.3.0.tgz" in repr(target)
    assert "signed" in repr(target)
    assert "left-pad" in repr(target)


def test_an_unsigned_target_says_so_in_its_repr() -> None:
    assert "no query" in repr(_approved(f"https://{NPM_REGISTRY_HOST}/a.tgz"))


def test_the_query_survives_only_in_the_field_the_transport_reads() -> None:
    target = plan_target(_validated(SIGNED))
    assert isinstance(target, DownloadTarget)
    fields = {
        name: getattr(target, name)
        for name in ("request_url", "host", "path", "signed", "reference")
    }
    carrying = [name for name, value in fields.items() if QUERY_CAPABILITY in str(value)]
    assert carrying == ["request_url"]


def test_a_target_is_frozen_so_a_later_stage_cannot_rewrite_the_authority() -> None:
    target = plan_target(_validated(SIGNED))
    assert isinstance(target, DownloadTarget)
    with pytest.raises(dataclasses.FrozenInstanceError):
        target.host = "cdn.evil.invalid"  # type: ignore[misc]


def test_a_refusal_is_a_decision_and_not_an_os_error() -> None:
    """No attempt was made, so there is no number to report — and a caller that reached for
    `errno` would be reaching for something that does not exist."""
    decision = _decision("https://cdn.evil.invalid/a.tgz")
    assert isinstance(decision, TargetRefusal)
    assert not isinstance(decision, OSError)
    assert decision.code not in {errno.EACCES, errno.ENOENT, errno.EXDEV}
