"""Provenance for machine-readable scan results.

A compliance finding is only comparable to another finding if you know what
produced it. Without that, two AIR Blackbox reports cannot be diffed with any
confidence: a rule change, an engine change, or a version bump all look
identical to a reader holding only the findings.

Every machine-readable result this server returns therefore carries a
`provenance` block:

    "provenance": {
      "engine": "builtin-rules",          # which scanner actually ran
      "scanner_version": "0.2.4",         # this MCP package
      "ruleset_id": "eu-ai-act-art9-15",  # which rule set
      "ruleset_version": "a1b2c3d4e5f6",  # content hash of the active rules
      "sdk_version": null                 # the SDK, when the SDK produced it
    }

Note on `engine`: it is a fixed property of each tool, not a runtime fallback.
The Tier 1-4 tools always run the built-in rule-based scanner; the Tier 5
tools always run the full air-blackbox SDK (or return an explicit error if it
is not installed). There is no silent switch between them.

Note on `sdk_version`: it names the SDK that *produced this result*, so it is
null for built-in results even when the SDK happens to be installed. Reporting
a version that contributed nothing would imply the SDK's rules ran.
"""
import hashlib
import os

from air_blackbox_mcp import __version__

# The rule set these checks implement. Bump the id (not just the hash) only if
# the scope changes - e.g. adding a different regulation alongside the AI Act.
RULESET_ID = "eu-ai-act-art9-15"

ENGINE_BUILTIN = "builtin-rules"
ENGINE_SDK = "air-blackbox-sdk"

_RULESET_VERSION_CACHE = None


def ruleset_version() -> str:
    """Content hash (SHA-256, first 12 hex) of the active built-in rule set.

    Hashed from scanner.py rather than maintained by hand, because a version
    string someone must remember to bump is a version string that eventually
    lies - and here it would lie about evidence.

    Deliberately conservative: it hashes the whole module, so ANY change to
    scanning logic changes the value, and an unrelated edit (a docstring, a
    helper) changes it too. Over-reporting a rule change is the safe
    direction: two results sharing a ruleset_version were produced by
    byte-identical rules.
    """
    global _RULESET_VERSION_CACHE
    if _RULESET_VERSION_CACHE is None:
        path = os.path.join(os.path.dirname(__file__), "scanner.py")
        try:
            with open(path, "rb") as f:
                _RULESET_VERSION_CACHE = hashlib.sha256(f.read()).hexdigest()[:12]
        except OSError:
            # Never fail a scan over provenance; say "unknown" rather than
            # claim a hash we could not compute.
            _RULESET_VERSION_CACHE = "unknown"
    return _RULESET_VERSION_CACHE


def sdk_version():
    """Installed air-blackbox SDK version, or None if it is not installed."""
    try:
        from importlib.metadata import version
    except ImportError:  # pragma: no cover - Python < 3.8
        return None
    try:
        return version("air-blackbox")
    except Exception:
        return None


def provenance(engine: str = ENGINE_BUILTIN) -> dict:
    """Build the provenance block for a result produced by `engine`."""
    return {
        "engine": engine,
        "scanner_version": __version__,
        "ruleset_id": RULESET_ID,
        "ruleset_version": ruleset_version(),
        "sdk_version": sdk_version() if engine == ENGINE_SDK else None,
    }


def stamp(result: dict, engine: str = ENGINE_BUILTIN) -> dict:
    """Attach provenance to a result dict in place and return it.

    Applied to error results too: knowing which version produced an error is
    as useful as knowing which version produced a finding.
    """
    result["provenance"] = provenance(engine)
    return result
