"""Scanning through the air-blackbox SDK, in the built-in scanner's shape.

Used for scan_code, scan_file and scan_project whenever the SDK is
installed. The result dicts have the same keys as air_blackbox_mcp.scanner
produces (frameworks, trust_layer_detected, findings, summary, plus file
info), so a client sees one shape whichever engine ran; `provenance.engine`
says which, and `language` says what the SDK read the code as.

Severity is derived from status as the built-in scanner does (fail HIGH,
warn MEDIUM, pass LOW). Findings outside EU AI Act articles 9-15 (the US
hiring checks the SDK runs for hiring code) carry article "US-HIRING".
"""
import os

from air_blackbox_mcp.provenance import ENGINE_SDK, stamp
from air_blackbox_mcp.scanner import detect_trust_layer

_SEVERITY = {"fail": "HIGH", "warn": "MEDIUM", "pass": "LOW"}


def available() -> bool:
    """True when an SDK with snippet scanning (>= 1.17) is importable."""
    try:
        from air_blackbox.compliance import snippet  # noqa: F401
    except ImportError:
        return False
    return True


def _finding(f: dict) -> dict:
    article = f.get("article")
    if f.get("framework") == "us_hiring" or article is None:
        article = "US-HIRING"
    return {
        "article": article,
        "name": f["name"],
        "status": f["status"],
        "severity": _SEVERITY.get(f["status"], "MEDIUM"),
        "evidence": f["evidence"],
        "fix_hint": f.get("fix_hint", ""),
    }


def _summary(findings: list) -> dict:
    total = len(findings)
    passing = sum(1 for f in findings if f["status"] == "pass")
    return {
        "total_checks": total,
        "passing": passing,
        "warnings": sum(1 for f in findings if f["status"] == "warn"),
        "failing": sum(1 for f in findings if f["status"] == "fail"),
        "score": f"{passing}/{total}",
    }


def scan_code(code: str, language: str = "", filename: str = "") -> dict:
    """Scan one piece of code; language stated, from the filename, or detected."""
    from air_blackbox.compliance.snippet import scan_snippet

    try:
        result = scan_snippet(code, language=language or None, filename=filename or None)
    except ValueError as e:
        return stamp({"error": str(e)}, ENGINE_SDK)
    findings = [_finding(f) for f in result["findings"] if f["name"] != "Source files"]
    return stamp({
        "language": result["language"],
        "language_source": result["language_source"],
        "frameworks": result["frameworks"],
        "trust_layer_detected": detect_trust_layer(code),
        "findings": findings,
        "summary": _summary(findings),
    }, ENGINE_SDK)


def scan_file(file_path: str) -> dict:
    """Read and scan a single source file."""
    if not os.path.exists(file_path):
        return stamp({"error": f"File not found: {file_path}"}, ENGINE_SDK)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()
    result = scan_code(code, filename=os.path.basename(file_path))
    result["file"] = file_path
    result["lines"] = len(code.splitlines())
    return result


def scan_project(directory: str) -> dict:
    """Scan a directory exactly as `air-blackbox comply --scan` does."""
    from air_blackbox.compliance.code_scanner import _find_source_files, scan_codebase
    from air_blackbox.compliance.engine import detect_frameworks
    from air_blackbox.compliance.sources import language_of, read_sources

    if not os.path.isdir(directory):
        return stamp({"error": f"Directory not found: {directory}"}, ENGINE_SDK)
    paths = _find_source_files(directory)
    if not paths:
        return stamp({"error": f"No source files found in {directory}",
                      "files_scanned": 0}, ENGINE_SDK)
    sources = read_sources(paths)
    findings = [_finding(f.__dict__) for f in scan_codebase(directory) if f.name != "Source files"]
    files = [{"file": os.path.relpath(p, directory), "language": language_of(p),
              "lines": len(src.splitlines())} for p, src in sources.items()]
    result = stamp({
        "frameworks": detect_frameworks(directory),
        "trust_layer_detected": any(detect_trust_layer(src) for src in sources.values()),
        "findings": findings,
        "summary": _summary(findings),
    }, ENGINE_SDK)
    result["directory"] = directory
    result["files_scanned"] = len(files)
    result["files"] = files[:20]  # Cap at 20 to keep response size manageable
    return result
