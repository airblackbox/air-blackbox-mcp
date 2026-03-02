"""
EU AI Act compliance scanner for Python AI agent code.
Checks Articles 9, 10, 11, 12, 14, and 15.
Tool-agnostic: detects real compliance patterns, not specific products.
"""

import re
import os
import json


# --- Framework Detection ---

FRAMEWORK_PATTERNS = {
    "langchain": [
        r"from\s+langchain", r"import\s+langchain",
        r"from\s+langchain_openai", r"from\s+langchain_core",
        r"from\s+langgraph", r"AgentExecutor", r"create_openai_tools_agent",
    ],
    "crewai": [
        r"from\s+crewai", r"import\s+crewai", r"Crew\s*\(", r"Agent\s*\(",
    ],
    "autogen": [
        r"from\s+autogen", r"import\s+autogen", r"from\s+ag2",
        r"AssistantAgent", r"UserProxyAgent", r"GroupChat",
    ],
    "openai": [
        r"from\s+openai", r"import\s+openai", r"OpenAI\s*\(",
        r"client\.chat\.completions", r"function_call", r"tools\s*=",
    ],
    "rag": [
        r"VectorStore", r"Chroma\s*\(", r"FAISS", r"Pinecone",
        r"retriever", r"RetrievalQA", r"from\s+llama_index",
    ],
    "huggingface": [
        r"from\s+transformers", r"import\s+transformers",
        r"AutoModel", r"pipeline\s*\(",
    ],
    "anthropic": [
        r"from\s+anthropic", r"import\s+anthropic", r"Anthropic\s*\(",
    ],
}

# --- Compliance Pattern Detection ---

RISK_PATTERNS = {
    "risk_classification": [
        r"risk[_\s]?level", r"risk[_\s]?class", r"risk[_\s]?score",
        r"risk[_\s]?categor", r"risk[_\s]?assess", r"RiskLevel",
        r"CRITICAL|HIGH|MEDIUM|LOW",
    ],
    "access_control": [
        r"@(?:requires?_auth|login_required|permission)", r"role[_\s]?based",
        r"rbac", r"access[_\s]?control", r"(?:is_)?authorized",
        r"authenticate", r"permission",
    ],
    "risk_audit": [
        r"risk.*(?:log|audit|record|review)", r"(?:log|audit|record).*risk",
    ],
}

DATA_PATTERNS = {
    "input_validation": [
        r"(?:from|import)\s+(?:pydantic|marshmallow|cerberus|jsonschema)",
        r"BaseModel", r"Schema\s*\(", r"validate\s*\(", r"@validator",
        r"field_validator", r"model_validator",
    ],
    "pii_handling": [
        r"pii", r"(?:redact|mask|anonymize|pseudonymize|tokenize)\s*\(",
        r"sensitive[_\s]?data", r"personal[_\s]?data", r"gdpr", r"encrypt",
    ],
    "data_schemas": [
        r"(?:from|import)\s+(?:dataclasses|attrs|pydantic|marshmallow)",
        r"@dataclass", r"TypedDict", r"NamedTuple",
    ],
    "data_provenance": [
        r"provenance", r"data[_\s]?source", r"data[_\s]?lineage", r"metadata",
    ],
}

DOC_PATTERNS = {
    "logging": [
        r"(?:from|import)\s+(?:logging|structlog|loguru)",
        r"logging\.(?:get)?[Ll]ogger", r"logger\.\w+\(",
        r"structlog\.get_logger",
    ],
    "documentation": [
        r'"""[\s\S]*?"""', r"'''[\s\S]*?'''",
    ],
    "type_hints": [
        r"def\s+\w+\([^)]*:\s*\w+", r"->\s*\w+",
        r":\s*(?:str|int|float|bool|list|dict|Optional|Union|List|Dict)",
    ],
}

RECORD_PATTERNS = {
    "structured_logging": [
        r"structlog", r"json[_\s]?log", r"log.*format.*json",
        r"logging\.handlers", r"extra\s*=\s*\{",
    ],
    "audit_trail": [
        r"audit[_\s]?(?:log|trail|record|entry|event)",
        r"(?:event|action)[_\s]?(?:log|record|store|write)",
    ],
    "timestamps": [
        r"datetime\.(?:now|utcnow)", r"time\.time\(\)", r"timestamp",
        r"created_at", r"updated_at", r"isoformat",
    ],
    "log_integrity": [
        r"hmac", r"hashlib", r"sha256", r"digest",
        r"tamper[_\s]?(?:evident|proof)", r"append[_\s]?only",
    ],
}

OVERSIGHT_PATTERNS = {
    "human_review": [
        r"human[_\s]?(?:in[_\s]?the[_\s]?loop|review|approval|oversight)",
        r"hitl", r"manual[_\s]?review", r"requires?[_\s]?approval",
        r"moderator", r"review[_\s]?(?:queue|status|step)",
    ],
    "override_mechanism": [
        r"(?:kill[_\s]?switch|emergency[_\s]?stop|circuit[_\s]?breaker)",
        r"(?:override|disable|pause|halt|abort|shutdown)\s*\(",
        r"safe[_\s]?mode", r"fallback",
    ],
    "notification": [
        r"notify|alert|warn|escalat",
        r"send[_\s]?(?:alert|notification|email|slack|webhook)",
    ],
}

SECURITY_PATTERNS = {
    "input_sanitization": [
        r"sanitiz", r"escape\s*\(", r"strip[_\s]?tags",
        r"injection", r"(?:prompt|sql|xss)[_\s]?(?:inject|attack|defense|detect|guard)",
        r"allow[_\s]?list|block[_\s]?list",
    ],
    "error_handling": [
        r"try\s*:", r"except\s+\w+", r"raise\s+\w+Error",
        r"finally\s*:", r"error[_\s]?handl",
    ],
    "testing": [
        r"(?:from|import)\s+(?:pytest|unittest|hypothesis)",
        r"def\s+test_", r"class\s+Test\w+", r"assert\s+",
    ],
    "rate_limiting": [
        r"rate[_\s]?limit", r"throttl", r"(?:max|limit)[_\s]?requests",
        r"cooldown", r"backoff",
    ],
}


def detect_framework(code: str) -> str:
    """Detect which AI agent framework the code uses."""
    scores = {}
    for framework, patterns in FRAMEWORK_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, code))
        if score > 0:
            scores[framework] = score
    if not scores:
        return "unknown"
    return max(scores, key=scores.get)


def detect_all_frameworks(code: str) -> list:
    """Detect all frameworks present in the code."""
    found = []
    for framework, patterns in FRAMEWORK_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, code))
        if score > 0:
            found.append(framework)
    return found or ["unknown"]


def _check_patterns(code: str, patterns: list) -> bool:
    return any(re.search(p, code, re.IGNORECASE | re.MULTILINE) for p in patterns)


def _find_evidence(code: str, patterns: list) -> list:
    evidence = []
    for p in patterns:
        for m in re.finditer(p, code, re.IGNORECASE | re.MULTILINE):
            line_num = code[:m.start()].count('\n') + 1
            evidence.append(f"Line {line_num}: {m.group()[:80]}")
    return evidence[:5]


def check_article_9(code: str) -> dict:
    has_risk = _check_patterns(code, RISK_PATTERNS["risk_classification"])
    has_access = _check_patterns(code, RISK_PATTERNS["access_control"])
    has_audit = _check_patterns(code, RISK_PATTERNS["risk_audit"])
    checks_passed = sum([has_risk, has_access, has_audit])
    passed = checks_passed >= 1
    severity = "LOW" if checks_passed >= 2 else ("MEDIUM" if checks_passed == 1 else "HIGH")
    return {
        "article": 9, "title": "Risk Management System", "passed": passed,
        "severity": severity,
        "checks": {"risk_classification": has_risk, "access_control": has_access, "risk_audit": has_audit},
        "evidence": (_find_evidence(code, RISK_PATTERNS["risk_classification"]) + _find_evidence(code, RISK_PATTERNS["access_control"]))[:3],
        "finding": f"Risk management patterns detected ({checks_passed}/3 checks pass)." if passed else
            "No risk classification or access control detected. Article 9 requires identifying and analyzing known risks with measures proportionate to risk level.",
        "fix": "Add risk classification (e.g., risk levels for operations), access control (e.g., role-based permissions), and risk logging." if not passed else None,
    }


def check_article_10(code: str) -> dict:
    has_validation = _check_patterns(code, DATA_PATTERNS["input_validation"])
    has_pii = _check_patterns(code, DATA_PATTERNS["pii_handling"])
    has_schemas = _check_patterns(code, DATA_PATTERNS["data_schemas"])
    has_provenance = _check_patterns(code, DATA_PATTERNS["data_provenance"])
    checks_passed = sum([has_validation, has_pii, has_schemas, has_provenance])
    passed = checks_passed >= 1
    severity = "LOW" if checks_passed >= 2 else ("MEDIUM" if checks_passed == 1 else "HIGH")
    return {
        "article": 10, "title": "Data and Data Governance", "passed": passed,
        "severity": severity,
        "checks": {"input_validation": has_validation, "pii_handling": has_pii, "data_schemas": has_schemas, "data_provenance": has_provenance},
        "evidence": (_find_evidence(code, DATA_PATTERNS["input_validation"]) + _find_evidence(code, DATA_PATTERNS["pii_handling"]))[:3],
        "finding": f"Data governance patterns detected ({checks_passed}/4 checks pass)." if passed else
            "No data validation or governance controls detected. Article 10 requires data governance including data quality criteria and PII protection.",
        "fix": "Add input validation (e.g., Pydantic, marshmallow, jsonschema), PII handling (e.g., redaction, masking), and data schemas." if not passed else None,
    }


def check_article_11(code: str) -> dict:
    has_logging = _check_patterns(code, DOC_PATTERNS["logging"])
    has_docs = _check_patterns(code, DOC_PATTERNS["documentation"])
    has_types = _check_patterns(code, DOC_PATTERNS["type_hints"])
    checks_passed = sum([has_logging, has_docs, has_types])
    passed = checks_passed >= 1
    severity = "LOW" if checks_passed >= 2 else "MEDIUM"
    return {
        "article": 11, "title": "Technical Documentation", "passed": passed,
        "severity": severity,
        "checks": {"logging": has_logging, "documentation": has_docs, "type_hints": has_types},
        "evidence": _find_evidence(code, DOC_PATTERNS["logging"])[:3],
        "finding": f"Documentation patterns detected ({checks_passed}/3 checks pass)." if passed else
            "No logging or documentation system detected. Article 11 requires structured logging and documentation.",
        "fix": "Add logging (e.g., Python logging, structlog, loguru), docstrings, and type hints." if not passed else None,
    }


def check_article_12(code: str) -> dict:
    has_structured = _check_patterns(code, RECORD_PATTERNS["structured_logging"])
    has_audit = _check_patterns(code, RECORD_PATTERNS["audit_trail"])
    has_timestamps = _check_patterns(code, RECORD_PATTERNS["timestamps"])
    has_integrity = _check_patterns(code, RECORD_PATTERNS["log_integrity"])
    checks_passed = sum([has_structured, has_audit, has_timestamps, has_integrity])
    passed = checks_passed >= 1
    severity = "LOW" if checks_passed >= 3 else ("MEDIUM" if checks_passed >= 1 else "CRITICAL")
    return {
        "article": 12, "title": "Record-Keeping", "passed": passed,
        "severity": severity,
        "checks": {"structured_logging": has_structured, "audit_trail": has_audit, "timestamps": has_timestamps, "log_integrity": has_integrity},
        "evidence": (_find_evidence(code, RECORD_PATTERNS["structured_logging"]) + _find_evidence(code, RECORD_PATTERNS["audit_trail"]))[:3],
        "finding": f"Record-keeping patterns detected ({checks_passed}/4 checks pass)." if passed else
            "No structured logging or audit trail detected. Article 12 requires automatic recording of events for regulatory verification.",
        "fix": "Add structured logging (e.g., structlog, JSON logging), audit trail events, timestamps, and consider log integrity (e.g., HMAC signatures)." if not passed else None,
    }


def check_article_14(code: str) -> dict:
    has_review = _check_patterns(code, OVERSIGHT_PATTERNS["human_review"])
    has_override = _check_patterns(code, OVERSIGHT_PATTERNS["override_mechanism"])
    has_notify = _check_patterns(code, OVERSIGHT_PATTERNS["notification"])
    checks_passed = sum([has_review, has_override, has_notify])
    passed = checks_passed >= 1
    severity = "LOW" if checks_passed >= 2 else ("MEDIUM" if checks_passed == 1 else "HIGH")
    return {
        "article": 14, "title": "Human Oversight", "passed": passed,
        "severity": severity,
        "checks": {"human_review": has_review, "override_mechanism": has_override, "notification": has_notify},
        "evidence": (_find_evidence(code, OVERSIGHT_PATTERNS["human_review"]) + _find_evidence(code, OVERSIGHT_PATTERNS["override_mechanism"]))[:3],
        "finding": f"Human oversight patterns detected ({checks_passed}/3 checks pass)." if passed else
            "No human oversight mechanism detected. Article 14 requires AI systems can be effectively overseen by natural persons.",
        "fix": "Add human-in-the-loop review for high-stakes decisions, a kill switch or circuit breaker, and notification/alerting." if not passed else None,
    }


def check_article_15(code: str) -> dict:
    has_sanitization = _check_patterns(code, SECURITY_PATTERNS["input_sanitization"])
    has_errors = _check_patterns(code, SECURITY_PATTERNS["error_handling"])
    has_testing = _check_patterns(code, SECURITY_PATTERNS["testing"])
    has_rate_limit = _check_patterns(code, SECURITY_PATTERNS["rate_limiting"])
    checks_passed = sum([has_sanitization, has_errors, has_testing, has_rate_limit])
    passed = checks_passed >= 1
    severity = "LOW" if checks_passed >= 3 else ("MEDIUM" if checks_passed >= 1 else "HIGH")
    return {
        "article": 15, "title": "Accuracy, Robustness & Cybersecurity", "passed": passed,
        "severity": severity,
        "checks": {"input_sanitization": has_sanitization, "error_handling": has_errors, "testing": has_testing, "rate_limiting": has_rate_limit},
        "evidence": (_find_evidence(code, SECURITY_PATTERNS["input_sanitization"]) + _find_evidence(code, SECURITY_PATTERNS["error_handling"]))[:3],
        "finding": f"Security patterns detected ({checks_passed}/4 checks pass)." if passed else
            "No cybersecurity defenses detected. Article 15 requires resilience against unauthorized third-party attempts.",
        "fix": "Add input sanitization, error handling (try/except), testing (pytest), and rate limiting." if not passed else None,
    }


def scan_code(code: str) -> dict:
    """Scan Python code for EU AI Act compliance across all 6 articles."""
    framework = detect_framework(code)
    frameworks = detect_all_frameworks(code)
    checks = [
        check_article_9(code), check_article_10(code), check_article_11(code),
        check_article_12(code), check_article_14(code), check_article_15(code),
    ]
    passed = sum(1 for c in checks if c["passed"])
    total = len(checks)
    return {
        "framework": framework,
        "frameworks_detected": frameworks,
        "compliance_score": f"{passed}/{total}",
        "passed": passed,
        "total": total,
        "articles": checks,
        "deadline": "August 2, 2026",
        "scanner_note": "This scanner is tool-agnostic. Use any libraries you prefer.",
    }


def scan_file(file_path: str) -> dict:
    if not os.path.isfile(file_path):
        return {"error": f"File not found: {file_path}"}
    if not file_path.endswith(".py"):
        return {"error": f"Not a Python file: {file_path}"}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()
    result = scan_code(code)
    result["file"] = file_path
    result["lines"] = len(code.splitlines())
    return result


def scan_project(directory: str) -> dict:
    if not os.path.isdir(directory):
        return {"error": f"Directory not found: {directory}"}
    py_files = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
        }]
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
    if not py_files:
        return {"error": f"No Python files found in {directory}"}
    file_results = []
    total_passed = 0
    total_checks = 0
    frameworks_found = set()
    for fp in py_files[:500]:
        result = scan_file(fp)
        if "error" not in result:
            file_results.append(result)
            total_passed += result["passed"]
            total_checks += result["total"]
            for fw in result.get("frameworks_detected", []):
                frameworks_found.add(fw)
    return {
        "directory": directory,
        "files_scanned": len(file_results),
        "total_py_files": len(py_files),
        "frameworks_detected": list(frameworks_found),
        "aggregate_score": f"{total_passed}/{total_checks}",
        "files": file_results,
        "scanner_note": "This scanner is tool-agnostic. Use any libraries you prefer.",
    }
