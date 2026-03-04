"""
EU AI Act compliance scanner for Python AI agent code.
Checks Articles 9, 10, 11, 12, 14, and 15.
Tool-agnostic: detects real compliance patterns, not specific products.
"""

import re
import os
import json
import tokenize
import io


# --- Code Preprocessing ---

def strip_comments_and_strings(code: str, keep_docstrings: bool = True) -> str:
    """Remove comments and string literals from Python code.

    When keep_docstrings=True, preserves true docstrings (triple-quoted
    strings that appear as the first statement in a module, class, or
    function body). When False, strips everything including docstrings.

    All other strings and comments are replaced to prevent false
    positive pattern matches from non-code content.
    """
    try:
        lines = code.split('\n')
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))

        # Find positions of true docstrings (first expr after def/class/module start)
        docstring_positions = set()
        for i, (tok_type, tok_string, start, end, line) in enumerate(tokens):
            if tok_type == tokenize.STRING:
                stripped = tok_string.strip()
                is_triple = stripped.startswith('"""') or stripped.startswith("'''")
                if is_triple:
                    # Check if this is a true docstring:
                    # Look backwards for def/class or module start
                    is_docstring = False
                    for j in range(i - 1, -1, -1):
                        prev_type = tokens[j][0]
                        if prev_type in (tokenize.NL, tokenize.NEWLINE,
                                         tokenize.INDENT, tokenize.COMMENT):
                            continue
                        if prev_type == tokenize.OP and tokens[j][1] == ':':
                            is_docstring = True  # follows a colon (def/class body)
                        break
                    # Module-level docstring: first string token in file
                    if i == 0 or all(tokens[k][0] in (tokenize.ENCODING,
                                     tokenize.NL, tokenize.NEWLINE,
                                     tokenize.INDENT, tokenize.COMMENT)
                                     for k in range(i)):
                        is_docstring = True
                    if is_docstring:
                        docstring_positions.add(i)

        # Now rebuild code, stripping comments and non-docstring strings
        preserve_set = docstring_positions if keep_docstrings else set()
        result_lines = list(lines)  # copy original lines
        for i, (tok_type, tok_string, start, end, _) in enumerate(tokens):
            if tok_type == tokenize.COMMENT:
                # Blank out comment on this line
                line_idx = start[0] - 1
                col = start[1]
                result_lines[line_idx] = result_lines[line_idx][:col] + ' ' * (len(result_lines[line_idx]) - col)
            elif tok_type == tokenize.STRING and i not in preserve_set:
                # Replace string content with empty string
                # Handle multi-line strings
                if start[0] == end[0]:
                    # Single-line string
                    line_idx = start[0] - 1
                    before = result_lines[line_idx][:start[1]]
                    after = result_lines[line_idx][end[1]:]
                    result_lines[line_idx] = before + '""' + after
                else:
                    # Multi-line string — blank out all lines
                    for line_no in range(start[0] - 1, end[0]):
                        if line_no == start[0] - 1:
                            result_lines[line_no] = result_lines[line_no][:start[1]] + '""'
                        elif line_no == end[0] - 1:
                            result_lines[line_no] = ' ' * len(result_lines[line_no])
                        else:
                            result_lines[line_no] = ''

        return '\n'.join(result_lines)
    except tokenize.TokenError:
        # If tokenization fails (syntax errors), fall back to raw code
        return code


# --- Framework Detection ---

FRAMEWORK_PATTERNS = {
    "langchain": [
        r"from\s+langchain", r"import\s+langchain",
        r"from\s+langchain_openai", r"from\s+langchain_core",
        r"from\s+langgraph", r"AgentExecutor", r"create_openai_tools_agent",
    ],
    "crewai": [
        r"from\s+crewai", r"import\s+crewai", r"Crew\s*\(", r"@CrewBase",
        r"CrewBase", r"Process\.sequential", r"Process\.hierarchical",
    ],
    "openai_agents_sdk": [
        r"from\s+agents\s+import", r"import\s+agents",
        r"Runner\.run", r"Runner\.run_streamed",
        r"\.as_tool\s*\(", r"handoffs\s*=\s*\[",
    ],
    "autogen": [
        r"from\s+autogen", r"import\s+autogen", r"from\s+ag2",
        r"AssistantAgent", r"UserProxyAgent", r"GroupChat",
    ],
    "openai": [
        r"from\s+openai", r"import\s+openai", r"OpenAI\s*\(",
        r"client\.chat\.completions", r"function_call",
    ],
    "rag": [
        r"VectorStore", r"Chroma\s*\(", r"FAISS", r"Pinecone",
        r"retriever", r"RetrievalQA", r"from\s+llama_index",
    ],
    "huggingface": [
        r"from\s+transformers", r"import\s+transformers",
        r"AutoModel", r"pipeline\s*\(",
    ],
    "pytorch": [
        r"import\s+torch", r"from\s+torch",
    ],
    "tensorflow": [
        r"import\s+tensorflow", r"from\s+tensorflow",
    ],
    "fastapi": [
        r"from\s+fastapi", r"import\s+fastapi", r"FastAPI\s*\(",
    ],
    "flask": [
        r"from\s+flask", r"import\s+flask", r"Flask\s*\(",
    ],
    "anthropic": [
        r"from\s+anthropic", r"import\s+anthropic", r"Anthropic\s*\(",
    ],
}


# --- Compliance Pattern Detection ---

# Article 9: Risk Management
RISK_PATTERNS = {
    "risk_classification": [
        r"risk[_\s]?level", r"risk[_\s]?class", r"risk[_\s]?score",
        r"risk[_\s]?categor", r"risk[_\s]?assess", r"risk[_\s]?tier",
        r"RiskLevel", r"\b(?:CRITICAL|HIGH|MEDIUM|LOW)\b",
    ],
    "access_control": [
        r"@(?:requires?_auth|login_required|permission)", r"role[_\s]?based",
        r"rbac", r"access[_\s]?control", r"(?:is_)?authorized",
        r"authenticate", r"permission", r"@requires_role",
    ],
    "risk_audit": [
        r"risk.*(?:log|audit|record|review)", r"(?:log|audit|record).*risk",
        r"risk.*(?:register|matrix|report)",
    ],
}

# Article 10: Data Governance
DATA_PATTERNS = {
    "input_validation": [
        r"(?:from|import)\s+(?:pydantic|marshmallow|cerberus|voluptuous|jsonschema|wtforms)",
        r"BaseModel", r"Schema\s*\(", r"validate\s*\(", r"@validator",
        r"field_validator", r"model_validator", r"@validates",
    ],
    "pii_handling": [
        r"pii", r"(?:redact|mask|anonymize|pseudonymize|tokenize)\s*\(",
        r"sensitive[_\s]?data", r"personal[_\s]?data", r"gdpr",
        r"data[_\s]?protection", r"encrypt",
    ],
    "data_schemas": [
        r"(?:from|import)\s+(?:dataclasses|attrs|pydantic|marshmallow)",
        r"@dataclass", r"TypedDict", r"NamedTuple",
        r"class\s+\w+\(BaseModel\)", r"Schema\s*\(",
    ],
    "data_provenance": [
        r"provenance", r"data[_\s]?source", r"data[_\s]?lineage",
        r"data[_\s]?origin", r"source[_\s]?track", r"metadata",
    ],
}

# Article 11: Technical Documentation
DOC_PATTERNS = {
    "logging": [
        r"(?:from|import)\s+(?:logging|structlog|loguru)",
        r"logging\.(?:get)?[Ll]ogger", r"logger\.\w+\(",
        r"structlog\.get_logger", r"loguru",
    ],
    "documentation": [
        r'"""[\s\S]*?"""', r"'''[\s\S]*?'''",
        r"# (?:TODO|FIXME|NOTE|HACK|XXX)",
    ],
    "type_hints": [
        r"def\s+\w+\([^)]*:\s*\w+", r"->\s*\w+",
        r":\s*(?:str|int|float|bool|list|dict|Optional|Union|List|Dict|Tuple|Set)",
    ],
}

# Article 12: Record-Keeping
RECORD_PATTERNS = {
    "structured_logging": [
        r"structlog\.get_logger", r"structlog\.configure",
        r"json[_\s]?log", r"log.*format.*json",
        r"logging\.handlers", r"(?:key|structured)[_\s]?value[_\s]?(?:log|pair)",
        r"extra\s*=\s*\{", r"logger\s*=\s*structlog",
    ],
    "audit_trail": [
        r"audit[_\s]?(?:log|trail|record|entry|event)",
        r"(?:event|action)[_\s]?(?:log|record|store|write)",
        r"create[_\s]?audit", r"log[_\s]?(?:event|action|decision)",
    ],
    "timestamps": [
        r"datetime\.(?:now|utcnow)", r"time\.time\(\)", r"timestamp",
        r"created_at", r"updated_at", r"isoformat",
    ],
    "log_integrity": [
        r"hmac\.new\s*\(", r"hashlib\.(?:sha256|sha512|md5)\s*\(",
        r"\.hexdigest\s*\(", r"\.digest\s*\(",
        r"tamper[_\s]?(?:evident|proof|detect)",
        r"append[_\s]?only", r"immutable[_\s]?log",
    ],
}

# Article 14: Human Oversight
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
        r"on_(?:error|failure|anomaly)",
    ],
}

# Article 15: Accuracy, Robustness, Cybersecurity
SECURITY_PATTERNS = {
    "input_sanitization": [
        r"sanitiz", r"escape\s*\(", r"strip[_\s]?tags",
        r"injection", r"(?:prompt|sql|xss)[_\s]?(?:inject|attack|defense|detect|guard|filter)",
        r"allow[_\s]?list|block[_\s]?list|whitelist|blacklist",
    ],
    "error_handling": [
        r"try\s*:", r"except\s+\w+", r"raise\s+\w+Error",
        r"finally\s*:", r"error[_\s]?handl",
    ],
    "testing": [
        r"(?:from|import)\s+(?:pytest|unittest|hypothesis|coverage)",
        r"def\s+test_", r"class\s+Test\w+", r"assert\s+",
        r"@pytest\.mark", r"mock\.|patch\(",
    ],
    "rate_limiting": [
        r"rate[_\s]?limit", r"throttl", r"(?:max|limit)[_\s]?requests",
        r"cooldown", r"backoff", r"retry.*(?:max|limit)",
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
    """Check if any pattern in the list matches the code."""
    return any(re.search(p, code, re.IGNORECASE | re.MULTILINE) for p in patterns)


def _find_evidence(code: str, patterns: list) -> list:
    """Find all matching evidence in the code."""
    evidence = []
    for p in patterns:
        matches = re.finditer(p, code, re.IGNORECASE | re.MULTILINE)
        for m in matches:
            line_num = code[:m.start()].count('\n') + 1
            evidence.append(f"Line {line_num}: {m.group()[:80]}")
    return evidence[:5]  # Cap at 5 evidence items


# --- Article Checks ---

def check_article_9(code: str) -> dict:
    """Article 9 — Risk Management System."""
    has_risk = _check_patterns(code, RISK_PATTERNS["risk_classification"])
    has_access = _check_patterns(code, RISK_PATTERNS["access_control"])
    has_audit = _check_patterns(code, RISK_PATTERNS["risk_audit"])
    
    checks_passed = sum([has_risk, has_access, has_audit])
    passed = checks_passed >= 1
    
    evidence = []
    if has_risk:
        evidence.extend(_find_evidence(code, RISK_PATTERNS["risk_classification"]))
    if has_access:
        evidence.extend(_find_evidence(code, RISK_PATTERNS["access_control"]))
    
    severity = "LOW" if checks_passed >= 2 else ("MEDIUM" if checks_passed == 1 else "HIGH")
    
    return {
        "article": 9,
        "title": "Risk Management System",
        "passed": passed,
        "severity": severity,
        "checks": {
            "risk_classification": has_risk,
            "access_control": has_access,
            "risk_audit": has_audit,
        },
        "evidence": evidence[:3],
        "finding": (
            f"Risk management patterns detected ({checks_passed}/3 checks pass)."
            if passed else
            "No risk classification or access control detected. "
            "Article 9 requires identifying and analyzing known risks "
            "with measures proportionate to risk level."
        ),
        "fix": (
            "Add risk classification (e.g., risk levels for different operations), "
            "access control (e.g., role-based permissions), and risk logging."
        ) if not passed else None,
    }


def check_article_10(code: str) -> dict:
    """Article 10 — Data and Data Governance."""
    has_validation = _check_patterns(code, DATA_PATTERNS["input_validation"])
    has_pii = _check_patterns(code, DATA_PATTERNS["pii_handling"])
    has_schemas = _check_patterns(code, DATA_PATTERNS["data_schemas"])
    has_provenance = _check_patterns(code, DATA_PATTERNS["data_provenance"])
    
    checks_passed = sum([has_validation, has_pii, has_schemas, has_provenance])
    passed = checks_passed >= 1
    
    evidence = []
    if has_validation:
        evidence.extend(_find_evidence(code, DATA_PATTERNS["input_validation"]))
    if has_pii:
        evidence.extend(_find_evidence(code, DATA_PATTERNS["pii_handling"]))
    
    severity = "LOW" if checks_passed >= 2 else ("MEDIUM" if checks_passed == 1 else "HIGH")
    
    return {
        "article": 10,
        "title": "Data and Data Governance",
        "passed": passed,
        "severity": severity,
        "checks": {
            "input_validation": has_validation,
            "pii_handling": has_pii,
            "data_schemas": has_schemas,
            "data_provenance": has_provenance,
        },
        "evidence": evidence[:3],
        "finding": (
            f"Data governance patterns detected ({checks_passed}/4 checks pass)."
            if passed else
            "No data validation or governance controls detected. "
            "Article 10 requires data governance and management practices "
            "including data quality criteria and PII protection."
        ),
        "fix": (
            "Add input validation (e.g., Pydantic, marshmallow, jsonschema), "
            "PII handling (e.g., redaction, masking, encryption), and data schemas."
        ) if not passed else None,
    }


def check_article_11(code: str) -> dict:
    """Article 11 — Technical Documentation."""
    has_logging = _check_patterns(code, DOC_PATTERNS["logging"])
    has_docs = _check_patterns(code, DOC_PATTERNS["documentation"])
    has_types = _check_patterns(code, DOC_PATTERNS["type_hints"])
    
    checks_passed = sum([has_logging, has_docs, has_types])
    passed = checks_passed >= 1
    
    evidence = []
    if has_logging:
        evidence.extend(_find_evidence(code, DOC_PATTERNS["logging"]))
    
    severity = "LOW" if checks_passed >= 2 else ("MEDIUM" if checks_passed == 1 else "MEDIUM")
    
    return {
        "article": 11,
        "title": "Technical Documentation",
        "passed": passed,
        "severity": severity,
        "checks": {
            "logging": has_logging,
            "documentation": has_docs,
            "type_hints": has_types,
        },
        "evidence": evidence[:3],
        "finding": (
            f"Documentation patterns detected ({checks_passed}/3 checks pass)."
            if passed else
            "No logging or documentation system detected. "
            "Article 11 requires a general description of the AI system "
            "kept up to date with structured logging."
        ),
        "fix": (
            "Add logging (e.g., Python logging, structlog, loguru), "
            "docstrings, and type hints for traceability."
        ) if not passed else None,
    }


def check_article_12(code: str) -> dict:
    """Article 12 — Record-Keeping."""
    has_structured = _check_patterns(code, RECORD_PATTERNS["structured_logging"])
    has_audit = _check_patterns(code, RECORD_PATTERNS["audit_trail"])
    has_timestamps = _check_patterns(code, RECORD_PATTERNS["timestamps"])
    has_integrity = _check_patterns(code, RECORD_PATTERNS["log_integrity"])
    
    checks_passed = sum([has_structured, has_audit, has_timestamps, has_integrity])
    passed = checks_passed >= 1
    
    evidence = []
    if has_structured:
        evidence.extend(_find_evidence(code, RECORD_PATTERNS["structured_logging"]))
    if has_audit:
        evidence.extend(_find_evidence(code, RECORD_PATTERNS["audit_trail"]))
    
    severity = "LOW" if checks_passed >= 3 else ("MEDIUM" if checks_passed >= 1 else "CRITICAL")
    
    return {
        "article": 12,
        "title": "Record-Keeping",
        "passed": passed,
        "severity": severity,
        "checks": {
            "structured_logging": has_structured,
            "audit_trail": has_audit,
            "timestamps": has_timestamps,
            "log_integrity": has_integrity,
        },
        "evidence": evidence[:3],
        "finding": (
            f"Record-keeping patterns detected ({checks_passed}/4 checks pass)."
            if passed else
            "No structured logging or audit trail detected. "
            "Article 12 requires automatic recording of events "
            "with enough detail for regulatory verification."
        ),
        "fix": (
            "Add structured logging (e.g., structlog, JSON logging), "
            "audit trail events, timestamps, and consider log integrity "
            "(e.g., HMAC signatures, append-only storage)."
        ) if not passed else None,
    }


def check_article_14(code: str) -> dict:
    """Article 14 — Human Oversight."""
    has_review = _check_patterns(code, OVERSIGHT_PATTERNS["human_review"])
    has_override = _check_patterns(code, OVERSIGHT_PATTERNS["override_mechanism"])
    has_notify = _check_patterns(code, OVERSIGHT_PATTERNS["notification"])
    
    checks_passed = sum([has_review, has_override, has_notify])
    passed = checks_passed >= 1
    
    evidence = []
    if has_review:
        evidence.extend(_find_evidence(code, OVERSIGHT_PATTERNS["human_review"]))
    if has_override:
        evidence.extend(_find_evidence(code, OVERSIGHT_PATTERNS["override_mechanism"]))
    
    severity = "LOW" if checks_passed >= 2 else ("MEDIUM" if checks_passed == 1 else "HIGH")
    
    return {
        "article": 14,
        "title": "Human Oversight",
        "passed": passed,
        "severity": severity,
        "checks": {
            "human_review": has_review,
            "override_mechanism": has_override,
            "notification": has_notify,
        },
        "evidence": evidence[:3],
        "finding": (
            f"Human oversight patterns detected ({checks_passed}/3 checks pass)."
            if passed else
            "No human oversight mechanism detected. Agent operates fully "
            "autonomous. Article 14 requires AI systems can be "
            "effectively overseen by natural persons."
        ),
        "fix": (
            "Add human-in-the-loop review for high-stakes decisions, "
            "a kill switch or circuit breaker, and notification/alerting "
            "for anomalies."
        ) if not passed else None,
    }


def check_article_15(code: str) -> dict:
    """Article 15 — Accuracy, Robustness & Cybersecurity."""
    has_sanitization = _check_patterns(code, SECURITY_PATTERNS["input_sanitization"])
    has_errors = _check_patterns(code, SECURITY_PATTERNS["error_handling"])
    has_testing = _check_patterns(code, SECURITY_PATTERNS["testing"])
    has_rate_limit = _check_patterns(code, SECURITY_PATTERNS["rate_limiting"])
    
    checks_passed = sum([has_sanitization, has_errors, has_testing, has_rate_limit])
    passed = checks_passed >= 1
    
    evidence = []
    if has_sanitization:
        evidence.extend(_find_evidence(code, SECURITY_PATTERNS["input_sanitization"]))
    if has_errors:
        evidence.extend(_find_evidence(code, SECURITY_PATTERNS["error_handling"]))
    
    severity = "LOW" if checks_passed >= 3 else ("MEDIUM" if checks_passed >= 1 else "HIGH")
    
    return {
        "article": 15,
        "title": "Accuracy, Robustness & Cybersecurity",
        "passed": passed,
        "severity": severity,
        "checks": {
            "input_sanitization": has_sanitization,
            "error_handling": has_errors,
            "testing": has_testing,
            "rate_limiting": has_rate_limit,
        },
        "evidence": evidence[:3],
        "finding": (
            f"Security patterns detected ({checks_passed}/4 checks pass)."
            if passed else
            "No cybersecurity defenses detected. Agent may be vulnerable to "
            "prompt injection and other attacks. Article 15 requires "
            "resilience against unauthorized third-party attempts."
        ),
        "fix": (
            "Add input sanitization (e.g., prompt injection detection), "
            "error handling (try/except with recovery), testing (pytest), "
            "and rate limiting."
        ) if not passed else None,
    }


# --- Main Scanner ---

def scan_code(code: str) -> dict:
    """Scan Python code for EU AI Act compliance across all 6 articles."""
    # Framework detection uses raw code (imports are real code)
    framework = detect_framework(code)
    frameworks = detect_all_frameworks(code)

    # Strip ALL comments, strings, AND docstrings for compliance checks
    # to avoid false positives from compliance terms in non-code content
    clean_code = strip_comments_and_strings(code, keep_docstrings=False)

    # Article 11 (Technical Documentation) uses raw code because
    # docstrings, logging imports, and type hints ARE documentation
    checks = [
        check_article_9(clean_code),
        check_article_10(clean_code),
        check_article_11(code),  # raw code — docstrings count as documentation
        check_article_12(clean_code),
        check_article_14(clean_code),
        check_article_15(clean_code),
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
    """Scan a single Python file."""
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
    """Scan all Python files in a directory."""
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
