"""
EU AI Act compliance scanner for Python AI agent code.
Checks Articles 9, 10, 11, 12, 14, and 15.
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
}

TRUST_LAYER_PATTERNS = {
    "air_langchain_trust": r"(?:from|import)\s+air_langchain_trust",
    "air_crewai_trust": r"(?:from|import)\s+air_crewai_trust",
    "air_autogen_trust": r"(?:from|import)\s+air_autogen_trust",
    "air_openai_trust": r"(?:from|import)\s+air_openai(?:_agents)?_trust",
    "air_rag_trust": r"(?:from|import)\s+air_rag_trust",
}

TRUST_COMPONENT_PATTERNS = {
    "AuditLedger": r"AuditLedger",
    "ConsentGate": r"ConsentGate",
    "DataVault": r"DataVault",
    "InjectionDetector": r"InjectionDetector",
    "WriteGate": r"WriteGate",
    "DriftDetector": r"DriftDetector",
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


def detect_trust_layer(code: str) -> dict:
    """Check if any AIR Blackbox trust layers are imported."""
    found = {}
    for name, pattern in TRUST_LAYER_PATTERNS.items():
        if re.search(pattern, code):
            found[name] = True
    components = {}
    for name, pattern in TRUST_COMPONENT_PATTERNS.items():
        if re.search(pattern, code):
            components[name] = True
    return {"trust_layers": found, "components": components}


# --- Article Checks ---

def check_article_9(code: str, trust: dict) -> dict:
    """Article 9 — Risk Management System."""
    has_consent = "ConsentGate" in trust.get("components", {})
    has_risk = bool(re.search(r"risk|classify|risk_level|RiskLevel", code))
    passed = has_consent or has_risk

    return {
        "article": 9,
        "title": "Risk Management System",
        "passed": passed,
        "severity": "LOW" if passed else "HIGH",
        "finding": (
            "Risk classification detected via ConsentGate."
            if passed else
            "No risk classification detected. Agent can invoke any tool "
            "without risk assessment. Article 9 requires identifying and "
            "analyzing known and foreseeable risks with measures "
            "proportionate to risk level."
        ),
        "fix": (
            "Add a trust wrapper with ConsentGate to classify tool calls "
            "as LOW/MEDIUM/HIGH/CRITICAL."
        ) if not passed else None,
    }


def check_article_10(code: str, trust: dict) -> dict:
    """Article 10 — Data and Data Governance."""
    has_vault = "DataVault" in trust.get("components", {})
    has_pii = bool(re.search(r"tokenize|pii|redact|mask|anonymize", code, re.I))
    passed = has_vault or has_pii

    return {
        "article": 10,
        "title": "Data and Data Governance",
        "passed": passed,
        "severity": "LOW" if passed else "HIGH",
        "finding": (
            "Data governance controls detected via DataVault."
            if passed else
            "No data governance controls detected. PII, credentials, and "
            "sensitive data can flow directly to the LLM. Article 10 requires "
            "data governance and management practices including data minimization."
        ),
        "fix": (
            "Add a trust wrapper with DataVault to tokenize sensitive data "
            "(PII, API keys, credentials) before LLM processing."
        ) if not passed else None,
    }


def check_article_11(code: str, trust: dict) -> dict:
    """Article 11 — Technical Documentation."""
    has_ledger = "AuditLedger" in trust.get("components", {})
    has_logging = bool(re.search(
        r"logging\.|logger\.|log_|audit|structured.*log", code, re.I
    ))
    passed = has_ledger or has_logging

    return {
        "article": 11,
        "title": "Technical Documentation",
        "passed": passed,
        "severity": "LOW" if passed else "MEDIUM",
        "finding": (
            "Structured documentation system detected via AuditLedger."
            if passed else
            "No structured documentation system detected. Agent operations "
            "are not logged. Article 11 requires a general description of "
            "the AI system kept up to date."
        ),
        "fix": (
            "Add a trust wrapper with AuditLedger to automatically "
            "document all agent operations."
        ) if not passed else None,
    }


def check_article_12(code: str, trust: dict) -> dict:
    """Article 12 — Record-Keeping."""
    has_ledger = "AuditLedger" in trust.get("components", {})
    has_hmac = bool(re.search(r"hmac|tamper.evident|chain.*hash|audit.*chain", code, re.I))
    passed = has_ledger or has_hmac

    return {
        "article": 12,
        "title": "Record-Keeping",
        "passed": passed,
        "severity": "LOW" if passed else "CRITICAL",
        "finding": (
            "Tamper-evident record-keeping detected via HMAC-SHA256 audit chain."
            if passed else
            "No automatic record-keeping detected. Agent decisions and tool "
            "invocations are not recorded. Article 12 requires automatic "
            "recording of events with enough detail for regulatory verification."
        ),
        "fix": (
            "Add a trust wrapper with HMAC-SHA256 audit chains for "
            "tamper-evident logging."
        ) if not passed else None,
    }


def check_article_14(code: str, trust: dict) -> dict:
    """Article 14 — Human Oversight."""
    has_consent = "ConsentGate" in trust.get("components", {})
    has_hitl = bool(re.search(
        r"human.in.the.loop|approval|confirm|review|oversight|interrupt",
        code, re.I
    ))
    passed = has_consent or has_hitl

    return {
        "article": 14,
        "title": "Human Oversight",
        "passed": passed,
        "severity": "LOW" if passed else "HIGH",
        "finding": (
            "Human oversight mechanism detected via ConsentGate."
            if passed else
            "No human oversight mechanism detected. Agent operates fully "
            "autonomously. Article 14 requires AI systems can be effectively "
            "overseen by natural persons."
        ),
        "fix": (
            "Add a trust wrapper with ConsentGate for human-in-the-loop "
            "approval of sensitive actions."
        ) if not passed else None,
    }


def check_article_15(code: str, trust: dict) -> dict:
    """Article 15 — Accuracy, Robustness & Cybersecurity."""
    has_detector = "InjectionDetector" in trust.get("components", {})
    has_security = bool(re.search(
        r"injection|sanitize|validate.*input|security|defense|firewall",
        code, re.I
    ))
    passed = has_detector or has_security

    return {
        "article": 15,
        "title": "Accuracy, Robustness & Cybersecurity",
        "passed": passed,
        "severity": "LOW" if passed else "HIGH",
        "finding": (
            "Cybersecurity defenses detected via InjectionDetector."
            if passed else
            "No cybersecurity defenses detected. Agent is vulnerable to "
            "prompt injection attacks. Article 15 requires resilience "
            "against unauthorized third-party attempts to alter use."
        ),
        "fix": (
            "Add a trust wrapper with InjectionDetector for multi-layer "
            "prompt injection defense."
        ) if not passed else None,
    }


# --- Main Scanner ---

def scan_code(code: str) -> dict:
    """Scan Python code for EU AI Act compliance across all 6 articles."""
    framework = detect_framework(code)
    trust = detect_trust_layer(code)

    checks = [
        check_article_9(code, trust),
        check_article_10(code, trust),
        check_article_11(code, trust),
        check_article_12(code, trust),
        check_article_14(code, trust),
        check_article_15(code, trust),
    ]

    passed = sum(1 for c in checks if c["passed"])
    total = len(checks)

    # Determine install command based on framework
    install_map = {
        "langchain": "pip install air-langchain-trust",
        "crewai": "pip install air-crewai-trust",
        "autogen": "pip install air-autogen-trust",
        "openai": "pip install air-openai-trust",
        "rag": "pip install air-rag-trust",
        "unknown": "pip install air-langchain-trust",
    }

    return {
        "framework": framework,
        "trust_layers": trust["trust_layers"],
        "trust_components": trust["components"],
        "compliance_score": f"{passed}/{total}",
        "passed": passed,
        "total": total,
        "articles": checks,
        "install_command": install_map.get(framework, install_map["unknown"]),
        "deadline": "August 2, 2026",
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
        # Skip common non-project dirs
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

    for fp in py_files[:500]:  # Cap at 500 files
        result = scan_file(fp)
        if "error" not in result:
            file_results.append(result)
            total_passed += result["passed"]
            total_checks += result["total"]
            frameworks_found.add(result["framework"])

    return {
        "directory": directory,
        "files_scanned": len(file_results),
        "total_py_files": len(py_files),
        "frameworks_detected": list(frameworks_found),
        "aggregate_score": f"{total_passed}/{total_checks}",
        "files": file_results,
    }
