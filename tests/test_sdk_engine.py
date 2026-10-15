"""With the air-blackbox SDK installed, the scanning tools run the SDK.

Skipped on a base install, where the built-in scanner runs and the rest of
the suite covers it.

Run with: python -m pytest tests/test_sdk_engine.py -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from air_blackbox_mcp import sdk_scanner  # noqa: E402
from air_blackbox_mcp.provenance import ENGINE_BUILTIN, ENGINE_SDK  # noqa: E402

pytestmark = pytest.mark.skipif(not sdk_scanner.available(), reason="air-blackbox SDK not installed")

PY = "import openai\nclient = openai.OpenAI()\n\ndef ask(q):\n    return client.chat.completions.create(model='gpt-4o', messages=q)\n"
KT = "import dev.langchain4j.model.chat.ChatModel\n\nfun ask(model: ChatModel, q: String): String = runCatching { model.chat(q) }.getOrDefault(\"\")\n"


def test_server_binds_scan_tools_to_the_sdk():
    import air_blackbox_mcp.server as server
    assert server._scan_code is sdk_scanner.scan_code
    assert server._scan_project is sdk_scanner.scan_project


def test_scan_code_reports_the_sdk_engine_and_language():
    result = sdk_scanner.scan_code(PY)
    assert result["provenance"]["engine"] == ENGINE_SDK
    assert result["provenance"]["sdk_version"]
    assert result["language"] == "python" and result["language_source"] == "detected"
    assert "openai" in result["frameworks"]
    by_name = {f["name"]: f for f in result["findings"]}
    assert by_name["LLM call error handling"]["status"] == "fail"
    assert by_name["LLM call error handling"]["severity"] == "HIGH"
    assert result["summary"]["total_checks"] == len(result["findings"]) > 10


def test_kotlin_is_scanned_with_the_same_shape():
    result = sdk_scanner.scan_code(KT)
    assert result["language"] == "kotlin"
    assert result["frameworks"] == ["LangChain4j"]
    assert {f["name"]: f["status"] for f in result["findings"]}["LLM call error handling"] == "pass"
    assert set(result) >= {"frameworks", "trust_layer_detected", "findings", "summary", "provenance"}


def test_stated_language_and_bad_language():
    assert sdk_scanner.scan_code(PY, language="python")["language_source"] == "stated"
    assert "error" in sdk_scanner.scan_code(PY, language="cobol")


def test_scan_file_uses_the_extension(tmp_path):
    f = tmp_path / "Agent.kt"
    f.write_text(KT)
    result = sdk_scanner.scan_file(str(f))
    assert result["language"] == "kotlin" and result["language_source"] == "filename"
    assert result["lines"] == 3 and result["file"] == str(f)
    assert "error" in sdk_scanner.scan_file(str(tmp_path / "nope.py"))


def test_scan_project_reads_every_language(tmp_path):
    (tmp_path / "a.py").write_text(PY)
    (tmp_path / "B.kt").write_text(KT)
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("require('openai')")
    result = sdk_scanner.scan_project(str(tmp_path))
    assert result["provenance"]["engine"] == ENGINE_SDK
    assert result["files_scanned"] == 2
    assert {f["language"] for f in result["files"]} == {"python", "kotlin"}
    assert sorted(result["frameworks"]) == ["LangChain4j", "openai"]
    assert "error" in sdk_scanner.scan_project(str(tmp_path / "missing"))
    empty = tmp_path / "empty"
    empty.mkdir()
    assert sdk_scanner.scan_project(str(empty))["files_scanned"] == 0


def test_builtin_scanner_still_says_builtin():
    from air_blackbox_mcp.scanner import scan_code
    assert scan_code(PY)["provenance"]["engine"] == ENGINE_BUILTIN
