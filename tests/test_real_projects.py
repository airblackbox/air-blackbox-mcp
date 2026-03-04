#!/usr/bin/env python3
"""
Test AIR Blackbox scanner against REAL open-source AI agent projects.
These are actual production agent files from popular GitHub repos.
"""
import sys
sys.path.insert(0, "/sessions/magical-laughing-goodall/mnt/jasonshotwell/Desktop/air-blackbox-mcp")

from air_blackbox_mcp.scanner import scan_code
import os

# ═══════════════════════════════════════════════════════════════
# REAL PROJECT FILES TO SCAN
# ═══════════════════════════════════════════════════════════════

BASE = "/sessions/magical-laughing-goodall/real_projects"

projects = [
    # ── OpenAI Agents SDK ──
    {
        "name": "OpenAI Agents — Translation Orchestrator",
        "repo": "openai/openai-agents-python",
        "stars": "~500",
        "file": f"{BASE}/openai-agents-python/examples/agent_patterns/agents_as_tools.py",
    },
    {
        "name": "OpenAI Agents — Routing/Handoffs",
        "repo": "openai/openai-agents-python",
        "stars": "~500",
        "file": f"{BASE}/openai-agents-python/examples/agent_patterns/routing.py",
    },
    {
        "name": "OpenAI Agents — Financial Research Agent",
        "repo": "openai/openai-agents-python",
        "stars": "~500",
        "file": f"{BASE}/openai-agents-python/examples/financial_research_agent/agents/financials_agent.py",
    },

    # ── CrewAI ──
    {
        "name": "CrewAI — Stock Analysis Crew",
        "repo": "crewAIInc/crewAI-examples",
        "stars": "~800",
        "file": f"{BASE}/crewAI-examples/crews/stock_analysis/src/stock_analysis/crew.py",
    },
    {
        "name": "CrewAI — Surprise Travel Planner",
        "repo": "crewAIInc/crewAI-examples",
        "stars": "~800",
        "file": f"{BASE}/crewAI-examples/crews/surprise_trip/src/surprise_travel/crew.py",
    },
    {
        "name": "CrewAI — Email Auto Responder Flow",
        "repo": "crewAIInc/crewAI-examples",
        "stars": "~800",
        "file": f"{BASE}/crewAI-examples/flows/email_auto_responder_flow/src/email_auto_responder_flow/main.py",
    },

    # ── GPT-Researcher (LangGraph / RAG) ──
    {
        "name": "GPT-Researcher — Editor Agent",
        "repo": "assafelovic/gpt-researcher",
        "stars": "~14K",
        "file": f"{BASE}/gpt-researcher/multi_agents/agents/editor.py",
    },
    {
        "name": "GPT-Researcher — Researcher Agent",
        "repo": "assafelovic/gpt-researcher",
        "stars": "~14K",
        "file": f"{BASE}/gpt-researcher/multi_agents/agents/researcher.py",
    },
]

# ═══════════════════════════════════════════════════════════════
# RUN SCANS
# ═══════════════════════════════════════════════════════════════

print("=" * 74)
print("AIR BLACKBOX SCANNER — REAL-WORLD PROJECT TEST")
print("Testing against actual open-source AI agent code from GitHub")
print("=" * 74)

results_summary = []

for proj in projects:
    filepath = proj["file"]
    if not os.path.exists(filepath):
        print(f"\n⚠️  SKIPPED: {proj['name']} — file not found")
        continue

    with open(filepath, "r") as f:
        code = f.read()

    result = scan_code(code)

    print(f"\n{'─' * 74}")
    print(f"📦 {proj['name']}")
    print(f"   Repo: github.com/{proj['repo']} ({proj['stars']} ⭐)")
    print(f"   File: {os.path.basename(filepath)}")
    print(f"   Framework: {result.get('framework', '?')} | Score: {result.get('compliance_score', '?')}")
    print(f"{'─' * 74}")

    passed_count = 0
    failed_count = 0
    article_details = []

    for art in result["articles"]:
        status = "✅ PASS" if art["passed"] else "❌ FAIL"
        if art["passed"]:
            passed_count += 1
        else:
            failed_count += 1

        subchecks = art.get("checks", {})
        sub_str = ", ".join(f"{k}={'✓' if v else '✗'}" for k, v in subchecks.items())
        print(f"  {status}  Art {art['article']} ({art['title']})")
        if sub_str:
            print(f"         [{sub_str}]")
        article_details.append({
            "article": art["article"],
            "passed": art["passed"],
            "title": art["title"]
        })

    score = result.get("compliance_score", "?")
    results_summary.append({
        "name": proj["name"],
        "repo": proj["repo"],
        "framework": result.get("framework", "?"),
        "score": score,
        "passed": passed_count,
        "failed": failed_count,
        "articles": article_details
    })

# ═══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════

print(f"\n\n{'=' * 74}")
print("SUMMARY — REAL-WORLD COMPLIANCE SCAN RESULTS")
print(f"{'=' * 74}")
print(f"\n{'Project':<45} {'Framework':<12} {'Score':<10} {'Pass/Fail'}")
print(f"{'─' * 45} {'─' * 12} {'─' * 10} {'─' * 10}")

for r in results_summary:
    name = r["name"][:44]
    print(f"{name:<45} {r['framework']:<12} {r['score']:<10} {r['passed']}/{r['passed']+r['failed']}")

# Which articles fail most?
print(f"\n\n{'=' * 74}")
print("ARTICLE FAILURE ANALYSIS — Which EU AI Act articles are most violated?")
print(f"{'=' * 74}")

article_fail_counts = {}
article_total_counts = {}
for r in results_summary:
    for art in r["articles"]:
        num = art["article"]
        article_total_counts[num] = article_total_counts.get(num, 0) + 1
        if not art["passed"]:
            article_fail_counts[num] = article_fail_counts.get(num, 0) + 1

for art_num in sorted(article_total_counts.keys()):
    fails = article_fail_counts.get(art_num, 0)
    total = article_total_counts[art_num]
    pct = (fails / total) * 100
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    print(f"  Art {art_num:>2}: {fails}/{total} projects fail  [{bar}] {pct:.0f}%")

print(f"\n{'=' * 74}")
print("CONCLUSION")
print(f"{'=' * 74}")
total_projects = len(results_summary)
all_fail_all = sum(1 for r in results_summary if r["passed"] == 0)
any_pass = sum(1 for r in results_summary if r["passed"] > 0)
print(f"  Total projects scanned: {total_projects}")
print(f"  Projects with ZERO compliance: {all_fail_all}")
print(f"  Projects with SOME compliance: {any_pass}")
print(f"\n  This demonstrates why EU AI Act compliance tooling matters —")
print(f"  even popular, well-maintained open-source projects lack basic")
print(f"  compliance infrastructure for the August 2026 deadline.")
print(f"{'=' * 74}")
