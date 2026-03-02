# Show HN: Open-source scanner that checks Python AI agents for EU AI Act compliance

**URL:** https://airblackbox.ai

**Posting Guidelines:**
- Post Tuesday-Thursday, 8-10am EST
- Keep title understated and factual — NO overclaiming
- Never say "100% compliant" — use "audit-ready" or "6/6 technical checks"
- Use builder voice, not marketing speak

---

## First Comment (Post Immediately After Submission)

Hey HN! This is AIR Blackbox, a CLI tool that scans Python AI agents for EU AI Act compliance.

**What it does:**
- Scans for 6 technical requirements from Articles 9, 10, 11, 12, 14, 15 of the EU AI Act
- Think of it as a linter for AI governance — it checks technical requirements, not legal compliance

**Framework support:**
- LangChain, CrewAI, AutoGen, OpenAI SDK, RAG patterns, Anthropic Claude Agent SDK

**Quick start:**
```bash
pip install air-compliance-checker
air-compliance scan your_file.py
```

**Architecture:**
- Rule-based scanner (always available, works offline)
- Optional local fine-tuned Llama model (runs on your machine via Ollama)
- Code never leaves your machine — everything is local-first

**What the trust layers do:**
The ecosystem includes drop-in packages for each framework:
- HMAC-SHA256 audit chains (who said what, when, and who approved it)
- Consent gating (explicit user approval before agent actions)
- PII tokenization (replace sensitive data with tokens, store mapping separately)
- Prompt injection detection (15 weighted attack patterns)

**Example use:**
```python
from structured_logging import ComplianceCallback

chain.add_callback(ComplianceCallback())
# Now your LangChain pipeline logs every decision and requires consent
```

**Resources:**
- GitHub: github.com/airblackbox
- PyPI: air-compliance-checker
- Demo: airblackbox.ai
- Docs: Full architecture guide at repo

**Honest limitation:**
This checks technical requirements, not legal compliance. Use this to audit-ready your codebase. For legal compliance, work with your legal and compliance teams.

Happy to answer questions about the architecture, how the scanner works, or EU AI Act requirements!
