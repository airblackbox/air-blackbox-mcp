# Awesome-List Submission Targets

Research and list of specific awesome-lists to submit PRs to, with exact format and submission details.

---

## 1. awesome-langchain

**GitHub Repo:** https://github.com/kyrolabs/awesome-langchain

**Section to Add To:** "Tools" or "Security & Compliance"

**Markdown Entry:**
```markdown
- [structured logging and input validation](https://github.com/airblackbox/structured logging and input validation) - Drop-in callback handler that adds HMAC audit chains, consent gating, and compliance logging to LangChain applications. EU AI Act technical checks.
```

**Accepts PRs:** Yes (check CONTRIBUTING.md)

**Notes:** This list is actively maintained. Format is alphabetical within sections. Keep description under 150 characters.

---

## 2. awesome-llm-security

**GitHub Repo:** https://github.com/rod-trent/awesome-llm-security

**Section to Add To:** "Compliance & Governance" (create if needed)

**Markdown Entry:**
```markdown
- [AIR Blackbox](https://github.com/airblackbox) - Open-source compliance scanner for Python AI agents. Checks 6 technical requirements from EU AI Act. Supports LangChain, CrewAI, AutoGen, OpenAI SDK, RAG patterns.
```

**Accepts PRs:** Yes

**Notes:** Accepts community contributions. This is a security-focused list, so emphasize the governance and audit aspects.

---

## 3. awesome-ai-safety

**GitHub Repo:** https://github.com/apple2373/awesome-ai-safety

**Section to Add To:** "Governance & Compliance" or "Interpretability & Auditing"

**Markdown Entry:**
```markdown
- [AIR Blackbox](https://github.com/airblackbox) - Compliance auditing toolkit for AI agents. Rule-based scanner + optional local fine-tuned Llama model for detecting non-compliance with EU AI Act requirements (Articles 9, 10, 11, 12, 14, 15).
```

**Accepts PRs:** Yes (community-driven)

**Notes:** Safety-focused audience. Emphasize auditing, transparency, and governance.

---

## 4. awesome-generative-ai-guide

**GitHub Repo:** https://github.com/aishwaryanr/awesome-generative-ai-guide

**Section to Add To:** "Tools & Frameworks" or "Governance & Ethics"

**Markdown Entry:**
```markdown
- [AIR Blackbox](https://github.com/airblackbox) - Compliance scanner and trust layer framework for Python AI agents. Multi-framework support (LangChain, CrewAI, AutoGen, OpenAI). Includes consent gating, audit chains, and PII tokenization.
```

**Accepts PRs:** Yes

**Notes:** Broad generative AI audience. Highlight the multi-framework support and practical use cases.

---

## 5. awesome-python

**GitHub Repo:** https://github.com/vinta/awesome-python

**Section to Add To:** "Artificial Intelligence" or "Machine Learning"

**Markdown Entry:**
```markdown
- [air-compliance-checker](https://github.com/airblackbox) - CLI tool for scanning Python AI agent code against EU AI Act technical requirements. Includes framework integrations for LangChain, CrewAI, AutoGen, and more.
```

**Accepts PRs:** Yes (very active, check guidelines)

**Notes:** This is one of the most popular awesome-lists. Follow their exact format and guidelines closely. Descriptions should be concise.

---

## 6. awesome-opentelemetry

**GitHub Repo:** https://github.com/magseminars/awesome-opentelemetry

**Section to Add To:** "Instrumentation Libraries" → "AI/ML" or "Integrations" → "AI"

**Markdown Entry:**
```markdown
- [structured logging for Anthropic](https://github.com/airblackbox/structured logging for Anthropic) - OpenTelemetry-compatible trust layer for Anthropic Claude Agent SDK. Adds compliance tracing, consent gating, and audit logging to agent workflows.
```

**Accepts PRs:** Yes

**Notes:** OpenTelemetry community is growing. Emphasize observability, tracing, and monitoring aspects. May need to add AI/ML subsection if it doesn't exist.

---

## Submission Strategy

**Priority Order (by impact):**
1. awesome-python (largest audience, Python-focused)
2. awesome-langchain (target LangChain users directly)
3. awesome-ai-safety (safety/governance community)
4. awesome-llm-security (security-focused practitioners)
5. awesome-generative-ai-guide (broad AI developers)
6. awesome-opentelemetry (observability practitioners)

**General PR Guidelines:**
- Check each repo's CONTRIBUTING.md before submitting
- Fork the repo, create a branch, add entry in alphabetical order
- Write a clear PR description: "Add AIR Blackbox compliance tooling"
- Reference the official website (airblackbox.ai) in PR description
- Keep descriptions concise (one line, under 150 characters)
- Don't oversell — focus on what the tool does technically
- Include only one framework package per list (unless it's a framework-specific awesome-list)

**Example PR Title:**
```
Add AIR Blackbox compliance tooling for AI agents
```

**Example PR Description:**
```
Added air-compliance-checker and relevant framework trust layers to [section].

AIR Blackbox is an open-source compliance scanner for Python AI agents, 
checking against EU AI Act technical requirements. Supports LangChain, 
CrewAI, AutoGen, OpenAI SDK, and more.

- Main tool: github.com/airblackbox
- Website: airblackbox.ai
```
