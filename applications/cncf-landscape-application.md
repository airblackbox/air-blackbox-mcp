# CNCF Cloud Native Landscape Application: AIR Blackbox

## Project Details

**Project Name:** AIR Blackbox

**Project URL:** https://airblackbox.ai

**GitHub URL:** https://github.com/airblackbox

**Logo URL:** (placeholder — will need to provide high-resolution PNG/SVG, 200x200px minimum)

**Open Source License:** MIT

**Primary Language:** Python

---

## Category Placement

**Primary Category:** Observability and Analysis > Observability

**Suggested New Subcategory:** AI Governance / AI Agent Compliance

(If "AI Governance" subcategory does not yet exist in the landscape, recommend creating one alongside "Observability" given the rapid growth of AI governance tools in cloud-native ecosystems.)

---

## Project Description (50 words max)

Open-source, local-first EU AI Act compliance scanner for AI agents. Framework-specific trust layers for LangChain, CrewAI, AutoGen, OpenAI, RAG, and Anthropic. HMAC-SHA256 audit chains, consent gating, injection detection. OpenTelemetry-compatible telemetry.

---

## Why AIR Blackbox Belongs in CNCF Landscape

### 1. **OpenTelemetry-Native Telemetry Export**
AIR Blackbox integrates seamlessly with OpenTelemetry for traces, metrics, and logs. Compliance events are exportable to standard observability backends (Jaeger, Datadog, New Relic, Prometheus). This aligns with CNCF's observability standards.

### 2. **CI/CD Integration via GitHub Actions**
AIR Blackbox provides GitHub Actions for automated compliance scanning in CI/CD pipelines. Fits the CNCF landscape's emphasis on cloud-native deployment and GitOps patterns.

### 3. **Local-First Architecture**
Code never leaves the developer's machine. No cloud dependencies, no external APIs. This aligns with CNCF values around security, decentralization, and operator control.

### 4. **Growing Regulatory Need**
EU AI Act (effective 2025-2026) mandates compliance for high-risk AI systems. Cloud-native teams increasingly need to bake compliance into their AI agent pipelines. AIR Blackbox addresses this emerging category.

### 5. **Community-Driven Open Source**
MIT-licensed, Python-based, with a focus on developer experience and interoperability. Fits the CNCF landscape's emphasis on vendor-neutral, interoperable tooling.

---

## How to Submit to CNCF Landscape

### Step 1: Prepare Submission Materials

1. **High-resolution logo** (PNG or SVG, 200x200px or larger)
2. **Project metadata**:
   - Official project name and URL
   - GitHub repository URL
   - Short description (see above)
   - Maturity level (if applicable: Sandbox, Incubating, Graduated)
3. **Categories**: Decide on primary and secondary categories

### Step 2: Submit via GitHub PR

The CNCF Landscape is maintained in the public GitHub repository: **https://github.com/cncf/landscape**

**Process:**

1. **Fork the repository**: `git clone https://github.com/cncf/landscape.git`
2. **Locate the landscape.yml file**: This is the main data file listing all projects.
3. **Add your project entry**:
   ```yaml
   - name: AIR Blackbox
     url: https://airblackbox.ai
     repo_url: https://github.com/airblackbox
     logo: air-blackbox-logo.svg
     crunchbase_uuid: (optional)
     twitter: (optional)
     license: MIT
     categories:
       - Observability and Analysis
     subcategories:
       - Observability
   ```
4. **Upload logo** to `/hosted_logos/` directory (or use a CDN URL).
5. **Submit PR** with a clear title and description:
   - Title: "Add AIR Blackbox to Observability category"
   - Description: Include project summary, why it belongs in CNCF, and link to GitHub repo.

### Step 3: Review and Approval

- CNCF maintainers will review your PR for completeness and fit.
- They may ask for clarification on maturity level, licensing, or category placement.
- Once approved and merged, your project will appear in the CNCF Landscape interactive map.

### Step 4: Post-Merge Promotion

After merge:
- Announce on Twitter, blog, and dev communities (HN, r/kubernetes, etc.)
- Update your project website to link to the CNCF Landscape listing
- Monitor analytics to track traffic and interest

---

## Maturity Level Recommendation

**Recommend:** Sandbox or Early Stage

(Once adoption increases and governance model is formalized, can graduate to Incubating/Graduated.)

---

## Contact & Next Steps

For questions or support during the CNCF submission process:
- **Repository Issues**: https://github.com/cncf/landscape/issues
- **CNCF Slack**: #landscape-questions (if applicable)
- **Project Contact**: Jason Shotwell (AIR Blackbox lead)

---

## Appendix: Category Recommendations

### Current CNCF Landscape Categories

- **Observability and Analysis**
  - CI/CD
  - Container Registry
  - Database
  - Horizontal Pod Autoscaling
  - Ingress
  - Key Management
  - Logging
  - Messaging
  - Monitoring
  - Observability (← recommended primary category)
  - Security Compliance
  - Service Mesh
  - Tracing

### Proposed New Subcategory

**AI Governance** (under Observability and Analysis or as standalone top-level category)
- AI Compliance Scanning
- AI Model Governance
- Prompt Injection Detection
- AI Agent Auditing

This would position AIR Blackbox alongside other emerging AI governance and trust tools in the cloud-native ecosystem.
