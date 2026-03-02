"""
EU AI Act Compliance Report Generator

Produces professional HTML and Markdown compliance reports from scan results.
Reports cover EU AI Act Articles 9-15 with detailed remediation guidance.
"""

from datetime import datetime, date
from typing import Dict, Optional
import json
from dataclasses import dataclass


DEADLINE_DATE = date(2026, 8, 2)
SCANNER_VERSION = "1.0.0"
ORANGE_ACCENT = "#f97316"
DARK_BG = "#0a0a0a"


@dataclass
class ComplianceMetrics:
    """Container for compliance metrics"""
    framework: str
    compliance_score: str
    passed: int
    total: int
    articles: list
    trust_layers: dict
    trust_components: dict
    install_command: Optional[str] = None
    deadline: str = "August 2, 2026"


def calculate_deadline_countdown() -> Dict:
    """
    Calculate days/months until EU AI Act deadline (August 2, 2026).
    
    Returns:
        dict: {
            "days": int,
            "months": int,
            "deadline": str,
            "urgency": "low|medium|high|critical"
        }
    """
    today = date.today()
    delta = DEADLINE_DATE - today
    
    days_remaining = delta.days
    months_remaining = (DEADLINE_DATE.year - today.year) * 12 + (DEADLINE_DATE.month - today.month)
    
    # Determine urgency level
    if days_remaining <= 0:
        urgency = "critical"
    elif days_remaining <= 30:
        urgency = "critical"
    elif days_remaining <= 90:
        urgency = "high"
    elif days_remaining <= 180:
        urgency = "medium"
    else:
        urgency = "low"
    
    return {
        "days": max(0, days_remaining),
        "months": max(0, months_remaining),
        "deadline": "August 2, 2026",
        "urgency": urgency
    }


def _get_severity_order(article: Dict) -> tuple:
    """Sort key for articles by severity (CRITICAL > HIGH > MEDIUM > LOW)"""
    severity_map = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    return (not article["passed"], severity_map.get(article.get("severity", "LOW"), 4))


def _format_code_snippet(code: str, max_lines: int = 10) -> str:
    """Extract and format code snippet for reports"""
    if not code or len(code) < 20:
        return ""
    
    lines = code.split("\n")[:max_lines]
    return "\n".join(lines)


def generate_compliance_report_md(
    scan_result: Dict,
    code: str = "",
    file_path: str = ""
) -> str:
    """
    Generate a professional Markdown compliance report.
    
    Args:
        scan_result: Output from scanner.py with articles, trust_layers, etc.
        code: Optional source code analyzed
        file_path: Optional path to analyzed file
    
    Returns:
        str: Markdown-formatted compliance report
    """
    metrics = ComplianceMetrics(
        framework=scan_result.get("framework", "unknown"),
        compliance_score=scan_result.get("compliance_score", "0/6"),
        passed=scan_result.get("passed", 0),
        total=scan_result.get("total", 6),
        articles=scan_result.get("articles", []),
        trust_layers=scan_result.get("trust_layers", {}),
        trust_components=scan_result.get("trust_components", {}),
        install_command=scan_result.get("install_command"),
        deadline=scan_result.get("deadline", "August 2, 2026")
    )
    
    countdown = calculate_deadline_countdown()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Build report
    report = []
    
    # Header
    report.append("# EU AI Act Compliance Report")
    report.append("")
    report.append(f"**Generated:** {generated_at}")
    report.append(f"**Framework:** {metrics.framework}")
    report.append(f"**File:** {file_path or '(inline code)'}")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append(f"**Compliance Score: {metrics.compliance_score}**")
    report.append("")
    report.append(f"Deadline: **{metrics.deadline}** ({countdown['days']} days remaining)")
    report.append(f"Urgency Level: **{countdown['urgency'].upper()}**")
    report.append("")
    
    if countdown['urgency'] == 'critical':
        report.append("> **⚠️ CRITICAL:** Your AI system must be compliant by August 2, 2026.")
        report.append("> Remediation should begin immediately.")
        report.append("")
    
    # Compliance Scorecard Table
    report.append("## Compliance Scorecard")
    report.append("")
    report.append("| Article | Title | Status | Severity |")
    report.append("|---------|-------|--------|----------|")
    
    for article in metrics.articles:
        status = "✓ PASS" if article["passed"] else "✗ FAIL"
        severity = article.get("severity", "—")
        title = article.get("title", f"Article {article.get('article', '?')}")
        article_num = article.get("article", "?")
        report.append(f"| {article_num} | {title} | {status} | {severity} |")
    
    report.append("")
    
    # Detailed Findings
    report.append("## Detailed Findings")
    report.append("")
    
    for article in metrics.articles:
        article_num = article.get("article", "?")
        title = article.get("title", "Unknown")
        finding = article.get("finding", "No details provided")
        fix = article.get("fix")
        severity = article.get("severity", "UNKNOWN")
        passed = article.get("passed", False)
        
        status_icon = "✓" if passed else "✗"
        report.append(f"### {status_icon} Article {article_num}: {title}")
        report.append("")
        report.append(f"**Severity:** {severity}")
        report.append("")
        report.append(f"**Finding:** {finding}")
        report.append("")
        
        if fix:
            report.append(f"**Remediation:** {fix}")
            report.append("")
        
        if metrics.install_command and not passed:
            report.append(f"```bash")
            report.append(f"{metrics.install_command}")
            report.append(f"```")
            report.append("")
    
    # Trust Layer Status
    report.append("## Trust Layer Status")
    report.append("")
    
    if metrics.trust_layers:
        for layer_name, installed in metrics.trust_layers.items():
            status = "✓ Installed" if installed else "✗ Not Installed"
            report.append(f"- {layer_name}: {status}")
    
    report.append("")
    
    if metrics.trust_components:
        report.append("### Trust Components")
        report.append("")
        for component_name, present in metrics.trust_components.items():
            status = "✓ Present" if present else "✗ Missing"
            report.append(f"- {component_name}: {status}")
        report.append("")
    
    # Remediation Priority List
    report.append("## Remediation Priority")
    report.append("")
    
    failed_articles = [a for a in metrics.articles if not a.get("passed", False)]
    sorted_articles = sorted(failed_articles, key=_get_severity_order)
    
    if sorted_articles:
        for i, article in enumerate(sorted_articles, 1):
            severity = article.get("severity", "LOW")
            title = article.get("title", "Unknown")
            article_num = article.get("article", "?")
            report.append(f"{i}. **[{severity}]** Article {article_num}: {title}")
    else:
        report.append("✓ No critical remediations required.")
    
    report.append("")
    
    # Disclaimer
    report.append("---")
    report.append("")
    report.append("## Disclaimer")
    report.append("")
    report.append(
        "This report checks technical requirements for EU AI Act compliance. "
        "It is not legal advice. Consult with legal counsel for authoritative guidance. "
        "This scanner detects compliance patterns but does not provide legal interpretation."
    )
    report.append("")
    
    # Footer
    report.append(f"**EU AI Act Scanner v{SCANNER_VERSION}** | Report generated {generated_at}")
    
    return "\n".join(report)


def generate_compliance_report_html(
    scan_result: Dict,
    code: str = "",
    file_path: str = ""
) -> str:
    """
    Generate a professional HTML compliance report with dark theme.
    
    Args:
        scan_result: Output from scanner.py
        code: Optional source code
        file_path: Optional file path
    
    Returns:
        str: Standalone HTML page with inline CSS
    """
    metrics = ComplianceMetrics(
        framework=scan_result.get("framework", "unknown"),
        compliance_score=scan_result.get("compliance_score", "0/6"),
        passed=scan_result.get("passed", 0),
        total=scan_result.get("total", 6),
        articles=scan_result.get("articles", []),
        trust_layers=scan_result.get("trust_layers", {}),
        trust_components=scan_result.get("trust_components", {}),
        install_command=scan_result.get("install_command"),
        deadline=scan_result.get("deadline", "August 2, 2026")
    )
    
    countdown = calculate_deadline_countdown()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Calculate percentage
    percentage = int((metrics.passed / metrics.total * 100) if metrics.total > 0 else 0)
    
    html_parts = []
    
    # DOCTYPE and head
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EU AI Act Compliance Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #0a0a0a;
            color: #e4e4e7;
            line-height: 1.6;
            padding: 40px 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        header {
            border-bottom: 2px solid #f97316;
            padding-bottom: 30px;
            margin-bottom: 40px;
        }
        
        h1 {
            font-size: 2.2rem;
            color: #fff;
            margin-bottom: 15px;
        }
        
        .header-meta {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            font-size: 0.9rem;
            color: #a1a1a1;
            margin-top: 15px;
        }
        
        .header-meta div {
            display: flex;
            flex-direction: column;
        }
        
        .header-meta label {
            color: #f97316;
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        
        .header-meta value {
            color: #e4e4e7;
            font-size: 1rem;
        }
        
        section {
            margin-bottom: 50px;
        }
        
        h2 {
            font-size: 1.6rem;
            color: #fff;
            margin-bottom: 20px;
            border-left: 3px solid #f97316;
            padding-left: 15px;
        }
        
        h3 {
            font-size: 1.2rem;
            color: #e4e4e7;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        .executive-summary {
            background-color: #18181b;
            border-radius: 8px;
            padding: 25px;
            border-left: 4px solid #f97316;
        }
        
        .score-display {
            font-size: 2.5rem;
            font-weight: 700;
            color: #f97316;
            margin-bottom: 15px;
        }
        
        .score-display .fraction {
            font-size: 1.2rem;
            color: #a1a1a1;
        }
        
        .countdown {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 15px;
        }
        
        .countdown-item {
            background-color: #27272a;
            padding: 12px;
            border-radius: 6px;
        }
        
        .countdown-label {
            font-size: 0.8rem;
            color: #a1a1a1;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        
        .countdown-value {
            font-size: 1.3rem;
            color: #fff;
            font-weight: 600;
        }
        
        .urgency-critical {
            background-color: rgba(239, 68, 68, 0.1);
            border-left: 4px solid #ef4444;
            padding: 15px;
            border-radius: 6px;
            margin-top: 15px;
            color: #fecaca;
        }
        
        .urgency-critical strong {
            color: #fca5a5;
        }
        
        .scorecard-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        
        .scorecard-table th {
            background-color: #18181b;
            color: #f97316;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #f97316;
        }
        
        .scorecard-table td {
            padding: 12px;
            border-bottom: 1px solid #27272a;
        }
        
        .scorecard-table tr:hover {
            background-color: #18181b;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        
        .badge-pass {
            background-color: rgba(34, 197, 94, 0.2);
            color: #86efac;
        }
        
        .badge-fail {
            background-color: rgba(239, 68, 68, 0.2);
            color: #fca5a5;
        }
        
        .severity-critical {
            background-color: rgba(239, 68, 68, 0.2);
            color: #fca5a5;
        }
        
        .severity-high {
            background-color: rgba(249, 115, 22, 0.2);
            color: #fed7aa;
        }
        
        .severity-medium {
            background-color: rgba(234, 179, 8, 0.2);
            color: #fef08a;
        }
        
        .severity-low {
            background-color: rgba(59, 130, 246, 0.2);
            color: #bfdbfe;
        }
        
        .finding-card {
            background-color: #18181b;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
            border-left: 4px solid #27272a;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .finding-card:hover {
            border-left-color: #f97316;
        }
        
        .finding-card.passed {
            border-left-color: #22c55e;
        }
        
        .finding-card.failed {
            border-left-color: #ef4444;
        }
        
        .finding-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            user-select: none;
        }
        
        .finding-title {
            font-weight: 600;
            color: #fff;
            font-size: 1.1rem;
        }
        
        .finding-toggle {
            color: #a1a1a1;
            transition: transform 0.3s ease;
        }
        
        .finding-card.expanded .finding-toggle {
            transform: rotate(180deg);
        }
        
        .finding-content {
            display: none;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #27272a;
        }
        
        .finding-card.expanded .finding-content {
            display: block;
        }
        
        .finding-field {
            margin-bottom: 12px;
        }
        
        .finding-label {
            font-size: 0.85rem;
            color: #a1a1a1;
            text-transform: uppercase;
            margin-bottom: 5px;
            font-weight: 600;
        }
        
        .finding-value {
            color: #e4e4e7;
        }
        
        .code-block {
            background-color: #0a0a0a;
            border: 1px solid #27272a;
            border-radius: 6px;
            padding: 12px;
            font-family: "Courier New", monospace;
            font-size: 0.9rem;
            overflow-x: auto;
            color: #22c55e;
            margin: 10px 0;
        }
        
        .trust-component-list {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .component {
            background-color: #18181b;
            padding: 15px;
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .component-status {
            font-weight: 600;
        }
        
        .remediation-list {
            list-style: none;
        }
        
        .remediation-item {
            background-color: #18181b;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 6px;
            border-left: 4px solid #f97316;
        }
        
        .remediation-number {
            display: inline-block;
            background-color: #f97316;
            color: #0a0a0a;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            text-align: center;
            line-height: 28px;
            font-weight: 600;
            margin-right: 10px;
        }
        
        .remediation-text {
            display: inline-block;
            vertical-align: middle;
        }
        
        .disclaimer {
            background-color: #18181b;
            border: 1px dashed #a1a1a1;
            border-radius: 8px;
            padding: 20px;
            font-size: 0.9rem;
            color: #a1a1a1;
            line-height: 1.8;
        }
        
        footer {
            border-top: 1px solid #27272a;
            padding-top: 20px;
            margin-top: 50px;
            text-align: center;
            color: #a1a1a1;
            font-size: 0.9rem;
        }
        
        .progress-bar-container {
            background-color: #18181b;
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 10px;
        }
        
        .progress-bar {
            background-color: #f97316;
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        @media print {
            body {
                background-color: white;
                color: #000;
            }
            
            .container {
                max-width: 100%;
            }
            
            h1, h2, h3 {
                color: #000;
            }
            
            section {
                page-break-inside: avoid;
            }
            
            .finding-card {
                page-break-inside: avoid;
            }
        }
        
        @media (max-width: 768px) {
            .header-meta {
                grid-template-columns: 1fr;
            }
            
            .trust-component-list {
                grid-template-columns: 1fr;
            }
            
            h1 {
                font-size: 1.8rem;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>EU AI Act Compliance</h1>
            <h2 style="border: none; margin: 0; font-size: 1rem; color: #a1a1a1; padding-left: 0; margin-top: 10px;">
                EU AI Act Compliance Report
            </h2>
            <div class="header-meta">
                <div>
                    <label>Generated</label>
                    <value>""" + generated_at + """</value>
                </div>
                <div>
                    <label>Framework</label>
                    <value>""" + metrics.framework + """</value>
                </div>
                <div>
                    <label>File</label>
                    <value>""" + (file_path or "(inline code)") + """</value>
                </div>
                <div>
                    <label>Score</label>
                    <value>""" + metrics.compliance_score + """</value>
                </div>
            </div>
        </header>
""")
    
    # Executive Summary Section
    html_parts.append("""        <section>
            <h2>Executive Summary</h2>
            <div class="executive-summary">
                <div class="score-display">
                    <span style="color: #f97316;">""" + str(metrics.passed) + """</span>
                    <span class="fraction">/ """ + str(metrics.total) + """</span>
                </div>
                <div class="progress-bar-container">
                    <div class="progress-bar" style="width: """ + str(percentage) + """%"></div>
                </div>
                <div class="countdown">
                    <div class="countdown-item">
                        <div class="countdown-label">Deadline</div>
                        <div class="countdown-value">""" + metrics.deadline + """</div>
                    </div>
                    <div class="countdown-item">
                        <div class="countdown-label">Days Remaining</div>
                        <div class="countdown-value">""" + str(countdown['days']) + """</div>
                    </div>
                </div>
""")
    
    if countdown['urgency'] == 'critical':
        html_parts.append("""                <div class="urgency-critical">
                    <strong>⚠️ CRITICAL:</strong> Your AI system must be compliant by August 2, 2026.
                    Remediation should begin immediately.
                </div>
""")
    
    html_parts.append("""            </div>
        </section>
""")
    
    # Scorecard Section
    html_parts.append("""        <section>
            <h2>Compliance Scorecard</h2>
            <table class="scorecard-table">
                <thead>
                    <tr>
                        <th>Article</th>
                        <th>Title</th>
                        <th>Status</th>
                        <th>Severity</th>
                    </tr>
                </thead>
                <tbody>
""")
    
    for article in metrics.articles:
        status = article.get("passed", False)
        status_badge = '<span class="badge badge-pass">✓ PASS</span>' if status else '<span class="badge badge-fail">✗ FAIL</span>'
        severity = article.get("severity", "—")
        severity_class = f"severity-{severity.lower()}"
        severity_badge = f'<span class="badge {severity_class}">{severity}</span>'
        title = article.get("title", f"Article {article.get('article', '?')}")
        article_num = article.get("article", "?")
        
        html_parts.append(f"""                    <tr>
                        <td>Article {article_num}</td>
                        <td>{title}</td>
                        <td>{status_badge}</td>
                        <td>{severity_badge}</td>
                    </tr>
""")
    
    html_parts.append("""                </tbody>
            </table>
        </section>
""")
    
    # Detailed Findings Section
    html_parts.append("""        <section>
            <h2>Detailed Findings</h2>
""")
    
    for article in metrics.articles:
        article_num = article.get("article", "?")
        title = article.get("title", "Unknown")
        finding = article.get("finding", "No details provided")
        fix = article.get("fix")
        severity = article.get("severity", "UNKNOWN")
        passed = article.get("passed", False)
        
        card_class = "passed" if passed else "failed"
        status_icon = "✓" if passed else "✗"
        
        html_parts.append(f"""            <div class="finding-card {card_class}">
                <div class="finding-header" onclick="this.parentElement.classList.toggle('expanded')">
                    <div class="finding-title">{status_icon} Article {article_num}: {title}</div>
                    <div class="finding-toggle">▼</div>
                </div>
                <div class="finding-content">
                    <div class="finding-field">
                        <div class="finding-label">Severity</div>
                        <div class="finding-value">
                            <span class="badge severity-{severity.lower()}">{severity}</span>
                        </div>
                    </div>
                    <div class="finding-field">
                        <div class="finding-label">Finding</div>
                        <div class="finding-value">{finding}</div>
                    </div>
""")
        
        if fix:
            html_parts.append(f"""                    <div class="finding-field">
                        <div class="finding-label">Remediation</div>
                        <div class="finding-value">{fix}</div>
                    </div>
""")
        
        if metrics.install_command and not passed:
            html_parts.append(f"""                    <div class="finding-field">
                        <div class="finding-label">Installation Command</div>
                        <div class="code-block">{metrics.install_command}</div>
                    </div>
""")
        
        html_parts.append("""                </div>
            </div>
""")
    
    html_parts.append("""        </section>
""")
    
    # Trust Layer Status
    html_parts.append("""        <section>
            <h2>Trust Layer Status</h2>
""")
    
    if metrics.trust_layers:
        html_parts.append("""            <h3>Installation Status</h3>
            <div class="trust-component-list">
""")
        for layer_name, installed in metrics.trust_layers.items():
            status_text = "✓ Installed" if installed else "✗ Not Installed"
            status_class = "badge-pass" if installed else "badge-fail"
            html_parts.append(f"""                <div class="component">
                    <span>{layer_name}</span>
                    <span class="badge {status_class} component-status">{status_text}</span>
                </div>
""")
        html_parts.append("""            </div>
""")
    
    if metrics.trust_components:
        html_parts.append("""            <h3 style="margin-top: 25px;">Trust Components</h3>
            <div class="trust-component-list">
""")
        for component_name, present in metrics.trust_components.items():
            status_text = "✓ Present" if present else "✗ Missing"
            status_class = "badge-pass" if present else "badge-fail"
            html_parts.append(f"""                <div class="component">
                    <span>{component_name}</span>
                    <span class="badge {status_class} component-status">{status_text}</span>
                </div>
""")
        html_parts.append("""            </div>
""")
    
    html_parts.append("""        </section>
""")
    
    # Remediation Priority
    html_parts.append("""        <section>
            <h2>Remediation Priority</h2>
""")
    
    failed_articles = [a for a in metrics.articles if not a.get("passed", False)]
    sorted_articles = sorted(failed_articles, key=_get_severity_order)
    
    if sorted_articles:
        html_parts.append("""            <ol class="remediation-list">
""")
        for i, article in enumerate(sorted_articles, 1):
            severity = article.get("severity", "LOW")
            title = article.get("title", "Unknown")
            article_num = article.get("article", "?")
            html_parts.append(f"""                <li class="remediation-item">
                    <span class="remediation-number">{i}</span>
                    <span class="remediation-text">
                        <strong>[{severity}]</strong> Article {article_num}: {title}
                    </span>
                </li>
""")
        html_parts.append("""            </ol>
""")
    else:
        html_parts.append("""            <p style="color: #22c55e; font-weight: 600;">✓ No critical remediations required.</p>
""")
    
    html_parts.append("""        </section>
""")
    
    # Disclaimer and Footer
    html_parts.append(f"""        <section>
            <h2>Disclaimer</h2>
            <div class="disclaimer">
                <p>
                    This report checks technical requirements for EU AI Act compliance. It is not legal advice.
                    Consult with legal counsel for authoritative guidance. This scanner detects compliance
                    patterns but does not provide legal interpretation.
                </p>
            </div>
        </section>
        
        <footer>
            <p><strong>EU AI Act Scanner v{SCANNER_VERSION}</strong> | Report generated {generated_at}</p>
        </footer>
    </div>
    
    <script>
        // Expand critical findings by default
        document.querySelectorAll('.finding-card.failed').forEach(card => {{
            card.classList.add('expanded');
        }});
    </script>
</body>
</html>
""")
    
    return "\n".join(html_parts)


def generate_compliance_report(
    scan_result: Dict,
    code: str = "",
    file_path: str = "",
    format: str = "markdown"
) -> str:
    """
    Generate a compliance report in the specified format.
    
    Args:
        scan_result: Output from scanner.py
        code: Optional source code analyzed
        file_path: Optional file path
        format: "markdown" or "html" (default: "markdown")
    
    Returns:
        str: Formatted compliance report
    """
    if format.lower() == "html":
        return generate_compliance_report_html(scan_result, code, file_path)
    else:
        return generate_compliance_report_md(scan_result, code, file_path)


if __name__ == "__main__":
    # Example usage
    example_scan = {
        "framework": "langchain",
        "trust_layers": {"structured_logging": True},
        "trust_components": {
            "risk_classification": True,
            "audit_logging": False,
            "pii_protection": True,
            "injection_detection": False
        },
        "compliance_score": "4/6",
        "passed": 4,
        "total": 6,
        "articles": [
            {
                "article": 9,
                "title": "Risk Management System",
                "passed": True,
                "severity": "HIGH",
                "finding": "Risk classification system detected",
                "fix": None
            },
            {
                "article": 10,
                "title": "Data and Record-Keeping",
                "passed": False,
                "severity": "CRITICAL",
                "finding": "No HMAC-SHA256 audit chain detected",
                "fix": "Add HMAC-SHA256 tamper-evident audit logging"
            },
            {
                "article": 11,
                "title": "Transparency Obligation",
                "passed": True,
                "severity": "HIGH",
                "finding": "Risk disclosure mechanism present",
                "fix": None
            },
            {
                "article": 12,
                "title": "Human Oversight",
                "passed": True,
                "severity": "MEDIUM",
                "finding": "User consent gates implemented",
                "fix": None
            },
            {
                "article": 14,
                "title": "Monitoring After Deployment",
                "passed": False,
                "severity": "HIGH",
                "finding": "No continuous monitoring system detected",
                "fix": "Implement continuous monitoring and audit logging"
            },
            {
                "article": 15,
                "title": "Incident Reporting",
                "passed": False,
                "severity": "CRITICAL",
                "finding": "No incident reporting mechanism found",
                "fix": "Add incident logging with alert thresholds"
            }
        ],
        "install_command": "pip install structlog pydantic",
        "deadline": "August 2, 2026"
    }
    
    # Generate markdown report
    md_report = generate_compliance_report(example_scan, format="markdown")
    print("=" * 80)
    print("MARKDOWN REPORT")
    print("=" * 80)
    print(md_report[:500] + "...\n")
    
    # Generate HTML report
    html_report = generate_compliance_report(example_scan, format="html")
    print("=" * 80)
    print("HTML REPORT")
    print("=" * 80)
    print(f"Generated HTML report ({len(html_report)} bytes)\n")
    
    # Test deadline countdown
    countdown = calculate_deadline_countdown()
    print("=" * 80)
    print("DEADLINE COUNTDOWN")
    print("=" * 80)
    print(f"Days remaining: {countdown['days']}")
    print(f"Months remaining: {countdown['months']}")
    print(f"Deadline: {countdown['deadline']}")
    print(f"Urgency: {countdown['urgency']}")
