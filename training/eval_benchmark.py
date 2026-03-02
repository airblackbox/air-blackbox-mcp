#!/usr/bin/env python3
"""
Evaluation benchmark for EU AI Act compliance scanner compliance model.

Loads eval_data.jsonl and tests against:
1. Rule-based scanner (baseline)
2. Ollama fine-tuned model (if available)

Produces accuracy metrics, confusion matrices, and reports.
"""

import json
import os
import sys
import time
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
from statistics import mean, stdev


# --- Ground Truth Parsing ---

def parse_ground_truth(output_text):
    """Extract article, severity, and compliance state from ground truth output."""
    result = {
        "article": None,
        "severity": None,
        "compliance": None,
    }
    
    lines = output_text.split("\n")
    for line in lines:
        if line.startswith("ARTICLE:"):
            try:
                result["article"] = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("SEVERITY:"):
            result["severity"] = line.split(":")[1].strip()
        elif line.startswith("PASS:"):
            result["compliance"] = "compliant"
        elif line.startswith("FINDING:"):
            if "SEVERITY:" in output_text:
                # This is a finding, look at compliance state metadata
                pass
    
    return result


def load_eval_data(eval_file):
    """Load eval_data.jsonl and parse ground truth."""
    examples = []
    with open(eval_file, "r") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                ground_truth = {
                    "article": obj["metadata"]["article"],
                    "compliance_state": obj["metadata"]["compliance_state"],
                    "framework": obj["metadata"]["framework"],
                }
                # Extract severity from output
                severity = None
                if "SEVERITY:" in obj["output"]:
                    for line in obj["output"].split("\n"):
                        if line.startswith("SEVERITY:"):
                            severity = line.split(":")[1].strip()
                            break
                ground_truth["severity"] = severity
                
                examples.append({
                    "code": obj["input"],
                    "ground_truth": ground_truth,
                    "instruction": obj["instruction"],
                })
    
    return examples


# --- Scanner Integration ---

# Import scanner at module level for efficiency
_SCANNER_CACHE = None

def get_scanner():
    """Get or load the scanner module."""
    global _SCANNER_CACHE
    if _SCANNER_CACHE is None:
        sys.path.insert(0, str(Path(__file__).parent.parent / "air_blackbox_mcp"))
        try:
            from scanner import scan_code
            _SCANNER_CACHE = scan_code
        except Exception as e:
            return None
    return _SCANNER_CACHE

def run_scanner(code):
    """Run rule-based scanner on code."""
    scan_code = get_scanner()
    if scan_code is None:
        return {"error": "Failed to load scanner"}
    try:
        result = scan_code(code)
        return result
    except Exception as e:
        return {"error": str(e)}


def extract_scanner_findings(scan_result):
    """Extract predicted article and findings from scanner output.
    
    The scanner may find multiple violations. Return the most severe one.
    If all articles pass, return None for article (no finding).
    """
    if "error" in scan_result:
        return {"article": None, "severity": None, "findings": []}
    
    findings = []
    passed_articles = []
    
    for article_check in scan_result.get("articles", []):
        if article_check.get("passed", True):
            passed_articles.append(article_check["article"])
        else:
            findings.append({
                "article": article_check["article"],
                "severity": article_check.get("severity"),
                "title": article_check.get("title"),
                "passed": False,
            })
    
    # If multiple findings, return the most severe
    if findings:
        severity_rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        primary = max(findings, 
                     key=lambda x: severity_rank.get(x["severity"], 0))
        return {
            "article": primary["article"],
            "severity": primary["severity"],
            "findings": findings,
        }
    
    # No findings - all articles passed
    return {"article": None, "severity": None, "findings": passed_articles}


# --- Ollama Model Integration ---

def check_ollama_available():
    """Check if Ollama is running and air-compliance-v2 model available."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=5,
            text=True
        )
        return "air-compliance-v2" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_ollama_model(code):
    """Run Ollama fine-tuned model on code."""
    prompt = f"""Analyze this Python AI agent code for EU AI Act compliance:

{code}

Identify which article (9, 10, 11, 12, 14, or 15) has compliance issues.
Respond with: ARTICLE: X, SEVERITY: CRITICAL/HIGH/MEDIUM/LOW"""
    
    try:
        result = subprocess.run(
            ["ollama", "run", "air-compliance-v2"],
            input=prompt,
            capture_output=True,
            timeout=30,
            text=True
        )
        return result.stdout
    except Exception as e:
        return f"Error: {e}"


# --- Metrics Calculation ---

class MetricsCalculator:
    def __init__(self):
        self.article_predictions = defaultdict(list)
        self.severity_predictions = defaultdict(list)
        self.compliance_predictions = []
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))
        self.framework_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
    
    def record_article_prediction(self, ground_truth, predicted):
        """Record article prediction."""
        self.article_predictions[ground_truth].append(predicted)
        self.confusion_matrix[ground_truth][predicted] += 1
    
    def record_severity_prediction(self, ground_truth, predicted):
        """Record severity prediction."""
        if ground_truth and predicted:
            self.severity_predictions[ground_truth].append(predicted)
    
    def record_compliance_prediction(self, correct):
        """Record binary compliance prediction."""
        self.compliance_predictions.append(correct)
    
    def record_framework_accuracy(self, framework, correct):
        """Record accuracy per framework."""
        self.framework_accuracy[framework]["total"] += 1
        if correct:
            self.framework_accuracy[framework]["correct"] += 1
    
    def compute_article_metrics(self):
        """Compute per-article accuracy, precision, recall, F1."""
        metrics = {}
        for article in range(9, 16):
            if article in [9, 10, 11, 12, 14, 15]:
                predictions = self.article_predictions.get(article, [])
                if not predictions:
                    metrics[article] = {
                        "accuracy": None,
                        "precision": None,
                        "recall": None,
                        "f1": None,
                        "count": 0,
                    }
                    continue
                
                correct = sum(1 for p in predictions if p == article)
                accuracy = correct / len(predictions)
                
                # Precision: true positives / (true positives + false positives)
                tp = sum(1 for ground, pred in self.confusion_matrix.items()
                        if ground == article and pred[article] > 0)
                fp = sum(1 for ground, pred in self.confusion_matrix.items()
                        if ground != article and pred[article] > 0)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                
                # Recall: true positives / (true positives + false negatives)
                fn = sum(1 for pred in predictions if pred != article)
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # F1
                f1 = 2 * (precision * recall) / (precision + recall) \
                    if (precision + recall) > 0 else 0
                
                metrics[article] = {
                    "accuracy": round(accuracy, 3),
                    "precision": round(precision, 3),
                    "recall": round(recall, 3),
                    "f1": round(f1, 3),
                    "count": len(predictions),
                }
        
        return metrics
    
    def compute_overall_metrics(self):
        """Compute overall accuracy metrics."""
        all_predictions = sum(len(preds) for preds in 
                             self.article_predictions.values())
        if all_predictions == 0:
            return {
                "overall_accuracy": 0,
                "severity_accuracy": None,
                "compliance_accuracy": None,
            }
        
        overall_correct = sum(
            sum(1 for p in preds if p == gt)
            for gt, preds in self.article_predictions.items()
        )
        overall_accuracy = overall_correct / all_predictions
        
        # Severity accuracy
        severity_correct = sum(
            sum(1 for p in preds if p == gt)
            for gt, preds in self.severity_predictions.items()
        )
        severity_total = sum(len(preds) for preds in 
                            self.severity_predictions.values())
        severity_accuracy = (severity_correct / severity_total 
                           if severity_total > 0 else None)
        
        # Compliance (binary pass/fail)
        compliance_correct = sum(self.compliance_predictions)
        compliance_accuracy = (compliance_correct / len(self.compliance_predictions)
                             if self.compliance_predictions else None)
        
        return {
            "overall_accuracy": round(overall_accuracy, 3),
            "severity_accuracy": round(severity_accuracy, 3) 
                                 if severity_accuracy else None,
            "compliance_accuracy": round(compliance_accuracy, 3)
                                   if compliance_accuracy else None,
        }


# --- Main Evaluation ---

def run_evaluation(eval_file, use_ollama=True, test_mode=False):
    """Run full evaluation.
    
    Args:
        eval_file: Path to eval_data.jsonl
        use_ollama: Whether to try using Ollama model
        test_mode: If True, only evaluate first 10 examples for quick testing
    """
    print("=" * 70)
    print("EU AI Act compliance scanner Evaluation Benchmark")
    if test_mode:
        print("(TEST MODE - limited to 10 examples)")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading eval data from {eval_file}...")
    examples = load_eval_data(eval_file)
    if test_mode:
        examples = examples[:10]
    print(f"Loaded {len(examples)} examples")
    
    # Check Ollama
    ollama_available = False
    if use_ollama:
        print("\nChecking Ollama availability...")
        ollama_available = check_ollama_available()
        if ollama_available:
            print("✓ Ollama available with air-compliance-v2 model")
        else:
            print("✗ Ollama not available - will use rule-based baseline only")
    
    # Run evaluations
    metrics_rule = MetricsCalculator()
    metrics_ollama = MetricsCalculator() if ollama_available else None
    
    print(f"\nRunning rule-based scanner on {len(examples)} examples...")
    rule_times = []
    
    for i, example in enumerate(examples):
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(examples)}")
        
        # Rule-based scanner
        start = time.time()
        scan_result = run_scanner(example["code"])
        rule_times.append((time.time() - start) * 1000)
        
        findings = extract_scanner_findings(scan_result)
        gt_article = example["ground_truth"]["article"]
        gt_severity = example["ground_truth"]["severity"]
        
        # Record article prediction
        if findings["article"]:
            metrics_rule.record_article_prediction(gt_article, 
                                                   findings["article"])
        
        # Record severity
        if findings["severity"] and gt_severity:
            metrics_rule.record_severity_prediction(gt_severity, 
                                                    findings["severity"])
        
        # Record compliance (match = correct)
        is_correct = findings["article"] == gt_article
        metrics_rule.record_compliance_prediction(is_correct)
        metrics_rule.record_framework_accuracy(
            example["ground_truth"]["framework"],
            is_correct
        )
        
        # Ollama model
        if ollama_available:
            ollama_output = run_ollama_model(example["code"])
            # TODO: Parse ollama output and record
    
    # Compute metrics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    rule_metrics = metrics_rule.compute_overall_metrics()
    article_metrics = metrics_rule.compute_article_metrics()
    
    print("\nRULE-BASED SCANNER METRICS:")
    print(f"  Overall Accuracy:    {rule_metrics['overall_accuracy']:.1%}")
    print(f"  Severity Accuracy:   {rule_metrics['severity_accuracy']:.1%}" 
          if rule_metrics['severity_accuracy'] else "  Severity Accuracy:   N/A")
    print(f"  Compliance Accuracy: {rule_metrics['compliance_accuracy']:.1%}"
          if rule_metrics['compliance_accuracy'] else "  Compliance Accuracy: N/A")
    print(f"  Avg Time per Scan:   {mean(rule_times):.1f}ms")
    if len(rule_times) > 1:
        print(f"  Std Dev Time:        {stdev(rule_times):.1f}ms")
    
    print("\nPER-ARTICLE METRICS (Rule-based):")
    print(f"{'Article':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<10} {'Count':<8}")
    print("-" * 64)
    for article in sorted(article_metrics.keys()):
        m = article_metrics[article]
        if m["accuracy"] is not None:
            print(f"Article {article:<3} {m['accuracy']:<11.1%} {m['precision']:<11.1%} {m['recall']:<11.1%} {m['f1']:<9.3f} {m['count']:<8}")
    
    # Framework breakdown
    print("\nFRAMEWORK ACCURACY (Rule-based):")
    print(f"{'Framework':<15} {'Accuracy':<12} {'Count':<8}")
    print("-" * 35)
    for fw in sorted(metrics_rule.framework_accuracy.keys()):
        acc_data = metrics_rule.framework_accuracy[fw]
        acc = acc_data["correct"] / acc_data["total"]
        print(f"{fw:<15} {acc:<11.1%} {acc_data['total']:<8}")
    
    # Confusion matrix
    print("\nCONFUSION MATRIX (Ground Truth vs Predicted Article):")
    print(f"{'GT':<5}", end="")
    for article in range(9, 16):
        print(f"{article:<8}", end="")
    print()
    print("-" * 60)
    for gt in range(9, 16):
        if gt in [9, 10, 11, 12, 14, 15]:
            print(f"{gt:<5}", end="")
            for pred in range(9, 16):
                if pred in [9, 10, 11, 12, 14, 15]:
                    count = metrics_rule.confusion_matrix[gt][pred]
                    print(f"{count:<8}", end="")
            print()
    
    # Save results
    results = {
        "timestamp": time.time(),
        "num_examples": len(examples),
        "rule_based_metrics": rule_metrics,
        "per_article_metrics": article_metrics,
        "framework_accuracy": dict(metrics_rule.framework_accuracy),
        "confusion_matrix": {
            str(k): {str(kk): v for kk, v in vv.items()}
            for k, vv in metrics_rule.confusion_matrix.items()
        },
        "avg_scan_time_ms": mean(rule_times),
        "ollama_available": ollama_available,
    }
    
    output_file = Path(eval_file).parent / "eval_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate EU AI Act compliance scanner compliance scanner"
    )
    parser.add_argument(
        "--eval-file",
        default=str(Path(__file__).parent / "eval_data.jsonl"),
        help="Path to eval_data.jsonl"
    )
    parser.add_argument(
        "--no-ollama",
        action="store_true",
        help="Skip Ollama model evaluation"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Quick test mode (only 10 examples)"
    )
    
    args = parser.parse_args()
    run_evaluation(
        args.eval_file,
        use_ollama=not args.no_ollama,
        test_mode=args.test_mode
    )
