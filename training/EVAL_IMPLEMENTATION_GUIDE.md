# Evaluation Benchmark Implementation Guide

## Overview

The evaluation benchmark script (`eval_benchmark.py`) provides a complete testing framework for EU AI Act compliance scanner's compliance scanner and fine-tuned model. This guide explains the implementation in detail.

## Architecture

```
eval_benchmark.py
├── Data Loading
│   └── load_eval_data() → 54 examples from eval_data.jsonl
├── Rule-Based Scanning (Baseline)
│   ├── get_scanner() → Load scanner module once
│   ├── run_scanner() → Execute on code
│   └── extract_scanner_findings() → Parse results
├── Ollama Model (Optional)
│   ├── check_ollama_available() → Detect model
│   └── run_ollama_model() → Get prediction
├── Metrics Calculation
│   ├── MetricsCalculator class
│   ├── compute_overall_metrics()
│   └── compute_article_metrics()
└── Output
    ├── Console tables
    └── eval_results.json
```

## Core Classes and Functions

### MetricsCalculator

Tracks all predictions and computes metrics:

```python
class MetricsCalculator:
    def __init__(self):
        # Store predictions by article
        self.article_predictions = defaultdict(list)  # gt_article → [pred1, pred2, ...]
        self.severity_predictions = defaultdict(list)  # gt_severity → [pred1, pred2, ...]
        self.compliance_predictions = []               # [True/False, ...]
        self.confusion_matrix = defaultdict(...)       # gt → pred → count
        self.framework_accuracy = defaultdict(...)     # framework → {correct, total}
```

**Methods**:
- `record_article_prediction(gt, pred)`: Store prediction, update confusion matrix
- `record_severity_prediction(gt, pred)`: Track severity matching
- `record_compliance_prediction(correct)`: Binary pass/fail tracking
- `record_framework_accuracy(framework, correct)`: Per-framework stats
- `compute_article_metrics()`: Calculate accuracy/precision/recall/F1 per article
- `compute_overall_metrics()`: Aggregate statistics across all articles

### Data Loading

```python
def load_eval_data(eval_file):
    """Load eval_data.jsonl and parse ground truth."""
    # Returns list of dicts:
    # {
    #     "code": "<Python source>",
    #     "ground_truth": {
    #         "article": 10,
    #         "compliance_state": "non_compliant",
    #         "framework": "langchain",
    #         "severity": "CRITICAL"
    #     },
    #     "instruction": "..."
    # }
```

Key features:
- Parses JSON lines format
- Extracts SEVERITY from output text
- Handles all compliance states

### Scanner Integration

**Caching optimization**:
```python
_SCANNER_CACHE = None

def get_scanner():
    """Load scanner module once at startup."""
    global _SCANNER_CACHE
    if _SCANNER_CACHE is None:
        from scanner import scan_code
        _SCANNER_CACHE = scan_code
    return _SCANNER_CACHE
```

Benefits:
- Imports module only once (not per iteration)
- ~0.1ms per scan (vs ~5ms with reload)
- Huge speedup for 54 examples

**Prediction extraction**:
```python
def extract_scanner_findings(scan_result):
    """Parse scanner output and extract primary finding."""
    # 1. Collect all failed articles
    findings = [article for article in articles if not article["passed"]]
    
    # 2. If multiple findings, return most severe
    if findings:
        severity_rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        primary = max(findings, key=lambda x: severity_rank[x["severity"]])
        return {
            "article": primary["article"],
            "severity": primary["severity"],
            "findings": findings,
        }
    
    # 3. If no findings, code is compliant
    return {"article": None, "severity": None, "findings": []}
```

Logic:
- Multiple articles may fail; report the most severe
- If all pass, return None (no violation detected)
- Matches ground truth evaluation

### Ollama Integration

```python
def check_ollama_available():
    """Check if Ollama running with air-compliance-v2 model."""
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        timeout=5,
        text=True
    )
    return "air-compliance-v2" in result.stdout

def run_ollama_model(code):
    """Send code to Ollama and get prediction."""
    prompt = f"""Analyze Python AI agent code...
    
Respond with: ARTICLE: X, SEVERITY: CRITICAL/HIGH/MEDIUM/LOW"""
    
    result = subprocess.run(
        ["ollama", "run", "air-compliance-v2"],
        input=prompt,
        capture_output=True,
        timeout=30,
        text=True
    )
    return result.stdout
```

Features:
- Automatic detection (graceful fallback if not available)
- 30-second timeout per model invocation
- Returns raw output (parsed externally if needed)

### Metrics Computation

**Overall metrics**:
```python
def compute_overall_metrics(self):
    """Aggregate accuracy across all articles."""
    # Overall accuracy: correct predictions / total predictions
    overall_correct = sum(
        sum(1 for p in preds if p == gt)
        for gt, preds in self.article_predictions.items()
    )
    overall_accuracy = overall_correct / total_predictions
    
    # Severity accuracy: severity predictions that match
    # (only for cases where both ground truth and prediction have severity)
    
    # Compliance accuracy: binary is_violation prediction correct
```

**Per-article metrics**:
```python
def compute_article_metrics(self):
    """Compute accuracy, precision, recall, F1 for each article."""
    for article in [9, 10, 11, 12, 14, 15]:
        predictions = self.article_predictions[article]
        correct = sum(1 for p in predictions if p == article)
        
        # Accuracy: within this article's predictions
        accuracy = correct / len(predictions)
        
        # Precision: TP / (TP + FP)
        # "When I predict article X, am I right?"
        tp = count_true_positives(article)
        fp = count_false_positives(article)
        precision = tp / (tp + fp)
        
        # Recall: TP / (TP + FN)
        # "When article X should be found, do I find it?"
        fn = count_false_negatives(article)
        recall = tp / (tp + fn)
        
        # F1: 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall)
```

## Evaluation Flow

### Step 1: Load Data
```python
examples = load_eval_data(eval_file)  # 54 examples by default, 10 in test mode
```

### Step 2: Initialize Metrics Tracker
```python
metrics_rule = MetricsCalculator()
metrics_ollama = MetricsCalculator() if ollama_available else None
```

### Step 3: Run Scanner on Each Example
```python
for i, example in enumerate(examples):
    # Rule-based scanner
    scan_result = run_scanner(example["code"])
    findings = extract_scanner_findings(scan_result)
    
    # Record metrics
    gt_article = example["ground_truth"]["article"]
    metrics_rule.record_article_prediction(gt_article, findings["article"])
    metrics_rule.record_framework_accuracy(
        example["ground_truth"]["framework"],
        findings["article"] == gt_article
    )
    
    # Ollama model (optional)
    if ollama_available:
        ollama_output = run_ollama_model(example["code"])
        # Parse and record (TODO in current implementation)
```

### Step 4: Compute and Display Metrics
```python
rule_metrics = metrics_rule.compute_overall_metrics()
article_metrics = metrics_rule.compute_article_metrics()

# Display as formatted tables
print("RULE-BASED SCANNER METRICS:")
print(f"Overall Accuracy: {rule_metrics['overall_accuracy']:.1%}")
# ... etc.

# Save to JSON
results = {
    "timestamp": time.time(),
    "num_examples": len(examples),
    "rule_based_metrics": rule_metrics,
    "per_article_metrics": article_metrics,
    # ... etc.
}
with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Input Data Format (eval_data.jsonl)

Each line is a JSON object representing one test case:

```json
{
  "instruction": "Analyze this Python AI agent code for EU AI Act Article 10 compliance.",
  "input": "from autogen import AssistantAgent...",
  "output": "FINDING: No PII protection...\nARTICLE: 10\nSEVERITY: CRITICAL\n...",
  "metadata": {
    "framework": "autogen",
    "article": 10,
    "compliance_state": "non_compliant",
    "template": "custom_tools"
  }
}
```

**Ground truth extraction**:
- `article`: From metadata.article
- `compliance_state`: From metadata.compliance_state
- `severity`: Extracted from "SEVERITY: X" in output text
- `framework`: From metadata.framework

## Output Format (eval_results.json)

```json
{
  "timestamp": 1772257869.788,
  "num_examples": 54,
  "rule_based_metrics": {
    "overall_accuracy": 0.074,
    "severity_accuracy": 0.2,
    "compliance_accuracy": 0.074
  },
  "per_article_metrics": {
    "9": {"accuracy": 0.0, "precision": 0, "recall": 0.0, "f1": 0, "count": 13},
    "12": {"accuracy": 1.0, "precision": 0.167, "recall": 1.0, "f1": 0.286, "count": 4},
    ...
  },
  "framework_accuracy": {
    "autogen": {"correct": 1, "total": 12},
    ...
  },
  "confusion_matrix": {
    "9": {"12": 13},
    "10": {"12": 12},
    ...
  },
  "avg_scan_time_ms": 0.1,
  "ollama_available": false
}
```

## Performance Characteristics

### Memory Usage
- **eval_data.jsonl**: ~450KB (54 examples)
- **In-memory**: ~10MB (parsed examples + metrics)
- **Output JSON**: ~5KB (eval_results.json)

### Execution Time
- **Rule-based baseline**: ~50-100ms total (0.1ms per scan × 54)
- **Ollama model**: ~30-60 seconds total (500-1000ms per scan × 54)
- **Startup**: ~100-200ms (module imports)

### Complexity Analysis
- **Time complexity**: O(n × m) where n=examples, m=articles (6)
- **Space complexity**: O(n × m) for storing predictions

## Extension Points

### Adding New Metrics

1. **Add to MetricsCalculator**:
```python
def record_model_confidence(self, confidence_score):
    self.confidence_scores.append(confidence_score)
```

2. **Add computation**:
```python
def compute_confidence_metrics(self):
    avg_confidence = mean(self.confidence_scores)
    # ... calibration analysis, etc.
```

3. **Output in run_evaluation**:
```python
confidence_metrics = metrics_rule.compute_confidence_metrics()
results["confidence_metrics"] = confidence_metrics
```

### Adding New Evaluators

Beyond rule-based and Ollama:

```python
def run_baseline_evaluator(code):
    """Alternative evaluation method."""
    # ... custom logic
    return {"article": X, "severity": Y}

# In run_evaluation():
for example in examples:
    baseline_result = run_baseline_evaluator(example["code"])
    metrics_baseline.record_article_prediction(gt, baseline_result["article"])
```

### Parsing Ollama Output

Current implementation sends code to Ollama but doesn't parse response. To enable:

```python
def parse_ollama_output(output_text):
    """Extract ARTICLE and SEVERITY from Ollama response."""
    result = {"article": None, "severity": None}
    for line in output_text.split("\n"):
        if line.startswith("ARTICLE:"):
            result["article"] = int(line.split(":")[1].strip())
        elif line.startswith("SEVERITY:"):
            result["severity"] = line.split(":")[1].strip()
    return result

# In run_evaluation():
if ollama_available:
    for example in examples:
        output = run_ollama_model(example["code"])
        prediction = parse_ollama_output(output)
        metrics_ollama.record_article_prediction(gt, prediction["article"])
```

## Testing & Validation

### Unit Test Pattern
```python
# Test metrics computation
calc = MetricsCalculator()
calc.record_article_prediction(9, 9)  # correct
calc.record_article_prediction(10, 12)  # wrong

metrics = calc.compute_overall_metrics()
assert metrics["overall_accuracy"] == 0.5
```

### Integration Test
```bash
# Quick validation
python3 eval_benchmark.py --test-mode --no-ollama

# Should complete in < 5 seconds
# Output should show results table
```

## Debugging Guide

### Issue: Script hangs
**Cause**: Slow module import  
**Solution**: Use `--test-mode` first (10 examples)

### Issue: All predictions are Article 12
**Cause**: Rule-based scanner bias (this is expected!)  
**Solution**: This is the baseline. Fine-tuned model should improve

### Issue: Ollama not detected
**Cause**: Model not running or wrong name  
**Solution**: Run `ollama list` to check; use `--no-ollama` to skip

### Issue: JSON parse errors
**Cause**: Malformed eval_data.jsonl  
**Solution**: Verify file with `python3 -m json --validate eval_data.jsonl`

## Future Improvements

1. **Markdown Report Generation**: Pretty HTML/PDF reports
2. **Cross-Validation**: k-fold evaluation with shuffled examples
3. **Confidence Scoring**: Model confidence calibration analysis
4. **Per-Template Analysis**: Breakdown by code template type
5. **ROC Curves**: Plot precision/recall tradeoffs
6. **Model Ensemble**: Compare multiple models side-by-side
7. **Incremental Learning**: Track accuracy as model improves
8. **Cost Analysis**: Complexity/accuracy tradeoffs

## References

- **Confusion Matrix**: Shows true positives, false positives, false negatives per class
- **Precision**: TP / (TP + FP) - "When I say X, am I right?"
- **Recall**: TP / (TP + FN) - "When X happens, do I catch it?"
- **F1-Score**: Harmonic mean - balances precision and recall
- **Accuracy**: (TP + TN) / All - overall correctness (misleading for imbalanced data)
