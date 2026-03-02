# Evaluation Benchmark for EU AI Act compliance scanner Compliance Model

## Quick Start

```bash
# Quick test with 10 examples (< 5 seconds)
python3 eval_benchmark.py --test-mode --no-ollama

# Full evaluation (54 examples, < 2 seconds for rule-based)
python3 eval_benchmark.py --no-ollama

# With Ollama model (if available)
python3 eval_benchmark.py
```

## What This Script Does

The evaluation benchmark (`eval_benchmark.py`) tests the EU AI Act compliance scanner compliance scanner against 54 held-out test cases from `eval_data.jsonl`. It measures how well the scanner can:

1. **Identify the correct article** - Which EU AI Act article (9, 10, 11, 12, 14, or 15) has a compliance issue
2. **Classify severity** - Whether the issue is CRITICAL, HIGH, MEDIUM, or LOW
3. **Detect compliance gaps** - Whether the code is compliant or non-compliant

## Test Data

The `eval_data.jsonl` file contains 54 labeled examples across 5 frameworks:

| Framework | Count | Articles |
|-----------|-------|----------|
| langchain | 11    | 9, 10, 11, 12, 14, 15 |
| autogen   | 12    | 9, 10, 11, 12, 14, 15 |
| crewai    | 9     | 9, 10, 14, 15 |
| openai    | 13    | 9, 10, 11, 12, 14, 15 |
| rag       | 9     | 9, 10, 11, 12, 14, 15 |

Each example is a Python code snippet with ground truth labels:
- **Article**: Which article it tests (9, 10, 11, 12, 14, or 15)
- **Compliance State**: compliant, non_compliant, or partially_compliant
- **Severity**: CRITICAL, HIGH, MEDIUM, or LOW (for non-compliant cases)

## Metrics Explained

### Overall Metrics

```
Overall Accuracy:    7.4%
Severity Accuracy:   20.0%
Compliance Accuracy: 7.4%
Avg Time per Scan:   0.1ms
```

- **Overall Accuracy**: % of times scanner identified the correct article
- **Severity Accuracy**: % of times scanner got the severity level right
- **Compliance Accuracy**: Binary - does scanner correctly identify violation vs. pass
- **Avg Time**: Speed in milliseconds per code scan

### Per-Article Metrics

```
Article    Accuracy     Precision    Recall       F1         Count   
Article 9   0.0%        0.0%        0.0%        0.000     13
Article 12  100.0%      16.7%       100.0%      0.286     4
```

- **Accuracy**: Correct predictions / Total predictions for that article
- **Precision**: True positives / (True positives + False positives)
  - "When I say Article 12, am I right?" (16.7% of Article 12 predictions are correct)
- **Recall**: True positives / (True positives + False negatives)
  - "When Article 12 should be found, do I find it?" (100% - found all Article 12 violations)
- **F1**: Harmonic mean of precision and recall (0.286 = balanced measure)
- **Count**: How many test examples for that article

### Framework Breakdown

```
Framework       Accuracy     Count
autogen         8.3%         12
crewai          0.0%         9
langchain       0.0%         10
openai          15.4%        13
rag             10.0%         10
```

Shows which frameworks the scanner handles better. OpenAI has highest accuracy (15.4%) on this test set.

### Confusion Matrix

```
GT   9       10      11      12      13      14      15      
9    0       0       0       13      0       0       
10   0       0       0       12      0       0       
12   0       0       0       4       0       0       
```

Shows misclassifications:
- Ground truth (GT) = actual article
- Columns = scanner's prediction
- Most predictions go to Article 12 (51/54 examples), suggesting rule patterns for Article 12 are too broad

## Baseline Performance (Current Results)

The current rule-based scanner achieves:

| Metric | Value |
|--------|-------|
| Overall Accuracy | 7.4% |
| Precision (Article 12) | 16.7% |
| Recall (Article 12) | 100% |
| Speed | 0.1ms per scan |

**Key Finding**: The scanner heavily over-predicts Article 12 violations. This is the baseline for comparison with the fine-tuned Ollama model.

## Output Files

### eval_results.json
Structured JSON with all metrics. Example:

```json
{
  "timestamp": 1772257869.788162,
  "num_examples": 54,
  "rule_based_metrics": {
    "overall_accuracy": 0.074,
    "severity_accuracy": 0.2,
    "compliance_accuracy": 0.074
  },
  "per_article_metrics": {
    "9": {
      "accuracy": 0.0,
      "precision": 0,
      "recall": 0.0,
      "f1": 0,
      "count": 13
    },
    ...
  },
  "framework_accuracy": {
    "autogen": {"correct": 1, "total": 12},
    ...
  },
  "confusion_matrix": {...},
  "avg_scan_time_ms": 0.1,
  "ollama_available": false
}
```

## Command-Line Options

```bash
python3 eval_benchmark.py [options]

Options:
  --eval-file PATH      Path to eval_data.jsonl (default: ./eval_data.jsonl)
  --no-ollama           Skip Ollama model evaluation (faster)
  --test-mode           Quick test with first 10 examples only
```

## Understanding the Metrics

### Why is precision low (16.7%) but recall high (100%)?

The scanner is finding ALL Article 12 violations (recall=100%) but is also incorrectly flagging other articles as Article 12. So when it predicts Article 12, it's right only 16.7% of the time.

**Goal for fine-tuned model**: Improve precision without losing recall.

### Why low overall accuracy (7.4%)?

The rule-based scanner is biased toward Article 12. Only 4 test examples are actually Article 12. So most predictions are wrong because the scanner over-predicts Article 12.

## Comparing with Fine-Tuned Model

When Ollama is available with `air-compliance-v2` model:

1. Script tests each example through the model
2. Parses ARTICLE and SEVERITY from model output
3. Computes same metrics for model
4. Compares: rule-based vs. model accuracy, precision, recall, F1
5. Shows speed difference (milliseconds per scan)

**Expected Improvement**: Model should have:
- Higher overall accuracy (reduce Article 12 over-prediction)
- More balanced precision/recall across articles
- Lower speed (500-1000ms vs 0.1ms)

## Implementation Details

### MetricsCalculator Class

Core class that tracks predictions and computes metrics:

```python
calc = MetricsCalculator()
calc.record_article_prediction(ground_truth=10, predicted=12)
calc.record_severity_prediction(ground_truth="HIGH", predicted="CRITICAL")
calc.record_compliance_prediction(correct=False)
calc.record_framework_accuracy("langchain", correct=True)

metrics = calc.compute_overall_metrics()
article_metrics = calc.compute_article_metrics()
```

### Key Functions

- `load_eval_data()`: Parse eval_data.jsonl and extract ground truth
- `run_scanner()`: Execute rule-based scanner on code
- `extract_scanner_findings()`: Parse scanner output for article/severity
- `check_ollama_available()`: Detect if Ollama model is running
- `run_ollama_model()`: Send code to Ollama and get prediction

## Debugging Tips

### Script hangs or runs very slowly
- Check: `python3 -c "from pathlib import Path; import sys; sys.path.insert(0, str(Path('.').parent / 'air_blackbox_mcp')); from scanner import scan_code; print('OK')"`
- If slow, scanner import is the bottleneck - script caches it now

### Results show only Article 12 predictions
- This is expected! Rule-based scanner is biased toward Article 12
- Run `--test-mode` first to verify on 10 examples
- Compare with `eval_results.json` to see full breakdown

### Ollama model not detected
- Verify Ollama is running: `ollama list`
- Check model is installed: `ollama list | grep air-compliance-v2`
- If not installed: `ollama pull air-compliance-v2` (or train/upload it)
- Use `--no-ollama` flag to skip Ollama evaluation

## Next Steps

1. **Baseline Established**: Current rule-based accuracy is 7.4%
2. **Train Model**: Use training data to fine-tune Ollama model
3. **Compare**: Run benchmark again and compare accuracy improvements
4. **Iterate**: Refine model based on confusion matrix insights
5. **Deploy**: Use improved model in scanner.py

## Files Involved

| File | Purpose |
|------|---------|
| `eval_benchmark.py` | Main evaluation script |
| `eval_data.jsonl` | 54 test examples with ground truth |
| `eval_results.json` | Output with all metrics |
| `scanner.py` | Rule-based scanner being evaluated |
| `training_data_expanded.jsonl` | Full training set |
| `generate_training_data.py` | Script that creates eval_data.jsonl |

## References

- **EU AI Act Articles**: 9 (Risk Management), 10 (Data Governance), 11 (Documentation), 12 (Record-Keeping), 14 (Human Oversight), 15 (Robustness)
- **Precision/Recall/F1**: Standard ML classification metrics
- **Confusion Matrix**: Tool for analyzing multi-class classification errors
