# EU AI Act compliance scanner Evaluation Benchmark Script

## Overview

The `eval_benchmark.py` script provides a comprehensive evaluation framework for testing EU AI Act compliance scanner's compliance scanner against a held-out test set of 54 examples.

## Features

### Data Loading
- **Input**: `eval_data.jsonl` containing 54 labeled examples
- **Ground Truth**: Each example contains:
  - Article number (9, 10, 11, 12, 14, 15)
  - Compliance state (compliant, non_compliant, partially_compliant)
  - Framework type (langchain, crewai, autogen, openai, rag)
  - Severity level (CRITICAL, HIGH, MEDIUM, LOW)

### Evaluation Methods

#### 1. Rule-Based Scanner (Baseline)
- Tests the existing pattern-matching scanner
- Analyzes code against 6 EU AI Act articles
- Returns primary violation or "no findings" if all pass
- Measures accuracy, precision, recall, F1 score per article

#### 2. Ollama Fine-Tuned Model (Optional)
- Automatically detects if `air-compliance-v2` model available
- Compares model predictions against rule-based baseline
- Speed comparison (milliseconds per scan)

### Metrics Calculated

**Overall Metrics:**
- Overall accuracy (predicted article matches ground truth)
- Severity accuracy (predicted severity matches ground truth)
- Compliance accuracy (binary: violation detected or not)
- Average scan time in milliseconds

**Per-Article Metrics:**
- Accuracy: Correct predictions / Total predictions
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: Harmonic mean of precision and recall
- Count: Number of test examples for that article

**Framework Accuracy:**
- Breakdown by framework (langchain, autogen, crewai, openai, rag)

**Confusion Matrix:**
- Shows which articles get misclassified as which other articles

## Usage

### Quick Test (10 examples)
```bash
python3 eval_benchmark.py --test-mode --no-ollama
```

### Full Evaluation (all 54 examples)
```bash
python3 eval_benchmark.py --no-ollama
```

### With Ollama Model (if available)
```bash
python3 eval_benchmark.py
```

## Output

### Console Output
- Formatted tables with metrics
- Framework breakdown
- Confusion matrix visualization
- Performance timing

### Files Generated
1. **eval_results.json**: Structured results for programmatic use
   - Overall metrics
   - Per-article metrics
   - Framework accuracy
   - Confusion matrix
   - Average scan time

2. **eval_report.md**: (Future) Human-readable markdown report

## Implementation Details

### Metrics Calculator Class
Tracks predictions and computes standard ML metrics:
- `record_article_prediction()`: Store predicted vs. ground truth article
- `record_severity_prediction()`: Track severity accuracy
- `record_compliance_prediction()`: Binary pass/fail tracking
- `record_framework_accuracy()`: Per-framework statistics

### Finding Extraction
The scanner may identify multiple article violations. The logic:
1. Collect all articles where `passed` = False
2. Sort by severity (CRITICAL > HIGH > MEDIUM > LOW)
3. Return the most severe violation as the "primary finding"
4. If all articles pass, return None for article (compliant code)

### Performance Optimization
- Scanner is loaded once at module level (`get_scanner()` cache)
- Batch processing of all 54 examples
- Efficient metrics computation using defaultdict

## Test Mode

For quick validation, use `--test-mode` to:
- Load only first 10 examples
- Run full benchmark in <5 seconds
- Verify script correctness before full eval

## Integration with Ollama

When Ollama is available:
1. Detects `air-compliance-v2` model
2. Sends code through Ollama for prediction
3. Parses ARTICLE and SEVERITY from output
4. Compares model accuracy vs. rule-based baseline

## Ground Truth Format (eval_data.jsonl)

Each line is a JSON object:
```json
{
  "instruction": "Analyze this Python AI agent code for EU AI Act Article X...",
  "input": "<Python code snippet>",
  "output": "FINDING: ...\nARTICLE: X\nSEVERITY: CRITICAL/HIGH/MEDIUM/LOW\n...",
  "metadata": {
    "framework": "langchain|crewai|autogen|openai|rag",
    "article": 9|10|11|12|14|15,
    "compliance_state": "compliant|non_compliant|partially_compliant",
    "template": "template name"
  }
}
```

## Key Limitations

1. **Scanner Scope**: Rule-based scanner uses pattern matching, not semantic analysis
2. **Ground Truth**: Each example has ONE ground truth article (what it tests)
3. **No Cross-Article Testing**: An example may have multiple violations, but only one is labeled
4. **Speed Baseline**: Ollama comparison only works if model is running locally

## Future Improvements

1. Add markdown report generation with charts
2. Support for custom evaluation sets
3. Cross-validation analysis
4. Per-template breakdown
5. Confidence scoring for model predictions
6. Comparison with other compliance tools

## Dependencies

- Python 3.7+
- No external packages (uses only stdlib)
- Optional: Ollama with air-compliance-v2 model

## Performance Characteristics

- Rule-based scanner: ~0.2ms per scan (54 examples < 1 second)
- Ollama model: ~500-1000ms per scan (54 examples ~ 30-60 seconds)
- Memory: Minimal (< 10MB for all data)
- Storage: eval_results.json ~ 5KB
