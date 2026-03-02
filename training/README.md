# EU AI Act compliance scanner Training Data Generation

## Overview

This directory contains the training data generation pipeline for EU AI Act compliance scanner's fine-tuned compliance model. The `generate_training_data.py` script generates **540+ synthetic training examples** for fine-tuning a Llama model to detect EU AI Act compliance issues in Python AI agent code.

## Generated Datasets

### training_data_expanded.jsonl
- **486 training examples** (90% of total)
- Size: ~393 KB
- Format: JSON Lines (one training example per line)

### eval_data.jsonl
- **54 evaluation examples** (10% of total)
- Size: ~44 KB
- Format: JSON Lines

## Training Example Structure

Each training example is a JSON object with the following structure:

```json
{
    "instruction": "Analyze this Python AI agent code for EU AI Act Article 12 (Record-Keeping) compliance.",
    "input": "<Python code snippet>",
    "output": "FINDING: No audit logging detected for tool calls...",
    "metadata": {
        "framework": "langchain",
        "article": 12,
        "compliance_state": "non_compliant",
        "template": "basic_chain"
    }
}
```

### Fields:
- **instruction**: The task description for the model
- **input**: A realistic Python code snippet (framework-specific)
- **output**: The expected compliance finding (with article, severity, evidence, recommendation)
- **metadata**: Tracking information for analysis

## Coverage

### EU AI Act Articles (6 total)
- **Article 9**: Risk Management System
- **Article 10**: Data Governance & Privacy
- **Article 11**: Technical Documentation
- **Article 12**: Record-Keeping & Audit Logging
- **Article 14**: Human Oversight & Kill Switch
- **Article 15**: Robustness & Input Validation

### AI Frameworks (5 total)
- **LangChain**: Basic chains, agents, RAG, memory
- **CrewAI**: Multi-agent orchestration with delegation
- **AutoGen**: Agent communication and group chat
- **OpenAI**: Chat completions, function calling, assistants API
- **RAG**: Vector stores and retrieval patterns

### Compliance States (3 total)
- **Non-compliant**: Missing required EU AI Act compliance scanner component
- **Partially compliant**: Has some but not complete implementation
- **Fully compliant**: Complete AIR trust layer integration

## Distribution

```
Total Examples: 540

By Article (90 examples each):
  Article 9:  90 (16.7%)
  Article 10: 90 (16.7%)
  Article 11: 90 (16.7%)
  Article 12: 90 (16.7%)
  Article 14: 90 (16.7%)
  Article 15: 90 (16.7%)

By Framework (108 examples each):
  LangChain: 108 (20.0%)
  CrewAI:    108 (20.0%)
  AutoGen:   108 (20.0%)
  OpenAI:    108 (20.0%)
  RAG:       108 (20.0%)

By Compliance State (180 examples each):
  Non-compliant:      180 (33.3%)
  Partially-compliant: 180 (33.3%)
  Fully-compliant:    180 (33.3%)
```

## Running the Script

### Requirements
- Python 3.6+ (no external dependencies)
- Write access to the training directory

### Command
```bash
python3 generate_training_data.py
```

### Output
- `training_data_expanded.jsonl` - Training examples
- `eval_data.jsonl` - Evaluation examples
- Console statistics summary

## Script Architecture

### 1. Code Templates (5 frameworks × 3+ templates each)
Located in `TEMPLATES` dictionary:
- `LANGCHAIN_TEMPLATES`: 5 patterns (basic_chain, agent_executor, retrieval_qa, langgraph_agent, memory_chain)
- `CREWAI_TEMPLATES`: 3 patterns (basic_crew, custom_tools, multi_agent_delegation)
- `AUTOGEN_TEMPLATES`: 3 patterns (two_agent, group_chat, function_calling)
- `OPENAI_TEMPLATES`: 3 patterns (chat_completions, function_calling, assistants_api)
- `RAG_TEMPLATES`: 3 patterns (chroma_rag, faiss_rag, llamaindex_engine)

### 2. Compliance Patterns (6 articles × 3 states)
Located in `FINDING_TEMPLATES` dictionary:
- Each article has 3 finding templates: non_compliant, partially_compliant, compliant
- Findings include:
  - Article number
  - Severity level (CRITICAL, HIGH, MEDIUM, LOW)
  - Evidence (line ranges)
  - Recommendation (what to add)

### 3. Variation Engine
Function `add_variation()` injects realistic diversity:
- Model names: gpt-4, gpt-3.5-turbo, claude-3-opus, claude-2, llama-2-70b
- Temperature values: 0.0, 0.3, 0.7, 0.9
- Tool names: search, calculator, weather, database, api, file_reader
- Role names: Analyst, Developer, Researcher, Planner
- Line ranges: Random 5-25 line spans

### 4. Generation Algorithm
```
For each framework:
  For each article:
    For each compliance state:
      For each variation (1-6):
        Generate unique example
```

Total combinations: 5 × 6 × 3 × 6 = **540 examples**

## EU AI Act compliance scanner Components Referenced

The training examples reference these EU AI Act compliance scanner trust layers:

| Article | Component | Purpose |
|---------|-----------|---------|
| 9 | risk classification system | Risk classification + user consent |
| 10 | PII protection layer | PII detection + encryption |
| 11 | structured audit logging | Decision records + model cards |
| 12 | HMAC audit chain | Tamper-evident logging |
| 14 | HITL + kill switch | Human-in-the-loop + emergency stop |
| 15 | prompt injection detection | Prompt injection protection |

## Fine-Tuning Integration

To fine-tune a Llama model with this data:

```bash
# Using Hugging Face Transformers
python3 -m transformers.training_script \
    --model llama-2-7b \
    --train_file training_data_expanded.jsonl \
    --eval_file eval_data.jsonl \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-5
```

## Reproducibility

The script uses `random.seed(42)` for reproducibility. Running the script multiple times will generate the same examples in the same order (though randomized by framework/article/state).

## File Locations

```
/Users/jasonshotwell/Desktop/air-blackbox-mcp/training/
├── generate_training_data.py        # Main generation script
├── training_data_expanded.jsonl     # 486 training examples
├── eval_data.jsonl                  # 54 evaluation examples
└── README.md                         # This file
```

## Customization

To generate more examples or adjust the distribution:

1. **Increase variations**: Change `variations_per_combination` variable (currently 6)
2. **Add frameworks**: Add to `TEMPLATES` dictionary
3. **Add articles**: Extend `ARTICLE_DESCRIPTIONS` and `FINDING_TEMPLATES`
4. **Adjust train/eval split**: Modify the 0.9 ratio in `main()`

Example: To generate 1000 examples, set `variations_per_combination = 12` (doubles the total).

## Statistics from Latest Run

```
Generated: 540 examples
Training set: 486 examples (90%)
Evaluation set: 54 examples (10%)
Balanced across:
  - 6 articles (90 each)
  - 5 frameworks (108 each)
  - 3 compliance states (180 each)
```

## Next Steps

1. **Fine-tune model**: Use training_data_expanded.jsonl with Ollama or Hugging Face
2. **Evaluate**: Use eval_data.jsonl to measure accuracy
3. **Iterate**: Regenerate with domain-specific templates as needed
4. **Deploy**: Use fine-tuned model in air-compliance-v2 Ollama container

---

**Generated**: 2026-02-28  
**Total Examples**: 540  
**Frameworks**: 5  
**Articles**: 6  
**States**: 3  
