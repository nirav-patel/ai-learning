# Reflection Pattern Demo

A learning POC that implements the **reflection design pattern** for agentic AI workflows, applied to data visualization.

## What it does

```
Instruction (natural language)
        │
        ▼
┌───────────────────┐
│ Stage 1: Generate │  Amazon Nova Lite (fast, cheap)
│   Chart Code V1   │  → code_output/code_v1.py
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Stage 2: Execute  │  Local Python exec
│   → chart_v1.png  │  → images/chart_v1.png
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Stage 3: Reflect  │  Claude Sonnet (robust, multimodal)
│   Evaluate chart  │  → JSON feedback (score, issues, suggestions)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Stage 4: Refine   │  Amazon Nova Lite (fast, cheap)
│   Code V2         │  → code_output/code_v2.py
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Stage 5: Execute  │  Local Python exec
│   → chart_v2.png  │  → images/chart_v2.png
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Stage 6: Compare  │  Pillow side-by-side stitch
│   Comparison img  │  → images/comparison.png
└───────────────────┘
```

## Models used

| Role | Model | Reason |
|------|-------|--------|
| Generator | `us.amazon.nova-2-lite-v1:0` | Fast, lightweight — good for code generation |
| Evaluator | `us.anthropic.claude-sonnet-4-6` | Robust multimodal — understands chart images |

## File structure

```
reflection-demo/
├── main.py              # Run the full workflow end-to-end
├── workflow.py          # Core reflection stages (generate → reflect + refine)
├── bedrock_client.py    # AWS Bedrock Converse API wrapper
├── utils.py             # Data loading, code execution, image utilities
├── coffee_sales.csv     # Sample dataset
├── images/              # Generated chart images (auto-created)
│   ├── chart_v1.png
│   ├── chart_v2.png
│   └── comparison.png
└── code_output/         # Generated Python code for review (auto-created)
    ├── code_v1.py
    └── code_v2.py
```

## Setup

AWS credentials must be configured (standard boto3 credential chain — e.g. `~/.aws/credentials`, env vars, or an IAM role).

```bash
# Install dependencies (already in parent requirements.txt)
pip install boto3 pandas matplotlib Pillow

# Run the workflow
python main.py
```

## Key learning points

- **Reflection pattern**: A second, more capable model reviews the output of the first and provides structured feedback.
- **Model specialisation**: Use a cheap/fast model for generation and a powerful model for evaluation.
- **Multimodal evaluation**: Claude Sonnet analyses the *rendered chart image*, not just the code.
- **Artifact persistence**: Both versions of the code and images are saved so improvements are easy to compare.
