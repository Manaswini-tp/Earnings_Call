# 📊 Earnings Call Q&A Environment

An OpenEnv-compliant environment for testing AI agents on real-world financial analysis tasks.

## 🎯 Problem Statement

Financial analysts spend hours manually extracting insights from earnings call transcripts. This environment benchmarks AI agents on this high-stakes task, testing their ability to:
- Extract precise financial metrics
- Identify key business risks
- Synthesize conflicting forward guidance

## 🏗️ Environment Design

### Three Tasks (Easy → Hard)

| Task | Difficulty | Description | Expected Score |
|------|------------|-------------|----------------|
| **Extract** | 🟢 Easy | Extract EPS and revenue numbers | 0.95+ |
| **Identify** | 🟡 Medium | List top 3 risk factors from transcript | 0.75+ |
| **Synthesize** | 🔴 Hard | Reconcile conflicting guidance into net outlook | 0.70+ |

### Observation Space
```python
Observation(
    transcript: str,      # Full earnings call transcript
    question: str,        # Analyst question
    task_type: str,       # "extract", "identify", or "synthesize"
    turn_number: int      # Current turn (max 1 per task)
)