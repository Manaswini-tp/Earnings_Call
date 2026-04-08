# 📊 Earnings Call Q&A Environment

An OpenEnv-compliant environment for testing AI agents on real-world financial analysis tasks.

## 🎯 Problem Statement

Financial analysts spend hours manually extracting insights from earnings call transcripts. This environment benchmarks AI agents on this high-stakes task, testing their ability to:
- Extract precise financial metrics
- Identify key business risks
- Synthesize conflicting forward guidance

## 🌍 Real-World Relevance

This environment directly reflects tasks performed by:

- 📈 Equity research analysts  
- 🏦 Investment banks  
- 💼 Hedge funds  
- 🤖 Financial AI systems  

It enables benchmarking of agents in **decision-critical scenarios where mistakes can have monetary impact**.

---

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


## 🎬 Action Space

```python
Action(
    answer: str
)
```

---

## 🎯 Reward Function

```
reward =
    0.5 * factual_accuracy
  + 0.3 * completeness
  - hallucination_penalty
  - verbosity_penalty
```

---

## 🧪 Grading Strategy

- **Extract** → Numeric tolerance matching  
- **Identify** → Keyword / overlap scoring  
- **Synthesize** → LLM-based evaluation  

All tasks produce scores in **[0.0, 1.0]**

---

## 🔄 OpenEnv API Compliance

- POST /reset  
- POST /step  
- GET /state  
- GET /health  

---
## ⚙️ Setup

```bash
pip install -r requirements.txt
python app.py
```

---

## 🐳 Docker

```bash
docker build -t earnings-env .
docker run -p 7860:7860 earnings-env
```

---

## 📊 Example Output

```
[START] task=extract env=earnings_call_env model=gpt-4o-mini
[STEP] step=1 action=EPS is 1.40 reward=1.00 done=true error=null
[END] success=true steps=1 score=1.00 rewards=1.00
```

---

## 🚀 Highlights

- Real-world financial task  
- Deterministic grading  
- Multi-difficulty setup  
- Hallucination-aware rewards  
- OpenEnv compliant  

---