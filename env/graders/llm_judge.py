# env/graders/llm_judge.py
import json

def llm_judge_grader(answer: str, question: str, transcript: str, rubric: dict = None) -> dict:
    """
    Simplified LLM judge for synthesis tasks (Task 3)
    Uses basic heuristics since we don't want to call external APIs
    """
    if rubric is None:
        rubric = {
            "factual_grounding": 0.4,
            "reasoning_quality": 0.3,
            "completeness": 0.3
        }
    
    breakdown = {}
    
    # Check factual grounding - look for claims supported by transcript
    sentences = answer.split('.')
    grounded_claims = 0
    total_claims = 0
    
    for sentence in sentences:
        if len(sentence.strip()) > 10:  # Treat as a claim
            total_claims += 1
            # Simple check: does this sentence contain words from the transcript?
            sentence_lower = sentence.lower()
            transcript_preview = transcript.lower()[:5000]  # Check first part
            
            # Extract potential key terms (numbers, percentages, key phrases)
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?%?', sentence)
            if numbers:
                for num in numbers:
                    if num in transcript_preview:
                        grounded_claims += 1
                        break
            else:
                # Check for key phrases
                key_words = ['revenue', 'earnings', 'growth', 'margin', 'guidance', 'quarter']
                for word in key_words:
                    if word in sentence_lower and word in transcript_preview:
                        grounded_claims += 1
                        break
    
    factual_score = grounded_claims / max(total_claims, 1)
    breakdown['factual_grounding'] = factual_score * rubric['factual_grounding']
    
    # Check reasoning quality - look for logical connectors
    reasoning_indicators = ['because', 'therefore', 'however', 'although', 'but', 'while']
    reasoning_count = sum(1 for word in reasoning_indicators if word in answer.lower())
    reasoning_score = min(reasoning_count / 3, 1.0)  # Cap at 3 indicators
    breakdown['reasoning_quality'] = reasoning_score * rubric['reasoning_quality']
    
    # Check completeness - answer length relative to question complexity
    expected_min_length = 50  # characters
    length_score = min(len(answer) / expected_min_length, 1.0)
    breakdown['completeness'] = length_score * rubric['completeness']
    
    total_score = sum(breakdown.values())
    
    feedback = f"Synthesis Score: {total_score:.3f}\n"
    feedback += f"  Factual grounding: {breakdown['factual_grounding']:.3f}\n"
    feedback += f"  Reasoning quality: {breakdown['reasoning_quality']:.3f}\n"
    feedback += f"  Completeness: {breakdown['completeness']:.3f}"
    total_score = max(0.01, min(0.99, total_score))
    return {
        "score": total_score,
        "breakdown": breakdown,
        "feedback": feedback
    }