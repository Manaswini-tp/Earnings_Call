# env/graders/exact_match.py
def exact_match_grader(answer: str, ground_truth: dict, tolerance: float = 0.01) -> dict:
    """
    Grade exact match questions (Task 1)
    """
    score = 0.0
    breakdown = {}
    feedback = []
    
    for key, expected in ground_truth.items():
        try:
            # Try to parse as float for numeric comparison
            expected_num = float(expected)
            try:
                answer_num = float(answer.strip())
                if abs(answer_num - expected_num) / expected_num <= tolerance:
                    score += 0.5
                    breakdown[key] = 1.0
                    feedback.append(f"✓ {key}: {answer} matches expected {expected}")
                else:
                    breakdown[key] = 0.0
                    feedback.append(f"✗ {key}: {answer} does not match expected {expected}")
            except ValueError:
                # If answer isn't a number, do exact string match
                if expected.lower() in answer.lower():
                    score += 0.5
                    breakdown[key] = 1.0
                    feedback.append(f"✓ {key}: Found in answer")
                else:
                    breakdown[key] = 0.0
                    feedback.append(f"✗ {key}: '{expected}' not found in answer")
        except (ValueError, TypeError):
            # Non-numeric ground truth
            if expected.lower() in answer.lower():
                score += 0.5
                breakdown[key] = 1.0
                feedback.append(f"✓ {key}: Found in answer")
            else:
                breakdown[key] = 0.0
                feedback.append(f"✗ {key}: '{expected}' not found in answer")
    
    return {
        "score": score,
        "breakdown": breakdown,
        "feedback": "\n".join(feedback)
    }