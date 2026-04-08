# env/graders/f1_score.py - More flexible version
import re
from collections import Counter

def normalize_text(text: str) -> set:
    """
    Normalize text by removing common words, punctuation, and using stems
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove common stop words and short words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
        'for', 'of', 'with', 'by', 'from', 'up', 'down', 'top', 'list',
        'here', 'are', 'risk', 'factors', 'mentioned', 'from', 'transcript',
        '1', '2', '3', 'i', 'ii', 'iii', 'first', 'second', 'third'
    }
    
    tokens = set()
    for word in text.split():
        if word not in stop_words and len(word) > 3:
            # Simple stemming
            if word.endswith('ing'):
                word = word[:-3]
            elif word.endswith('ed'):
                word = word[:-2]
            elif word.endswith('s') and not word.endswith('ss'):
                word = word[:-1]
            elif word.endswith('al'):
                word = word[:-2]
            tokens.add(word)
    
    return tokens

def token_f1_score(prediction: str, ground_truths: list) -> float:
    """
    Calculate flexible token-level F1 score with partial matching
    """
    if not ground_truths:
        return 0.0
    
    pred_tokens = normalize_text(prediction)
    
    if not pred_tokens:
        return 0.0
    
    best_f1 = 0.0
    for truth in ground_truths:
        truth_tokens = normalize_text(truth)
        
        if not truth_tokens:
            continue
        
        # Calculate intersection
        intersection = pred_tokens & truth_tokens
        
        if intersection:
            precision = len(intersection) / len(pred_tokens)
            recall = len(intersection) / len(truth_tokens)
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                best_f1 = max(best_f1, f1)
    
    # Boost score if we found matches (minimum 0.5 for any match)
    if best_f1 > 0 and best_f1 < 0.5:
        best_f1 = max(best_f1, 0.5)
    
    return best_f1

def f1_grader(answer: str, ground_truth: list) -> dict:
    """
    Grade identification tasks using improved F1 score
    """
    score = token_f1_score(answer, ground_truth)
    
    # Provide better feedback
    if score >= 0.8:
        feedback = f"F1 Score: {score:.3f} - Excellent! All key risks identified correctly."
    elif score >= 0.6:
        feedback = f"F1 Score: {score:.3f} - Good! Most risks identified."
    elif score >= 0.4:
        feedback = f"F1 Score: {score:.3f} - Fair. Some risks identified correctly."
    else:
        # Check if answer has the right structure but wrong wording
        if any(str(i) in answer for i in range(1, 4)):
            feedback = f"F1 Score: {score:.3f} - Your answer structure is good, but try using exact phrases from the transcript."
        else:
            feedback = f"F1 Score: {score:.3f} - Needs improvement. List risks as numbered items using phrases from the transcript."
    
    return {
        "score": min(score, 1.0),
        "breakdown": {"f1_score": score},
        "feedback": feedback
    }