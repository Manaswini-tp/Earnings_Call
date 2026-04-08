# env/graders/f1_score.py - Improved version
import re
from collections import Counter

def normalize_text(text: str) -> set:
    """
    Normalize text by removing common words and punctuation
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                  'for', 'of', 'with', 'by', 'from', 'up', 'down', 'top', 'list',
                  'here', 'are', 'risk', 'factors', 'mentioned', 'from', 'transcript'}
    
    tokens = set()
    for word in text.split():
        if word not in stop_words and len(word) > 2:
            # Also try to get root word (simple stemming)
            if word.endswith('ing'):
                word = word[:-3]
            elif word.endswith('ed'):
                word = word[:-2]
            elif word.endswith('s') and not word.endswith('ss'):
                word = word[:-1]
            tokens.add(word)
    
    return tokens

def token_f1_score(prediction: str, ground_truths: list) -> float:
    """
    Calculate flexible token-level F1 score
    """
    if not ground_truths:
        return 0.0
    
    pred_tokens = normalize_text(prediction)
    
    best_f1 = 0.0
    for truth in ground_truths:
        truth_tokens = normalize_text(truth)
        
        if not pred_tokens and not truth_tokens:
            f1 = 1.0
        elif not pred_tokens or not truth_tokens:
            f1 = 0.0
        else:
            intersection = pred_tokens & truth_tokens
            precision = len(intersection) / len(pred_tokens) if pred_tokens else 0
            recall = len(intersection) / len(truth_tokens) if truth_tokens else 0
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
        
        best_f1 = max(best_f1, f1)
    
    return best_f1

def f1_grader(answer: str, ground_truth: list) -> dict:
    """
    Grade identification tasks using improved F1 score
    """
    score = token_f1_score(answer, ground_truth)
    
    # Give partial credit for good attempts
    if score < 0.3 and len(answer) > 50:
        # Check if answer contains key terms
        answer_lower = answer.lower()
        matched_any = False
        for truth in ground_truth:
            truth_keywords = normalize_text(truth)
            for keyword in truth_keywords:
                if keyword in answer_lower:
                    matched_any = True
                    break
            if matched_any:
                break
        
        if matched_any:
            score = max(score, 0.5)  # Boost score if keywords are present
    
    # Provide detailed feedback
    if score >= 0.8:
        feedback = f"F1 Score: {score:.3f} - Excellent! All key risks identified."
    elif score >= 0.6:
        feedback = f"F1 Score: {score:.3f} - Good! Most risks identified correctly."
    elif score >= 0.4:
        feedback = f"F1 Score: {score:.3f} - Fair. Some risks identified, but missing key elements."
    else:
        feedback = f"F1 Score: {score:.3f} - Needs improvement. Make sure to list risks using phrases from the transcript."
    
    return {
        "score": min(score, 1.0),  # Cap at 1.0
        "breakdown": {"f1_score": score},
        "feedback": feedback
    }