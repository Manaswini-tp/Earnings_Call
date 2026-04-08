# env/environment.py
from typing import Dict, Any
from .models import Observation, Action, Reward
from .graders.exact_match import exact_match_grader
from .graders.f1_score import f1_grader
from .graders.llm_judge import llm_judge_grader

class EarningsCallEnv:
    def __init__(self, transcript: str, questions: Dict[str, Any], ground_truth: Dict[str, Any]):
        """
        Initialize environment with a specific transcript and questions
        
        Args:
            transcript: Full transcript text
            questions: Dict with keys 'extract', 'identify', 'synthesize' containing questions
            ground_truth: Dict with answers for each task
        """
        self.transcript = transcript
        self.questions = questions
        self.ground_truth = ground_truth
        self.current_task = None
        self.turn = 0
        
    def reset(self, task_type: str = None) -> Observation:
        """
        Reset environment for a new task
        
        Args:
            task_type: 'extract', 'identify', or 'synthesize'
        """
        if task_type is None:
            task_type = 'extract'  # Default
        
        self.current_task = task_type
        self.turn = 0
        
        question = self.questions.get(task_type, f"Answer the question based on the transcript")
        
        return Observation(
            transcript=self.transcript,
            question=question,
            task_type=task_type,
            turn_number=1
        )
    
    def step(self, action: Action) -> tuple[Observation, Reward, bool]:
        """
        Take an action (agent's answer) and return reward
        
        Args:
            action: Agent's answer text
            
        Returns:
            observation: Next observation (None if episode done)
            reward: Reward object with score and feedback
            done: Whether episode is complete
        """
        self.turn += 1
        answer = action.answer
        
        # Grade based on task type
        if self.current_task == 'extract':
            result = exact_match_grader(answer, self.ground_truth.get('extract', {}))
        elif self.current_task == 'identify':
            result = f1_grader(answer, self.ground_truth.get('identify', []))
        elif self.current_task == 'synthesize':
            result = llm_judge_grader(
                answer, 
                self.questions.get('synthesize', ''),
                self.transcript,
                self.ground_truth.get('synthesize_rubric')
            )
        else:
            result = {'score': 0.0, 'breakdown': {}, 'feedback': 'Unknown task type'}
        
        # Calculate final reward with penalties
        score = result['score']
        
        # Hallucination penalty (simplified)
        hallucination_penalty = 0
        if self.current_task in ['extract', 'identify']:
            # Check if answer contains numbers not in transcript
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', answer)
            transcript_preview = self.transcript[:10000]
            for num in numbers:
                if num not in transcript_preview and float(num) > 100:  # Ignore small numbers
                    hallucination_penalty -= 0.1
        
        # Efficiency penalty for verbose answers
        efficiency_penalty = -min(len(answer.split()) / 200, 0.1)  # Max -0.1 for very long answers
        
        final_score = score + hallucination_penalty + efficiency_penalty
        final_score = max(0, min(1, final_score))  # Clip between 0 and 1
        
        reward = Reward(
            score=final_score,
            breakdown={
                'task_score': score,
                'hallucination_penalty': hallucination_penalty,
                'efficiency_penalty': efficiency_penalty,
                'details': result['breakdown']
            },
            feedback=result['feedback'] + f"\nFinal reward: {final_score:.3f}"
        )
        
        # Episode ends after one turn per task
        return None, reward, True
    
    def get_task_list(self):
        """Return available tasks"""
        return list(self.questions.keys())
    # Add to EarningsCallEnv class:

def state(self) -> dict:
    """Return current environment state for OpenEnv compliance"""
    return {
        "current_task": self.current_task,
        "turn": self.turn,
        "transcript_length": len(self.transcript),
        "question": self.questions.get(self.current_task, "") if self.current_task else ""
    }

# env/environment.py - Updated step() method with explicit reward function

def step(self, action: Action) -> tuple:
    """
    OpenEnv compliant step method with explicit reward calculation
    """
    self.turn += 1
    answer = action.answer
    
    # ============================================
    # STEP 1: Get base scores from grader
    # ============================================
    if self.current_task == 'extract':
        result = exact_match_grader(answer, self.ground_truth.get('extract', {}))
    elif self.current_task == 'identify':
        result = f1_grader(answer, self.ground_truth.get('identify', []))
    elif self.current_task == 'synthesize':
        result = llm_judge_grader(
            answer, 
            self.questions.get('synthesize', ''),
            self.transcript,
            self.ground_truth.get('synthesize_rubric')
        )
    else:
        result = {'score': 0.0, 'breakdown': {}, 'feedback': 'Unknown task type'}
    
    # ============================================
    # STEP 2: Calculate reward components
    # ============================================
    
    # Component 1: Factual Accuracy (50% of reward)
    # This comes directly from the grader's score
    factual_accuracy = result['score']
    
    # Component 2: Completeness (30% of reward)
    # For extract: both numbers found? For identify: all risks found?
    if self.current_task == 'extract':
        # Completeness = how many required fields were found
        expected_fields = len(self.ground_truth.get('extract', {}))
        found_fields = sum(1 for v in result['breakdown'].values() if v == 1.0)
        completeness = found_fields / expected_fields if expected_fields > 0 else 1.0
    elif self.current_task == 'identify':
        # Completeness = F1 score already measures this
        completeness = factual_accuracy
    else:  # synthesize
        # Completeness = from LLM judge breakdown
        completeness = result['breakdown'].get('completeness', 0.3) / 0.3  # Normalize
    
    # Component 3: Hallucination Penalty
    hallucination_penalty = 0
    import re
    transcript_preview = self.transcript[:10000]
    
    # Extract numbers from agent's answer
    numbers_in_answer = re.findall(r'\d+(?:\.\d+)?%?', answer)
    # Check if any number is NOT in transcript
    for num in numbers_in_answer:
        if num not in transcript_preview and len(num) > 2:  # Ignore small numbers
            hallucination_penalty = -0.3  # Fixed penalty for hallucination
            break
    
    # Component 4: Verbosity Penalty (extra_turns penalty)
    # We use answer length as proxy for verbosity since single-turn
    word_count = len(answer.split())
    if word_count > 100:
        verbosity_penalty = -0.1
    elif word_count > 50:
        verbosity_penalty = -0.05
    elif word_count > 20:
        verbosity_penalty = -0.02
    else:
        verbosity_penalty = 0
    
    # ============================================
    # STEP 3: Calculate FINAL REWARD
    # ============================================
    # reward = factual_accuracy * 0.5 + completeness * 0.3 + hallucination_penalty + verbosity_penalty
    reward_score = (factual_accuracy * 0.5) + (completeness * 0.3) + hallucination_penalty + verbosity_penalty
    
    # Clamp between 0 and 1
    reward_score = max(0.0, min(1.0, reward_score))
    
    # ============================================
    # STEP 4: Create reward breakdown for debugging
    # ============================================
    reward_breakdown = {
        'factual_accuracy': factual_accuracy,
        'factual_contribution': factual_accuracy * 0.5,
        'completeness': completeness,
        'completeness_contribution': completeness * 0.3,
        'hallucination_penalty': hallucination_penalty,
        'verbosity_penalty': verbosity_penalty,
        'final_reward': reward_score,
        'grader_details': result['breakdown']
    }
    
    # ============================================
    # STEP 5: Create feedback message
    # ============================================
    feedback = f"""
    REWARD CALCULATION:
    ─────────────────────────────────────────
    factual_accuracy: {factual_accuracy:.3f} × 0.5 = {factual_accuracy * 0.5:.3f}
    completeness:      {completeness:.3f} × 0.3 = {completeness * 0.3:.3f}
    hallucination:                          {hallucination_penalty:.3f}
    verbosity:                              {verbosity_penalty:.3f}
    ─────────────────────────────────────────
    FINAL REWARD: {reward_score:.3f}
    
    Grader feedback: {result['feedback']}
    """
    
    # ============================================
    # STEP 6: Return OpenEnv-compliant response
    # ============================================
    done = True  # Episode ends after one turn
    
    info = {
        "task_type": self.current_task,
        "turn": self.turn,
        "reward_breakdown": reward_breakdown,
        "grader_feedback": result['feedback']
    }
    
    return None, reward_score, done, info

def _compute_reward(self, action: Action) -> tuple:
    """Internal reward computation"""
    # Your existing grading logic here
    # Returns (reward_obj, done)
    pass

def _get_current_observation(self) -> Observation:
    """Get current observation"""
    return Observation(
        transcript=self.transcript,
        question=self.questions.get(self.current_task, ""),
        task_type=self.current_task,
        turn_number=self.turn
    )