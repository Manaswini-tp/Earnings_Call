# env/environment.py
from typing import Dict, Any, Tuple
from .models import Observation, Action, Reward
from .graders.exact_match import exact_match_grader
from .graders.f1_score import f1_grader
from .graders.llm_judge import llm_judge_grader
import re

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
            task_type = 'extract'
        
        self.current_task = task_type
        self.turn = 0
        
        question = self.questions.get(task_type, f"Answer the question based on the transcript")
        
        return Observation(
            transcript=self.transcript,
            question=question,
            task_type=task_type,
            turn_number=1
        )
    
    def step(self, action: Action) -> Tuple:
        """
        Execute one step in the environment.
        
        Returns:
            observation: None (episode ends after one step)
            reward: float (0.0 to 1.0)
            done: bool (True since single-turn episode)
            info: dict (additional information)
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
        
        # Hallucination penalty
        hallucination_penalty = 0
        if self.current_task in ['extract', 'identify']:
            numbers = re.findall(r'\d+(?:\.\d+)?', answer)
            transcript_preview = self.transcript[:10000]
            for num in numbers:
                if num not in transcript_preview and len(num) > 2:
                    hallucination_penalty = -0.1
                    break
        
        # Efficiency penalty for verbose answers
        efficiency_penalty = -min(len(answer.split()) / 200, 0.1)
        
        final_score = score + hallucination_penalty + efficiency_penalty
        final_score = max(0.01, min(0.99, final_score))
        
        # Return 4 values: (observation, reward, done, info)
        observation = None  # Episode ends after one step
        reward = final_score
        done = True
        info = {
            "task_type": self.current_task,
            "turn": self.turn,
            "grader_details": result['breakdown'],
            "feedback": result['feedback']
        }
        
        return observation, reward, done, info
    
    def state(self) -> dict:
        """Return current environment state for OpenEnv compliance"""
        return {
            "current_task": self.current_task,
            "turn": self.turn,
            "transcript_length": len(self.transcript),
            "question": self.questions.get(self.current_task, "") if self.current_task else "",
            "task_completed": self.turn > 0
        }
    
    def get_task_list(self):
        """Return available tasks"""
        return list(self.questions.keys())