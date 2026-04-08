# env/models.py
from pydantic import BaseModel
from typing import Dict, Any, Optional

class Observation(BaseModel):
    transcript: str
    question: str
    task_type: str  # "extract", "identify", or "synthesize"
    turn_number: int = 1

class Action(BaseModel):
    answer: str

class Reward(BaseModel):
    score: float
    breakdown: Dict[str, Any]
    feedback: str