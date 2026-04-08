# inference.py - Required for hackathon submission
import os
import json
from openai import OpenAI
from env.environment import EarningsCallEnv
from env.models import Action
from sample_data import SAMPLE_TRANSCRIPT, SAMPLE_QUESTIONS, SAMPLE_GROUND_TRUTH

# Get environment variables (set these in HF Space secrets)
API_BASE_URL = os.getenv('API_BASE_URL', 'https://api.groq.com/openai/v1')
MODEL_NAME = os.getenv('MODEL_NAME', 'llama-3.3-70b-versatile')
HF_TOKEN = os.getenv('HF_TOKEN', '')

# Initialize OpenAI client (works with Groq too)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

def run_inference():
    """Run baseline inference on all tasks"""
    env = EarningsCallEnv(SAMPLE_TRANSCRIPT, SAMPLE_QUESTIONS, SAMPLE_GROUND_TRUTH)
    results = {}
    
    print("[START] Earnings Call Environment Evaluation")
    
    for task in ['extract', 'identify', 'synthesize']:
        print(f"[STEP] Starting task: {task}")
        
        observation = env.reset(task_type=task)
        
        # Call LLM
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a financial analyst. Answer based ONLY on the transcript."},
                {"role": "user", "content": f"Transcript:\n{observation.transcript}\n\nQuestion: {observation.question}"}
            ],
            temperature=0.2,
            max_tokens=300,
        )
        
        answer = response.choices[0].message.content
        action = Action(answer=answer)
        _, reward, done, info = env.step(action)
        
        results[task] = reward
        print(f"[STEP] Task {task} completed with score: {reward:.3f}")
    
    print(json.dumps(results, indent=2))
    print(f"[END] Average score: {sum(results.values()) / len(results):.3f}")

if __name__ == "__main__":
    run_inference()