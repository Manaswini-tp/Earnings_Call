# app.py - OpenEnv compliant version

import os
import traceback
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

print("Starting Earnings Call Environment...")

# Import environment
try:
    from env.environment import EarningsCallEnv
    from env.models import Action
    from sample_data import SAMPLE_TRANSCRIPT, SAMPLE_QUESTIONS, SAMPLE_GROUND_TRUTH
    print("✓ Modules imported")
except Exception as e:
    print(f"✗ Import error: {e}")
    traceback.print_exc()

# Initialize environment
try:
    env = EarningsCallEnv(SAMPLE_TRANSCRIPT, SAMPLE_QUESTIONS, SAMPLE_GROUND_TRUTH)
    print("✓ Environment initialized")
except Exception as e:
    print(f"✗ Environment error: {e}")
    traceback.print_exc()
    env = None


# -------------------------
# FastAPI APP (OpenEnv)
# -------------------------
api_app = FastAPI()


class StepRequest(BaseModel):
    answer: str


@api_app.get("/health")
def health():
    return {"status": "ok"}


@api_app.post("/reset")
def reset():
    if env is None:
        return {"error": "Environment not initialized"}

    observation = env.reset()

    return {
        "observation": observation,
        "reward": 0.0,
        "done": False,
        "info": {}
    }

@api_app.post("/step")
def step(request: StepRequest):
    if env is None:
        return {"error": "Environment not initialized"}

    try:
        action = Action(answer=request.answer)
        observation, reward, done, info = env.step(action)

        return {
            "observation": observation if observation is not None else {},
            "reward": reward,
            "done": done,
            "info": info
        }

    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }


@api_app.get("/state")
def state():
    if env is None:
        return {"error": "Environment not initialized"}

    return env.state()

if __name__ == "__main__":
    uvicorn.run(api_app, host="0.0.0.0", port=7860)