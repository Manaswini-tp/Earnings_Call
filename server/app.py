# app.py - OpenEnv compliant version (FINAL FIXED)

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
# FastAPI APP (IMPORTANT: must be named 'app')
# -------------------------
app = FastAPI()


class StepRequest(BaseModel):
    answer: str


# Root endpoint (debug + prevents 404 confusion)
@app.get("/")
def root():
    return {"message": "Earnings Call API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset():
    if env is None:
        return {"error": "Environment not initialized"}

    observation = env.reset()

    return {
        "observation": observation if observation is not None else {},
        "reward": 0.0,
        "done": False,
        "info": {}
    }


@app.post("/step")
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


@app.get("/state")
def state():
    if env is None:
        return {"error": "Environment not initialized"}

    return env.state()


# -------------------------
# RUN SERVER
# -------------------------
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()