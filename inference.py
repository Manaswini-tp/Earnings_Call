import os
from openai import OpenAI
from env.environment import EarningsCallEnv
from env.models import Action
from sample_data import SAMPLE_TRANSCRIPT, SAMPLE_QUESTIONS, SAMPLE_GROUND_TRUTH

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)


def log_start(task):
    print(f"[START] task={task} env=earnings_call_env model={MODEL_NAME}")


def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action[:80]} reward={reward:.2f} done={done_val} error={error_val}")


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}")


def run_inference():
    env = EarningsCallEnv(SAMPLE_TRANSCRIPT, SAMPLE_QUESTIONS, SAMPLE_GROUND_TRUTH)

    for task in ["extract", "identify", "synthesize"]:
        rewards = []
        steps = 0

        log_start(task)

        observation = env.reset(task_type=task)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a financial analyst. Answer ONLY using the transcript."},
                {"role": "user", "content": f"Transcript:\n{observation.transcript}\n\nQuestion: {observation.question}"}
            ],
            temperature=0.2,
            max_tokens=300,
        )

        answer = response.choices[0].message.content.strip()
        action = Action(answer=answer)

        _, reward, done, info = env.step(action)

        rewards.append(reward)
        steps = 1

        log_step(1, answer, reward, done)

        score = reward
        success = score > 0.1

        log_end(success, steps, score, rewards)


if __name__ == "__main__":
    run_inference()