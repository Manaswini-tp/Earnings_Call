# app.py - Complete working version
import os
import sys
import gradio as gr
import traceback

print("Starting Earnings Call Environment...")

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

def run_agent(task_type: str, agent_answer: str):
    """Run a single task - returns exactly (score, feedback)"""
    if env is None:
        return 0.0, "Environment failed to initialize"
    
    if not agent_answer or agent_answer.strip() == "":
        return 0.0, "Please enter an answer before submitting."
    
    try:
        observation = env.reset(task_type=task_type)
        action = Action(answer=agent_answer)
        _, reward, done, info = env.step(action)
        
        # Format feedback
        feedback = f"**Score: {reward:.3f}**\n\n"
        feedback += f"**Task:** {task_type}\n\n"
        
        if "grader_details" in info:
            feedback += f"**Grading Details:**\n```\n{info['grader_details']}\n```\n"
        
        if "feedback" in info:
            feedback += f"**Feedback:** {info['feedback']}\n"
        
        return reward, feedback
        
    except Exception as e:
        error_msg = f"**Error:** {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return 0.0, error_msg

def get_transcript():
    return SAMPLE_TRANSCRIPT

# Create Gradio interface
with gr.Blocks(title="Earnings Call Q&A Environment") as demo:
    gr.Markdown("# 📊 Earnings Call Q&A Agent Environment")
    gr.Markdown("Test an AI agent's ability to analyze earnings call transcripts.")
    
    with gr.Accordion("📄 View Full Transcript", open=False):
        gr.Code(value=SAMPLE_TRANSCRIPT, label="Earnings Call Transcript")
    
    with gr.Tabs():
        for task in ["extract", "identify", "synthesize"]:
            with gr.TabItem(task.capitalize()):
                difficulty = {
                    "extract": "🟢 Easy - Extract exact numbers",
                    "identify": "🟡 Medium - List key risks", 
                    "synthesize": "🔴 Hard - Synthesize conflicting guidance"
                }[task]
                gr.Markdown(f"### {difficulty}")
                
                question = SAMPLE_QUESTIONS.get(task, "")
                gr.Markdown(f"**Question:** {question}")
                
                gr.Markdown("---")
                
                answer_input = gr.Textbox(
                    label="Your Answer", 
                    lines=6,
                    placeholder="Type your answer here...",
                    elem_id=f"answer_{task}"
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Submit & Grade", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear", scale=0)
                
                with gr.Row():
                    score_output = gr.Number(label="Score (0.0 - 1.0)", precision=3, value=None)
                
                feedback_output = gr.Markdown(label="Feedback", value="Waiting for submission...")
                
                # Example answers
                with gr.Accordion("💡 Example Answers", open=False):
                    if task == "extract":
                        gr.Markdown("```EPS was $1.40 and revenue was $85.8 billion```")
                    elif task == "identify":
                        gr.Markdown("```\n1. Supply chain constraints in Southeast Asia\n2. Regulatory pressure from EU Digital Markets Act\n3. Slowing consumer demand in China```")
                    else:
                        gr.Markdown("```While revenue guidance suggests modest growth, margin pressure from increased costs may limit earnings upside. Net outlook is cautiously positive with moderate EPS growth.```")
                
                # Wire up the buttons
                submit_btn.click(
                    fn=run_agent,
                    inputs=[gr.State(task), answer_input],
                    outputs=[score_output, feedback_output]
                )
                clear_btn.click(
                    fn=lambda: ("", None, "Waiting for submission..."),
                    outputs=[answer_input, score_output, feedback_output]
                )
    
    gr.Markdown("---")
    gr.Markdown("**OpenEnv Compliant** | Built for Hackathon")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)