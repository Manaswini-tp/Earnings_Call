# app.py - HF Space deployment with Gradio
import gradio as gr
import os
import json
from env.environment import EarningsCallEnv
from env.models import Action
from sample_data import SAMPLE_TRANSCRIPT, SAMPLE_QUESTIONS, SAMPLE_GROUND_TRUTH

# Initialize environment
env = EarningsCallEnv(SAMPLE_TRANSCRIPT, SAMPLE_QUESTIONS, SAMPLE_GROUND_TRUTH)

def run_agent(task_type: str, agent_answer: str):
    """Run a single task with given answer"""
    try:
        observation = env.reset(task_type=task_type)
        action = Action(answer=agent_answer)
        _, reward, done, info = env.step(action)
        
        # Format feedback nicely
        feedback = info.get("grader_details", {})
        feedback_str = json.dumps(feedback, indent=2)
        
        return reward, feedback_str, done
    except Exception as e:
        return 0.0, f"Error: {str(e)}", True

def get_question(task_type: str):
    """Get question for task type"""
    return SAMPLE_QUESTIONS.get(task_type, "Question not found")

def get_transcript_preview():
    """Get transcript preview"""
    return SAMPLE_TRANSCRIPT[:1000] + "..."

# Create Gradio interface
with gr.Blocks(title="Earnings Call Q&A Environment", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📊 Earnings Call Q&A Agent Environment")
    gr.Markdown("""
    Test an AI agent's ability to analyze earnings call transcripts. 
    This environment simulates the work of a financial analyst.
    """)
    
    # Show transcript
    with gr.Accordion("📄 View Transcript", open=False):
        gr.Markdown(f"```\n{SAMPLE_TRANSCRIPT}\n```")
    
    with gr.Tabs():
        for task in ["extract", "identify", "synthesize"]:
            with gr.TabItem(task.capitalize()):
                # Difficulty badge
                difficulty = {"extract": "🟢 Easy", "identify": "🟡 Medium", "synthesize": "🔴 Hard"}[task]
                gr.Markdown(f"**Difficulty:** {difficulty}")
                
                # Question
                question = SAMPLE_QUESTIONS.get(task, "")
                gr.Markdown(f"**Question:** {question}")
                
                # Answer input
                answer_input = gr.Textbox(
                    label="Agent Answer", 
                    lines=5,
                    placeholder="Type your answer here..."
                )
                
                # Submit button
                submit_btn = gr.Button("Submit & Grade", variant="primary")
                
                # Results
                with gr.Row():
                    score_output = gr.Number(label="Score (0.0-1.0)", precision=3)
                feedback_output = gr.Textbox(label="Detailed Feedback", lines=10)
                
                # Example answers
                with gr.Accordion("💡 Example Answers", open=False):
                    if task == "extract":
                        gr.Markdown("EPS was $1.40 and revenue was $85.8 billion, beating analyst estimates.")
                    elif task == "identify":
                        gr.Markdown("``` 1. Supply chain constraints in Southeast Asia \n2. Regulatory pressure from EU Digital Markets Act\n3. Slowing consumer demand in China1```")
                    else:
                        gr.Markdown("```While revenue guidance suggests modest growth (+5% YoY), margin pressure from increased costs may limit earnings upside.Net outlook is cautiously positive with 2-3% EPS growth.```")
                
                submit_btn.click(
                    fn=run_agent,
                    inputs=[gr.State(task), answer_input],
                    outputs=[score_output, feedback_output]
                )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("**OpenEnv Compliant** | Built for Hackathon | [Documentation](https://github.com/your-repo)")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)