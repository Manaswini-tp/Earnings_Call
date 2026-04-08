# run_baseline.py - Updated with only working models
import os
from env.environment import EarningsCallEnv
from env.models import Action
from sample_data import SAMPLE_TRANSCRIPT, SAMPLE_QUESTIONS, SAMPLE_GROUND_TRUTH

# Load API key from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded API key from .env file")
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")

def run_with_groq():
    """
    Run baseline with Groq API
    """
    # Get API key
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        print("❌ GROQ_API_KEY not found!")
        api_key = input("Enter your Groq API key: ").strip()
        if not api_key:
            print("No API key provided. Exiting.")
            return
    
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        print("✓ Groq client initialized")
        
        env = EarningsCallEnv(SAMPLE_TRANSCRIPT, SAMPLE_QUESTIONS, SAMPLE_GROUND_TRUTH)
        
        # Only working models
        print("\n✓ Available Groq models (tested and working):")
        print("1. llama-3.3-70b-versatile (Best quality - Recommended for synthesis)")
        print("2. llama-3.1-8b-instant (Fast - Good for extraction/identification)")
        
        model_choice = input("\nChoose model (1-2, default=1): ").strip()
        
        if model_choice == "2":
            model = "llama-3.1-8b-instant"
            print("\n✓ Using fast model: llama-3.1-8b-instant")
        else:
            model = "llama-3.3-70b-versatile"
            print("\n✓ Using high-quality model: llama-3.3-70b-versatile")
        
        results = {}
        
        for task in ['extract', 'identify', 'synthesize']:
            print(f"\n{'='*60}")
            print(f"Running {task.upper()} task")
            print(f"{'='*60}")
            
            observation = env.reset(task_type=task)
            print(f"\nQuestion: {observation.question}")
            
            # Task-specific prompts
            if task == 'extract':
                system_prompt = """You are a precise financial analyst. Extract EXACT numbers from the transcript.
Answer format: 'EPS was $X.XX and revenue was $X.X billion'
Only use numbers that appear in the transcript."""
            elif task == 'identify':
                system_prompt = """You are a precise financial analyst. List the key risk factors mentioned.
Format as a numbered list. Use exact phrases from the transcript."""  
            else:  # synthesize
                system_prompt = """You are a senior financial analyst. Provide a balanced synthesis.
Show reasoning, cite specific numbers, address conflicting signals. Keep to 2-3 sentences."""
            
            user_prompt = f"""Transcript:
{observation.transcript}

Question: {observation.question}

Answer:"""
            
            try:
                print("Calling Groq API...")
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=300,
                )
                
                answer = response.choices[0].message.content
                print(f"\nAgent Answer:\n{answer}")
                
                # Grade the answer
                action = Action(answer=answer)
                _, reward, _ = env.step(action)
                
                print(f"\n{'─'*40}")
                print(f"Score: {reward.score:.3f}")
                print(f"Breakdown: {reward.breakdown}")
                print(f"\nFeedback: {reward.feedback}")
                
                results[task] = reward.score
                
            except Exception as e:
                print(f"❌ Error on {task}: {e}")
                results[task] = 0.0
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 FINAL RESULTS")
        print(f"{'='*60}")
        for task, score in results.items():
            if score >= 0.8:
                status = "✅ Excellent"
            elif score >= 0.6:
                status = "👍 Good"
            elif score >= 0.4:
                status = "⚠️ Acceptable"
            else:
                status = "❌ Needs work"
            print(f"{task.capitalize():12}: {score:.3f} - {status}")
        
        avg_score = sum(results.values()) / len(results)
        print(f"\n{'─'*40}")
        print(f"Average Score: {avg_score:.3f}")
        
        if avg_score >= 0.8:
            print("\n🎉 Outstanding! Your agent is performing exceptionally well!")
        elif avg_score >= 0.6:
            print("\n👍 Good job! Fine-tune prompts to improve further.")
        elif avg_score >= 0.4:
            print("\n⚠️ Decent start. Try adjusting the prompts.")
        else:
            print("\n❌ Needs improvement. Check the transcript format.")
        
    except ImportError:
        print("❌ Groq package not installed. Run: pip install groq")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Earnings Call Environment with Groq")
    print("=" * 40)
    run_with_groq()