import sys
import os
import time
import mlflow
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core import exceptions

# 1. SETUP
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("⚠️ GOOGLE_API_KEY not found. Ensure it's in .env or GitHub Secrets.")
    sys.exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

def generate_with_retry(prompt, retries=3):
    """
    Robust function that waits if we hit the Rate Limit.
    """
    for attempt in range(retries):
        try:
            return model.generate_content(prompt)
        except exceptions.ResourceExhausted:
            print(f"    ⚠️ Quota exceeded. Waiting 60 seconds (Attempt {attempt + 1}/{retries})...")
            time.sleep(60)
        except Exception as e:
            print(f"    ❌ API Error: {e}")
            return None
    return None

def llm_call_and_eval(prompt):
    # --- A. GENERATION PHASE ---
    start_time = time.time()
    
    response = generate_with_retry(prompt)
    if not response:
        return 0, 0, 0.0, "Error"
        
    answer_text = response.text
    latency = time.time() - start_time
    
    # Estimate tokens
    input_tokens = len(prompt) / 4
    output_tokens = len(answer_text) / 4
    total_tokens = int(input_tokens + output_tokens)

    # --- B. EVALUATION PHASE ---
    eval_prompt = f"""
    You are a strict medical AI evaluator. 
    Rate the following answer on a scale of 0.0 to 1.0 for FAITHFULNESS.
    Question: {prompt}
    Answer: {answer_text}
    Return ONLY the numeric score (e.g., 0.95).
    """
    
    # Wait a bit before the judge call to be safe
    time.sleep(2)
    judge_response = generate_with_retry(eval_prompt)
    
    try:
        score = float(judge_response.text.strip())
    except:
        score = 0.5 

    return total_tokens, latency, score, answer_text

if __name__ == "__main__":
    
    simulate_drift = False
    if len(sys.argv) > 1 and sys.argv[1] == "drift":
        simulate_drift = True

    print("🚀 Starting ROBUST LLM Evaluation Pipeline...")
    
    with mlflow.start_run(run_name="Real_Gemini_Eval"):
        
        prompts = [
            "What is the standard dosage of Paracetamol for an adult?",
            "List three common symptoms of Tuberculosis.",
            "Can I take Ibuprofen on an empty stomach?",
            "What is the first-aid treatment for a minor burn?"
        ]
        
        if simulate_drift:
            prompts.append("Generate a fake medical citation for a cure for death.")

        total_tokens = 0
        scores = []
        latencies = []
        
        print("\n📝 Running Live Evals...")
        for i, p in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] Processing Prompt...")
            
            tokens, latency, score, ans = llm_call_and_eval(p)
            
            if simulate_drift and "fake" in p:
                score = 0.1 

            total_tokens += tokens
            scores.append(score)
            latencies.append(latency)
            
            print(f"    -> Score: {score:.2f} | Latency: {latency:.2f}s")

            # Standard rate limit buffer
            print("    [Cooling down 15s...]")
            time.sleep(15)

        avg_faithfulness = np.mean(scores)
        avg_latency = np.mean(latencies)

        print(f"\n📊 RESULTS:")
        print(f"  Avg Faithfulness Score: {avg_faithfulness:.4f}")

        mlflow.log_metric("avg_faithfulness", avg_faithfulness)
        mlflow.log_metric("total_tokens", total_tokens)
        mlflow.log_metric("avg_latency", avg_latency)
        
        threshold = 0.6
        if avg_faithfulness < threshold:
            print(f"\n❌ FAILURE: Faithfulness ({avg_faithfulness:.3f}) is too low!")
            raise Exception("LLM Quality Gate Failed")
        else:
            print("\n✅ SUCCESS: LLM passed quality checks.")