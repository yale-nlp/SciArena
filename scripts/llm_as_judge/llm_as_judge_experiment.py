import json
import asyncio
from typing import Dict, List
from pydantic import BaseModel
from datetime import datetime
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from common.openai_utils import process_data, client_dict

# Define response model
class JudgmentResponse(BaseModel):
    choice: str

# Define prompt template
PAIRWISE_COMPARISON_PROMPT = """
You are an expert in scientific literature synthesis. Your task is to evaluate the quality of two AI-generated citation-attributed responses to a user's question. Assess both responses for relevance, accuracy, clarity, and appropriate use of citations. Then, select the response, Output (a) or Output (b), that best address the user's question.

User Question:
{question}

Output (a):
{response_a}

Output (b):
{response_b}

Which is best, Output(a) or Output (b)?
Answer with only: "A" or "B"
""".strip()

def build_messages(example: Dict) -> List[Dict[str, str]]:
    """Build messages for LLM judgment"""
    prompt = PAIRWISE_COMPARISON_PROMPT.format(
        question=example["question"],
        response_a=example["responseA"],
        response_b=example["responseB"]
    )
    
    return [
        {"role": "user", "content": prompt}
    ]

def post_process(example: Dict, parsed_response: JudgmentResponse, model: str) -> Dict:
    """Post-processing function to add LLM judgment results to example"""
    result = example.copy()
    result["llm_judgment"] = parsed_response.choice
    result["model_used"] = model
    return result

async def run_llm_as_judge_experiment(input_file: str, output_file: str = None):
    """Run LLM as judge experiment"""
    
    # Load data
    print(f"Loading data file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} data entries")
    
    # Use gpt-4.1 model for judgment
    models = ["gpt-4.1"]
    
    print("Starting LLM judgment...")
    results = await process_data(
        data=data,
        client_dict=client_dict,
        build_messages=build_messages,
        response_model=JudgmentResponse,
        models=models,
        enable_structured_output=True,
        post_process=post_process,
        max_concurrency=50,  # Control concurrency
        max_tokens=2048        # Only need to return A or B
    )
    
    # Filter successful results
    successful_results = [r for r in results if r is not None and "llm_judgment" in r]
    failed_count = len(data) - len(successful_results)
    
    print(f"Successfully completed: {len(successful_results)} entries")
    print(f"Failed: {failed_count} entries")
    
    # Calculate accuracy
    correct_predictions = 0
    total_predictions = len(successful_results)
    
    for result in successful_results:
        original_vote = result["vote"]
        llm_judgment = result["llm_judgment"]
        
        # Normalize judgment results (handle potential case issues)
        llm_judgment = llm_judgment.upper().strip()
        original_vote = original_vote.upper().strip()
        
        if llm_judgment == original_vote:
            correct_predictions += 1
    
    # Calculate self consistency (accuracy)
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\n=== Experiment Results ===")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Total predictions: {total_predictions}")
        print(f"Self Consistency (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
    else:
        print("No successful prediction results")
        accuracy = 0.0
    
    # Detailed statistics
    judgment_distribution = {}
    vote_distribution = {}
    
    for result in successful_results:
        llm_judgment = result["llm_judgment"].upper().strip()
        original_vote = result["vote"].upper().strip()
        
        judgment_distribution[llm_judgment] = judgment_distribution.get(llm_judgment, 0) + 1
        vote_distribution[original_vote] = vote_distribution.get(original_vote, 0) + 1
    
    print(f"\nLLM judgment distribution: {judgment_distribution}")
    print(f"Original label distribution: {vote_distribution}")
    
    # Save results
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/llm_as_judge_results_{timestamp}.json"
    
    output_data = {
        "experiment_info": {
            "input_file": input_file,
            "model_used": "gpt-4.1",
            "total_data": len(data),
            "successful_predictions": total_predictions,
            "failed_predictions": failed_count,
            "accuracy": accuracy,
            "correct_predictions": correct_predictions
        },
        "statistics": {
            "llm_judgment_distribution": judgment_distribution,
            "original_vote_distribution": vote_distribution
        },
        "results": successful_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return accuracy, successful_results

async def main():
    """Main function"""
    # if sampled
    input_file = "data/sampled_2000.json" 
    
    try:
        accuracy, results = await run_llm_as_judge_experiment(input_file)
        print(f"\nExperiment completed! Final accuracy: {accuracy:.4f}")
        
    except FileNotFoundError:
        print(f"Error: File not found {input_file}")
    except Exception as e:
        print(f"Error during experiment: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 