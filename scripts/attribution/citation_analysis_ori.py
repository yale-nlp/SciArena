import asyncio
import json
import random
import sys
import os
from typing import List, Dict
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from common.openai_utils import process_data, client_dict


class CitationClassification(BaseModel):
    classification: str  # "support", "irrelevant", "contradict"
    reasoning: str


def build_messages(example: Dict) -> List[Dict[str, str]]:
    """Build prompt messages for classification"""
    response = example["response"]
    concise_authors = example["concise_authors"] 
    content = example["content"]
    
    prompt = f"""Please analyze whether the following citation is consistent with the content mentioned in the response.

Response content:
{response}

Citation author: {concise_authors}
Citation original content:
{content}

Please judge whether the citation's original content is consistent with the statements in the response based on the specific content mentioned about "{concise_authors}" in the response.

Special attention:
1. Please pay attention to the letters at the end of concise_authors (such as a, b, c, d, etc.) to ensure correct citation matching
2. Carefully compare the specific statements about this author in the response with the citation's original content
3. Classification criteria:
   - support: The citation content supports the statements in the response
   - irrelevant: The citation content is unrelated or irrelevant to the statements in the response
   - contradict: The citation content contradicts the statements in the response

Please classify based on the above analysis and provide a brief explanation."""

    return [
        {"role": "system", "content": "You are a professional literature citation analysis expert who needs to judge the relationship between citation content and text statements."},
        {"role": "user", "content": prompt}
    ]


def filter_and_sample_data(data: List[Dict], sample_size: int = 3000) -> List[Dict]:
    """Filter and sample data that meets the criteria"""
    
    def check_citations_in_response(citations: List[Dict], response: str) -> bool:
        """Check if all citation concise_authors appear in response"""
        if not citations:
            return False
        
        for citation in citations:
            concise_authors = citation.get("concise_authors", "")
            if concise_authors not in response:
                return False
        return True
    
    valid_data = []
    for item in data:
        citations_a = item.get("citations_a", [])
        citations_b = item.get("citations_b", [])
        response_a = item.get("responseA", "")
        response_b = item.get("responseB", "")
        vote = item.get("vote", "")
        
        # Skip tie or bad votes
        if vote in ["tie", "bad"]:
            continue
        
        # Check if both A and B meet criteria
        if (citations_a and check_citations_in_response(citations_a, response_a) and
            citations_b and check_citations_in_response(citations_b, response_b)):
            valid_data.append(item)
    
    # Random sampling
    if len(valid_data) > sample_size:
        valid_data = random.sample(valid_data, sample_size)
    
    print(f"Found {len(valid_data)} valid data elements, sampling {min(len(valid_data), sample_size)}")
    return valid_data


def prepare_citation_data_for_llm(sampled_data: List[Dict]) -> List[Dict]:
    """Convert sampled data to LLM analysis format"""
    llm_data = []
    
    for item in sampled_data:
        citations_a = item.get("citations_a", [])
        citations_b = item.get("citations_b", [])
        response_a = item.get("responseA", "")
        response_b = item.get("responseB", "")
        
        # Process citations_a
        for citation in citations_a:
            citation_item = {
                "id": item["id"],
                "modelA": item["modelA"],
                "modelB": item["modelB"], 
                "vote": item["vote"],
                "response": response_a,
                "concise_authors": citation["concise_authors"],
                "content": citation["content"],
                "response_type": "A",
                "response_length_a": len(response_a),
                "response_length_b": len(response_b)
            }
            llm_data.append(citation_item)
        
        # Process citations_b
        for citation in citations_b:
            citation_item = {
                "id": item["id"],
                "modelA": item["modelA"],
                "modelB": item["modelB"],
                "vote": item["vote"], 
                "response": response_b,
                "concise_authors": citation["concise_authors"],
                "content": citation["content"],
                "response_type": "B",
                "response_length_a": len(response_a),
                "response_length_b": len(response_b)
            }
            llm_data.append(citation_item)
    
    return llm_data


def aggregate_results_by_data_id(results: List[Dict]) -> List[Dict]:
    """Aggregate results by data ID"""
    data_stats = {}
    
    for result in results:
        if not result or "id" not in result or "classification" not in result:
            print(f"Skipping invalid result: {result}")
            continue

        data_id = result["id"]
        response_type = result["response_type"]
        classification = result["classification"]
        
        if data_id not in data_stats:
            data_stats[data_id] = {
                "id": data_id,
                "modelA": result["modelA"],
                "modelB": result["modelB"],
                "vote": result["vote"],
                "support_count_a": 0,
                "irrelevant_count_a": 0,
                "contradict_count_a": 0,
                "support_count_b": 0,
                "irrelevant_count_b": 0,
                "contradict_count_b": 0,
                "total_count_a": 0,
                "total_count_b": 0,
                "response_length_a": result.get("response_length_a", 0),
                "response_length_b": result.get("response_length_b", 0)
            }
        
        # Count classification results
        if response_type == "A":
            data_stats[data_id]["total_count_a"] += 1
            if classification == "support":
                data_stats[data_id]["support_count_a"] += 1
            elif classification == "irrelevant":
                data_stats[data_id]["irrelevant_count_a"] += 1
            elif classification == "contradict":
                data_stats[data_id]["contradict_count_a"] += 1
        else:  # response_type == "B"
            data_stats[data_id]["total_count_b"] += 1
            if classification == "support":
                data_stats[data_id]["support_count_b"] += 1
            elif classification == "irrelevant":
                data_stats[data_id]["irrelevant_count_b"] += 1
            elif classification == "contradict":
                data_stats[data_id]["contradict_count_b"] += 1
    
    return list(data_stats.values())


def format_final_output(aggregated_results: List[Dict]) -> List[Dict]:
    """Format final output"""
    vote_mapping = {"A": "model_a", "B": "model_b", "tie": "tie", "bad": "tie"}
    
    final_output = []
    for result in aggregated_results:
        formatted_result = {
            "model_a": result["modelA"],
            "model_b": result["modelB"],
            "winner": vote_mapping.get(result["vote"], "tie"),
            "support_count_a": result["support_count_a"],
            "support_count_b": result["support_count_b"],
            "contradict_count_a": result["contradict_count_a"],
            "contradict_count_b": result["contradict_count_b"],
            "irrelevant_count_a": result["irrelevant_count_a"],
            "irrelevant_count_b": result["irrelevant_count_b"],
            "total_count_a": result["total_count_a"],
            "total_count_b": result["total_count_b"],
            "response_length_a": result["response_length_a"],
            "response_length_b": result["response_length_b"]
        }
        final_output.append(formatted_result)
    
    return final_output


async def analyze_citations():
    """Main function: analyze citations and perform classification"""
    
    with open("data/train_full.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Read {len(data)} raw data records")
    
    sampled_data = filter_and_sample_data(data, 3000)
    
    if not sampled_data:
        print("No qualifying data found")
        return
    
    llm_data = prepare_citation_data_for_llm(sampled_data)
    print(f"Prepared {len(llm_data)} citations for analysis")
    
    print("Starting LLM citation classification...")
    results = await process_data(
        data=llm_data,
        client_dict=client_dict,
        build_messages=build_messages,
        response_model=CitationClassification,
        models=["o4-mini"],
        enable_structured_output=True,
        max_concurrency=100,
        max_tokens=4096
    )
    
    # Filter out None results
    results = [r for r in results if r is not None]
    print(f"Successfully processed {len(results)} citations")
    
    aggregated_results = aggregate_results_by_data_id(results)
    print(f"Aggregated {len(aggregated_results)} data element statistics")
    
    final_output = format_final_output(aggregated_results)
    
    with open("data/citation.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    print(f"Analysis complete! Results saved to citation.json ({len(final_output)} data elements)")

if __name__ == "__main__":
    asyncio.run(analyze_citations()) 