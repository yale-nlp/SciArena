import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List

from openai import AzureOpenAI, APIError, APITimeoutError
from tqdm import tqdm

# This variable sets the shot number (0=zero-shot, 1=one-shot, 2=two-shot)
N_SHOT = 2   

INPUT_JSON  = Path("Sciarena-Eval-2000.json")
OUTPUT_JSON = Path("Sciarena-Eval-2000_results.json")

AZURE_CONF = {}

client = AzureOpenAI(
    api_key       = AZURE_CONF["KEY"],
    azure_endpoint= AZURE_CONF["ENDPOINT"],
    api_version   = AZURE_CONF["API_VERSION"],
    max_retries   = 1,
    timeout       = AZURE_CONF["TIMEOUT"],
)

GOLD_EXAMPLES: List[dict] = [
    {
        "question": "How do recent accelerated ageing studies better predict natural paper degradation?",
        "responseA": "Recent accelerated ageing studies have improved predictions of natural paper degradation by refining testing protocols, developing advanced kinetic models, and validating correlations between laboratory and real-world conditions. Cyclic ageing methods that periodically reintroduce moisture to paper samples better replicate natural hydrolytic degradation mechanisms. Experiments have shown 1.5-1.6 times faster cellulose degradation compared to continuous ageing due to enhanced water absorption (Mihajlović et al. 2023a).\n\nConfined artificial ageing in controlled humidity environments has been validated to mimic natural degradation pathways. Studies confirm that such methods align with observed oxidation and hydrolysis patterns in historical artifacts (Mirković et al. 2024a). Accelerated protocols now incorporate standardized time-temperature equivalences, such as 72 hours at 105°C corresponding to 25 years of natural ageing, enabling direct lifespan predictions for materials like traditional Xuan paper (Dong et al. 2025a).\n\nKinetic models, including modifications to the Ekenstam framework, account for non-linear degradation rates and environmental factors (e.g., temperature, humidity, pH), allowing multi-century predictions of cellulose chain scissions and mechanical property losses (Tétreault et al. 2023a). These models integrate activation energy (Ea) values derived from accelerated tests, with alkaline-filled papers exhibiting higher Ea (∼129 kJ/mol) and greater durability than acidic counterparts (Tétreault et al. 2023a). Improved measurement techniques, such as rheometry and viscometry for tracking degree of polymerization (DP), ensure accurate degradation data by standardizing solvent preparation and temperature conditions (Maraghechi et al. 2023b). Material-specific studies, such as those on soymilk-sized papers, demonstrate how accelerated ageing quantifies long-term stability by correlating protein degradation rates with cellulose oxidation under controlled thermal stress (Wu et al. 2023a).\n\nFurthermore, non-destructive spectroscopic methods (e.g., Raman, FTIR) enable real-time monitoring of molecular changes during accelerated tests, bridging laboratory findings with natural ageing markers like carbonyl group formation (Botti et al. 2022a). Validation efforts include multi-year studies comparing artificially aged samples with naturally aged counterparts, confirming that accelerated protocols preserve degradation mechanisms observed in historical collections (Samir et al. 2022a). These advances collectively enhance the reliability of accelerated ageing in predicting paper longevity, though challenges remain in fully replicating complex environmental interactions (e.g., pollutant exposure, microbial activity) over extended timescales (Mirković et al. 2024a, Dong et al. 2025a).",
        "responseB": "Accelerated ageing studies have been developed to better predict natural paper degradation. Recent research has focused on improving these methods to more accurately reflect real-life degradation processes. For instance, studies have shown that cyclic ageing can accelerate paper degradation by 1.5-1.6 times compared to continuous ageing, due to the more pronounced effect of hydrolytic mechanisms (Mihajlović et al. 2023b). Additionally, the initial water content in cellulose can increase the ageing rate two to three times more than in dry conditions (Mihajlović et al. 2023b).\n\nThe use of alternative dielectric fluids, such as natural esters, has also been investigated, with results indicating that they can slow down the degradation rate of Kraft paper (Montero et al. 2023a, Gutiérrez et al. 2023a, Wilhelm et al. 2022a). Furthermore, research has demonstrated that the integration of degree of polymerization, equilibrium moisture content, and pH into kinetic models can predict the decay rate of different types of papers under various environmental conditions (Tétreault et al. 2023a, Tétreault et al. 2023b). These advancements aim to provide more accurate predictions of paper degradation, ultimately informing conservation strategies for paper artifacts.",
        "vote": "A"
    },
    {
        "question": "What roles does AI play in drug target discovery and virtual screening?",
        "responseA": "Artificial intelligence (AI) plays a significant role in drug target discovery and virtual screening, revolutionizing the pharmaceutical industry. AI-driven approaches, such as machine learning and network-based analyses, have accelerated the identification of new targets associated with different diseases (Khanam et al. 2024a). In virtual screening, AI algorithms can rapidly analyze chemical databases and predict the binding affinity of small molecules to target proteins, allowing for the identification of lead compounds with a higher likelihood of success (Tripathy et al. 2024a).\n\nAI-powered virtual screening techniques, such as molecular docking and deep learning approaches, have notably broadened the chemical space for exploration, uncovering molecules that traditional screening methods might have overlooked (Vișan et al. 2024b). Additionally, AI can optimize lead compounds to enhance their drug-like characteristics and safety (Khanam et al. 2024a).\n\nAI also plays a crucial role in target identification, enabling researchers to analyze vast datasets, integrate diverse data sources, and utilize advanced algorithms to predict potential drug targets (Gupta et al. 2021a). Overall, AI has transformed the drug discovery process, making it faster, cheaper, and more effective while minimizing repeated work (Khan et al. 2024a). By streamlining processes, AI can improve the efficiency and success rate of identifying viable drug candidates (Ji et al. 2022b).",
        "responseB": "Artificial intelligence (AI) plays significant roles in both drug target discovery and virtual screening, streamlining early phases of pharmaceutical research. In target discovery, AI leverages machine learning and network-based analyses to process vast, complex biological datasets, including multi-omics data, identifying novel therapeutic targets and understanding disease-target associations (Vijayan et al. 2021a). AI platforms can predict protein structures, aiding druggability assessments (Ocaña et al. 2025a). AI can also analyze genomic or proteomic information to uncover previously unknown oncogenic vulnerabilities or disease mechanisms, thereby improving the crucial target selection and validation process (Nizhenkovska et al. 2024a, Gupta et al. 2021a).\n\nFor virtual screening, AI algorithms computationally evaluate extensive chemical libraries much faster than traditional methods (Suleiman et al. 2023a). These AI tools, employing techniques like deep learning and support vector machines, analyze molecular structures and predict binding affinities between compounds and protein targets (Nizhenkovska et al. 2024a, Pache et al. 2025a), effectively prioritizing molecules with a higher likelihood of therapeutic efficacy for experimental validation (Serrano et al. 2024b). AI enhances conventional virtual screening approaches like molecular docking by improving scoring functions (Son et al. 2018a)and allows exploration of wider chemical spaces to identify promising hit and lead compounds (Gawade et al. 2023a), ultimately accelerating drug discovery and making it more efficient (Vișan et al. 2024c).",
        "vote": "B"
    },
    # Can add more example if needed
]

def build_fewshot_prompt(item: dict, n_shot: int = 0) -> str:
    head = (
        "You are an expert in scientific literature synthesis. "
        "Your task is to evaluate the quality of two AI-generated citation-attributed responses to a user's question. "
        "Assess both responses for relevance, accuracy, clarity, and appropriate use of citations. "
        "Then, select the response, Output (a) or Output (b), that best address the user's question.\n"
    )
    shots = ""
    n = min(n_shot, len(GOLD_EXAMPLES))
    for i in range(n):
        eg = GOLD_EXAMPLES[i]
        shots += (
            f"\nExample {i+1}:\n"
            f"User Question:\n{eg['question']}\n\n"
            f"Output (a):\n{eg['responseA']}\n\n"
            f"Output (b):\n{eg['responseB']}\n\n"
            f"Expert choice: {eg['vote']}\n"
            "---\n"
        )
    tail = (
        "\nNow, evaluate a new question in the same way. For the following, answer only with \"A\" or \"B\":\n\n"
        f"User Question:\n{item.get('question', '')}\n\n"
        f"Output (a):\n{item.get('responseA', '')}\n\n"
        f"Output (b):\n{item.get('responseB', '')}\n\n"
        "Which is best, Output(a) or Output(b)?\n"
        "Answer with only: \"A\" or \"B\""
    )
    return head + shots + tail

vote_pat = re.compile(r"\b([AB])\b", flags=re.I)

def normalize_vote(text: str) -> str:
    if not text:
        return ""
    m = vote_pat.search(text.strip())
    return m.group(1).upper() if m else ""

def score_one(idx: int, item: dict, n_shot: int) -> Tuple[int, dict]:
    question   = item.get("question")
    response_a = item.get("responseA")
    response_b = item.get("responseB")

    if not (question and response_a and response_b):
        item["predicted_output"] = ""
        return idx, item

    prompt = build_fewshot_prompt(item, n_shot=n_shot)

    for attempt in range(1, AZURE_CONF["RETRY"] + 1):
        try:
            resp  = client.chat.completions.create(
                model    = AZURE_CONF["DEPLOY"],
                messages = [{"role": "user", "content": prompt}],
            )
            vote = normalize_vote(resp.choices[0].message.content)
            item["predicted_output"] = vote
            return idx, item

        except (APIError, APITimeoutError, Exception) as e:
            if attempt == AZURE_CONF["RETRY"]:
                print(f"[ERROR] idx={idx} failed after {attempt} tries: {e}")
                item["predicted_output"] = ""
            time.sleep(2 ** attempt)

    return idx, item

def main():
    print(f"Few-shot setting: n_shot={N_SHOT} (max available gold examples: {len(GOLD_EXAMPLES)})")
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"{INPUT_JSON} does not exist")

    with INPUT_JSON.open(encoding="utf-8") as f:
        data = json.load(f)

    results = [None] * len(data)

    with ThreadPoolExecutor(max_workers=AZURE_CONF["MAX_WORKERS"]) as pool:
        futures = {pool.submit(score_one, i, sample, N_SHOT): i
                   for i, sample in enumerate(data)}

        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Scoring"):
            idx, scored_item = fut.result()
            results[idx] = scored_item

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done ✔ Saved {len(results)} items to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()