# SciArena

A research tool for analyzing AI model performance in scientific literature review, focusing on citation attribution analysis, model preference evaluation, and bias detection.

## Features

- **Citation attribution Analysis**: Automatically analyze consistency between citations and response content using LLMs
- **Preference Evaluation**: Calculate AI model preference scores using Bradley-Terry model
- **LLM as Judge**: Use LLM to evaluate and compare AI model responses in pairwise comparison experiments
- **Statistical Analysis**: Provide bootstrap confidence intervals and variance analysis

## Project Structure

```
sciarena/
├── common/
│   ├── openai_utils.py      # OpenAI API utilities
│   └── utils.py             # Bradley-Terry model and statistical analysis
├── scripts/
│   ├── attribution/
│   │   └── citation_analysis.py     # Citation analysis script
│   ├── bias_analysis/
│   │   └── preference_calculate.py  # Preference calculation script
│   └── llm_as_judge/
│       └── Sciarena-Eval_evaluation.py  # LLM as judge experiment script
│       └── Sciarena-Eval_evaluation_light.py # Light version of LLM as judge experiment script
│       └── Sciarena-Eval_check_evaluation.py # Compare LLM as judge with gold answer
├── requirements.txt
└── README.md

```

## Installation

### Requirements

- Python 3.8+

### Setup

```bash
# Clone repository
git clone SciArena
cd sciarena

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### Configure API Keys

Create a `.env` file in the project root:

```bash
# OpenAI API Configuration
# Get API key: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

```

**Note**: API keys are required for citation analysis and LLM as judge experiments. Preference analysis can use existing `citation_attribution.json` files.

## Usage

### 1. Citation Analysis

Analyze citation accuracy in AI model responses:

```bash
python scripts/attribution/citation_analysis.py

```

This script:

- Reads original data from `data/train_full.json`
- Uses LLM to analyze citation accuracy (support/irrelevant/contradict)
- Outputs results to `data/citation.json`

### 2. Preference and Bias Analysis

Calculate model preference scores based on citation analysis:

```bash
python scripts/bias_analysis/preference_calculate.py

```

This script:

- Reads citation analysis results from `data/citation_attribution.json`
- Calculates Bradley-Terry ratings and confidence intervals
- Analyzes impact of citation count, response length, etc.
- Outputs detailed analysis to `results/` directory

### 3. LLM as Judge Experiment

Evaluate AI model responses using LLM as a judge in pairwise comparisons:

```bash
python scripts/llm_as_judge/Sciarena-Eval_evaluation.py

```

This script:

- Reads sampled data from `Sciarena-Eval-2000.json`
- Uses a chosen model to judge which response is better in pairwise comparisons

```bash
python scripts/llm_as_judge/Sciarena-Eval_check_evaluation.py

```

This script:

- Calculates self-consistency (accuracy) between LLM judgments and original votes
- Outputs detailed experiment results including accuracy metrics and judgment distribution

## Configuration

### Citation Analysis

- `sample_size`: Number of samples to analyze (default: 3000)
- `max_concurrency`: Concurrent requests (default: 100)
- `max_tokens`: LLM response length limit (default: 4096)

### Preference Analysis

- `anchor_model`: Reference model (default: 'GPT-4.1')
- `anchor_rating`: Reference rating (default: 1000)
- `num_bootstrap_samples`: Bootstrap samples (default: 100)

### LLM as Judge

- `model`: Judge model (default: 'gpt-4.1')
- `max_concurrency`: Concurrent requests (default: 50)
- `max_tokens`: Judge response length limit (default: 2048)
- `input_file`: Input data file (default: 'data/train_sampled_2000.json')

## Data Format

### Input Data

```json
{
  "id": "unique_id",
  "modelA": "model_name_a",
  "modelB": "model_name_b",
  "responseA": "response_content_a",
  "responseB": "response_content_b",
  "citations_a": [
    {
      "concise_authors": "Author et al.",
      "content": "citation_content"
    }
  ],
  "vote": "A"
}

```

### Output Data

```json
{
    "model_a": "Gemini-2.5-Flash",
    "model_b": "Qwen3-235B-A22B",
    "winner": "model_b",
    "support_count_a": 10,
    "support_count_b": 7,
    "irrelevant_count_a": 1,
    "irrelevant_count_b": 3,
    "total_count_a": 11,
    "total_count_b": 10,
    "response_length_a": 1680,
    "response_length_b": 2156
}

```

---

## LLM as Judge: Evaluation Results

This section summarizes the pairwise evaluation results of several models using the "LLM as Judge" paradigm, both **with**and **without** citation information.

### (a) Results Without Citation

| Model | Zero-Shot | One-Shot | Two-Shot | Three-Shot | Four-Shot |
| --- | --- | --- | --- | --- | --- |
| o4-mini | 64.05 | 63.65 | 64.35 | 64.50 | 64.75 |
| GPT-4.1 | 61.10 | 60.05 | 60.30 | 59.40 | 58.60 |
| DeepSeek-R1-0528 | 60.05 | 49.50 | 50.90 | 48.90 | 48.75 |

### (b) Results With Citation

| Model | Zero-Shot | One-Shot | Two-Shot | Three-Shot |
| --- | --- | --- | --- | --- |
| o4-mini | 58.55 | 62.75 | 61.06 | 62.54 |
| GPT-4.1 | 59.35 | 59.60 | 58.55 | 58.05 |
| DeepSeek-R1-0528 | 59.85 | 54.65 | 55.05 | 51.20 |

### Key Insights

- **o4-mini** achieves the highest overall scores in most scenarios.
    - *Without citation*: Shows steady improvement as more demonstration shots are added.
    - *With citation*: Lower zero-shot score, but quickly recovers with examples, indicating that examples are crucial for handling citations.
- **GPT-4.1** is relatively stable, with a slight performance decrease as shots increase, and is less sensitive to citation inclusion.
- **DeepSeek-R1-0528**:
    - *Without citation*: Adding demonstration examples significantly decreases performance, suggesting this model fails to extract meaningful guidance from examples.
    - *With citation*: Strong zero-shot performance (even tied for best), but further examples reduce scores.
- **Adding citation information generally increases task difficulty** (average scores drop for all models), but models respond very differently:
    - o4-mini needs examples to offset citation-induced complexity,
    - GPT-4.1 is minimally affected,
    - DeepSeek benefits from citations in the zero-shot case, but adding more examples can be detrimental.

---

## Troubleshooting

**ModuleNotFoundError**: Ensure you're running scripts from the project root directory.

**API Errors**: Check your API key configuration and account balance.

**Memory Issues**: Reduce `sample_size` or `max_concurrency` for large datasets.

**Missing Files**: Ensure `data/train.json` exists before running analysis.

## License

MIT License - see [LICENSE](https://poe.com/chat/LICENSE) file for details.

---
*This project is intended for academic research purposes.* 
