import json
from pathlib import Path
from collections import Counter

INPUT_JSON  = Path("Sciarena-Eval-2000.json")
OUTPUT_JSON = Path("Sciarena-Eval-2000_results.json")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    input_data  = load_json(INPUT_JSON)
    output_data = load_json(OUTPUT_JSON)

    if len(input_data) != len(output_data):
        print(f"❗ Sample count mismatch: {len(input_data)} vs {len(output_data)}")
    N = min(len(input_data), len(output_data))

    agree, disagree, skip = 0, 0, 0
    confusion = Counter()
    for i in range(N):
        gold = str(input_data[i].get("vote", "")).strip().upper()
        pred = str(output_data[i].get("predicted_output", "")).strip().upper()

        if gold not in {"A", "B"} or pred not in {"A", "B"}:
            skip += 1
            continue

        if gold == pred:
            agree += 1
        else:
            disagree += 1

        confusion[(gold, pred)] += 1

    print(f"\nTotal samples: {N}")
    print(f"Effective comparisons: {agree + disagree}")
    print(f"Agreement (consensus): {agree}")
    print(f"Disagreement: {disagree}")
    print(f"Invalid/missing (skipped): {skip}")
    print("\nConfusion matrix (gold, pred):")
    for key in sorted(confusion):
        print(f"  gold={key[0]}, pred={key[1]} : {confusion[key]}")

    if agree + disagree > 0:
        acc = agree / (agree + disagree)
        print(f"\nConsensus accuracy: {acc:.2%}")

if __name__ == "__main__":
    main()