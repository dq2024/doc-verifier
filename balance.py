#!/usr/bin/env python
"""
Balance the merged dataset to 30% positive rate.
"""

import json
import random
from pathlib import Path

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def balance_dataset(input_file, output_file, target_positive_rate=0.30, seed=42):
    """
    Balance dataset to target positive rate by undersampling negatives.
    """
    random.seed(seed)
    
    print(f"Loading data from: {input_file}")
    data = read_jsonl(input_file)
    
    positive = [ex for ex in data if ex['label'] == 1]
    negative = [ex for ex in data if ex['label'] == 0]
    
    print(f"\nBefore balancing:")
    print(f"  Positive: {len(positive):,}")
    print(f"  Negative: {len(negative):,}")
    print(f"  Total: {len(data):,}")
    print(f"  Positive rate: {len(positive)/len(data)*100:.2f}%")
    
    # Calculate how many negatives we need for 30% positive rate
    # positive / (positive + negative) = 0.30
    # positive = 0.30 * (positive + negative)
    # positive = 0.30 * positive + 0.30 * negative
    # 0.70 * positive = 0.30 * negative
    # negative = (0.70 / 0.30) * positive
    
    n_positive = len(positive)
    n_negative_needed = int(n_positive * (1 - target_positive_rate) / target_positive_rate)
    
    print(f"\nTarget: {target_positive_rate*100:.0f}% positive rate")
    print(f"  Keep all {n_positive:,} positives")
    print(f"  Need {n_negative_needed:,} negatives")
    
    # Sample hard negatives (highest retrieval score)
    negative_sorted = sorted(negative, key=lambda x: x['retrieval_score'], reverse=True)
    sampled_negative = negative_sorted[:n_negative_needed]
    
    # Combine and shuffle
    balanced = positive + sampled_negative
    random.shuffle(balanced)
    
    final_pos = len([x for x in balanced if x['label'] == 1])
    final_neg = len([x for x in balanced if x['label'] == 0])
    
    print(f"\nAfter balancing:")
    print(f"  Positive: {final_pos:,}")
    print(f"  Negative: {final_neg:,}")
    print(f"  Total: {len(balanced):,}")
    print(f"  Positive rate: {final_pos/len(balanced)*100:.2f}%")
    
    # Save
    print(f"\nSaving to: {output_file}")
    write_jsonl(balanced, output_file)
    
    return balanced

def main():
    input_file = '/scratch/dq2024/doc-verifier/verifier_training_data/merged_3_retrievers_1k_queries.jsonl'
    output_file = '/scratch/dq2024/doc-verifier/verifier_training_data/verifier_train_20pct_positive.jsonl'
    
    balanced = balance_dataset(
        input_file=input_file,
        output_file=output_file,
        target_positive_rate=0.20
    )

if __name__ == "__main__":
    main()