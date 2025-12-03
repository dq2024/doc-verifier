#!/usr/bin/env python
"""
Convert verifier data to Llama-3.2 instruction format.
UPDATED: Split by query ID so no query appears in both train and validation.
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

def read_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def write_jsonl(data: List[Dict], file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def convert_to_llama_format(data: List[Dict]) -> List[Dict]:
    """
    Convert to Llama-3.2 chat format.
    """
    formatted = []
    
    system_prompt = "You are a document relevance verifier. Determine if a document directly answers a question. Respond only with YES or NO."
    
    for item in data:
        # Truncate document if too long
        doc_text = item['document_text']
        if len(doc_text) > 2000:
            doc_text = doc_text[:2000] + "..."
        
        user_message = f"""Question: {item['query']}

Document: {doc_text}

Does this document directly answer the question?"""
        
        assistant_message = "YES" if item['label'] == 1 else "NO"
        
        formatted_item = {
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message},
                {'role': 'assistant', 'content': assistant_message}
            ],
            'label': item['label'],
            'query_id': item['query_id']  # Keep for verification
        }
        
        formatted.append(formatted_item)
    
    return formatted

def split_by_query(data: List[Dict], train_ratio: float = 0.9, seed: int = 42) -> tuple:
    """
    Split data by query ID so no query appears in both train and validation.
    
    Args:
        data: List of examples with 'query_id' field
        train_ratio: Fraction of queries to use for training
        seed: Random seed for reproducibility
    
    Returns:
        (train_data, val_data)
    """
    random.seed(seed)
    
    # Group examples by query ID
    query_to_examples = defaultdict(list)
    for item in data:
        query_id = item['query_id']
        query_to_examples[query_id].append(item)
    
    # Get unique query IDs and shuffle
    query_ids = list(query_to_examples.keys())
    random.shuffle(query_ids)
    
    # Split query IDs
    split_idx = int(len(query_ids) * train_ratio)
    train_query_ids = set(query_ids[:split_idx])
    val_query_ids = set(query_ids[split_idx:])
    
    # Assign examples based on query ID
    train_data = []
    val_data = []
    
    for query_id, examples in query_to_examples.items():
        if query_id in train_query_ids:
            train_data.extend(examples)
        else:
            val_data.extend(examples)
    
    # Shuffle within each split
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    return train_data, val_data

def main():
    # Configuration
    input_file = '/scratch/dq2024/doc-verifier/verifier_training_data/verifier_train_30pct_positive.jsonl'
    output_dir = Path('/scratch/dq2024/doc-verifier/verifier_training_data')
    
    train_file = output_dir / 'train_llama_query_split.jsonl'
    val_file = output_dir / 'val_llama_query_split.jsonl'
    
    print(f"Loading data from: {input_file}")
    data = read_jsonl(input_file)
    print(f"Total examples: {len(data):,}")
    
    # Count unique queries
    unique_queries = set(item['query_id'] for item in data)
    print(f"Unique queries: {len(unique_queries):,}")
    
    # Split by query ID
    print("\nSplitting by query ID (90/10)...")
    train_data, val_data = split_by_query(data, train_ratio=0.9, seed=42)
    
    # Verify no overlap
    train_queries = set(item['query_id'] for item in train_data)
    val_queries = set(item['query_id'] for item in val_data)
    overlap = train_queries & val_queries
    
    print(f"\nTrain queries: {len(train_queries):,}")
    print(f"Val queries: {len(val_queries):,}")
    print(f"Query overlap: {len(overlap)} (should be 0)")
    
    if overlap:
        print("WARNING: Found overlapping queries!")
        print(f"  Overlapping IDs: {list(overlap)[:5]}...")
    
    print(f"\nTrain examples: {len(train_data):,}")
    print(f"Val examples: {len(val_data):,}")
    
    # Convert to Llama format
    train_formatted = convert_to_llama_format(train_data)
    val_formatted = convert_to_llama_format(val_data)
    
    # Stats
    train_pos = sum(1 for x in train_data if x['label'] == 1)
    val_pos = sum(1 for x in val_data if x['label'] == 1)
    
    print(f"\nTrain: {train_pos:,} positive / {len(train_formatted):,} total ({train_pos/len(train_formatted)*100:.1f}%)")
    print(f"Val: {val_pos:,} positive / {len(val_formatted):,} total ({val_pos/len(val_formatted)*100:.1f}%)")
    
    # Save
    write_jsonl(train_formatted, train_file)
    write_jsonl(val_formatted, val_file)
    
    print(f"\nSaved to:")
    print(f"  {train_file}")
    print(f"  {val_file}")
    
    # Print example
    print("\n" + "="*70)
    print("EXAMPLE:")
    print("="*70)
    example = train_formatted[0]
    for msg in example['messages']:
        print(f"\n[{msg['role'].upper()}]")
        print(msg['content'][:300] + ("..." if len(msg['content']) > 300 else ""))
    
    # Verify split integrity
    print("\n" + "="*70)
    print("SPLIT VERIFICATION:")
    print("="*70)
    print(f"Train query IDs (first 5): {sorted(train_queries)[:5]}")
    print(f"Val query IDs (first 5): {sorted(val_queries)[:5]}")

if __name__ == "__main__":
    main()