#!/usr/bin/env python
"""
Convert verifier data to Llama-3.2 instruction format.
"""

import json
import random
from pathlib import Path
from typing import List, Dict

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
    Uses the standard Llama-3 message format.
    """
    formatted = []
    
    system_prompt = "You are a document relevance verifier. Determine if a document directly answers a question. Respond only with YES or NO."
    
    for item in data:
        # Truncate document if too long (Llama-3.2-3B has 128k context but we'll be conservative)
        doc_text = item['document_text']
        if len(doc_text) > 2000:  # ~500 tokens
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
            'label': item['label']  # Keep for validation
        }
        
        formatted.append(formatted_item)
    
    return formatted

def main():
    input_file = '/scratch/dq2024/doc-verifier/verifier_training_data/verifier_train_30pct_positive.jsonl'
    output_dir = Path('/scratch/dq2024/doc-verifier/verifier_training_data')
    
    train_file = output_dir / 'train_llama.jsonl'
    val_file = output_dir / 'val_llama.jsonl'
    
    print(f"Loading data from: {input_file}")
    data = read_jsonl(input_file)
    print(f"Total examples: {len(data):,}")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(data)
    
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Train: {len(train_data):,}")
    print(f"Val: {len(val_data):,}")
    
    # Convert
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

if __name__ == "__main__":
    main()