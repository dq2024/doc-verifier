#!/usr/bin/env python
"""
Filter documents using the trained classifier model.
Reads a JSON file with questions and candidate documents (ctxs),
verifies each document, and outputs only verified documents.
"""

import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from tqdm import tqdm
import argparse
from pathlib import Path


class LlamaClassifier(nn.Module):
    """Llama with a classification head."""
    
    def __init__(self, base_model, num_labels=2, dtype=torch.bfloat16):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels, dtype=dtype)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state of last non-padding token
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        
        last_hidden_state = outputs.hidden_states[-1]
        pooled = last_hidden_state[torch.arange(batch_size, device=input_ids.device), sequence_lengths]
        
        logits = self.classifier(pooled)
        return logits


def load_model(model_dir, base_model_name="meta-llama/Llama-3.2-1B", device="cuda", hf_token=None):
    """Load the trained classifier model."""
    print(f"Loading tokenizer from {model_dir}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    print(f"Loading base model: {base_model_name}...", flush=True)
    base_model = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        token=hf_token
    )
    
    print(f"Loading LoRA weights from {model_dir}...", flush=True)
    base_model = PeftModel.from_pretrained(base_model, model_dir)
    
    # Create classifier and load head weights
    model = LlamaClassifier(base_model, num_labels=2, dtype=torch.bfloat16)
    
    classifier_path = Path(model_dir) / "classifier_head.pt"
    print(f"Loading classifier head from {classifier_path}...", flush=True)
    model.classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def verify_document(model, tokenizer, question, document, device="cuda", max_length=1024):
    """
    Verify if a document answers a question.
    Returns: (prediction, probability)
        prediction: 1 for YES, 0 for NO
        probability: confidence score for the prediction
    """
    # Format input same as training
    text = f"Question: {question}\n\nDocument: {document}\n\nDoes this document directly answer the question?"
    
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=-1)
        prediction = logits.argmax(dim=-1).item()
        confidence = probs[0, prediction].item()
    
    return prediction, confidence


def verify_batch(model, tokenizer, questions, documents, device="cuda", max_length=1024, batch_size=16):
    """
    Batch verification for efficiency.
    Returns: list of (prediction, probability) tuples
    """
    results = []
    
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i+batch_size]
        batch_documents = documents[i:i+batch_size]
        
        texts = [
            f"Question: {q}\n\nDocument: {d}\n\nDoes this document directly answer the question?"
            for q, d in zip(batch_questions, batch_documents)
        ]
        
        encoding = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)
            
            for j in range(len(batch_questions)):
                pred = predictions[j].item()
                conf = probs[j, pred].item()
                results.append((pred, conf))
    
    return results


def process_file(input_file, output_file, model, tokenizer, device="cuda", 
                 max_length=1024, batch_size=16, confidence_threshold=0.5,
                 keep_unverified=False):
    """
    Process input JSONL file and filter documents.
    """
    print(f"\nReading {input_file}...", flush=True)
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"Processing {len(data)} questions...", flush=True)
    
    total_docs = 0
    kept_docs = 0
    
    for item in tqdm(data, desc="Processing questions"):
        question = item.get('question', item.get('input', ''))
        ctxs = item.get('ctxs', [])
        
        if not ctxs:
            continue
        
        # Prepare batch
        questions = [question] * len(ctxs)
        documents = [f"{ctx.get('title', '')} {ctx.get('text', '')}" for ctx in ctxs]
        
        # Verify all documents for this question
        results = verify_batch(model, tokenizer, questions, documents, 
                              device=device, max_length=max_length, batch_size=batch_size)
        
        # Filter documents
        verified_ctxs = []
        for ctx, (pred, conf) in zip(ctxs, results):
            total_docs += 1
            
            if pred == 1 and conf >= confidence_threshold:
                ctx['verified'] = True
                ctx['verification_confidence'] = round(conf, 4)
                verified_ctxs.append(ctx)
                kept_docs += 1
            elif keep_unverified:
                ctx['verified'] = False
                ctx['verification_confidence'] = round(conf, 4)
                verified_ctxs.append(ctx)
        
        item['ctxs'] = verified_ctxs
    
    print(f"\nResults:")
    print(f"  Total documents processed: {total_docs}")
    print(f"  Documents kept (verified): {kept_docs}")
    print(f"  Documents removed: {total_docs - kept_docs}")
    print(f"  Keep rate: {100 * kept_docs / total_docs:.1f}%")
    
    # Write output as JSONL (same format as input)
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print("Done!")
    return data


def main():
    parser = argparse.ArgumentParser(description="Filter documents using trained classifier")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    parser.add_argument("--model-dir", "-m", required=True, 
                        help="Path to trained model directory")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-1B",
                        help="Base model name (default: meta-llama/Llama-3.2-1B)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for inference (default: 16)")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Max sequence length (default: 1024)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold to keep documents (default: 0.5)")
    parser.add_argument("--keep-unverified", action="store_true",
                        help="Keep unverified docs with verified=False flag")
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()
    
    # Get HF token
    import os
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    # Load model
    model, tokenizer = load_model(
        args.model_dir, 
        base_model_name=args.base_model,
        device=device,
        hf_token=hf_token
    )
    
    # Process file
    process_file(
        input_file=args.input,
        output_file=args.output,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        confidence_threshold=args.threshold,
        keep_unverified=args.keep_unverified
    )


if __name__ == "__main__":
    main()