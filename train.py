#!/usr/bin/env python
"""
Fine-tune Llama-3.2-3B (base) for document verification with multi-GPU support.
"""

import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from pathlib import Path
import os

class VerifierDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract from messages format
        question = item['messages'][1]['content'].replace("Question: ", "").replace("\n\nDocument:", "").strip()
        user_content = item['messages'][1]['content']
        if "Document:" in user_content:
            document = user_content.split("Document:")[1].split("Does this document")[0].strip()
        else:
            document = ""
        
        answer = item['messages'][2]['content']
        
        # Simple format for base model
        text = f"Question: {question}\n\nDocument: {document}\n\nAnswer: {answer}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()

def train_epoch(model, train_loader, optimizer, device, grad_accum_steps, local_rank):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    if local_rank == 0:
        pbar = tqdm(train_loader, desc="Training")
    else:
        pbar = train_loader
    
    for i, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / grad_accum_steps
        loss.backward()
        
        if (i + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += outputs.loss.item()
    
    return total_loss / len(train_loader)

def eval_epoch(model, val_loader, device, local_rank):
    model.eval()
    total_loss = 0
    
    if local_rank == 0:
        pbar = tqdm(val_loader, desc="Evaluating")
    else:
        pbar = val_loader
    
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
    
    return total_loss / len(val_loader)

def main():
    # Setup distributed
    local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # Paths
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    TRAIN_FILE = "/scratch/dq2024/doc-verifier/verifier_training_data/train_llama.jsonl"
    VAL_FILE = "/scratch/dq2024/doc-verifier/verifier_training_data/val_llama.jsonl"
    OUTPUT_DIR = "/scratch/dq2024/doc-verifier/models/llama-3.2-1b-verifier"
    
    # GET TOKEN FROM ENVIRONMENT VARIABLE
    HF_TOKEN = os.environ.get('HF_TOKEN', None)
    
    # Hyperparameters
    BATCH_SIZE = 4  # Per GPU
    GRAD_ACCUM_STEPS = 4
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    MAX_LENGTH = 2048
    
    if local_rank == 0:
        print(f"Using {torch.cuda.device_count()} GPUs")
        print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS * torch.cuda.device_count()}")
    
    # Load model WITH TOKEN
    if local_rank == 0:
        print("Loading tokenizer and model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN  # ADD TOKEN HERE
    ).to(device)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])
    
    # Load data
    if local_rank == 0:
        print("Loading datasets...")
    
    train_dataset = VerifierDataset(TRAIN_FILE, tokenizer, MAX_LENGTH)
    val_dataset = VerifierDataset(VAL_FILE, tokenizer, MAX_LENGTH)
    
    # Use DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    if local_rank == 0:
        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Train
    best_val_loss = float('inf')
    if local_rank == 0:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        # Set epoch for sampler
        train_sampler.set_epoch(epoch)
        
        if local_rank == 0:
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, GRAD_ACCUM_STEPS, local_rank)
        val_loss = eval_epoch(model, val_loader, device, local_rank)
        
        if local_rank == 0:
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Saving best model...")
                # Save from rank 0 only, unwrap DDP
                model.module.save_pretrained(OUTPUT_DIR)
                tokenizer.save_pretrained(OUTPUT_DIR)
    
    if local_rank == 0:
        print(f"\nTraining complete! Model saved to {OUTPUT_DIR}")
    
    cleanup_distributed()

if __name__ == "__main__":
    main()