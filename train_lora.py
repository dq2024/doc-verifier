#!/usr/bin/env python
"""
Fine-tune Llama-3.2-1B for document verification using LoRA.
Key improvements over original:
- LoRA for parameter-efficient fine-tuning (prevents overfitting)
- Loss computed only on answer tokens (YES/NO)
- Early stopping
- Learning rate scheduler with warmup
"""

import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
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
        user_content = item['messages'][1]['content']
        
        # Parse question
        question = user_content.replace("Question: ", "").split("\n\nDocument:")[0].strip()
        
        # Parse document
        if "Document:" in user_content:
            document = user_content.split("Document:")[1].split("Does this document")[0].strip()
        else:
            document = ""
        
        # Get answer (YES or NO)
        answer = item['messages'][2]['content'].strip()
        
        # Build prompt and full text
        prompt = f"Question: {question}\n\nDocument: {document}\n\nDoes this document directly answer the question? Answer YES or NO.\n\nAnswer:"
        full_text = f"{prompt} {answer}"
        
        # Tokenize prompt alone to find where answer starts
        prompt_encoding = self.tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]
        
        # Tokenize full text
        full_encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = full_encoding['input_ids'].squeeze()
        attention_mask = full_encoding['attention_mask'].squeeze()
        
        # Create labels: only compute loss on answer tokens
        labels = input_ids.clone()
        labels[:prompt_length] = -100  # Mask the prompt
        labels[attention_mask == 0] = -100  # Mask padding
        
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


def train_epoch(model, train_loader, optimizer, scheduler, device, grad_accum_steps, local_rank):
    model.train()
    total_loss = 0
    num_batches = 0
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += outputs.loss.item()
        num_batches += 1
        
        if local_rank == 0 and isinstance(pbar, tqdm):
            pbar.set_postfix({'loss': f'{outputs.loss.item():.4f}'})
    
    return total_loss / num_batches


def eval_epoch(model, val_loader, device, local_rank):
    model.eval()
    total_loss = 0
    num_batches = 0
    
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
            num_batches += 1
    
    # Gather losses across all processes
    loss_tensor = torch.tensor([total_loss, num_batches], device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    
    return loss_tensor[0].item() / loss_tensor[1].item()


def main():
    # Setup distributed
    local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # Paths - UPDATE THESE FOR YOUR SETUP
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    TRAIN_FILE = "/scratch/dq2024/doc-verifier/verifier_training_data/train_llama.jsonl"
    VAL_FILE = "/scratch/dq2024/doc-verifier/verifier_training_data/val_llama.jsonl"
    OUTPUT_DIR = "/scratch/dq2024/doc-verifier/models/llama-3.2-1b-verifier-lora"
    
    # Get token from environment variable
    HF_TOKEN = os.environ.get('HF_TOKEN', None)
    
    # Hyperparameters - tuned to prevent overfitting
    BATCH_SIZE = 4          # Per GPU
    GRAD_ACCUM_STEPS = 4
    LEARNING_RATE = 1e-4    # Higher LR is fine with LoRA
    NUM_EPOCHS = 10         # More epochs since we have early stopping
    MAX_LENGTH = 2048
    WARMUP_RATIO = 0.1
    PATIENCE = 3            # Early stopping patience
    
    # LoRA config
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    
    if local_rank == 0:
        print(f"Using {torch.cuda.device_count()} GPUs")
        print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS * torch.cuda.device_count()}")
        print(f"Using LoRA with r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    
    # Load tokenizer
    if local_rank == 0:
        print("Loading tokenizer and model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN
    )
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    
    if local_rank == 0:
        model.print_trainable_parameters()
    
    model = model.to(device)
    
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
    
    # Optimizer - only optimize LoRA parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * NUM_EPOCHS // GRAD_ACCUM_STEPS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    if local_rank == 0:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        # Set epoch for sampler (important for shuffling)
        train_sampler.set_epoch(epoch)
        
        if local_rank == 0:
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
            print(f"{'='*50}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, 
            device, GRAD_ACCUM_STEPS, local_rank
        )
        val_loss = eval_epoch(model, val_loader, device, local_rank)
        
        if local_rank == 0:
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            print(f"LR:         {scheduler.get_last_lr()[0]:.2e}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"✓ New best validation loss! Saving model...")
                
                # Save LoRA weights (from rank 0 only)
                model.module.save_pretrained(OUTPUT_DIR)
                tokenizer.save_pretrained(OUTPUT_DIR)
                
                # Also save training state
                torch.save({
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(OUTPUT_DIR, 'training_state.pt'))
            else:
                patience_counter += 1
                print(f"✗ No improvement. Patience: {patience_counter}/{PATIENCE}")
                
                if patience_counter >= PATIENCE:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        # Broadcast early stopping decision to all processes
        stop_tensor = torch.tensor([patience_counter >= PATIENCE], device=device)
        dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item():
            break
    
    if local_rank == 0:
        print(f"\n{'='*50}")
        print(f"Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved to: {OUTPUT_DIR}")
        print(f"{'='*50}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()