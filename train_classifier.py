#!/usr/bin/env python
"""
Alternative approach: Train a sequence classification model instead of causal LM.
This is often better for binary classification tasks like document verification.

Uses the last hidden state with a classification head instead of next-token prediction.
"""

import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from pathlib import Path
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class VerifierDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(jsonl_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                
                user_content = item['messages'][1]['content']
                question = user_content.replace("Question: ", "").split("\n\nDocument:")[0].strip()
                
                if "Document:" in user_content:
                    document = user_content.split("Document:")[1].split("Does this document")[0].strip()
                else:
                    document = ""
                
                answer = item['messages'][2]['content'].strip().upper()
                label = 1 if answer == "YES" else 0
                
                self.data.append({
                    'question': question,
                    'document': document,
                    'label': label
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        text = f"Question: {item['question']}\n\nDocument: {item['document']}\n\nDoes this document directly answer the question?"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }


class LlamaClassifier(nn.Module):
    """Llama with a classification head."""
    
    def __init__(self, base_model, num_labels=2, dropout=0.1, dtype=torch.bfloat16):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        # Match dtype to base model to avoid mat1/mat2 dtype mismatch
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels, dtype=dtype)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state of last token (like GPT-style classification)
        # Find the last non-padding token for each sequence
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        
        last_hidden_state = outputs.hidden_states[-1]
        pooled = last_hidden_state[torch.arange(batch_size, device=input_ids.device), sequence_lengths]
        
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {'loss': loss, 'logits': logits}


def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def train_epoch(model, train_loader, optimizer, scheduler, device, grad_accum_steps, local_rank):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc="Training") if local_rank == 0 else train_loader
    
    for i, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss'] / grad_accum_steps
        loss.backward()
        
        if (i + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += outputs['loss'].item()
        
        preds = outputs['logits'].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        if local_rank == 0 and isinstance(pbar, tqdm):
            pbar.set_postfix({'loss': f'{outputs["loss"].item():.4f}'})
    
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(train_loader), acc


def eval_epoch(model, val_loader, device, local_rank):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc="Evaluating") if local_rank == 0 else val_loader
    
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs['loss'].item()
            
            preds = outputs['logits'].argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Gather across processes
    loss_tensor = torch.tensor([total_loss, len(val_loader)], device=device, dtype=torch.float32)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    
    avg_loss = loss_tensor[0].item() / loss_tensor[1].item()
    
    # Local metrics (for rank 0 logging)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    return avg_loss, acc, precision, recall, f1


def main():
    local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    world_size = dist.get_world_size()
    
    # Paths
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    TRAIN_FILE = "/scratch/dq2024/doc-verifier/verifier_training_data/train_llama_query_split.jsonl"
    VAL_FILE = "/scratch/dq2024/doc-verifier/verifier_training_data/val_llama_query_split.jsonl"
    OUTPUT_DIR = "/scratch/dq2024/doc-verifier/models/llama-3.2-1b-verifier-classifier-query_split-dp0.5-ldp0.2"
    
    HF_TOKEN = os.environ.get('HF_TOKEN', None)
    
    # Hyperparameters
    BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = 2
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10
    MAX_LENGTH = 1024  # Shorter since we don't need to generate
    PATIENCE = 3
    CLASSIFIER_DROPOUT = 0.5
    
    # LoRA config
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.2
    
    if local_rank == 0:
        print(f"\n{'='*60}")
        print("Document Verifier - Classification Approach")
        print(f"{'='*60}")
        print(f"GPUs: {world_size}")
        print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS * world_size}")
        print(f"{'='*60}\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model (not for causal LM)
    if local_rank == 0:
        print("Loading model...")
    
    base_model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN
    )
    
    # Apply LoRA to base model
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION  # Not causal LM
    )
    base_model = get_peft_model(base_model, lora_config)
    
    if local_rank == 0:
        base_model.print_trainable_parameters()
    
    # Create classifier model (match dtype to base model)
    model = LlamaClassifier(base_model, num_labels=2, dropout=CLASSIFIER_DROPOUT, dtype=torch.bfloat16)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
    
    # Load data
    if local_rank == 0:
        print("Loading datasets...")
    
    train_dataset = VerifierDataset(TRAIN_FILE, tokenizer, MAX_LENGTH)
    val_dataset = VerifierDataset(VAL_FILE, tokenizer, MAX_LENGTH)
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler,
                            num_workers=4, pin_memory=True)
    
    if local_rank == 0:
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Optimizer - include classifier parameters
    optimizer = torch.optim.AdamW([
        {'params': model.module.base_model.parameters(), 'lr': LEARNING_RATE},
        {'params': model.module.classifier.parameters(), 'lr': LEARNING_RATE * 10}  # Higher LR for new head
    ], weight_decay=0.01)
    
    num_training_steps = len(train_loader) * NUM_EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * num_training_steps), num_training_steps)
    
    # Training
    best_val_loss = float('inf')
    best_f1 = 0
    patience_counter = 0
    
    if local_rank == 0:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    dist.barrier()
    
    try:
        for epoch in range(NUM_EPOCHS):
            train_sampler.set_epoch(epoch)
            
            if local_rank == 0:
                print(f"\n{'='*50}")
                print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
                print(f"{'='*50}")
            
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scheduler, device, GRAD_ACCUM_STEPS, local_rank
            )
            val_loss, val_acc, precision, recall, f1 = eval_epoch(model, val_loader, device, local_rank)
            
            dist.barrier()
            
            if local_rank == 0:
                print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
                print(f"Precision:  {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                
                # Save based on F1 score (better metric for potentially imbalanced data)
                if f1 > best_f1:
                    best_f1 = f1
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(f"✓ New best F1! Saving model...")
                    
                    # Save LoRA weights and classifier
                    model.module.base_model.save_pretrained(OUTPUT_DIR)
                    tokenizer.save_pretrained(OUTPUT_DIR)
                    torch.save(model.module.classifier.state_dict(), 
                              os.path.join(OUTPUT_DIR, 'classifier_head.pt'))
                    torch.save({
                        'epoch': epoch,
                        'best_f1': best_f1,
                        'best_val_loss': best_val_loss,
                    }, os.path.join(OUTPUT_DIR, 'training_state.pt'))
                else:
                    patience_counter += 1
                    print(f"✗ No improvement. Patience: {patience_counter}/{PATIENCE}")
            
            stop_tensor = torch.tensor([patience_counter >= PATIENCE], device=device, dtype=torch.int)
            dist.broadcast(stop_tensor, src=0)
            dist.barrier()
            
            if stop_tensor.item():
                if local_rank == 0:
                    print(f"\nEarly stopping after {epoch + 1} epochs")
                break
        
        if local_rank == 0:
            print(f"\n{'='*50}")
            print(f"Training complete!")
            print(f"Best F1: {best_f1:.4f}")
            print(f"Best Val Loss: {best_val_loss:.4f}")
            print(f"{'='*50}")
    
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()