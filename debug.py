#!/usr/bin/env python
"""Debug script to diagnose verifier performance issues."""

import json
import torch
from collections import Counter
import regex
import unicodedata

# ============================================================================
# Copy your has_answer code here
# ============================================================================
def _normalize(text):
    return unicodedata.normalize('NFD', text)

class SimpleTokenizer:
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'
    def __init__(self):
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )
    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

def has_answer(answers, text, tokenizer):
    text = _normalize(text)
    text_tokens = tokenizer.tokenize(text, uncased=True)
    for answer in answers:
        if not answer:
            continue
        answer = _normalize(answer)
        answer_tokens = tokenizer.tokenize(answer, uncased=True)
        if not answer_tokens:
            continue
        for i in range(0, len(text_tokens) - len(answer_tokens) + 1):
            if answer_tokens == text_tokens[i: i + len(answer_tokens)]:
                return True
    return False

# ============================================================================
# Load model (copy from filter_documents.py)
# ============================================================================
from filter_documents import load_model, verify_batch

def main():
    import os
    
    # CONFIGURE THESE PATHS
    MODEL_DIR = "/scratch/dq2024/doc-verifier/models/llama-3.2-1b-verifier-classifier"
    VAL_FILE = "/scratch/dq2024/doc-verifier/verifier_training_data/val_llama.jsonl"
    TEST_FILE = "/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_corpus_finetuned_base_t100.json"  # Your test file
    ANSWERS_FILE = "/scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl"  # File with answer strings
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_token = os.environ.get('HF_TOKEN')
    
    model, tokenizer_model = load_model(MODEL_DIR, device=device, hf_token=hf_token)
    str_tokenizer = SimpleTokenizer()
    
    # ========================================================================
    # TEST 1: Validation data (should match training performance ~85%)
    # ========================================================================
    print("="*70)
    print("TEST 1: Validation data (expect ~85% accuracy)")
    print("="*70)
    
    val_data = []
    with open(VAL_FILE) as f:
        for line in f:
            val_data.append(json.loads(line))
    
    # Sample 200 for speed
    val_sample = val_data[:200]
    
    questions = []
    documents = []
    true_labels = []
    
    for item in val_sample:
        user_content = item['messages'][1]['content']
        q = user_content.replace("Question: ", "").split("\n\nDocument:")[0].strip()
        d = user_content.split("Document:")[1].split("Does this document")[0].strip()
        label = 1 if item['messages'][2]['content'].strip().upper() == "YES" else 0
        
        questions.append(q)
        documents.append(d)
        true_labels.append(label)
    
    preds = verify_batch(model, tokenizer_model, questions, documents, device=device)
    pred_labels = [p[0] for p in preds]
    
    correct = sum(1 for p, t in zip(pred_labels, true_labels) if p == t)
    print(f"Validation accuracy: {correct}/{len(true_labels)} = {100*correct/len(true_labels):.1f}%")
    
    # Confusion matrix
    tp = sum(1 for p, t in zip(pred_labels, true_labels) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred_labels, true_labels) if p == 1 and t == 0)
    tn = sum(1 for p, t in zip(pred_labels, true_labels) if p == 0 and t == 0)
    fn = sum(1 for p, t in zip(pred_labels, true_labels) if p == 0 and t == 1)
    
    print(f"\nConfusion Matrix (Val):")
    print(f"  TP={tp}, FP={fp}")
    print(f"  FN={fn}, TN={tn}")
    print(f"  Precision: {tp/(tp+fp):.3f}" if tp+fp > 0 else "  Precision: N/A")
    print(f"  Recall: {tp/(tp+fn):.3f}" if tp+fn > 0 else "  Recall: N/A")
    
    # ========================================================================
    # TEST 2: Check test data with ground truth
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 2: Test data with has_answer ground truth")
    print("="*70)
    
    # Load test data and answers
    # ADJUST THIS BASED ON YOUR FILE FORMAT
    test_data = []
    with open(TEST_FILE) as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    # Load answers file
    answers_map = {}  # question -> answers list
    with open(ANSWERS_FILE) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                q = item.get('question', item.get('question_text', ''))
                # Adjust based on your answer format
                answers = item.get('answers', [])
                if 'answer_list' in item:
                    for a in item['answer_list']:
                        if isinstance(a, dict):
                            answers.extend(a.get('aliases', []))
                            if 'answer_text' in a:
                                answers.append(a['answer_text'])
                        else:
                            answers.append(a)
                answers_map[q] = [a for a in answers if a]
    
    # Process test examples
    all_questions = []
    all_documents = []
    all_gt_labels = []
    
    for item in test_data[:500]:  # Sample for speed
        question = item.get('question', item.get('input', ''))
        answers = answers_map.get(question, [])
        
        for ctx in item.get('ctxs', [])[:10]:  # Top 10 docs
            doc_text = ctx.get('text', '')
            gt_label = 1 if has_answer(answers, doc_text, str_tokenizer) else 0
            
            all_questions.append(question)
            all_documents.append(doc_text)
            all_gt_labels.append(gt_label)
    
    print(f"Ground truth positive rate: {sum(all_gt_labels)/len(all_gt_labels)*100:.1f}%")
    
    # Get predictions
    preds = verify_batch(model, tokenizer_model, all_questions, all_documents, device=device)
    pred_labels = [p[0] for p in preds]
    
    print(f"Model positive rate: {sum(pred_labels)/len(pred_labels)*100:.1f}%")
    
    # Confusion matrix
    tp = sum(1 for p, t in zip(pred_labels, all_gt_labels) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred_labels, all_gt_labels) if p == 1 and t == 0)
    tn = sum(1 for p, t in zip(pred_labels, all_gt_labels) if p == 0 and t == 0)
    fn = sum(1 for p, t in zip(pred_labels, all_gt_labels) if p == 0 and t == 1)
    
    print(f"\nConfusion Matrix (Test):")
    print(f"  TP={tp}, FP={fp}")
    print(f"  FN={fn}, TN={tn}")
    
    precision = tp/(tp+fp) if tp+fp > 0 else 0
    recall = tp/(tp+fn) if tp+fn > 0 else 0
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    
    # ========================================================================
    # TEST 3: Show disagreement examples
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 3: Examples where model disagrees with ground truth")
    print("="*70)
    
    print("\n--- FALSE NEGATIVES (GT=YES, Model=NO) ---")
    fn_count = 0
    for i, (q, d, gt, (pred, conf)) in enumerate(zip(all_questions, all_documents, all_gt_labels, preds)):
        if gt == 1 and pred == 0 and fn_count < 3:
            answers = answers_map.get(q, [])
            print(f"\nExample {fn_count+1}:")
            print(f"  Q: {q[:100]}")
            print(f"  Answers: {answers[:3]}")
            print(f"  Doc: {d[:200]}...")
            print(f"  Model confidence for NO: {conf:.3f}")
            fn_count += 1
    
    print("\n--- FALSE POSITIVES (GT=NO, Model=YES) ---")
    fp_count = 0
    for i, (q, d, gt, (pred, conf)) in enumerate(zip(all_questions, all_documents, all_gt_labels, preds)):
        if gt == 0 and pred == 1 and fp_count < 3:
            answers = answers_map.get(q, [])
            print(f"\nExample {fp_count+1}:")
            print(f"  Q: {q[:100]}")
            print(f"  Answers: {answers[:3]}")
            print(f"  Doc: {d[:200]}...")
            print(f"  Model confidence for YES: {conf:.3f}")
            fp_count += 1

if __name__ == "__main__":
    main()