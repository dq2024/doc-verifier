import json
import random
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from tqdm import tqdm
import regex
import unicodedata


def _normalize(text):
    return unicodedata.normalize('NFD', text)


class SimpleTokenizer(object):
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


def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


def get_all_answer_aliases(gt_item: Dict) -> List[str]:
    all_aliases = []
    
    if 'answer_list' in gt_item:
        for answer_obj in gt_item['answer_list']:
            if isinstance(answer_obj, dict):
                if 'aliases' in answer_obj:
                    all_aliases.extend(answer_obj['aliases'])
                if 'answer_text' in answer_obj:
                    all_aliases.append(answer_obj['answer_text'])
            elif isinstance(answer_obj, str):
                all_aliases.append(answer_obj)
    elif 'answers' in gt_item:
        all_aliases = gt_item['answers']
    
    return [a for a in all_aliases if a and a.strip()]


def read_jsonl(file_path: str) -> List[Dict]:
    """Read a JSONL or JSON file."""
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Parse as JSONL
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return data


def write_jsonl(data: List[Dict], file_path: str):
    """Write list of dictionaries to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')


def merge_and_label_documents(
    retriever_files: Dict[str, str],
    ground_truth_file: str,
    output_file: str,
    sampled_indices: List[int],
    n_docs_per_retriever: int = 100
) -> tuple:

    
    tokenizer = SimpleTokenizer()
    
    # Load ground truth
    print(f"\nLoading ground truth: {ground_truth_file}")
    ground_truth = read_jsonl(ground_truth_file)
    print(f"  Loaded {len(ground_truth)} ground truth entries")
    
    # Load all retriever data
    retriever_data = {}
    for name, path in retriever_files.items():
        print(f"Loading {name}: {path}")
        retriever_data[name] = read_jsonl(path)
        print(f"  Loaded {len(retriever_data[name])} queries")
    
    print(f"\nProcessing {len(sampled_indices)} sampled queries...")
    
    # Statistics
    stats = {
        'n_queries_sampled': len(sampled_indices),
        'n_retrievers': len(retriever_files),
        'n_docs_per_retriever': n_docs_per_retriever,
        'total_docs_before_dedup': 0,
        'total_docs_after_dedup': 0,
        'docs_per_retriever': defaultdict(int),
        'total_positive': 0,
        'total_negative': 0,
        'positive_per_retriever': defaultdict(int),
        'negative_per_retriever': defaultdict(int),
        'queries_with_positives': 0,
    }
    
    merged_data = []
    positives_per_query = []
    
    for idx in tqdm(sampled_indices, desc="Merging and labeling"):
        gt_item = ground_truth[idx]
        answers = get_all_answer_aliases(gt_item)
        question = gt_item.get('question_text', gt_item.get('question', ''))
        
        seen_doc_ids = set()
        query_positive_count = 0
        
        for retriever_name, retriever_items in retriever_data.items():
            if idx >= len(retriever_items):
                continue
            
            item = retriever_items[idx]
            docs = item.get('ctxs', [])[:n_docs_per_retriever]
            
            for rank, doc in enumerate(docs):
                doc_id = doc.get('id', '')
                doc_text = doc.get('text', '')
                
                stats['total_docs_before_dedup'] += 1
                stats['docs_per_retriever'][retriever_name] += 1
                
                if doc_id in seen_doc_ids:
                    continue
                
                seen_doc_ids.add(doc_id)
                
                is_positive = has_answer(answers, doc_text, tokenizer) if answers else False
                
                labeled_doc = {
                    'query_id': idx,
                    'query': question,
                    'document_id': doc_id,
                    'document_title': doc.get('title', ''),
                    'document_text': doc_text,
                    'label': 1 if is_positive else 0,
                    'source_retriever': retriever_name,
                    'retrieval_score': float(doc.get('score', 0)),
                    'retrieval_rank': rank + 1
                }
                
                merged_data.append(labeled_doc)
                stats['total_docs_after_dedup'] += 1
                
                if is_positive:
                    stats['total_positive'] += 1
                    stats['positive_per_retriever'][retriever_name] += 1
                    query_positive_count += 1
                else:
                    stats['total_negative'] += 1
                    stats['negative_per_retriever'][retriever_name] += 1
        
        positives_per_query.append(query_positive_count)
        if query_positive_count > 0:
            stats['queries_with_positives'] += 1
    
    stats['avg_positives_per_query'] = sum(positives_per_query) / len(positives_per_query)
    
    print(f"\nSaving merged data to: {output_file}")
    write_jsonl(merged_data, output_file)
    
    return merged_data, stats


def print_statistics(stats: Dict[str, Any]):
    """Print statistics in format for mentor."""
    
    print("\n" + "=" * 70)
    print("DATA STATISTICS")
    print("=" * 70)
    
    print(f"\n# Questions: {stats['n_queries_sampled']:,}")
    print(f"  - {stats['n_retrievers']} retrievers, {stats['n_docs_per_retriever']} documents each")
    
    print(f"\n# Total Docs (before dedup): {stats['total_docs_before_dedup']:,}")
    print(f"# Total Docs (after dedup): {stats['total_docs_after_dedup']:,}")
    
    dedup_removed = stats['total_docs_before_dedup'] - stats['total_docs_after_dedup']
    dedup_rate = dedup_removed / stats['total_docs_before_dedup'] * 100
    print(f"# Duplicates removed: {dedup_removed:,} ({dedup_rate:.1f}%)")
    
    total = stats['total_positive'] + stats['total_negative']
    pct_positive = stats['total_positive'] / total * 100
    
    print(f"\n# Exs: {total:,}")
    print(f"# Positive: {stats['total_positive']:,}")
    print(f"# Negative: {stats['total_negative']:,}")
    print(f"% Positive Label: {pct_positive:.2f}%")
    
    print(f"\nQueries with ≥1 positive: {stats['queries_with_positives']}/{stats['n_queries_sampled']}")
    print(f"Avg positives per query: {stats['avg_positives_per_query']:.2f}")
    
    print(f"\nPer Retriever:")
    for retriever in stats['docs_per_retriever'].keys():
        pos = stats['positive_per_retriever'][retriever]
        neg = stats['negative_per_retriever'][retriever]
        total_r = pos + neg
        pct = pos / total_r * 100 if total_r > 0 else 0
        print(f"  {retriever}: {pos:,} pos / {total_r:,} total ({pct:.1f}%)")


def main():
    
    retriever_files = {
        'contriever': '/scratch/dq2024/diverse_retriever/data/contriever_training_file_retrieved.json',
        'qwen3': '/scratch/dq2024/diverse_retriever/data/qwen3_training_file_retrieved.json',
        'infly': '/scratch/dq2024/diverse_retriever/data/infly_training_file_retrieved.json',
    }
    
    ground_truth_file = '/scratch/dq2024/diverse_retriever/train_data.jsonl'
    
    output_dir = Path('/scratch/dq2024/doc-verifier/verifier_training_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    merged_file = output_dir / 'merged_3_retrievers_1k_queries.jsonl'
    
    # Parameters
    N_QUERIES_SAMPLE = 1000
    N_DOCS_PER_RETRIEVER = 100
    
    print("=" * 70)
    print("Sampling queries")
    print("=" * 70)
    
    gt_data = read_jsonl(ground_truth_file)
    total_queries = len(gt_data)
    print(f"Total queries: {total_queries}")
    
    sampled_indices = sorted(random.sample(range(total_queries), N_QUERIES_SAMPLE))
    print(f"Sampled {len(sampled_indices)} queries")
    
    # Save indices
    indices_file = output_dir / 'sampled_query_indices.json'
    with open(indices_file, 'w') as f:
        json.dump(sampled_indices, f)
    
    print("\n" + "=" * 70)
    print("Merging and labeling documents")
    print("=" * 70)
    
    merged_data, stats = merge_and_label_documents(
        retriever_files=retriever_files,
        ground_truth_file=ground_truth_file,
        output_file=str(merged_file),
        sampled_indices=sampled_indices,
        n_docs_per_retriever=N_DOCS_PER_RETRIEVER
    )
    
    # Print statistics
    print_statistics(stats)
    
    # Save statistics
    stats_file = output_dir / 'data_statistics.json'
    stats_serializable = {k: dict(v) if isinstance(v, defaultdict) else v for k, v in stats.items()}
    with open(stats_file, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    
    print(f"\nOutput files:")
    print(f"  {merged_file}")
    print(f"  {stats_file}")
    print(f"  {indices_file}")


if __name__ == "__main__":
    main()
