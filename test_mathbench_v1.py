#!/usr/bin/env python3

import os
import json
import re
from datasets import Dataset
from test import (
    FastLanguageModel,
    generateResponse,
    instructions,
    max_seq_length,
    lora_rank,
    answerStart,
    answerEnd,
    extract_last_number,
)

def extract_guess(text):
    """
    Extract answer guess from model output:
    - First try to find a single letter A-D.
    - Otherwise fallback to numeric extraction.
    """
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1)
    return text.strip()

def load_level(level_dir):
    """
    Load all JSONL files in the given level directory.
    Converts single-choice questions to cloze format by mapping the
    "answer" letter/index to the corresponding text in "options".
    """
    data = []
    for fname in os.listdir(level_dir):
        if not fname.endswith('.jsonl'):
            continue
        path = os.path.join(level_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if 'single_choice' in fname:
                    opts = item.get('options', [])
                    # Append options to question text
                    labels = ["A", "B", "C", "D"]
                    q = item.get('question', "")
                    option_strs = []
                    for i, opt in enumerate(opts):
                        label = labels[i] if i < len(labels) else str(i)
                        option_strs.append(f"{label}. {opt}")
                    # Update question to include options
                    item['question'] = q + " " + " ".join(option_strs)
                    # Keep the original answer letter and remove options list
                    item.pop('options', None)
                elif 'cloze' in fname:
                    # Already cloze format: "question" and text "answer"
                    pass
                else:
                    continue
                data.append({'question': item['question'], 'answer': item['answer']})
    return Dataset.from_list(data)

def test_process(model, tokenizer, dataset, instruction, max_samples=5):
    """
    Evaluate a Dataset with the provided model, tokenizer, and instruction.
    Returns a tuple (correct, total).
    """
    correct = 0
    total = 0
    for ix, example in enumerate(dataset):
        if ix >= max_samples:
            break
        question = example["question"]
        gold = str(example["answer"]).strip()
        # print(question, gold)
        pred = generateResponse(model, tokenizer, instruction, question)
        start = pred.find(answerStart)
        end = pred.find(answerEnd)
        if start != -1 and end != -1:
            text = pred[start + len(answerStart):end].strip()
        else:
            text = pred.strip()
        guess = extract_guess(text)
        # guess = text
        if guess == gold:
            correct += 1
            print(f"{ix+1}/{len(dataset)} -> ac ({guess} / {gold})")
        else:
            print(f"{ix+1}/{len(dataset)} -> wa ({guess} / {gold})")
        total += 1
    return correct, total

def main():
    root = 'mathbench_v1'
    print("Loading LoRA fine-tuned model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/root/autodl-fs/Qwen3-4B-FA-0612",
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.7,
    )
    total_correct = 0
    total_count = 0
    for level in sorted(os.listdir(root)):
        level_dir = os.path.join(root, level)
        if not os.path.isdir(level_dir):
            continue
        dataset = load_level(level_dir)
        if len(dataset) == 0:
            continue
        correct, count = test_process(
            model, tokenizer, dataset, instructions.minimal3
        )
        acc = correct / count * 100 if count > 0 else 0.0
        print(f"Level: {level} -> Accuracy: {acc:.2f}% ({correct}/{count})")
        total_correct += correct
        total_count += count
    overall_acc = total_correct / total_count * 100 if total_count > 0 else 0.0
    print(f"Overall Accuracy: {overall_acc:.2f}% ({total_correct}/{total_count})")

if __name__ == '__main__':
    main()