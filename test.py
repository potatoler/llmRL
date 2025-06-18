from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from tqdm import tqdm
import instructions
from datasets import Dataset
import os
import json
import re

max_seq_length = 4096
lora_rank = 32

def extractHashAnswer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()

def processDataset(example):
    example["answer"] = extractHashAnswer(example["answer"])
    return example

# symbolsStart = "<symbols>"
# symbolsEnd = "</symbols>"
variablesStart = "<variables>"
variablesEnd = "</variables>"
formulaStart = "<formula>"
formulaEnd = "</formula>"
calculationStart = "<calculation>"
calculationEnd = "</calculation>"
answerStart = "<answer>"
answerEnd = "</answer>"

instruction = \
f"""You are given a math or physics problem. Please solve the problem in a structured, step-by-step manner as follows:

Step 1: Variable Extraction
Carefully read the problem and extract all relevant variables given in the question.
Clearly list these variables with their symbols (use letters like a, b, c, \alpha, \beta, \gamma, etc.) and their meanings or values.
Place this list between {variablesStart} and {variablesEnd} tags.

Step 2: Formula Derivation
Based on the extracted variables, identify and derive the appropriate mathematical or physical formula(s) needed to solve the problem.
Explain the reasoning behind choosing or deriving the formula.
Use LaTeX for all formulas and mathematical expressions.
Place the formula derivation between {formulaStart} and {formulaEnd} tags.

Step 3: Calculation Process
Using the derived formula(s) and extracted variables, perform the detailed calculation step-by-step.
Show all intermediate steps and provide explanations if necessary.
Use LaTeX for all calculations.
Place this calculation process between {calculationStart} and {calculationEnd} tags.

Final:
Provide the final numerical answer clearly.
Place the answer between {answerStart} and {answerEnd} tags.

Then your response should be in the following format:

<variables>
Extracted variables and their definitions
</variables>

<formula>
Derived formula(s) with explanation
</formula>

<calculation>
Step-by-step calculation
</calculation>

<answer>
Final numerical answer
</answer>
"""

promptTemplate ="""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}
"""

def generateResponse(model, tokenizer, instruction, question, answerLimit=2048):
    FastLanguageModel.for_inference(model)
    inputs = tokenizer(
        promptTemplate.format(instruction, question, ""),
        return_tensors = "pt"
    ).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=answerLimit, use_cache=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:")[1].strip()
    return response

def checkAnswer(guess, true_answer):
    if guess is None: return 0
    if guess == true_answer: return 1
    elif guess.strip() == true_answer.strip(): return 0.5
    else: return 0

import re

def extract_last_number(text):
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else ""

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

def eval(model, tokenizer, dataset, instruction, maxSamples=None):
    correct = 0
    total = 0
    if maxSamples is None:
        maxSamples = len(dataset)
    for ix, example in enumerate(dataset):
        if ix >= maxSamples:
            break
        question = example["question"]
        gold = str(example["answer"]).strip()
        pred = generateResponse(model, tokenizer, instruction, question)

        answer_start = pred.find(answerStart)
        answer_end = pred.find(answerEnd)
        if answer_start != -1 and answer_end != -1:
            answer_text = pred[answer_start+len(answerStart):answer_end].strip()
        else:
            answer_text = pred.strip()

        # guess = extract_last_number(answer_text)
        guess = extract_guess(answer_text)
        if guess == gold:
            correct += 1
            print(f"{ix+1}/{len(dataset)} -> ac ({guess} / {gold})", flush=True)
        else:
            print(f"{ix+1}/{len(dataset)} -> wa ({guess} / {gold})", flush=True)
        total += 1

    print(f"Accuracy: {correct/total*100:.2f}% ({correct}/{total})")
    return correct, total

def testMathbench(model, tokenizer, datasetRoot, instruction, maxSamples=None):
    root = datasetRoot
    total_correct = 0
    total_count = 0
    for level in sorted(os.listdir(root)):
        level_dir = os.path.join(root, level)
        if not os.path.isdir(level_dir):
            continue
        dataset = load_level(level_dir)
        if len(dataset) == 0:
            continue
        correct, count = eval(
            model, tokenizer, dataset, instruction, maxSamples
        )
        acc = correct / count * 100 if count > 0 else 0.0
        print(f"Level: {level} -> Accuracy: {acc:.2f}% ({correct}/{count})")
        total_correct += correct
        total_count += count
    overall_acc = total_correct / total_count * 100 if total_count > 0 else 0.0
    print(f"Overall Accuracy: {overall_acc:.2f}% ({total_correct}/{total_count})")

# test_dataset = load_dataset("openai/gsm8k", "main", split="test")
# test_dataset = test_dataset.map(processDataset)

print("===== Base =====")
print("Loading base model...")
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-Base",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.8, # Reduce if out of memory
)
# eval(base_model, base_tokenizer, test_dataset, instructions.taskOnly)
testMathbench(base_model, base_tokenizer, "mathbench_v1", instructions.taskOnly)

# print("===== Finetuned =====")
# print("Loading LoRA fine-tuned model...")
# finetuned_model, finetuned_tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "/root/autodl-fs/Qwen3-4B-FA-0612",
#     max_seq_length = max_seq_length,
#     load_in_4bit = False, # False for LoRA 16bit
#     fast_inference = True, # Enable vLLM fast inference
#     max_lora_rank = lora_rank,
#     gpu_memory_utilization = 0.8, # Reduce if out of memory
# )
# eval(finetuned_model, finetuned_tokenizer, test_dataset, instructions.taskOnly)
# testMathbench(finetuned_model, finetuned_tokenizer, "mathbench_v1", instructions.minimal3)