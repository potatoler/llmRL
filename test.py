from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from tqdm import tqdm
import instructions

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

def eval(model, tokenizer, dataset, instruction, batch_size=4, maxSamples=None):
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

        guess = extract_last_number(answer_text)
        if guess == gold:
            correct += 1
            print(f"{ix+1}/{len(dataset)} -> ac ({guess} / {gold})")
        else:
            print(f"{ix+1}/{len(dataset)} -> wa ({guess} / {gold})")
        total += 1

    print(f"Accuracy: {correct/total*100:.2f}% ({correct}/{total})")
    return correct, total

# def eval(model, tokenizer, dataset, instruction, batch_size=4):
#     correct = 0
#     total = 0
#     for ix, example in enumerate(dataset):
#         if ix >= 100:
#             break
#         question = example["question"]
#         gold = example["answer"]
#         pred = generateResponse(model, tokenizer, instruction, question)

#         answer_start = pred.find(answerStart)
#         answer_end = pred.find(answerEnd)
#         if answer_start != -1 and answer_end != -1:
#             guess = pred[answer_start+len(answerStart):answer_end].strip()
#         else:
#             guess = ""
#         if guess.strip() == str(gold).strip():
#             correct += 1
#             print(f"{ix+1}/{len(dataset)} -> ac ({guess.strip()} / {str(gold).strip()})")
#         else:
#             print(f"{ix+1}/{len(dataset)} -> wa ({guess.strip()} / {str(gold).strip()})")
#         total += 1
#     print(f"Accuracy: {correct/total*100:.2f}% ({correct}/{total})")
#     return correct, total

test_dataset = load_dataset("openai/gsm8k", "main", split="test")
test_dataset = test_dataset.map(processDataset)

# print("Loading base model...")
# base_model, base_tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "unsloth/Qwen3-0.6B-Base",
#     max_seq_length = max_seq_length,
#     load_in_4bit = False, # False for LoRA 16bit
#     fast_inference = True, # Enable vLLM fast inference
#     max_lora_rank = lora_rank,
#     gpu_memory_utilization = 0.7, # Reduce if out of memory
# )

print("Loading LoRA fine-tuned model...")
finetuned_model, finetuned_tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/root/autodl-tmp/GrpoOut_0611/checkpoint-1400",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7, # Reduce if out of memory
)

# print("===== Base =====")
# eval(base_model, base_tokenizer, test_dataset, instructions.taskOnly)

# print("===== Finetuned =====")
eval(finetuned_model, finetuned_tokenizer, test_dataset, instructions.minimal3)

example = test_dataset[0]["question"]
# print(generateResponse(finetuned_model, finetuned_tokenizer, instructions.minimal3, example))
# print(generateResponse(base_model, base_tokenizer, instructions.suggested, example))
# print(generateResponse(base_model, base_tokenizer, instructions.deepseekStyle, example))
# print(generateResponse(base_model, base_tokenizer, instructions.detailedZeroShot, example))
# print(generateResponse(base_model, base_tokenizer, instructions.detailedOneShot, example))