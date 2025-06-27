import os
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from datasets import load_dataset
import re

#! === Global configuration constants ===
modelPrompterName = "unsloth/Qwen3-0.6B-Base"
modelCandidateName = "unsloth/Qwen3-0.6B-Base"
max_seq_length = 4096
loraRank = 32

#! === Model loading interfaces ===
"""
load prompter and candidate model \\
return `modelPrompter, tokenizerPrompter, modelCandidate, tokenitzerCandidate`
"""

modelPrompter, tokenizerPrompter = FastLanguageModel.from_pretrained(
    model_name=modelPrompterName,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=loraRank,
    gpu_memory_utilization=0.7,
)

modelPrompter = FastLanguageModel.get_peft_model(
    modelPrompter,
    r=loraRank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=loraRank * 2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

os.system("nvidia-smi --query-gpu=memory.free,memory.used --format=csv")

modelCandidate, tokenizerCandidate = FastLanguageModel.from_pretrained(
    model_name=modelCandidateName,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    # fast_inference=True,
    gpu_memory_utilization=0.2,
)

FastLanguageModel.for_inference(modelCandidate)

#! === Prompt generation interface ===

instructionForPrompter = f"""
You are given a question that belongs to a specific type, such as fill-in-the-blank or multiple-choice. Your task is to generate a thinking guide to teach another model "how to reason appropriately about the question and respond in the correct format".

When generating your guiding prompt, ensure the following:
1. It should instruct the other model to carefully analyze the question, including any required calculations, logic, or background knowledge.
2. It must explicitly state how to format the final answer:
    - For fill-in-the-blank questions, the final answer should be a number, enclosed in <answer>...</answer>.
    - For multiple-choice questions, the final answer should be the option label (e.g., A, B, C, or D), enclosed in <answer>...</answer>.

The guiding you write must NOT answer the question directly, but should instead clearly direct the model on how to think and how to format its answer.
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

#! === Answer generation interface ===

def generateResponse(model, tokenizer, instruction: str, question: str, answerLimit: int = 2048) -> str:
    #? utility function to generate response from the model
    FastLanguageModel.for_inference(model)
    inputs = tokenizer(
        promptTemplate.format(
            instruction,
            question,
            "",
        ),
        return_tensors = "pt"
    ).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=answerLimit, use_cache=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:")[1].strip()
    return response

#!  === Reward functions ===

answerStartTag = "<answer>"
answerEndTag = "</answer>"

def extractInputQuestion(prompt: str) -> str:
    inputTag = "### Input:"
    responseTag = "### Response:"
    promptInputStart = prompt.find(inputTag)
    promptResponseStart = prompt.find(responseTag)
    if promptInputStart == -1 or promptResponseStart == -1 or promptResponseStart <= promptInputStart:
        return ""
    inputQuestion = prompt[promptInputStart + len(inputTag):promptResponseStart]
    return inputQuestion.strip()



def hasContextNumber(text: str) -> bool:
    """
    判断字符串中是否含有非项目标号的数字
    """
    all_numbers = set(re.findall(r'\d+', text))
    bullet_numbers = set()
    bullet_pattern = re.compile(r'^\s*(\d+)[\.\)、,，]\s', re.MULTILINE)
    for match in bullet_pattern.finditer(text):
        bullet_numbers.add(match.group(1))
    for num in all_numbers:
        if num not in bullet_numbers:
            return True
    return False



def checkAnswer(prompts, completions, answer, **kwargs):
    scores = []

    for prompt, response, trueAnswer in zip(prompts, completions, answer):
        # print("=== Prompter Response ===")
        # print(response)

        question = extractInputQuestion(prompt)
        # print("=== Candidate's Question ===")
        # print(question)

        candidateResponse = generateResponse(modelCandidate, tokenizerCandidate, response, question)
        candidateAnswerStart = candidateResponse.find(answerStartTag)
        candidateAnswerEnd = candidateResponse.find(answerEndTag)
        candidateAnswer = None

        if candidateAnswerStart != -1 and candidateAnswerEnd != -1:
            content = response[candidateAnswerStart+len(answerStartTag):candidateAnswerEnd].strip()
            candidateAnswer = content if content else None
        
        # print("=== Candidate's Answer ===")
        # print(candidateAnswer)

        # print("=== True Answer ===")
        # print(answer)

        score = 0
        if response.find(answerStartTag) != -1:
            score += 1.0
        if response.find(answerEndTag) != -1:
            score += 1.0
        if hasContextNumber(response) == True:
            score -= 2.0
        if candidateAnswer is None:
            score -= 2.0
        elif candidateAnswer == trueAnswer:
            score += 4.0
        elif candidateAnswer.strip() == trueAnswer.strip():
            score += 2.0
        else:
            score -= 2.0
        scores.append(score)
    return scores

#! === Dataset loading and processing ===

def extractHashAnswer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()

def processDataset(example):
    example["answer"] = extractHashAnswer(example["answer"])
    example["prompt"] = promptTemplate.format(
        instructionForPrompter,
        example["question"],
        ""
    )
    return example

dataset = load_dataset("openai/gsm8k", "main", split="train")
dataset = dataset.map(processDataset)
# print(dataset[0]['question'], dataset[0]['prompt'], dataset[0]['answer'])

#! === Build GRPO trainer ===

from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    min_p = 0.1,
    top_p = 1.0,
    top_k = -1,
    seed = 3407,
    stop = [tokenizerPrompter.eos_token],
    include_stop_str_in_output = True,
)

max_prompt_length = 2048
max_completion_length = 2048

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    vllm_sampling_params = vllm_sampling_params,
    temperature = 1.0,
    learning_rate = 5e-6,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 2000,
    save_strategy = "steps",
    save_steps = 200,
    report_to = "none", # Can use Weights & Biases
    output_dir = "/root/autodl-tmp/checkpointPrompter_0627",

    # For optional training + evaluation
    # fp16_full_eval = True,
    # per_device_eval_batch_size = 4,
    # eval_accumulation_steps = 1,
    # eval_strategy = "steps",
    # eval_steps = 1,
)

trainer = GRPOTrainer(
    model = modelPrompter,
    processing_class = tokenizerPrompter,
    reward_funcs = [
        checkAnswer
    ],
    args = training_args,
    train_dataset = dataset,

    # For optional training + evaluation
    # train_dataset = new_dataset["train"],
    # eval_dataset = new_dataset["test"],
)

if __name__ == "__main__":
    trainer.train()