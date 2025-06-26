import os
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from datasets import load_dataset

#! === Global configuration constants ===
modelPrompterName = "unsloth/Qwen3-0.6B-Base"
modelCandidateName = "unsloth/Qwen3-0.6B-Base"
max_seq_length = 2048
loraRank = 32

#! Training hyperparameters
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
LR_SCHEDULER_TYPE = "linear"
OPTIMIZER = "adamw_8bit"
NUM_GENERATIONS = 4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
MAX_PROMPT_LENGTH = 2048
MAX_COMPLETION_LENGTH = 2048
MAX_STEPS = 1400
SAVE_STRATEGY = "steps"
SAVE_STEPS = 200
OUTPUT_DIR = "./grpo_output"

#! === Model loading interfaces ===
def loadModels():
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
        gpu_memory_utilization=0.3,
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
        fast_inference=True,
        gpu_memory_utilization=0.3,
    )

    FastLanguageModel.for_inference(modelCandidate)
    return modelPrompter, tokenizerCandidate, modelCandidate, tokenizerCandidate
#! === Prompt generation interface ===

instructionForPrompter = f"""
You are given a question that belongs to a specific type, such as fill-in-the-blank or multiple-choice. Your goal is to generate a guiding prompt that will help another model reason appropriately about the question and respond in the correct format.

When generating your guiding prompt, ensure the following:
1. It should instruct the other model to carefully analyze the question, including any required calculations, logic, or background knowledge.
2. It must explicitly state how to format the final answer:
    - For fill-in-the-blank questions, the final answer should be a number, enclosed in <answer>...</answer>.
    - For multiple-choice questions, the final answer should be the option label (e.g., A, B, C, or D), enclosed in <answer>...</answer>.

The guiding prompt you write should not answer the question directly, but instead clearly direct the model on how to think and how to format its answer.
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

#! === Answer generation interface ===
def generateAnswer(model, tokenizer, prompt: str, maxNewTokens: int = 512) -> str:
    """
    Generate answer from model B given a prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=maxNewTokens, use_cache=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# === 4. Reward functions ===
def matchFormat(responses: list[str], **kwargs) -> list[float]:
    """
    Score based on presence of tags: <variables>, <formula>, <calculation>, <answer>.
    """
    scores = []
    tags = [
        "<variables>", "</variables>",
        "<formula>", "</formula>",
        "<calculation>", "</calculation>",
        "<answer>", "</answer>"
    ]
    for resp in responses:
        score = 0.0
        for tag in tags:
            score += 1.0 if resp.count(tag) == 1 else -1.0
        scores.append(score)
    return scores

def checkAnswer(prompts: list[str], responses: list[str], answers: list[str], **kwargs) -> list[float]:
    """
    Score based on correctness of the extracted answer.
    """
    scores = []
    for resp, trueAns in zip(responses, answers):
        start = resp.find("<answer>")
        end = resp.find("</answer>")
        if start == -1 or end == -1:
            scores.append(-2.0)
            continue
        guess = resp[start+len("<answer>"):end].strip()
        if guess == trueAns:
            scores.append(5.0)
        elif guess.strip() == trueAns.strip():
            scores.append(3.5)
        else:
            scores.append(-2.5)
    return scores

# === 5. Dataset loading and processing ===
def loadAndProcessDataset(split: str = "train"):
    """
    Load dataset and prepare prompts and answers.
    Returns processed dataset.
    """
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    def process(example):
        ansText = example["answer"]
        trueAns = ansText.split("####")[1].strip() if "####" in ansText else None
        prompt = generatePrompt(example["question"])
        return {"prompt": prompt, "answer": trueAns}
    return dataset.map(process)

# === 6. Build GRPO trainer ===
def buildTrainer(modelA, tokenizerA, trainDataset):
    """
    Configure GRPOConfig and initialize GRPOTrainer.
    """
    samplingParams = SamplingParams(
        min_p=0.1,
        top_p=1.0,
        top_k=-1,
        seed=3407,
        stop=[tokenizerA.eos_token],
        include_stop_str_in_output=True,
    )
    config = GRPOConfig(
        vllm_sampling_params=samplingParams,
        temperature=1.0,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        optim=OPTIMIZER,
        logging_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_generations=NUM_GENERATIONS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_steps=MAX_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEPS,
        report_to="none",
        output_dir=OUTPUT_DIR,
    )
    trainer = GRPOTrainer(
        model=modelA,
        processing_class=tokenizerA,
        reward_funcs=[matchFormat, checkAnswer],
        args=config,
        train_dataset=trainDataset,
    )
    return trainer

# === 7. Main entry ===
if __name__ == "__main__":
    modelPrompter, tokenizerPrompter, modelCandidte, tokenizerCandidate = loadModels()
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.map(processDataset)
    print(dataset[0])
    exit()