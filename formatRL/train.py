from unsloth import FastLanguageModel
import torch
from datasets import load_dataset

#! === MODEL LOADING ===

max_seq_length = 4096 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower
modelToTrain = "unsloth/Qwen3-4B-Base"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = modelToTrain,
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)

#! === PROMPT FORMATTING ===

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
Answer the question with a number
Place the step within <answer> </answer> tag.

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
Number as the answer
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

def doProblem(instruction: str, question: str):
    #? utility function to do a problem
    return generateResponse(
        model = model,
        tokenizer = tokenizer,
        instruction = instruction,
        question = question
    )

#! === REWARD FUNCTIONS ===

def printCompletions(completions):
    print("\n=== DEBUG Completions ===")
    print(completions[0])

def matchFormat(completions, **kwargs):
    scores = []
    for response in completions:
        # print("===Completion===")
        # print(completion)
        try:
            # print("===Response1===")
            # print(response)
            score = 0
            # score += 0.5 if response.count(symbolsEnd) == 1 else -1.0
            # score += 0.5 if response.count(variablesStart) == 1 else -1.0
            score += 1.0 if response.count(variablesEnd) == 1 else -1.0
            score += 1.0 if response.count(formulaStart) == 1 else -1.0
            score += 1.0 if response.count(formulaEnd) == 1 else -1.0
            score += 1.0 if response.count(calculationStart) == 1 else -1.0
            score += 1.0 if response.count(calculationEnd) == 1 else -1.0
            score += 1.0 if response.count(answerStart) == 1 else -1.0
            score += 1.0 if response.count(answerEnd) == 1 else -1.0
            scores.append(score)
        except Exception as e:
            print(f"Error processing completion: {e}")
            scores.append(-5.0)
    return scores

def checkAnswer(prompts, completions, answer, **kwargs):
    extractedResponses = []
    for response in completions:
        try:
            # print("===Response2===")
            # print(response)
            answer_start = response.find(answerStart)
            answer_end = response.find(answerEnd)
            if answer_start != -1 and answer_end != -1:
                content = response[answer_start+len(answerStart):answer_end].strip()
                extractedResponses.append(content if content else None)
            else:
                extractedResponses.append(None)
        except Exception as e:
            print(f"Error processing completion in checkAnswer: {e}")
            extractedResponses.append(None)

    scores = []
    for guess, true_answer in zip(extractedResponses, answer):
        score = 0
        if guess is None:
            scores.append(-2.0)
            continue
        # Fully correct answer gets 5 points
        if guess == true_answer:
            score += 5.0
        # Partially correct answer gets 3.5 points
        elif guess.strip() == true_answer.strip():
            score += 3.5
        else:
            score -= 2.5  # Incorrect answer gets -2.5 points
        scores.append(score)
    return scores

#! === DATA LOADING ===

from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main", split="train")

def extractHashAnswer(text):
    if "####" not in text: return None
    return text.split("####")[1].strip()

def processDataset(example):
    example["answer"] = extractHashAnswer(example["answer"])
    example["prompt"] = promptTemplate.format(
        instruction,
        example["question"],
        ""
    )
    return example

dataset = dataset.map(processDataset)

#! === TRAINING ===

from vllm import SamplingParams
vllm_sampling_params = SamplingParams(
    min_p = 0.1,
    top_p = 1.0,
    top_k = -1,
    seed = 3407,
    stop = [tokenizer.eos_token],
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
    max_steps = 1400,
    save_strategy = "steps",
    save_steps = 200,
    report_to = "none", # Can use Weights & Biases
    output_dir = "/root/autodl-tmp/GrpoOut_0612_FA",

    # For optional training + evaluation
    # fp16_full_eval = True,
    # per_device_eval_batch_size = 4,
    # eval_accumulation_steps = 1,
    # eval_strategy = "steps",
    # eval_steps = 1,
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        matchFormat,
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