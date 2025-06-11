# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen3-1.7B-Base")
model = AutoModelForCausalLM.from_pretrained("unsloth/Qwen3-1.7B-Base")