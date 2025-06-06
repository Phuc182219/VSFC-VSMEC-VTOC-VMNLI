from datasets import load_dataset
from transformers import AutoTokenizer
import os

task_name = "mnli"  # VMNLI tương ứng với tên Hugging Face là "mnli"
data_dir = "D:/Data_luat_VN/SCRIPT NOW/VieGLUE/data"
dataset = load_dataset("json", data_dir=os.path.join(data_dir, task_name))

model_name = "Qwen2.5-0.5B_v2/checkpoint-116000"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

label2id = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}

def format_prompt(premise, hypothesis):
    return f"Premise: {premise}\nHypothesis: {hypothesis}"

def preprocess_function(examples):
    texts = [format_prompt(p, h) for p, h in zip(examples["premise"], examples["hypothesis"])]
    tokenized = tokenizer(texts, truncation=True)
    return {
        "input_ids": tokenized["input_ids"],
        "label": [label2id[label] for label in examples["label"]]
    }

encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset.save_to_disk("vmnli_preprocessed")
