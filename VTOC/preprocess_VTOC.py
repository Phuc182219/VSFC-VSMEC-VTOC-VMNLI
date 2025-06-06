from datasets import load_dataset
from transformers import AutoTokenizer
import os

task_name = "vtoc"
data_dir = "D:/Data_luat_VN/SCRIPT NOW/VieGLUE/data"
dataset = load_dataset("json", data_dir=os.path.join(data_dir, task_name))

model_name = "Qwen2.5-0.5B_v2/checkpoint-116000"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Bạn có thể tùy chỉnh lại số label theo đúng dataset nếu cần
topics = ['Automobile', 'Business', 'Digital', 'Education', 'Entertainment',
          'Health', 'Law', 'Life', 'News', 'Perspective',
          'Relax', 'Science', 'Sports', 'Travel', 'World']
label2id = {topic: i for i, topic in enumerate(topics)}

def preprocess_function(examples):
    tokenized = tokenizer(examples["sentence"], truncation=True)
    return {
        "input_ids": tokenized["input_ids"],
        "label": [label2id[label] for label in examples["label"]]
    }

encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset.save_to_disk("vtoc_preprocessed")
