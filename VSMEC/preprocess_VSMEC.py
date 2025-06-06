from datasets import load_dataset
from transformers import AutoTokenizer

if __name__ =='__main__':
    dataset = load_dataset("json",data_dir="D:/Data_luat_VN/SCRIPT NOW/VieGLUE/data/vsmec")
    dataset.pop("test", None)
    model_name = "Qwen2.5-0.5B_v2/checkpoint-116000"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    label_map = {
        "enjoyment": 0,
        "disgust": 1,
        "sadness": 2,
        "anger": 3,
        "fear": 4,
        "surprise": 5,
        "other": 6
    }

    def preprocess_function(examples):
        tokenized = tokenizer(examples["sentence"], truncation=True)
        return {
            "input_ids": tokenized["input_ids"],
            "label": [label_map[label] for label in examples["label"]],
        }

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    encoded_dataset.save_to_disk("vsmec_preprocessed")

