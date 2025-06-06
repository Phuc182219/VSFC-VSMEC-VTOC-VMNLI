from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
label2id = {v: k for k, v in id2label.items()}

model_name = "Qwen_VMNLI_results/checkpoint-116000"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, trust_remote_code=True, label2id=label2id, id2label=id2label)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer, trust_remote_code=True)

def predict(premise, hypothesis):
    prompt = f"Premise: {premise}\nHypothesis: {hypothesis}"
    return classifier(prompt)

# Ví dụ
premise = "Chính phủ đã ban hành chính sách mới về giáo dục."
hypothesis = "Giáo dục sẽ được cải cách theo chính sách mới."
print(predict(premise, hypothesis))


