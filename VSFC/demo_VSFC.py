from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

id2label = {0: "positive", 1: "neutral", 2: "negative"}
label2id = {v: k for k, v in id2label.items()}

model_name = "Qwen_VSFC_results/checkpoint-116000"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, trust_remote_code=True, label2id=label2id, id2label=id2label)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer, trust_remote_code=True)

def predict(text):
    return classifier(text)

# Ví dụ
sentence = "Bài giảng hôm nay rất dễ hiểu và thú vị."
print(predict(sentence))
