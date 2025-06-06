from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

topics = ['Automobile', 'Business', 'Digital', 'Education', 'Entertainment',
          'Health', 'Law', 'Life', 'News', 'Perspective',
          'Relax', 'Science', 'Sports', 'Travel', 'World']
id2label = {i: topic for i, topic in enumerate(topics)}
label2id = {v: k for k, v in id2label.items()}

model_name = "Qwen_VTOC_results/checkpoint-116000"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=15, trust_remote_code=True, label2id=label2id, id2label=id2label)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer, trust_remote_code=True)

def predict(text):
    return classifier(text)

# Ví dụ
sentence = "Chính phủ vừa ban hành chính sách mới về giáo dục đại học."
print(predict(sentence))

