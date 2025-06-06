from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer
from datasets import load_from_disk
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

dataset = load_from_disk("vsmec_preprocessed")
model_name = "Qwen2.5-0.5B_v2/checkpoint-116000"
label2id = {
    "enjoyment": 0,
    "disgust": 1,
    "sadness": 2,
    "anger": 3,
    "fear": 4,
    "surprise": 5,
    "other": 6
}
id2label = {v: k for k, v in label2id.items()}

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7, trust_remote_code=True, label2id=label2id, id2label=id2label)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

training_args = TrainingArguments(
    output_dir="Qwen_VSMEC_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    lr_scheduler_type="cosine",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    bf16=True,
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
