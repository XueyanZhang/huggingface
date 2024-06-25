# VeRA: Vector-based Random Matrix Adaptation
# https://huggingface.co/docs/peft/main/en/package_reference/vera

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer
from datasets import tokenized_datasets, load_dataset
from peft import VeraConfig, get_peft_model
import evaluate
import numpy as np


# TRAIN
base_model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
config = VeraConfig(r=128)
model = get_peft_model(base_model, config)

print(model.print_trainable_parameters())

dataset = load_dataset("yelp_review_full")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

training_args = TrainingArguments(
    output_dir="vera_trainer",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained("output_dir")
