import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_metric
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding

device = torch.device("mps")
tensor = torch.randn(size=(3, 4, 5), device=device)
print(tensor)
print(tensor.device)

# Load the dataset
dataset = load_dataset('imdb')

# Split the dataset into train, validation, and test sets
train_val_dataset = dataset['train'].train_test_split(test_size=0.2)
train_dataset = train_val_dataset['train']
val_dataset = train_val_dataset['test']
test_dataset = dataset['test']

# Load pre-trained model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example['text'], truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metric
metric = load_metric('accuracy')
precision_metric = load_metric('precision')
recall_metric = load_metric('recall')
f1_metric = load_metric('f1')

# Compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)['accuracy']
    precision = precision_metric.compute(predictions=predictions, references=labels, average='binary')['precision']
    recall = recall_metric.compute(predictions=predictions, references=labels, average='binary')['recall']
    f1 = f1_metric.compute(predictions=predictions, references=labels, average='binary')['f1']
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# Evaluate pre-trained model on test set
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
print("Evaluating pre-trained model...")
pretrained_results = trainer.evaluate(eval_dataset=test_dataset)
print(pretrained_results)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Fine-tune the model
print("Fine-tuning the model...")
trainer.train()

# Evaluate fine-tuned model on test set
print("Evaluating fine-tuned model...")
finetuned_results = trainer.evaluate(eval_dataset=test_dataset)
print(finetuned_results)