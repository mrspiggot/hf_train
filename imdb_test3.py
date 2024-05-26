import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding, TextClassificationPipeline

# Load the dataset
dataset = load_dataset('imdb')

# Split the dataset into train, validation, and test sets
train_val_dataset = dataset['train'].train_test_split(test_size=0.2)
train_dataset = train_val_dataset['train']
val_dataset = train_val_dataset['test']
test_dataset = dataset['test']

# Load pre-trained model and tokenizer
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

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

# Evaluate pre-trained model on test set using pipeline
print("Evaluating pre-trained model...")

pretrained_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, truncation=True, padding=True)

# Take ten samples from the test dataset
sample_indices = np.random.choice(len(test_dataset), size=10, replace=False).tolist()
sample_texts = [test_dataset[i]['text'] for i in sample_indices]
sample_labels = [test_dataset[i]['label'] for i in sample_indices]
label_map = {0: "Negative", 1: "Positive"}

# Print the samples and their actual sentiments
print("\nSample Reviews and Actual Sentiments:")
for i, (text, label) in enumerate(zip(sample_texts, sample_labels)):
    print(f"Review {i+1}:")
    print(text)
    print(f"Actual Sentiment: {label_map[label]}")
    print()

# Run these samples through the pre-trained model
print("Predictions by Pre-trained Model:")
for i, text in enumerate(sample_texts):
    prediction = pretrained_pipeline(text)
    predicted_label = prediction[0]['label']
    predicted_label = int(predicted_label.split('_')[-1])  # Convert 'LABEL_0'/'LABEL_1' to 0/1
    print(f"Review {i+1} Prediction: {label_map[predicted_label]} (Confidence: {prediction[0]['score']:.4f})")

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

# Evaluate fine-tuned model on test set using pipeline
print("Evaluating fine-tuned model...")

finetuned_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, truncation=True, padding=True)

# Run inference on test dataset
finetuned_predictions = [finetuned_pipeline(text)[0]['label'] == 'LABEL_1' for text in sample_texts]

# Calculate metrics for fine-tuned model
finetuned_accuracy = metric.compute(predictions=finetuned_predictions, references=sample_labels)['accuracy']
finetuned_precision = precision_metric.compute(predictions=finetuned_predictions, references=sample_labels, average='binary')['precision']
finetuned_recall = recall_metric.compute(predictions=finetuned_predictions, references=sample_labels, average='binary')['recall']
finetuned_f1 = f1_metric.compute(predictions=finetuned_predictions, references=sample_labels, average='binary')['f1']

print(f"Fine-tuned model results: Accuracy = {finetuned_accuracy}, Precision = {finetuned_precision}, Recall = {finetuned_recall}, F1 = {finetuned_f1}")
