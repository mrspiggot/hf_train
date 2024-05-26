import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding, TextClassificationPipeline
import pandas as pd
import re

device = torch.device("mps")
# Load the dataset
dataset = load_dataset('imdb')
TRAIN_SIZE = 150 # Default Sample 1000, raise this with more powerful GPU
EVAL_SIZE = 75 # Default Sample 1000, raise this with more powerful GPU
SAMPLE_SIZE = 10

def clean_text(text):
    # Replace line breaks and backspaces with spaces
    text = text.replace('\n', ' ').replace('\r', ' ').replace('', ' ')
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove excessive spaces
    text = re.sub(' +', ' ', text)
    # Remove problematic characters
    text = re.sub(r'[<>/\\]', ' ', text)
    # Remove any remaining non-printable characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
    return text

# Split the dataset into train, validation, and test sets
train_val_dataset = dataset['train'].train_test_split(test_size=0.2)
train_dataset = train_val_dataset['train']
val_dataset = train_val_dataset['test']
test_dataset = dataset['test']

train_dataset = train_dataset.shuffle(seed=42).select(range(TRAIN_SIZE))  # Default Sample 1000, raise this with more powerful GPU
val_dataset = val_dataset.shuffle(seed=42).select(range(EVAL_SIZE))    # Default Sample 1000, raise this with more powerful GPU


# Load pre-trained sentiment analysis model and tokenizer
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset_clean = train_dataset.map(lambda x: {"text": clean_text(x["text"])})
eval_dataset_clean = val_dataset.map(lambda x: {"text": clean_text(x["text"])})
test_dataset_clean = test_dataset.map(lambda x: {"text": clean_text(x["text"])})

train_df = pd.DataFrame(train_dataset_clean)[["text", "label"]]
eval_df = pd.DataFrame(eval_dataset_clean)[["text", "label"]]
test_df = pd.DataFrame(test_dataset_clean)[["text", "label"]]

l = [train_df, eval_df, test_df]
for d in l:
    print(d.info())
    print(d.head())

# Save the DataFrames as Excel spreadsheets
train_df.to_excel("train_dataset.xlsx", index=False)
eval_df.to_excel("eval_dataset.xlsx", index=False)
test_df.to_excel("test_dataset.xlsx", index=False)

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
sample_indices = np.random.choice(len(test_dataset), size=SAMPLE_SIZE, replace=False).tolist()
sample_texts = [test_dataset[i]['text'] for i in sample_indices]
sample_labels = [test_dataset[i]['label'] for i in sample_indices]
label_map = {0: "Negative", 1: "Positive"}
reverse_label_map = {"NEGATIVE": 0, "POSITIVE": 1}

# Print the samples and their actual sentiments
print("\nSample Reviews and Actual Sentiments:")
for i, (text, label) in enumerate(zip(sample_texts, sample_labels)):
    print(f"Review {i+1}:")
    print(text)
    print(f"Actual Sentiment: {label_map[label]}")
    print()

# Run these samples through the pre-trained model
print("Predictions by Pre-trained Model:")
pretrained_predictions = []
for i, text in enumerate(sample_texts):
    prediction = pretrained_pipeline(text)
    predicted_label = reverse_label_map[prediction[0]['label']]
    pretrained_predictions.append(predicted_label)
    print(f"Review {i+1} Prediction: {label_map[predicted_label]} (Confidence: {prediction[0]['score']:.4f})")

# Calculate metrics for pre-trained model
pretrained_accuracy = metric.compute(predictions=pretrained_predictions, references=sample_labels)['accuracy']
pretrained_precision = precision_metric.compute(predictions=pretrained_predictions, references=sample_labels, average='binary')['precision']
pretrained_recall = recall_metric.compute(predictions=pretrained_predictions, references=sample_labels, average='binary')['recall']
pretrained_f1 = f1_metric.compute(predictions=pretrained_predictions, references=sample_labels, average='binary')['f1']

print(f"\nPre-trained model results: Accuracy = {pretrained_accuracy:.4f}, Precision = {pretrained_precision:.4f}, Recall = {pretrained_recall:.4f}, F1 = {pretrained_f1:.4f}")

# Evaluate pre-trained model on the entire test set
print("Evaluating pre-trained model on the entire test set...")

test_texts = test_dataset['text']
test_labels = test_dataset['label']

pretrained_predictions_full = [reverse_label_map[pretrained_pipeline(text)[0]['label']] for text in test_texts]

# Calculate metrics for pre-trained model on the entire test set
pretrained_accuracy_full = metric.compute(predictions=pretrained_predictions_full, references=test_labels)['accuracy']
pretrained_precision_full = precision_metric.compute(predictions=pretrained_predictions_full, references=test_labels, average='binary')['precision']
pretrained_recall_full = recall_metric.compute(predictions=pretrained_predictions_full, references=test_labels, average='binary')['recall']
pretrained_f1_full = f1_metric.compute(predictions=pretrained_predictions_full, references=test_labels, average='binary')['f1']

print(f"\nPre-trained model results on the entire test set: Accuracy = {pretrained_accuracy_full:.4f}, Precision = {pretrained_precision_full:.4f}, Recall = {pretrained_recall_full:.4f}, F1 = {pretrained_f1_full:.4f}")


# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
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
finetuned_predictions = [reverse_label_map[finetuned_pipeline(text)[0]['label']] for text in sample_texts]

# Calculate metrics for fine-tuned model
finetuned_accuracy = metric.compute(predictions=finetuned_predictions, references=sample_labels)['accuracy']
finetuned_precision = precision_metric.compute(predictions=finetuned_predictions, references=sample_labels, average='binary')['precision']
finetuned_recall = recall_metric.compute(predictions=finetuned_predictions, references=sample_labels, average='binary')['recall']
finetuned_f1 = f1_metric.compute(predictions=finetuned_predictions, references=sample_labels, average='binary')['f1']

print(f"\nFine-tuned model results: Accuracy = {finetuned_accuracy:.4f}, Precision = {finetuned_precision:.4f}, Recall = {finetuned_recall:.4f}, F1 = {finetuned_f1:.4f}")
