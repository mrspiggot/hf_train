import evaluate  # Import the evaluate library
from transformers import pipeline, Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import numpy as np


# Load pre-trained model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
pre_trained_pipeline = pipeline("sentiment-analysis", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the Yelp Full Review dataset
dataset = load_dataset("yelp_review_full")

# Evaluate pre-trained model
accuracy_metric = evaluate.load("accuracy")

def compute_metrics_old(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

def compute_metrics(eval_pred):
    if isinstance(eval_pred, tuple):
        predictions, labels = eval_pred
    else:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

def extract_star_rating(result):
    if not result:
        return -1
    try:
        label = result["label"]
        if isinstance(label, str):
            # Handle cases where the label is a string
            label_parts = label.split(' ')
            if label_parts:
                return int(label_parts[0])
            else:
                return -1
        else:
            # Handle cases where the label is an integer
            return int(label)
    except (KeyError, ValueError):
        return -1

def extract_star_rating_old(result):
    if not result:
        return -1
    try:  # Try to extract star rating directly
        return int(result[0]["label"])
    except ValueError:
        # Handle cases where the label is not a single integer
        label_parts = result[0]["label"].split(' ')
        if label_parts:
            return int(label_parts[0])
        else:
            return -1

pre_trained_results = pre_trained_pipeline(dataset["test"]["text"][:1000], truncation=True)
pre_trained_labels = [extract_star_rating(result) for result in pre_trained_results]

# Filter out invalid labels (-1) for accuracy calculation
valid_indices = [i for i, label in enumerate(pre_trained_labels) if label != -1]
pre_trained_accuracy = compute_metrics((
    np.array([pre_trained_labels[i] for i in valid_indices]),
    np.array(dataset["test"]["label"][:1000])[valid_indices]
))["accuracy"]

print(f"Pre-trained model accuracy: {pre_trained_accuracy:.2f}")




# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Split the dataset

train_testvalid = dataset["train"].train_test_split(test_size=0.2)
train_dataset = train_testvalid['train']
testvalid_dataset = train_testvalid['test']

test_valid = dataset["test"].train_test_split(test_size=0.5)
eval_dataset = test_valid['train']
test_dataset = test_valid['test']

# Tokenize the datasets
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_eval = eval_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# Fine-tune the model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()

# Load the fine-tuned model
fine_tuned_pipeline = pipeline("sentiment-analysis", model=trainer.model, tokenizer=tokenizer)

# Evaluate the fine-tuned model
fine_tuned_results = fine_tuned_pipeline(test_dataset["text"], truncation=True)
fine_tuned_accuracy = compute_metrics((
    np.array([extract_star_rating(result) for result in fine_tuned_results]),
    test_dataset["label"]
))["accuracy"]

print(f"Fine-tuned model accuracy: {fine_tuned_accuracy:.2f}")
