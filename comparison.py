from transformers import pipeline
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer

# Load the pre-trained sentiment analysis model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
EPOCHS=1
pre_trained_pipeline = pipeline("sentiment-analysis", model=model_name)

# Load a sample dataset
dataset = load_dataset("imdb", split="test[:1%]")

# Evaluate the pre-trained model
pre_trained_results = []
for example in dataset:
    # Truncate the input text to the maximum allowed length
    truncated_text = example["text"][:510]  # Leave space for special tokens

    # Perform sentiment analysis on the truncated text
    result = pre_trained_pipeline(truncated_text)
    pre_trained_results.append(result[0]["label"])

# Calculate accuracy or other metrics (unchanged)
pre_trained_accuracy = sum([1 for x, y in zip(pre_trained_results, dataset["label"]) if (x == "POSITIVE" and y == 1) or (x == "NEGATIVE" and y == 0)]) / len(dataset)
print(f"Pre-trained model accuracy: {pre_trained_accuracy:.2f}")




# Load pre-trained model and tokenizer
# model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Split the dataset into train, validation, and test sets
train_testvalid = dataset.train_test_split(test_size=0.2)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
train_dataset = train_testvalid['train']
eval_dataset = test_valid['train']
test_dataset = test_valid['test']

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Fine-tune the model
training_args = TrainingArguments(output_dir="./results", evaluation_strategy="epoch", num_train_epochs=EPOCHS)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_datasets, eval_dataset=tokenized_datasets)

trainer.train()

# Load the fine-tuned model
fine_tuned_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Evaluate the fine-tuned model
fine_tuned_results = []
for example in dataset:
    truncated_text = example["text"][:510]  # Leave space for special tokens
    result = fine_tuned_pipeline(truncated_text)
    fine_tuned_results.append(result[0]["label"])

# Calculate accuracy or other metrics
fine_tuned_accuracy = sum([1 for x, y in zip(fine_tuned_results, dataset["label"]) if (x == "POSITIVE" and y == 1) or (x == "NEGATIVE" and y == 0)]) / len(dataset)
print(f"Fine-tuned model accuracy: {fine_tuned_accuracy:.2f}")
