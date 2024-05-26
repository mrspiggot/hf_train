from transformers import pipeline, Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, DatasetDict

# Load the pre-trained sentiment analysis model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
EPOCHS = 1
pre_trained_pipeline = pipeline("sentiment-analysis", model=model_name)

# Load a sample dataset and split it into train, validation, and test sets
dataset = load_dataset("imdb")
dataset = dataset["test"].train_test_split(test_size=0.2)
dataset["validation"] = dataset.pop("test")
dataset = dataset["train"].train_test_split(test_size=0.25)
dataset["train"], dataset["test"] = dataset["train"], dataset.pop("test")

# Function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Load pre-trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Fine-tune the model
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)

trainer.train()

# Load the fine-tuned model
fine_tuned_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Evaluate the pre-trained model on the test set
pre_trained_results = []
for example in dataset["test"]:
    truncated_text = example["text"][:510]  # Leave space for special tokens
    result = pre_trained_pipeline(truncated_text)
    pre_trained_results.append(result[0]["label"])

pre_trained_accuracy = sum([1 for x, y in zip(pre_trained_results, dataset["test"]["label"]) if (x == "POSITIVE" and y == 1) or (x == "NEGATIVE" and y == 0)]) / len(dataset["test"])
print(f"Pre-trained model accuracy: {pre_trained_accuracy:.2f}")

# Evaluate the fine-tuned model on the test set
fine_tuned_results = []
for example in dataset["test"]:
    truncated_text = example["text"][:510]  # Leave space for special tokens
    result = fine_tuned_pipeline(truncated_text)
    fine_tuned_results.append(result[0]["label"])

fine_tuned_accuracy = sum([1 for x, y in zip(fine_tuned_results, dataset["test"]["label"]) if (x == "POSITIVE" and y == 1) or (x == "NEGATIVE" and y == 0)]) / len(dataset["test"])
print(f"Fine-tuned model accuracy: {fine_tuned_accuracy:.2f}")
