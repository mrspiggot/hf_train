from transformers import pipeline, Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import numpy as np

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
pre_trained_pipeline = pipeline("sentiment-analysis", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load a sample dataset
dataset = load_dataset("imdb", split="test[:1%]")

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Fine-tune the model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets
)
trainer.train()

# Define Evaluation Metrics
import evaluate
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Evaluate the fine-tuned model (with metrics)
trainer.evaluate()
