from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import Trainer



def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])

dataset = load_dataset("rotten_tomatoes")  # doctest: +IGNORE_RESULT
model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")

# my_config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", n_heads=12)
# my_model = AutoModel.from_config(my_config)


training_args = TrainingArguments(
    output_dir="./save_folder",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
)

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
dataset = dataset.map(tokenize_dataset, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)  # doctest: +SKIP

trainer.train()