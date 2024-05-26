from datasets import load_dataset

dataset = load_dataset("yelp_review_full")
print(dataset["train"][88])

for i in range(0, 100):
    print(dataset["train"][i])