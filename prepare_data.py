from datasets import load_dataset

# Load SciQ dataset
dataset = load_dataset("allenai/sciq")

def format_example(example):
    # We combine the question and correct answer into a simple prompt-response format
    # This is the format the model will learn to replicate during training
    text = f"### Question: {example['question']}\n### Answer: {example['correct_answer']}"
    return {"text": text}

# Apply the formatting to all splits (train, validation, test)
# This transforms every raw entry into our clean prompt-response format
formatted = dataset.map(format_example)

# Keep only the 'text' column — that's all the trainer needs
formatted = formatted.remove_columns(['question', 'distractor1', 'distractor2', 'distractor3', 'correct_answer', 'support'])

# Save the formatted dataset to disk so we can load it during training
formatted.save_to_disk("sciq_formatted")

print("Dataset formatted and saved!")
print("\nExample formatted entry:")
print(formatted["train"][0]["text"])
print("\nTotal training examples:", len(formatted["train"]))