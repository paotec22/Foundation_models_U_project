#!/usr/bin/env python
# coding: utf-8

# # Exercise: Full-fine tuning BERT
# 
# In this exercise, you will create a BERT sentiment classifier (actually DistilBERT) using the [Hugging Face Transformers](https://huggingface.co/transformers/) library. You will use the [IMDB movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/) to complete a full fine-tuning and evaluate your model.
# 
# The IMDB dataset contains movie reviews that are labeled as either positive or negative.

# In[1]:


# Install the required version of datasets in case you have an older version
# You will need to choose "Kernel > Restart Kernel" from the menu after executing this cell
get_ipython().system('pip install -q "datasets==2.15.0"')
get_ipython().system('pip install -q transformers datasets torch scikit-learn')
get_ipython().system('pip install --upgrade datasets')


# In[2]:


# Load the sms_spam dataset
# See: https://huggingface.co/datasets/sms_spam

# Load the IMDB dataset
from datasets import load_dataset

# Load the IMDB dataset directly
dataset = load_dataset("imdb")

# Optionally, reduce dataset size for faster training (for demonstration purposes)
splits = ["train", "test"]
ds = {split: dataset[split] for split in splits}

# Optionally shuffle and reduce the size for a faster demonstration
for split in splits:
    ds[split] = ds[split].shuffle(seed=42).select(range(500))

# View the dataset characteristics (train split)
print(ds["train"])


# Let's look at the first example!

# In[3]:


# Inspect the first example. Do you think this is spam or not?
dataset["train"][0]


# ## Pre-process datasets
# 
# Now we are going to process our datasets by converting all the text into tokens for our models.

# In[5]:


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Let's use a lambda function to tokenize all the examples
tokenized_dataset = {}
for split in splits:
    tokenized_dataset[split] = dataset[split].map(
        lambda x: tokenizer(x["text"], truncation=True), batched=True
    )

# Inspect the available columns in the dataset
tokenized_dataset["train"]


# ## Load and set up the model
# 
# In this case we are doing a full fine tuning, so we will want to unfreeze all parameters.

# In[6]:


# Replace <MASK> with the code to unfreeze all the model parameters

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label={0: "not spam", 1: "spam"},
    label2id={"not spam": 0, "spam": 1},
)

# Unfreeze all the model parameters.
# Hint: Check the documentation at https://huggingface.co/transformers/v4.2.2/training.html
for param in model.parameters():
    param.requires_grad = True


# In[7]:


print(model)


# ## Let's train it!
# 
# Now it's time to train our model. We'll use the `Trainer` class.
# 
# First we'll define a function to compute our accuracy metreic then we make the `Trainer`.
# 
# In this instance, we will fill in some of the training arguments
# 

# In[9]:


# Replace <MASK> with the Training Arguments of your choice

import numpy as np
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


# The HuggingFace Trainer class handles the training and eval loop for PyTorch for us.
# Read more about it here https://huggingface.co/docs/transformers/main_classes/trainer
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./data/spam_not_spam",   # Specify where to save the model outputs
        learning_rate=2e-5,                  # Set the learning rate (you can adjust this value)
        per_device_train_batch_size=16,      # Set the training batch size
        per_device_eval_batch_size=64,       # Set the evaluation batch size
        evaluation_strategy="epoch",         # Evaluate the model after each epoch
        save_strategy="epoch",               # Save the model after each epoch
        num_train_epochs=2,                  # Number of training epochs
        weight_decay=0.01,                   # Weight decay to avoid overfitting
        load_best_model_at_end=True,         # Load the best model at the end of training
    ),
    train_dataset=tokenized_dataset["train"],  # The training dataset
    eval_dataset=tokenized_dataset["test"],   # The evaluation dataset
    tokenizer=tokenizer,                      # The tokenizer
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),  # Padding for batches
    compute_metrics=compute_metrics,          # Function to compute metrics (accuracy)
)

# Start the training process
trainer.train()


# ## Evaluate the model
# 
# Evaluating the model is as simple as calling the evaluate method on the trainer object. This will run the model on the test set and compute the metrics we specified in the compute_metrics function.

# In[12]:


# Show the performance of the model on the test set
# What do you think the evaluation accuracy will be?
trainer.evaluate()


# ### View the results
# 
# Let's look at a few examples

# In[15]:


# Make a dataframe with the predictions and the text and the labels
import pandas as pd

# Select items from the test dataset for manual review
items_for_manual_review = tokenized_dataset["test"].select(
    [0, 1, 22, 31, 43, 292, 448, 487]
)

# Make predictions
results = trainer.predict(items_for_manual_review)

# Create a DataFrame with the results
df = pd.DataFrame(
    {
        "sms": [item["text"] for item in items_for_manual_review],  # The text of the SMS
        "predictions": results.predictions.argmax(axis=1),  # Predicted label (0 or 1)
        "labels": results.label_ids,  # True label (ground truth)
    }
)

# Display the DataFrame with no column width restrictions
pd.set_option("display.max_colwidth", None)
df


# ### End of the exercise
# 
# Great work! Congrats on making it to the end of the exercise!
