#!/usr/bin/env python
# coding: utf-8

# # Exercise: Create a BERT sentiment classifier
# 
# In this exercise, you will create a BERT sentiment classifier (actually DistilBERT) using the [Hugging Face Transformers](https://huggingface.co/transformers/) library. 
# 
# You will use the [IMDB movie review dataset](https://huggingface.co/datasets/imdb) to train and evaluate your model. The IMDB dataset contains movie reviews that are labeled as either positive or negative. 

# In[1]:


# Install the required version of datasets in case you have an older version
# You will need to choose "Kernel > Restart Kernel" from the menu after executing this cell
# ! pip install -q "datasets==2.15.0"
get_ipython().system(' pip install -q transformers datasets torch scikit-learn')
get_ipython().system(' pip install --upgrade datasets')


# In[10]:


# Step 1: Import necessary libraries
from datasets import load_dataset
import pandas as pd

# Step 2: Attempt to load the IMDb dataset
try:
    # Load the IMDb dataset
    print("Attempting to load the IMDb dataset...")
    ds = load_dataset("imdb")
    print("Dataset successfully loaded.")
except Exception as e:
    print(f"Error loading IMDb dataset: {e}")
    print("Attempting fallback solutions...")

    # Step 3: Attempt to load an alternative dataset
    try:
        print("Loading alternative 'yelp_polarity' dataset...")
        ds = load_dataset("yelp_polarity")
        print("Alternative dataset loaded successfully.")
    except Exception as e_alt:
        print(f"Error loading alternative dataset: {e_alt}")
        print("Falling back to local download...")

        # Step 4: Fallback to local dataset
        try:
            print("Downloading and loading IMDb dataset locally...")
            get_ipython().system('wget -q https://raw.githubusercontent.com/datasets/imdb/master/data/imdb.csv')
            df = pd.read_csv("imdb.csv")
            print("Local dataset successfully loaded.")
            print("Sample data:")
            print(df.head())
        except Exception as e_local:
            print(f"Error loading local dataset: {e_local}")
            print("Please check your internet connection or dataset availability.")

# Step 5: Inspect the dataset structure if successfully loaded
try:
    print("\nDataset structure:")
    print(ds)
except NameError:
    print("Dataset not available in memory.")


# ## Pre-process datasets
# 
# Now we are going to process our datasets by converting all the text into tokens for our models. You may ask, why isn't the text converted already? Well, different models may use different tokenizers, so by converting at train time we retain more flexibility.

# In[14]:


from transformers import AutoTokenizer
from datasets import load_dataset

# Load the IMDb dataset
ds = load_dataset("imdb")

# Define the splits you want to use
splits = ["train", "test"]

# Initialize the tokenizer for distilbert
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Preprocess function to tokenize the IMDB dataset
def preprocess_function(examples):
    """Preprocess the imdb dataset by returning tokenized examples."""
    return tokenizer(examples["text"], padding=True, truncation=True)

# Create a tokenized dataset
tokenized_ds = {}
for split in splits:
    tokenized_ds[split] = ds[split].map(preprocess_function, batched=True)

# Show the first example of the tokenized training set
print("Tokenized example:", tokenized_ds["train"][0]["input_ids"])

# Optionally, inspect the actual tokenized output to understand any discrepancies



# ## Load and set up the model
# 
# We will now load the model and freeze most of the parameters of the model: everything except the classification head.

# In[15]:


# Replace <MASK> with your code freezes the base model

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},  # For converting predictions to strings
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
)

# Freeze all the parameters of the base model
# Hint: Check the documentation at https://huggingface.co/transformers/v4.2.2/training.html
for param in model.base_model.parameters():
    param.requires_grad = False

model.classifier


# In[16]:


print(model)


# ## Let's train it!
# 
# Now it's time to train our model. We'll use the `Trainer` class from the 🤗 Transformers library to do this. The `Trainer` class provides a high-level API that abstracts away a lot of the training loop.
# 
# First we'll define a function to compute our accuracy metreic then we make the `Trainer`.
# 
# Let's take this opportunity to learn about the `DataCollator`. According to the HuggingFace documentation:
# 
# > Data collators are objects that will form a batch by using a list of dataset elements as input. These elements are of the same type as the elements of train_dataset or eval_dataset.
# 
# > To be able to build batches, data collators may apply some processing (like padding).
# 

# In[17]:


# Replace <MASK> with your DataCollatorWithPadding argument(s)

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
        output_dir="./data/sentiment_analysis",
        learning_rate=2e-3,
        # Reduce the batch size if you don't have enough memory
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    ),
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()


# ## Evaluate the model
# 
# Evaluating the model is as simple as calling the evaluate method on the trainer object. This will run the model on the test set and compute the metrics we specified in the compute_metrics function.

# In[18]:


# Show the performance of the model on the test set
# What do you think the evaluation accuracy will be?
trainer.evaluate()


# ### View the results
# 
# Let's look at two examples with labels and predicted values.

# In[ ]:


import pandas as pd

df = pd.DataFrame(tokenized_ds["test"])
df = df[["text", "label"]]

# Replace <br /> tags in the text with spaces
df["text"] = df["text"].str.replace("<br />", " ", regex=False)

# Add the model predictions to the dataframe
predictions = trainer.predict(tokenized_ds["test"])
df["predicted_label"] = np.argmax(predictions[0], axis=1)

df.head(2)


# ### Look at some of the incorrect predictions
# 
# Let's take a look at some of the incorrectly-predcted examples

# In[ ]:


# Show full cell output
pd.set_option("display.max_colwidth", None)

df[df["label"] != df["predicted_label"]].head(2)


# ## End of exercise
# 
# Great work! 🤗
