---
layout: single
title: "BERT Model Compression through Layer Freezing: An Analysis of Performance and Efficiency"
excerpt: "A walkthrough of my predictive model and how I improved accuracy using feature engineering."
date: 2025-01-01
author: Ernest Essel-Kaitoo
read_time: true
comments: true
share: true
related: true
header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: "/assets/images/ml-project-banner.jpg"
  caption: "Exploring prediction with Random Forest"
class: wide
---

# Problem Overview
With the abundance of large models, businesses are faced with the challenge of computational resources, making them unable to adopt these models to their advantage. Pretrained models have already been trained, and their weights are stored; these are general models. Due to this, model compression techniques exist to solve these issues. There is a need to analyze the trade-offs between using layer freezing as a compression strategy. This study offers a comparative analysis of the effects of freezing strategies to help businesses make informed decisions when considering the trade-offs and expenditure involved.

# Research Questions

In order to carry out this study, the following research questions will guide the investigation:

1. How much performance is sacrificed when freezing BERT layers, and what efficiency gains are achieved?
2. Which freezing strategy provides the best trade-off between model performance and computational efficiency?
3. Can we maintain acceptable task performance while significantly reducing training time and resource requirements through selective layer freezing?

# Dataset

The rotten_tomatoes dataset will be used of this study. It consists of movie reviews from the Rotten Tomatoes website, where the task is to classify each review's sentiment as either positive or negative. The following code will load the dataset and split it into training and test set.

```python
from datasets import load_dataset
tomatoes = load_dataset("rotten_tomatoes")
train_data, test_data = tomatoes["train"], tomatoes["test"]
```
## (Full Code Available On Github)[]
# Methodology

## Bert Model (Base Model)
We skip data exploration for this study. We move on to loading the model since we hae loaded the data. We will load the base BERT model and the tokenizer.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Load model and tokenizer
model_id = "bert-base-cased"
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

I prepare the data using Hugging Face's Transformers library. The text sequence are tokenized and dynamically padded per batch using DataCollatorWithPadding for memory efficiency. The preprocessing pipeline employed batched tokenization with automatic truncation, converting raw text into model-ready inputs for both training and test splits.

#### insert Image

After the first run, model evaluation revealed suboptimal performance with an F1 score of 35.4% and a loss value of 0.693, indicating the model is currently performing at near-random guessing levels for this binary classification task. Despite the model's limited predictive capability, the evaluation processed approximately 9 samples per second over 117 seconds, demonstrating adequate computational efficiency despite.

## Bert Model (Freezing Layers)

I use the Hugging Face Transformers to freeze certain layers of the network. I froze the main BERT model and allow only updates to pass through the classification head. This is to provide a  great comparison as everything will be the same, except for freezing specific layers.

```python
# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```
The code below demonstrate a selective fine-tuning approach where the pre-trained transformer base layers were frozen to preserve general language understanding, while only the task-specific classification head was set as trainable. 

```python
for name, param in model.named_parameters():
  # Trainable classification head
  if name.startswith("classifier"):
    param.requires_grad = True
    # Freeze everything else
  else:
    param.requires_grad = False
```
### Insert image

After unfreezing only the classification head, this yielded performance with an F1 score of 59.7% and a loss value of 0.67. 

## Summary
The selective fine-tuning strategy yielded substantial improvements, with F1 score increasing by 68.5% from 0.354 to 0.597, while loss decreased from random guessing levels (0.693) to more confident predictions (0.677). This demonstrates the effectiveness of training only the classification head while preserving the pre-trained model's linguistic knowledge. The model maintained consistent inference speed (~8.9 samples/second) while achieving significantly better sentiment classification capability.

