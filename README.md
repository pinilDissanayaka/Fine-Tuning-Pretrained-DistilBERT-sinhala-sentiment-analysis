# Fine-Tuning Pretrained DistilBERT for Sinhala Sentiment Analysis

This repository contains the implementation for fine-tuning a pretrained DistilBERT model for sentiment analysis in the Sinhala language. The primary objective is to train a model that can effectively classify text as positive, negative, or neutral using a Sinhala language dataset.

## Overview

Sentiment analysis is a common task in natural language processing (NLP) where the goal is to determine the sentiment expressed in a piece of text. By leveraging a pretrained transformer model like DistilBERT, we can achieve efficient and accurate results, even in low-resource languages like Sinhala.

## Dataset

We use the [Sinhala Sentiment Analysis dataset](https://huggingface.co/datasets/sinhala-nlp/sinhala-sentiment-analysis) provided by Sinhala NLP. This dataset includes text samples in Sinhala labeled with sentiment categories. The distribution of data across these categories is suitable for training and evaluating sentiment analysis models.

### Dataset Features
- **Language**: Sinhala
- **Labels**: Positive, Negative, Neutral
- **Source**: Collected and shared via the Hugging Face `datasets` library

## Model and Architecture

### DistilBERT
[DistilBERT](https://arxiv.org/abs/1910.01108) is a smaller, faster, cheaper, and lighter version of BERT. It retains 97% of BERT's performance while being 60% faster and reducing model size by 40%.

### Fine-Tuning
The fine-tuning process involves:
- Adding a classification head to the pretrained DistilBERT model.
- Training the model using the Sinhala sentiment dataset.
- Optimizing using a suitable loss function for classification (e.g., cross-entropy loss).

## License
This project is licensed under the MIT License.


