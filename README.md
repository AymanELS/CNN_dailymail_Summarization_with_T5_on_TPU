# CNN_dailymail_Summarization_with_T5_on_TPU
Finetuning T5-base-english for summarization on TPU

## Introduction
In this project, we use transfer learning with the model T5 to summarize new articles from CNN and the dailymail.

## Text-To-Text Transfer Transformer (T5)
T5 is a transformer-based model that introduced a novel concept of framing all NLP tasks as text-to-text problems. This means that both the input and ouput are sequences of text. This model is pre-trained on a large corpus of text in an unsupervised way, then it is fine-tuned for various NLP tasks like translation, question answering, sentiment classification, etc. [1](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf)
In this project, we use fine-tuned T5 for text summarization.
