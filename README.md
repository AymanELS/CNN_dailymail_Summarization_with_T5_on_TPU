# CNN_dailymail_Summarization_with_T5_on_TPU

## Introduction
In this project, we use transfer learning with the model T5 to summarize new articles from CNN and the dailymail. In the implementation, we use the [Huggingface accelerate] (https://github.com/huggingface/accelerate) so it can run on TPU. 
The code in this project was inpired by: [https://github.com/NielsRogge/Transformers-Tutorials/tree/master/T5](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/T5)

## Text-To-Text Transfer Transformer (T5)
T5 is a transformer-based model that introduced a novel concept of framing all NLP tasks as text-to-text problems. This means that both the input and ouput are sequences of text. This model is pre-trained on a large corpus of text in an unsupervised way, then it is fine-tuned for various NLP tasks like translation, question answering, sentiment classification, etc. [\[1\]](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf)
In this project, we use fine-tuned T5 for text summarization.

![alt text](https://github.com/AymanELS/CNN_dailymail_Summarization_with_T5_on_TPU/blob/main/T5.png)
The T5 was first introduced by Google AI in the following paper: [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf)

## Dataset
We use in this project the CNN_DailyMail dataset, which is an English-language dataset containing 312k unique news articles written by journalists at CNN and the Daily Mail. The current version supports both extractive and abstractive summarization, though the original version was created for machine reading and comprehension and abstractive question answering. Source: [https://huggingface.co/datasets/cnn_dailymail] (https://huggingface.co/datasets/cnn_dailymail)

