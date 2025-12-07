Project : Urdu Conversational Chatbot — Transformer with Multi-Head Attention

A custom-built Urdu conversational chatbot created using a Transformer Encoder–Decoder model, implemented completely from scratch using PyTorch (no pre-trained models).
The chatbot generates coherent, context-aware Urdu responses and provides an interactive UI using Streamlit/Gradio.

📌 Objective

The goal of this project is to:

Build a fully custom NLP pipeline for Urdu conversation.

Implement a Transformer model from scratch using:

Multi-Head Attention

Positional Encoding

Encoder–Decoder architecture

Train the model using an Urdu conversational dataset.

Evaluate using standard NLP metrics.

Deploy a real-time Urdu chatbot using Streamlit/Gradio.

Publish a final Medium blog summarizing results and learning.

👥 Group Members

Maximum 2 students allowed.

AI assistance is permitted, but:

Complete AI-generated code is strictly prohibited (0 marks if detected).

📅 Deadline

20 October 2025
No extensions will be provided.

📂 Dataset

We use the following Urdu conversational dataset:

🔗 Kaggle Dataset:
https://www.kaggle.com/datasets/muhammadahmedansari/urdu-dataset-20000

Dataset includes 20,000+ Urdu conversational pairs suitable for dialogue modeling.

🛠️ Tasks & Workflow
1. Preprocessing

Normalize Urdu text:

Remove diacritics

Standardize Arabic characters (Alef, Yeh alternates)

Tokenization (custom Urdu tokenizer or regex-based)

Build vocabulary

Data split:

Train: 80%

Validation: 10%

Test: 10%

2. Model Architecture (From Scratch)

Implement a Transformer Encoder–Decoder using PyTorch:

🔹 Encoder

Multi-Head Self-Attention

Layer Normalization

Feed Forward Network

Positional Encoding

🔹 Decoder

Masked Multi-Head Self-Attention

Encoder–Decoder Attention

FFN

Positional Encoding

🔹 Additional Components

Token & Positional Embeddings

Dropout layers

Teacher forcing during training

3. Training & Hyperparameters
Component	Suggested Values
Embedding Size	256 / 512
Attention Heads	2
Encoder Layers	2
Decoder Layers	2
Dropout	0.1 – 0.3
Batch Size	32 / 64
Learning Rate	1e-4 – 5e-4 (Adam)

✔ Save the best model based on validation BLEU score.

✔ Use teacher forcing while training.

📊 4. Evaluation
Automatic Metrics

BLEU

ROUGE-L

chrF

Perplexity

Human Evaluation

Rate the chatbot output on a scale of 1–5 for:

Fluency

Relevance

Adequacy

Qualitative Analysis

Include examples comparing:

Model Output

Ground Truth

Human Notes

💬 5. Inference & User Interface

A real-time chatbot UI built using Streamlit or Gradio.

Features:

Urdu text input box

Generated Urdu response

Conversation history

Option to select decoding:

Greedy Search

Beam Search

Proper right-to-left (RTL) Urdu text rendering
