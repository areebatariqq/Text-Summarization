# Text-Summarization

This project presents a hybrid **text summarization system** that condenses lengthy articles, blogs, or news reports into concise and coherent summaries. It implements both **extractive summarization** (using NLP techniques) and **abstractive summarization** (using state-of-the-art transformer models).

---

## 📁 Dataset

- **Source**: [CNN/Daily Mail Dataset on Hugging Face](https://huggingface.co/datasets/cnn_dailymail)
- The dataset contains over 300,000 news articles paired with human-written summaries, originally compiled for training and evaluating summarization models.

---

## 📌 Project Objective

- To build a dual summarization system:
  - **Extractive Summarizer**: Identifies and extracts key sentences from the original text.
  - **Abstractive Summarizer**: Generates novel summaries using deep learning models.
- To fine-tune transformer models for enhanced summarization quality.
- To test and evaluate the system on real-world content.

---

## 🔧 Tools and Technologies Used

- **Google Colab** (for development)
- **Python**
- **spaCy** (for NLP-based extractive summarization)
- **Hugging Face Transformers** (for abstractive summarization)
- **Pretrained models**: BART, T5, or GPT-2 (via Hugging Face)
- **Evaluation Metrics**: ROUGE, BLEU

---

## 📊 Steps Performed

### 1. Data Preprocessing
- Loaded and cleaned the CNN/Daily Mail dataset.
- Tokenized and normalized text for both summarization approaches.

### 2. Extractive Summarization (using spaCy)
- Applied sentence scoring based on:
  - Named Entity Recognition (NER)
  - Term frequency–inverse document frequency (TF-IDF)
  - Sentence position
- Selected top-ranked sentences to form the summary.

### 3. Abstractive Summarization (using Transformers)
- Used pre-trained models like **BART**, **T5**, and **GPT-2** via Hugging Face.
- Implemented model inference using `pipeline("summarization")`.
- Fine-tuned models (optional step) for improved coherence and readability.

### 4. Evaluation
- Evaluated summaries using:
  - **ROUGE** scores (Recall-Oriented Understudy for Gisting Evaluation)
  - **BLEU** scores (for fluency and n-gram overlap)
- Compared extractive and abstractive outputs for quality and informativeness.

---

## ✅ Results

- **Extractive Model**: Simple, fast, and retains factual accuracy but lacks paraphrasing.
- **Abstractive Model**: More human-like summaries with better flow and coherence.
- **Best Model**: Fine-tuned **BART-base** achieved the best ROUGE-L and readability scores.

---

## 💡 Key Highlights

- Supports both summarization types, allowing flexible application.
- Easy to test on real-world data (e.g., articles from websites or blogs).
- Highly interpretable and extensible for future improvements.

---
