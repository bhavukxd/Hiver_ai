# 📧 Hiver AI Challenge – GenAI Email Reply Assistant

## Overview

This project is an AI-powered email reply assistant built for the Hiver Open Challenge.

The system generates professional customer support replies using a Retrieval-Augmented Generation (RAG) pipeline and evaluates the quality of generated responses using semantic similarity and an LLM-based evaluation framework.

---

## Features

- AI-generated customer support email replies
- Retrieval-Augmented Generation (RAG)
- Uses a custom dataset of 520 email-response pairs
- Semantic retrieval using Sentence Transformers
- Evaluation using:
  - Embedding-based Semantic Similarity
  - Style Matching
  - LLM-as-a-Judge (Tone, Completeness, Empathy, Actionability)
- Interactive Streamlit interface
- Batch evaluation pipeline with detailed reports

---

## Project Structure

```
.
├── dataset.json
├── generator.py
├── evaluator.py
├── run.py
├── streamlit_app.py
├── results.json
├── requirements.txt
└── README.md
```

---

## Approach

### 1. Retrieval-Augmented Generation (RAG)

Instead of directly asking an LLM to generate a reply, the system first retrieves the most similar customer emails from the dataset using sentence embeddings.

Pipeline:

Customer Email
↓
Embedding Generation
↓
Similarity Search
↓
Retrieve Top-3 Similar Examples
↓
LLM Prompt Construction
↓
AI Reply Generation

This allows the model to follow the style and tone of previous support responses while adapting the reply to the current customer's issue.

---

### 2. Reply Generation

The retrieved examples are added as few-shot demonstrations inside the prompt.

The LLM generates a response that is:

- Professional
- Empathetic
- Actionable
- Context-aware
- Free from unsupported assumptions

---

### 3. Evaluation

Every generated response is compared against the expected human-written response.

Metrics:

- Semantic Similarity (Sentence Transformer embeddings)
- Style Matching
- LLM Judge
    - Tone
    - Completeness
    - Actionability
    - Empathy
    - Overall Score

A final weighted score is produced for each response along with aggregate statistics.

---

## Dataset

The project uses a dataset containing **520 customer support email-response pairs**.

The dataset covers scenarios such as:

- Refund requests
- Billing issues
- Password resets
- Technical support
- Feature requests
- Meeting scheduling
- Team management
- Report generation
- Account issues
- Customer complaints

---

## Running the Application

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit interface

```bash
streamlit run streamlit_app.py
```

Run batch evaluation

```bash
python run.py
```

Evaluate only a subset

```bash
python run.py --limit 10
```

---

## Technologies Used

- Python
- Streamlit
- Groq API
- Llama 3.3
- Sentence Transformers
- NumPy
- python-dotenv

---

## Why RAG?

A standard LLM may generate generic customer support replies.

Using Retrieval-Augmented Generation allows the model to leverage similar historical email-response examples, improving consistency, tone, and relevance while reducing hallucinations.

---

## Future Improvements

- Hybrid search (BM25 + embeddings)
- Conversation history support
- Retrieval confidence score
- Citation of retrieved examples
- Automatic hallucination detection
- Response streaming
- Multi-language support

---