# LLM Lab Task 3

Complete implementation of Parts B-F for LLM and LangChain lab assignment.

## Overview

This project demonstrates:
- **Part B**: Fine-tuning DistilGPT-2 on Agriculture-QA dataset
- **Part C**: Model evaluation with perplexity, BLEU, ROUGE, and human metrics
- **Part D**: Local LLM deployment with 4-bit quantization and privacy discussion
- **Part E**: LangChain PDF chatbot with RAG pipeline
- **Part F**: Analysis of training and evaluation challenges

## Project Structure

```
lab-task-3/
├── part_b_finetune.py / .ipynb      # Fine-tune DistilGPT-2 on Agriculture-QA
├── part_c_evaluation.py / .ipynb     # Perplexity, BLEU, ROUGE, human eval
├── part_d_local_llm.py / .ipynb      # TinyLlama 4-bit local inference + privacy discussion
├── part_e_langchain_chatbot.py / .ipynb  # PDF Chatbot RAG pipeline
├── part_f_challenges.py / .ipynb     # Training & evaluation challenges analysis
├── convert_to_notebooks.py           # .py → .ipynb converter
├── requirements.txt                  # All dependencies
├── setup.ps1                         # Venv creation + install script
├── data/                             # Dataset cache (auto-loaded from HF)
├── pdfs/                             # Place your PDF here for Part E
└── outputs/                          # Saved models, graphs, FAISS index
```

## Setup

1. **Clone and setup environment:**
   ```powershell
   git clone https://github.com/karthik14344/DL-Task-3.git
   cd DL-Task-3
   .\setup.ps1
   ```

2. **Login to Hugging Face:**
   ```bash
   huggingface-cli login
   ```

3. **Place a PDF in `pdfs/sample.pdf`** for Part E

## Key Features

- **100% Local Execution**: No cloud APIs, all models run locally
- **GPU Optimized**: 4-bit quantization for memory efficiency
- **Agriculture Domain**: Fine-tuned on Agriculture-QA dataset
- **Modern LangChain**: Uses LCEL syntax for RAG pipeline
- **Comprehensive Evaluation**: Multiple metrics + human evaluation

## Models Used

| Part | Model | Purpose |
|------|-------|---------|
| **B** | DistilGPT-2 (82M) | Fine-tuning on agriculture Q&A |
| **C** | Base vs Fine-tuned | Evaluation comparison |
| **D** | TinyLlama-1.1B-Chat | Local inference with 4-bit quantization |
| **E** | TinyLlama + MiniLM-L6-v2 | PDF chatbot RAG pipeline |

## Hardware Requirements

- **GPU**: RTX 5070 (or any CUDA GPU with 8GB+ VRAM)
- **RAM**: 16GB recommended
- **Storage**: 10GB free space

## Results Summary

- **Perplexity improvement**: 87.8% reduction after fine-tuning
- **BLEU/ROUGE**: Significant improvement in agricultural domain
- **Human evaluation**: Fluency 1.0→3.0, Relevance 1.0→4.5, Correctness 1.0→2.5
- **Local LLM**: ~0.6GB VRAM usage with 4-bit quantization
- **PDF Chatbot**: Functional RAG pipeline with source attribution

## Dependencies

See `requirements.txt` for complete package list including:
- PyTorch with CUDA 12.8
- Transformers & BitsAndBytes
- LangChain ecosystem
- Evaluation metrics (BLEU, ROUGE)
- FAISS vector database

## Notes

- All models run locally without internet (after initial download)
- Dataset automatically loaded from Hugging Face
- Part F provides detailed analysis of challenges
- Interactive chatbot mode available in Part E
