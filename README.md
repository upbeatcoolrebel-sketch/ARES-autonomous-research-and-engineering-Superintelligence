# ARES-autonomous-research-and-engineering-superintelligence

Welcome to **ARES** (Autonomous Research and Engineering Superintelligence), an AI-powered chatbot designed to provide witty, tech-focused responses. Built with a transformer model (~10.8M parameters), ARES is trained on ~10,000 tech-filtered samples and leverages conversation history for context-aware interactions. This project is optimized for Google Colab's T4 GPU and is ideal for exploring AI, robotics, and engineering topics.

## Overview
- **Model**: Transformer decoder with 4 layers, 8 heads, and a 256-dimensional embedding.
- **Dataset**: ~10,000 sentences from WikiText and RSS feeds (e.g., KDnuggets, MachineLearningMastery), filtered for tech relevance.
- **Features**: Context-aware chat, memory system, beam search generation, and early stopping training.
- **Hardware**: Optimized for Colab's T4 GPU (~15GB VRAM, ~12.7GB RAM).

## Prerequisites
- **Python 3.11+**
- Required libraries: `datasets`, `torch`, `feedparser`, `requests`, `beautifulsoup4`
- Google Colab account with T4 GPU runtime

## Setup
### 1. Clone the Repository
In a Google Colab notebook:
```bash
!git clone https://github.com/yourusername/ARES-autonomous-research-and-engineering-superintelligence.git
%cd ARES-autonomous-research-and-engineering-superintelligence
