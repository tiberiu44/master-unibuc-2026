# Lab 2: Vanilla Transformer

## Lab Overview
This lab is divided into two main parts:
1.  **Part 1: Sentiment Classification**: Build a Bidirectional Transformer Encoder to classify movie reviews. Compare its performance against a standard MLP baseline.
2.  **Part 2: Math GPT**: Build a Causal (Decoder-Only) Transformer to solve algebraic addition problems.

## Getting Started

### 1. Environment Setup
We recommend using a virtual environment (Conda or `venv`). Ensure you have Python 3.9+ installed.

```bash
# Optional: Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```


### 2. Running the Lab
Open `student_lab.ipynb` in your favorite Jupyter environment (VS Code, JupyterLab, etc.) and follow the instructions.

## Key Components to Implement
-   **Positional Encoding**: Sinusoidal absolute encodings.
-   **Multi-Head Attention**: Scaled dot-product attention logic.
-   **Transformer Block**: Post-LayerNorm residual connections.
-   **Causal Masking**: For the generative math model.

## Troubleshooting
If you encounter any issues with dataset downloads, ensure you have an active internet connection for the Hugging Face `datasets` library.
