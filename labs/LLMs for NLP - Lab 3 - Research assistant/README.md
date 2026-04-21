# Safe AutoML & Tutor Agent Lab

Welcome to the Safe AutoML & AI Tutor Agent lab! In this project, you will build an interactive AI Agent that acts as a Data Science Tutor. The agent can take natural language requests to train real text classifiers on datasets and then conversationally explain the model dynamics back to the user.

A critical part of this lab involves implementing robust ML guardrails — you will be building checks for prompt injections and enforcing safe execution using a Human-in-the-Loop constraint.

## Directory Structure

This lab contains the following directories:

- `data/`: Contains the `setup_data.py` script necessary to download our test dataset.
- `starter_code/`: The directory you will be working in! Various Python files here contain `TODO:` statements guiding your implementation. All scaffolding and boilerplate have been provided for you.
- `solution_repo/`: The complete, fully working reference implementations for instructors (or if you get hopelessly stuck).

## Initial Setup

1. **Create a Virtual Environment**:
   It is highly recommended to use [uv](https://github.com/astral-sh/uv) to manage your virtual environment for significantly faster dependency resolution.
   ```bash
   uv venv
   # On macOS/Linux:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Generate the Datasets**:
   Run the data preparation script to download the IMDb datasets:
   ```bash
   python data/setup_data.py
   ```
   You should now see `real_reviews_clean.csv` pop up in the `data/` folder.

4. **Start your Ollama Server**:
   This agent uses the local `ollama` model provider, ensuring privacy and eliminating API costs.
   Ensure that you have [Ollama](https://ollama.com/) installed on your actual machine.
   Run the following terminal command to pull and start the Tool-Calling capable `gemma4:e4b` model:
   ```bash
   ollama run gemma4:e4b
   ```

## Workflow & Execution

### 1. Fill out the TODOs
Navigate to `starter_code/` and implement the remaining logic marked by `TODO:` comments across three main scripts:
- **`guardrails.py`**: Safety rules engine. 
- **`tools.py`**: The model training implementations (Scikit-Learn Regression & Huggingface DistilBERT).
- **`agent.py`**: The AI orchestrator handling conversation memory and tool execution dispatch running on Ollama.

### 2. Validation Testing
You can test your implementation at any time using the provided unit tester:
```bash
cd starter_code
python lab_tester.py
```
*Note: Make sure you've run the dataset generator first.*

### 3. Running the Agent
Once you are confident with your implementation, you can instantiate and connect to your agent to watch it work interactively using the provided `run_agent.py` script.

Navigating to the lab root and running the interactive script:
```bash
python run_agent.py
```

This will launch a conversational session with the AI Tutor Agent in your terminal. You can chat with it, ask it to train models on datasets, and see the interactive guardrails (like human-in-the-loop approvals) in real time. Be sure to provide the dataset path in your prompt directly.

Example Interaction:
```text
=========================================================
🤖 Safe AutoML Tutor Agent Initialized (Ollama Backend).
Type 'exit' or 'quit' to end the chat.
=========================================================

[System] Ready! Say hello to your AI tutor. Remember to give it the dataset path like 'data/real_reviews_clean.csv' in your prompt.

🧑 You: Train a fast sklearn logical regression model on data/real_reviews_clean.csv. Tell me about the results.
```

Have fun, and good luck building a safer ML ecosystem!
