import json
from typing import List, Dict, Any
from openai import OpenAI
import config
from guardrails import check_prompt_injection, check_resource_limits
from tools import train_sklearn_pipeline, train_bert_pipeline, plot_confusion_matrix

class AutoMLAgent:
    def __init__(self):
        # Initialize identity and memory
        self.memory = []
        self.client = OpenAI(
            base_url=getattr(config, "BASE_URL", "http://localhost:11434/v1"),
            api_key=getattr(config, "OPENROUTER_API_KEY", "ollama"),
        )
        
        system_prompt = (
            "You are an AI ML Tutor, designed to help students train machine learning models and understand their performance. "
            "You have access to tools to train sklearn models, train BERT models, and plot confusion matrices. "
            "When tools return results like accuracy, classification reports, and confusion matrices, "
            "read these metrics and explain the model's dynamics (e.g., false positives/negatives) "
            "in plain English to the student. Be encouraging and instructional."
        )
        # TODO: Add the system_prompt to self.memory mimicking the dictionary format {"role": "system", "content": ...}
        
        # Tools structure is provided for you
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "train_sklearn_pipeline",
                    "description": "Train a Scikit-Learn TF-IDF + LogisticRegression pipeline.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "train_path": {"type": "string", "description": "Path to the training dataset CSV."},
                            "test_path": {"type": "string", "description": "Path to the evaluation/test dataset CSV."},
                            "text_col": {"type": "string"},
                            "label_col": {"type": "string"}
                        },
                        "required": ["train_path", "test_path", "text_col", "label_col"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "train_bert_pipeline",
                    "description": "Fine-tune a DistilBERT model.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "train_path": {"type": "string", "description": "Path to the training dataset CSV."},
                            "test_path": {"type": "string", "description": "Path to the evaluation/test dataset CSV."},
                            "text_col": {"type": "string"},
                            "label_col": {"type": "string"},
                            "epochs": {"type": "integer"}
                        },
                        "required": ["train_path", "test_path", "text_col", "label_col", "epochs"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "plot_confusion_matrix",
                    "description": "Plot a confusion matrix and save it to disk.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "matrix_data": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "integer"}}
                            },
                            "labels": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["matrix_data", "labels"]
                    }
                }
            }
        ]

    def process_request(self, user_prompt: str) -> str:
        """Process a natural language request from a student to train a model."""
        
        # 1) Append user_prompt to self.memory
        self.memory.append({"role": "user", "content": f"User prompt: {user_prompt}"})
        
        # 2) Run input guardrails
        # TODO: Call check_prompt_injection. If it returns False, return a "Safety Violation" string.
            
        # 3) Pass memory and tools to LLM (Strictly OpenAI API)
        # TODO: Use self.client.chat.completions.create to get a response from the LLM. 
        # Remember to pass: model=config.MODEL, messages=self.memory, tools=self.tools.
        # Assign the message object to `msg` (ensure it's converted to a dict for memory).
        msg = {} # Placeholder
        
        pass 
        
        self.memory.append(msg)
        
        # 4) If LLM chooses a tool
        if msg.get('tool_calls'):
            # Deduplicate tool calls
            unique_tool_calls = []
            seen_calls = set()
            for tc in msg['tool_calls']:
                call_key = (tc['function']['name'], json.dumps(tc['function']['arguments'], sort_keys=True))
                if call_key not in seen_calls:
                    seen_calls.add(call_key)
                    unique_tool_calls.append(tc)
                
            for tool_call in unique_tool_calls:
                func_name = tool_call['function']['name']
                args = tool_call['function']['arguments']
                if isinstance(args, str):
                    args = json.loads(args)
                
                # Check backend parameters
                if func_name == "train_bert_pipeline":
                    epochs = args.get("epochs", 1)
                    if not check_resource_limits(epochs):
                        return "Safety Violation: Requested epochs exceed resource limits."

                # TODO: Implement a Human-in-the-Loop check using `input("Approve tool execution? (y/n): ")`. Return a rejection string if they answer no.
                print(f"Agent wants to execute '{func_name}' with arguments: {args}")
                
                print(f"\nExecuting '{func_name}'... Please wait, this may take a while.\n")
                
                # 5) Execute tool
                result = ""
                # TODO: Based on `func_name`, call the correct local function (train_sklearn_pipeline, 
                # train_bert_pipeline, or plot_confusion_matrix) passing the **args dictionary. Assign output to `result`.
                    
                self.memory.append({
                    "role": "tool",
                    "name": func_name,
                    "content": result if result else "{}"
                })
                
            # 6) Call LLM again to get Tutor analytical explanation
            # TODO: Create a second completion request using self.client.chat.completions.create
            # with your updated self.memory, append the newly generated message to memory, and return its content text.
            pass
            
        return msg['content']
