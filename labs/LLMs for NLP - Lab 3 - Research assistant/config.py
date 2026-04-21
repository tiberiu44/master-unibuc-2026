import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration (OpenRouter - Comment out to use Ollama below)
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# BASE_URL = "https://openrouter.ai/api/v1"
# MODEL = "openrouter/free"

# Ollama Configuration (Local - Currently Active)
BASE_URL = "http://localhost:11434/v1"
MODEL = "gemma4:e4b"
