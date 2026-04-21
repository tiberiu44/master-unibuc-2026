import re
import pandas as pd





def check_prompt_injection(text: str) -> bool:
    """
    Check if the text contains potential prompt injection attempts.
    Return False if detected, True if safe.
    """
    malicious_words = [
        "ignore previous instructions",
        "system prompt",
        "override",
        "forget all",
        "you are now",
        "bypassing"
    ]
    text_lower = text.lower()
    for word in malicious_words:
        if word in text_lower:
            return False
            
    return True

def check_resource_limits(epochs: int) -> bool:
    """
    Ensure the number of epochs does not exceed a reasonable limit.
    Return False if epochs exceed a reasonable threshold (e.g., 5). True if safe.
    """
    return epochs <= 5
