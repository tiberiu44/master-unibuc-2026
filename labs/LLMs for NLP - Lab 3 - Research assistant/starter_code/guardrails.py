import re
import pandas as pd




def check_prompt_injection(text: str) -> bool:
    """
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
    
    # TODO: Write a loop to check if any of the malicious_words are in text_lower.
    # Return False if you find any.
    
    return True

def check_resource_limits(epochs: int) -> bool:
    """
    Return False if epochs exceed 5. True if safe.
    """
    # TODO: Return True if epochs <= 5, otherwise False.
    pass
