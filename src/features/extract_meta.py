import re
import pandas as pd

def get_url_count(text):
    """
    Extracts the total number of hyperlinks in the email body.
    """
    if not isinstance(text, str):
        return 0
    # Regex to find URLs
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return len(urls)

def get_text_length(text):
    """
    Calculates the total character count of the email.
    """
    if not isinstance(text, str):
        return 0
    return len(text)

def get_special_char_ratio(text):
    """
    Calculates the ratio of special characters to total characters.
    """
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    special_chars = re.findall(r'[^\w\s]', text)
    return len(special_chars) / len(text)

def get_keyword_flags(text):
    """
    Returns a dictionary of binary flags for suspicious keywords.
    """
    if not isinstance(text, str):
        return {
            'has_urgent': 0,
            'has_verify': 0,
            'has_password': 0,
            'has_suspension': 0,
            'has_financial': 0,
            'has_prize': 0
        }
    
    text = text.lower()
    
    flags = {
        'has_urgent': 1 if any(word in text for word in ["urgent", "immediate action", "asap", "hurry", "expire"]) else 0,
        'has_verify': 1 if any(word in text for word in ["verify", "validate", "confirm", "update account", "action required"]) else 0,
        'has_password': 1 if any(word in text for word in ["password", "credential", "login", "security", "unauthorized"]) else 0,
        'has_suspension': 1 if any(word in text for word in ["suspended", "locked", "restricted", "closed", "blocked", "terminated"]) else 0,
        'has_financial': 1 if any(word in text for word in ["bank", "invoice", "payment", "billing", "credit card", "refund", "statement"]) else 0,
        'has_prize': 1 if any(word in text for word in ["winner", "congratulations", "prize", "gift card", "reward"]) else 0
    }
    
    return flags

if __name__ == "__main__":
    # Quick test
    sample_text = "URGENT: Please verify your account at https://fake-bank.com/login immediately! Password reset required."
    print(f"URL Count: {get_url_count(sample_text)}")
    print(f"Text Length: {get_text_length(sample_text)}")
    print(f"Special Char Ratio: {get_special_char_ratio(sample_text):.4f}")
    print(f"Keyword Flags: {get_keyword_flags(sample_text)}")
