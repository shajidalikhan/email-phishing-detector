from src.predict import predict_email

def test_phishing():
    text = "Your account is suspended. Click here to verify"
    assert predict_email(text) == "Phishing"

def test_safe():
    text = "Hey bro, let's meet tomorrow"
    assert predict_email(text) == "Safe"
