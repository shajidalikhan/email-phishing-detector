# How to Run the Phishing Detection Model

**Model File:** `phishing_model.pkl` | **Framework:** scikit-learn

---

## Step 1: Install Requirements

```bash
pip install scikit-learn joblib
```

---

## Step 2: Load the Model

Place `phishing_model.pkl` in the same folder as your script, then:

```python
import joblib

model = joblib.load('phishing_model.pkl')
```

---

## Step 3: Run a Prediction

```python
email = "URGENT: Click here to claim your prize!!!"

prediction = model.predict([email])[0]        # 0 = Safe, 1 = Phishing
score      = model.decision_function([email])[0]   # confidence score

if prediction == 1:
    print(f"PHISHING  (score: {score:.4f})")
else:
    print(f"SAFE      (score: {score:.4f})")
```

> **Note:** Always wrap the email in a list — `model.predict([email])`, not `model.predict(email)`

---

## Step 4: Understand the Output

- **Prediction:** `0` = Safe, `1` = Phishing
- **Score > 0** → Phishing (higher = more confident)
- **Score < 0** → Safe (lower = more confident)

---

## Common Errors

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: sklearn` | `pip install scikit-learn` |
| `FileNotFoundError: phishing_model.pkl` | Keep `.pkl` in the same folder as your script |
| `ValueError: Expected 2D array` | Use `model.predict([email])` not `model.predict(email)` |
