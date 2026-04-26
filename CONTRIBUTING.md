# Contributing Guidelines

## Getting Started

### 1. Prerequisites (Python Version Management)

This project requires **Python 3.10.x** to avoid compatibility issues with ML libraries (like TensorFlow).

**Check your current Python version:**
```bash
python --version
```

If you do not have Python 3.10 installed, **do not uninstall your system Python**. Instead, install it locally just for this project using `pyenv`:

1.  **Install `pyenv`:**
    *   **Windows:** Install [pyenv-win](https://github.com/pyenv-win/pyenv-win) (run this in PowerShell):
        ```powershell
        Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
        ```
    *   **Mac/Linux:**
        ```bash
        curl https://pyenv.run | bash
        ```
2.  **Install Python 3.10 just for this project:**
    ```bash
    pyenv install 3.10.11
    pyenv local 3.10.11  # Tells this directory to use 3.10.11
    ```

---

### 2. Setup the Project

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd email-phishing-detector
   ```

2. **Create a virtual environment:**
   Verify your python version is 3.10, then create your isolated environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Workflow

1. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git commit -m "Add some feature"
   ```

3. Push your branch and open a Pull Request.
