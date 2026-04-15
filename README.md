# Doc2JSON

Doc2JSON is an advanced document-to-JSON converter primarily built for extracting data from documents (like PDFs, DOCX, TEXT) into structured JSON for AI training data. It uses multiple engines (PyMuPDF, pdfplumber) and provides Mistral AI and Tesseract integration for handling images and diagrams within documents.

## Local Setup & Activation

Follow these steps to run the Doc2JSON project locally on your machine.

### Prerequisites

1. **Python 3.8+**: The one-click launcher will try to install Python automatically if it's missing (Windows: `winget`, Mac: Homebrew, Linux: distro package manager). If auto-install is unavailable, install Python manually from [python.org](https://www.python.org/downloads/).
2. **Tesseract OCR (Optional but Recommended)**: Since this application features image OCR using Tesseract as a local fallback, you will need the actual Tesseract binary installed. 
   - **Windows**: Download the installer from the [UB-Mannheim repository](https://github.com/UB-Mannheim/tesseract/wiki).
   - **Mac**: Run `brew install tesseract`.
   - **Linux**: Run `sudo apt-get install tesseract-ocr`.

### Follow These Steps

**1. Clone the repository** (if you haven't recently):
```bash
git clone https://github.com/PrincePatel-027/Doc2JSON.git
cd Doc2JSON
```

**2. One-click start (recommended)**:
After cloning, just run the starter script. It will automatically install Python (if missing), create `.venv`, install dependencies, start the backend, and open the frontend in your browser.

- **Windows**: Double-click `doc2json-windows.bat`
- **Mac**: Double-click `doc2json-mac.command`
- **Linux**:
  ```bash
  ./doc2json-linux.sh
  ```

**3. Manual setup (alternative)**:
If you prefer doing setup manually, use the steps below.

**4. Create a Virtual Environment**:
It is strongly recommended to use a virtual environment so the app's dependencies remain isolated.
```bash
python -m venv .venv
```

**5. Activate the Virtual Environment**:
- **Windows (Command Prompt / PowerShell)**:
  ```powershell
  .venv\Scripts\activate
  ```
- **Mac/Linux**:
  ```bash
  source .venv/bin/activate
  ```

**6. Install Dependencies**:
With the virtual environment active, install the required packages using pip:
```bash
pip install -r requirements.txt
```

**7. (Optional) Set up Environment Variables**:
If you plan to use Mistral AI for sophisticated OCR or OpenAI for structural validation, you will need to provide API keys as environment variables before running:
```bash
# Windows
set MISTRAL_API_KEY=your_mistral_key_here
set OPENAI_API_KEY=your_openai_key_here

# Mac/Linux
export MISTRAL_API_KEY="your_mistral_key_here"
export OPENAI_API_KEY="your_openai_key_here"
```

**8. Run the Application**:
Launch the built-in Flask development server using python:
```bash
python app.py
```

**9. Open the App in Your Browser**:
Once the terminal outputs that the application is running, open a web browser and go to your localhost:
[http://localhost:5000](http://localhost:5000)

---

### Troubleshooting

- **Mistral/OpenAI Keys**: If certain API-reliant processes are skipped or failing, verify you have the credentials set correctly in your environment. Mistral OCR processes full PDFs and images optimally when activated.
- **OCR Engine Not Found**: If you see missing `tesseract` fallback errors, make sure the actual Tesseract OCR engine software is installed on your OS and the executable path is accessible via your system's PATH variables.
