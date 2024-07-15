# PDF Search Engine
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wToEH__f8vQOhSnMHm1rSC5TlnB8eyX6?usp=sharing)

The PDF Search Engine allows users to upload PDF files, extract text from them, and query the extracted content using OpenAI's language models. The extracted content is indexed using FAISS, allowing for efficient similarity searches to answer user questions. The application also provides functionality to save query history, generate word clouds, and manage settings.

## Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/ethan-haas/PDFSearcher.git
    cd PDFSearcher
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Application:**
    ```bash
    python PDFSearcher.py
    ```

2. **Open the Web Interface:**
    - The web interface will automatically open in your default browser. If not, open [http://127.0.0.1:7860](http://127.0.0.1:7860).

3. **Upload PDF Files:**
    - Use the "Home" tab to upload PDF files.
    - Once uploaded, you can ask questions about their contents.

4. **Query PDF Files:**
    - Enter your question in the input box and click "Ask".
    - The answer will be displayed below.

5. **Manage Settings:**
    - Use the "Settings" tab to change the model, chunk size, prompts, and API key.
    - Click "Save Settings" to apply changes.

6. **View and Export Query History:**
    - Use the "Question History" tab to view past queries and their answers.
    - Export history to a CSV file or clear the history as needed.

7. **Generate Word Cloud:**
    - Use the "Visualization" tab to generate a word cloud of the query history.

## File Structure

- `PDFSearcher.py`: Main script to run the application.
- `settings.json`: Configuration file for settings.
- `qa_history.db`: SQLite database for storing query history.
- `requirements.txt`: List of required Python packages.
