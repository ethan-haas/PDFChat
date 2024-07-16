# PDF Chat
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wToEH__f8vQOhSnMHm1rSC5TlnB8eyX6?usp=sharing)


## Overview
**PDF Chat** is an advanced tool designed to interactively query and analyze the contents of PDF documents using the power of OpenAI's models. It processes PDFs, extracts and indexes their text, and provides an intuitive interface for users to ask questions about the document contents. This application is especially useful for research, legal documents, academic papers, and any other scenario where in-depth document analysis is required.


### Key Features
- **Text Extraction and Chunking**: Efficiently extracts text from PDFs and splits them into manageable chunks for processing.
- **Similarity Search**: Utilizes [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss) for quick and efficient similarity search on document chunks.
- **AI-Powered Querying**: Integrates with OpenAI's GPT-3.5-Turbo, GPT-4o, GPT-4-Turbo models to provide accurate and context-aware answers to user queries based on the document content.
- **User-Friendly Interface**: Leverages Gradio to offer a clean and easy-to-use web interface for interacting with the system.
- **Settings Management**: Supports loading and saving configuration settings, including the OpenAI API key and model preferences.
- **History and Visualization**: Maintains a history of questions and answers, and generates word clouds from the question history to visualize frequently asked topics.


## Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/ethan-haas/PDFChat.git
    cd PDFChat
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Application:**
    ```bash
    python PDFChat.py
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
