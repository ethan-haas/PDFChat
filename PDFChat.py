import os
import json
import PyPDF2
from openai import OpenAI
import faiss
import numpy as np
from typing import List, Dict
import sqlite3
import time
import concurrent.futures
from colorama import init, Fore, Style
import gradio as gr
import webbrowser
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import threading

class PDFChat:
    def __init__(self):
        self.pdf_files = []
        self.chunks: List[Dict[str, str]] = []
        self.index = None
        self.embeddings = []
        self.dimension = None
        self.load_settings()
        self.stop_query_event = threading.Event()
        self.client = None  # We'll initialize this later when we have the API key

    def load_settings(self):
        default_settings = {
            'model': "gpt-4o",
            'chunk_size': 1000,
            'default_prompt': """Provide a clear and well-sourced answer to the question, utilizing the context provided. Following each statement, appropriately cite the source(s) used by including them in square brackets. If a statement draws from multiple sources, cite each of them for comprehensive referencing. Format your answer as per the following guidelines:

- Statement 1. [Source: Document: X, Page: Y]
- Statement 2. [Source: Document: A, Page: B] [Source: Document: C, Page: D]
- Continue using this formatting for subsequent details.""",
            'system_content': "You are a precision-driven assistant tasked with delivering answers based on specific contexts. With each piece of information you provide, ensure that you include accurate and clear source references. Each part of your response should identify the document and page number, affirming the reliability and relevance of the sources to the query.",
            'api_key': ""
        }
        
        if os.path.exists('settings.json'):
            with open('settings.json', 'r') as f:
                saved_settings = json.load(f)
            # Update default settings with saved settings
            default_settings.update(saved_settings)
        
        self.model = default_settings['model']
        self.chunk_size = default_settings['chunk_size']
        self.default_prompt = default_settings['default_prompt']
        self.system_content = default_settings['system_content']
        self.api_key = default_settings['api_key']

        # Initialize the OpenAI client if API key is available
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def save_settings(self):
        settings = {
            'model': self.model,
            'chunk_size': self.chunk_size,
            'default_prompt': self.default_prompt,
            'system_content': self.system_content,
            'api_key': self.api_key
        }
        with open('settings.json', 'w') as f:
            json.dump(settings, f)

    def set_pdf_files(self, files: List[str]):
        self.pdf_files = files
        self.chunks = []
        self.index = None
        self.embeddings = []
        self.dimension = None

    def process_pdfs(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for filepath in self.pdf_files:
                futures.append(executor.submit(self.extract_text_from_pdf, filepath))
            concurrent.futures.wait(futures)

    def extract_text_from_pdf(self, filepath: str):
        local_chunks = []
        try:
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(reader.pages, 1):
                    if self.stop_query_event.is_set():
                        return
                    text = page.extract_text()
                    chunks = self.create_chunks(text, self.chunk_size)
                    for chunk in chunks:
                        if self.stop_query_event.is_set():
                            return
                        local_chunks.append({
                            'text': chunk,
                            'document': os.path.basename(filepath),
                            'page': page_num
                        })
        except Exception as e:
            print(Fore.RED + f"[Error] {str(e)}" + Style.RESET_ALL)
        self.chunks.extend(local_chunks)

    @staticmethod
    def create_chunks(text: str, chunk_size: int) -> List[str]:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    def create_index(self):
        if not self.chunks:
            self.process_pdfs()

        if self.chunks:
            self.embeddings = self.get_embeddings([chunk['text'] for chunk in self.chunks])
            self.dimension = len(self.embeddings[0])
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.array(self.embeddings).astype('float32'))

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.client:
            raise ValueError("API key is not set. Please add your OpenAI API key in the Settings tab.")

        batch_size = 100
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(input=batch, model="text-embedding-3-small")
            all_embeddings.extend([embedding.embedding for embedding in response.data])
        return all_embeddings

    def query(self, question: str) -> str:
        if not self.api_key:
            return "API key is not set. Please add your OpenAI API key in the Settings tab."

        if self.client is None:
            self.client = OpenAI(api_key=self.api_key)

        if self.stop_query_event.is_set():
            return "Query stopped by user."

        if self.index is None or not self.embeddings:
            self.create_index()

        question_embedding = self.get_embeddings([question])[0]

        if self.stop_query_event.is_set():
            return "Query stopped by user."

        _, indices = self.index.search(np.array([question_embedding]).astype('float32'), k=5)

        relevant_chunks = [self.chunks[i] for i in indices[0]]
        context = "\n".join([f"{chunk['text']} [Source: Document: {chunk['document']}, Page: {chunk['page']}]" for chunk in relevant_chunks])

        prompt = f"""Question: {question}
Context: {context}
{self.default_prompt}"""

        if self.stop_query_event.is_set():
            return "Query stopped by user."

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_content},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    def update_settings(self, model, chunk_size, default_prompt, system_content, api_key):
        self.model = model
        self.chunk_size = chunk_size
        self.default_prompt = default_prompt
        self.system_content = system_content
        self.api_key = api_key
        self.save_settings()
        # Reset the index and chunks when settings change
        self.chunks = []
        self.index = None
        self.embeddings = []
        self.dimension = None
        # Update the client with the new API key
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None


qa_system = PDFChat()
current_query_thread = None


def init_db():
    with sqlite3.connect('qa_history.db') as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS qa_pairs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      question TEXT,
                      answer TEXT,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')


def add_qa_pair(question, answer):
    with sqlite3.connect('qa_history.db') as conn:
        conn.execute("INSERT INTO qa_pairs (question, answer) VALUES (?, ?)", (question, answer))


def get_qa_history():
    with sqlite3.connect('qa_history.db') as conn:
        return conn.execute("SELECT question, answer FROM qa_pairs ORDER BY timestamp DESC LIMIT 100").fetchall()


def clear_qa_history():
    with sqlite3.connect('qa_history.db') as conn:
        conn.execute("DELETE FROM qa_pairs")
    return "History cleared successfully."


def set_pdf_files(files):
    if not files:
        return "No files selected. Please choose PDF files to analyze."

    pdf_files = [f for f in files if f.name.lower().endswith('.pdf')]
    if not pdf_files:
        return "No PDF files found in the selection. Please choose PDF files to analyze."

    qa_system.set_pdf_files([f.name for f in pdf_files])
    return f"PDF files set successfully. Selected {len(pdf_files)} PDF files."


def query_pdf(question: str, progress=gr.Progress()):
    if not qa_system.pdf_files:
        return "Please select PDF files first."

    qa_system.stop_query_event.clear()

    progress(0.1, "Processing query...")
    answer = qa_system.query(question)
    progress(0.9, "Finalizing answer...")

    if not qa_system.stop_query_event.is_set():
        add_qa_pair(question, answer)
        progress(1.0, "Query completed.")
        return answer
    else:
        return "Query stopped by user."


def stop_current_query():
    qa_system.stop_query_event.set()
    return "Query stopping..."


def load_history():
    history = get_qa_history()
    # Convert history to a list of lists for DataFrame
    history_data = [[q, a] for q, a in history]
    return history_data


def save_history_to_csv():
    history = get_qa_history()
    df = pd.DataFrame(history, columns=["Question", "Answer"])
    df.to_csv("qa_history.csv", index=False)
    return "History exported to qa_history.csv"


def generate_wordcloud():
    all_text = " ".join([q for q, _ in get_qa_history()])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig("wordcloud.png")
    return "Word Cloud generated and saved as wordcloud.png"


def save_and_reload_settings(model, chunk_size, default_prompt, system_content, api_key):
    qa_system.update_settings(model, int(chunk_size), default_prompt, system_content, api_key)
    return "Settings saved and updated successfully.", qa_system.model, qa_system.chunk_size, qa_system.default_prompt, qa_system.system_content, qa_system.api_key


def load_current_settings():
    return qa_system.model, qa_system.chunk_size, qa_system.default_prompt, qa_system.system_content, qa_system.api_key


def main():
    init(autoreset=True)  # Initialize colorama with autoreset
    init_db()

    # Define Gradio interface
    with gr.Blocks(title="PDF Chat") as demo:
        gr.Markdown("# PDF Chat")

        with gr.Tabs():
            with gr.TabItem("Home"):
                with gr.Row():
                    with gr.Column(scale=2):
                        pdf_files = gr.File(label="PDF Files", file_count="multiple", type="filepath")
                        file_status = gr.Markdown("No files selected. Please choose PDF files to analyze.")
                    with gr.Column(scale=3):
                        question_input = gr.Textbox(label="Enter your question", placeholder="Type your question here...")
                        with gr.Row():
                            ask_button = gr.Button("Ask", variant="primary")
                            stop_button = gr.Button("Stop")

                answer_output = gr.Markdown(label="Answer")

                # Instructions
                with gr.Accordion("Instructions", open=False):
                    gr.Markdown("""
                    1. Upload one or more PDF files.
                    2. Once valid PDF files are selected, you can ask questions about their contents.
                    3. Type your question in the input box and click 'Ask' to get an answer.
                    4. If a query is taking too long, you can click 'Stop' to interrupt it.

                    Note: You must select at least one PDF file before asking questions.
                    """)

            with gr.TabItem("Question History"):
                history_output = gr.DataFrame(
                    headers=["Question", "Answer"],
                    datatype=["str", "str"],
                    interactive=False
                )
                with gr.Row():
                    refresh_history_button = gr.Button("Refresh History")
                    export_history_button = gr.Button("Export History to CSV")
                    clear_history_button = gr.Button("Clear History", variant="stop")  # Red button

            with gr.TabItem("Settings"):
                model_dropdown = gr.Dropdown(
                    choices=["gpt-3.5-turbo","gpt-4o-mini","gpt-4o","gpt-4","gpt-4-turbo"],
                    label="Select Model",
                    value=qa_system.model
                )
                chunk_size_slider = gr.Slider(
                    minimum=500,
                    maximum=10000,
                    step=100,
                    label="Chunk Size",
                    value=qa_system.chunk_size
                )
                default_prompt_input = gr.Textbox(
                    label="Default Prompt",
                    value=qa_system.default_prompt,
                    lines=3
                )
                system_content_input = gr.Textbox(
                    label="System Content",
                    value=qa_system.system_content,
                    lines=3
                )
                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    value=qa_system.api_key,
                    type="password"
                )
                save_settings_button = gr.Button("Save Settings")
                settings_status = gr.Markdown()

            with gr.TabItem("Visualization"):
                with gr.Row():
                    wordcloud_button = gr.Button("Generate Word Cloud")
                    save_wordcloud_status = gr.Markdown()

        # Set up event handlers
        pdf_files.change(
            set_pdf_files,
            inputs=[pdf_files],
            outputs=[file_status]
        )

        ask_button.click(
            query_pdf,
            inputs=[question_input],
            outputs=[answer_output]
        )

        stop_button.click(
            stop_current_query,
            outputs=[answer_output]
        )

        refresh_history_button.click(
            load_history,
            outputs=[history_output]
        )

        export_history_button.click(
            save_history_to_csv,
            outputs=[]
        )

        clear_history_button.click(
            clear_qa_history,
            outputs=[clear_history_button]
        )

        wordcloud_button.click(
            generate_wordcloud,
            outputs=[save_wordcloud_status]
        )

        save_settings_button.click(
            save_and_reload_settings,
            inputs=[model_dropdown, chunk_size_slider, default_prompt_input, system_content_input, api_key_input],
            outputs=[settings_status, model_dropdown, chunk_size_slider, default_prompt_input, system_content_input, api_key_input]
        )

        # Add an event handler to ensure settings are loaded initially
        demo.load(
            load_current_settings,
            outputs=[model_dropdown, chunk_size_slider, default_prompt_input, system_content_input, api_key_input]
        )

    # Launch Gradio interface
    webbrowser.open('http://127.0.0.1:7860')
    demo.launch(share=True)


if __name__ == '__main__':
    main()
