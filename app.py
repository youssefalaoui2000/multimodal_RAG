import spaces
import gradio as gr
import pdfplumber
import pandas as pd
import uuid

import torch
import torch.nn as nn

from PIL import Image
import os
from rag import ChatbotRAG

# Chat bot


chatbot_rag = ChatbotRAG()


# Interface Gradio

with gr.Blocks() as demo:
    
    gr.Markdown("<h1><center>Chatbot RAG pour Documents PDF</center></h1>")

    with gr.Tab("Télécharger des PDF"):
        pdf_files = gr.File(label="Téléchargez un ou plusieurs PDF", file_types=['.pdf'], file_count="multiple")
        upload_button = gr.Button("Créer le Retriever")
        status = gr.Textbox(label="Statut", interactive=False)

    with gr.Tab("Chatbot"):
        chatbot = gr.Chatbot()
        question = gr.Textbox(label="Votre question", placeholder="Entrez votre question ici...")
        ask_button = gr.Button("Poser la question")
    
    def create_retriever(files):
        pdf_paths = []
        for file in files:
            pdf_path = file.name
            pdf_paths.append(pdf_path)
        chatbot_rag.process_pdf_and_create_retriever(pdf_paths)
        status_text = "Retriever créé avec succès."
        return status_text
    
    def chat(question_text, chat_history):
        answer = chatbot_rag.answer_with_rag(question_text)
        chat_history.append((question_text, answer))
        return "", chat_history
    
    upload_button.click(
        fn=create_retriever,
        inputs=[pdf_files],
        outputs=[status]
    )

    ask_button.click(
        fn=chat,
        inputs=[question, chatbot],
        outputs=[question, chatbot]
    )

if __name__ == "__main__":
    demo.launch()
