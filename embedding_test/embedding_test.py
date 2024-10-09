# 필요한 패키지 설치:
# pip install --upgrade langchain langchain-openai langchain-community langchain-huggingface langchain-cohere pydantic pandas openpyxl faiss-cpu openai sentence-transformers cohere spacy python-dotenv

import time
import tkinter as tk
from tkinter import filedialog
from typing import List, Tuple
import pandas as pd
from dotenv import load_dotenv
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_community.embeddings import SpacyEmbeddings

# .env 파일 로드
load_dotenv()

# API 키 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

def load_excel_file(file_path: str) -> List[Document]:
    df = pd.read_excel(file_path)
    documents = []
    for _, row in df.iterrows():
        content = " ".join(str(cell) for cell in row)
        documents.append(Document(page_content=content, metadata={"source": file_path}))
    return documents

def load_text_file(file_path: str) -> List[Document]:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return [Document(page_content=content, metadata={"source": file_path})]

def load_documents(excel_path: str, txt_path1: str, txt_path2: str) -> List[Document]:
    documents = []
    documents.extend(load_excel_file(excel_path))
    documents.extend(load_text_file(txt_path1))
    documents.extend(load_text_file(txt_path2))
    return documents

def create_vectorstore(documents: List[Document], embedding_function) -> FAISS:
    return FAISS.from_documents(documents=documents, embedding=embedding_function)

def search_vectorstore(vectorstore: FAISS, query: str, k: int = 3) -> List[Tuple[Document, float]]:
    return vectorstore.similarity_search_with_score(query, k=k)

def test_embedding_function(name: str, embedding_function, documents: List[Document], query: str):
    print(f"\nTesting {name}")
    start_time = time.time()
    
    vectorstore = create_vectorstore(documents, embedding_function)
    creation_time = time.time() - start_time
    print(f"Vector store creation time: {creation_time:.2f} seconds")
    
    start_time = time.time()
    results = search_vectorstore(vectorstore, query)
    search_time = time.time() - start_time
    print(f"Search time: {search_time:.2f} seconds")
    
    print("Top 3 results:")
    for doc, score in results:
        print(f"- {doc.page_content[:100]}... (Score: {score:.4f})")

def select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return file_path

def main():
    excel_path = select_file("Select Excel file", [("Excel files", "*.xlsx")])
    txt_path1 = select_file("Select first text file", [("Text files", "*.txt")])
    txt_path2 = select_file("Select second text file", [("Text files", "*.txt")])

    if not all([excel_path, txt_path1, txt_path2]):
        print("File selection cancelled.")
        return

    documents = load_documents(excel_path, txt_path1, txt_path2)
    query = input("Enter your search query: ")

    embedding_functions = [
        ("OpenAI", OpenAIEmbeddings(openai_api_key=openai_api_key)),
        ("HuggingFace", HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")),
        ("Cohere", CohereEmbeddings(model="embed-multilingual-v2.0", cohere_api_key=cohere_api_key)),
        ("Spacy", SpacyEmbeddings())
    ]

    for name, embedding_function in embedding_functions:
        try:
            test_embedding_function(name, embedding_function, documents, query)
        except Exception as e:
            print(f"Error testing {name}: {str(e)}")

if __name__ == "__main__":
    main()