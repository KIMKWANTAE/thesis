import chromadb
import os
import sys
import faiss
import numpy as np
import time
import openai
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
import hashlib

# .env 파일 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# OpenAI 임베딩 함수
def get_embedding(text: str) -> List[float]:
    result = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return result.data[0].embedding

# 문서 임베딩 함수
def get_embeddings(texts: List[str]) -> List[List[float]]:
    return [get_embedding(text) for text in texts]

# 파일 읽기 함수 (기존과 동일)
def read_files_from_directory(directory):
    documents = []
    ids = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read().strip()
                documents.append(content)
                ids.append(filename)
    return documents, ids

# 샘플 파일 디렉토리 설정
sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_files")
if not os.path.exists(sample_dir):
    print(f"오류: {sample_dir} 디렉토리가 존재하지 않습니다.")
    sys.exit(1)

# 문서 읽기
documents, ids = read_files_from_directory(sample_dir)

# 미리 정의된 질문
predefined_query = "강백호가 강사장에게 물건을 어디에 두라고 했어?"

# 문서 임베딩
embeddings = get_embeddings(documents)

# ChromaDB 설정 및 검색 함수
def setup_and_search_chroma():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="sample_documents")
    collection.add(documents=documents, ids=ids, embeddings=embeddings)
    
    query = predefined_query
    query_embedding = get_embedding(query)
    start_time = time.time()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    end_time = time.time()
    
    print("\nChromaDB 검색 결과:")
    for id, distance in zip(results['ids'][0], results['distances'][0]):
        print(f"문서: {id}, 거리: {distance}")
    print(f"검색 시간: {end_time - start_time:.4f}초")
    
    # 상세 인덱스 통계 출력
    print("\n상세 인덱스 통계:")
    print(f"총 벡터 수: {collection.count()}")
    
    chroma_client.delete_collection("sample_documents")

# Faiss 설정 및 검색 함수
def setup_and_search_faiss():
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    query = predefined_query
    query_vector = get_embedding(query)
    
    start_time = time.time()
    D, I = index.search(np.array([query_vector]).astype('float32'), k=5)
    end_time = time.time()
    
    print("\nFaiss 검색 결과:")
    for idx, distance in zip(I[0], D[0]):
        print(f"문서: {ids[idx]}, 거리: {distance}")
    print(f"검색 시간: {end_time - start_time:.4f}초")
    
    # 상세 인덱스 통계 출력
    print("\n상세 인덱스 통계:")
    print(f"총 벡터 수: {index.ntotal}")

# 메인 실행
if __name__ == "__main__":
    setup_and_search_chroma()
    setup_and_search_faiss()
