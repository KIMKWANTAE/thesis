from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import time
from typing import List
import hashlib

# Load environment variables
load_dotenv()

# Initialize OpenAI and Pinecone
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# OpenAI embedding function
def get_embedding(text: str) -> List[float]:
    result = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return result.data[0].embedding

# Read files from directory
def read_files_from_directory(directory):
    documents = []
    ids = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read().strip()
                documents.append(content)
                # 파일 이름을 해시로 변환하여 ASCII ID 생성
                ascii_id = hashlib.md5(filename.encode('utf-8')).hexdigest()
                ids.append(ascii_id)
                filenames.append(filename)
    return documents, ids, filenames

# Set up Pinecone index
index_name = "langchain-demo"
index = pc.Index(index_name)

# Read documents from sample_files
sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_files")
documents, ids, filenames = read_files_from_directory(sample_dir)

# Embed and upsert documents
vectors = []
for doc, id, filename in zip(documents, ids, filenames):
    embedding = get_embedding(doc)
    vectors.append((id, embedding, {"text": doc, "original_filename": filename}))

upsert_response = index.upsert(vectors=vectors, namespace="sample_docs")
print("Upsert response:", upsert_response)

# Wait for a moment
time.sleep(5)

# Check index stats
print("Index stats after upserting:", index.describe_index_stats())

# Query
query = "강백호가 강사장에게 물건을 어디에 두라고 했어?"
query_embedding = get_embedding(query)

start_time = time.time()
response = index.query(
    vector=query_embedding,
    top_k=5,
    namespace="sample_docs",
    include_metadata=True
)
end_time = time.time()

print("\nPinecone 검색 결과:")
if response['matches']:
    for match in response['matches']:
        print(f"문서: {match['metadata'].get('original_filename', match['id'])}, 거리: {1 - match['score']}")
else:
    print("No matches found.")
print(f"검색 시간: {end_time - start_time:.4f}초")

stats = index.describe_index_stats()
print("\n상세 인덱스 통계:")
print(f"총 벡터 수: {stats['total_vector_count']}")
print("네임스페이스:")
for ns, ns_stats in stats['namespaces'].items():
    print(f"  {ns}: {ns_stats['vector_count']} 벡터")
