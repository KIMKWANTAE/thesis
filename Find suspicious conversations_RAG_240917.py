from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import re

load_dotenv()

def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                doc = Document(page_content=content, metadata={"source": filename})
                documents.append(doc)
    return documents

def preprocess_text(text):
    # 따옴표로 둘러싸인 단어 강조
    text = re.sub(r"'(\w+)'", r"SUSPICIOUS_TERM:\1", text)
    return text

def analyze_suspicion(text):
    suspicious_patterns = [
        (r"SUSPICIOUS_TERM:\w+", 2),  # 따옴표로 둘러싸인 단어
        (r"\b(정원|꽃|화분|비료|영양제)\b", 1),  # 의심 키워드
        (r"\b(다음 주|월요일|아무도 모르게)\b", 1.5),  # 시간 및 은밀함 관련 표현
    ]
    
    suspicion_score = 0
    for pattern, weight in suspicious_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        suspicion_score += len(matches) * weight
    
    return suspicion_score

folder_path = "D:\Python test\RAG\data"
docs = load_documents_from_folder(folder_path)

# 텍스트 전처리 및 의심 점수 계산
preprocessed_docs = []
for doc in docs:
    preprocessed_text = preprocess_text(doc.page_content)
    suspicion_score = analyze_suspicion(preprocessed_text)
    preprocessed_docs.append(Document(
        page_content=preprocessed_text,
        metadata={"source": doc.metadata["source"], "suspicion_score": suspicion_score}
    ))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(preprocessed_docs)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

retriever = vectorstore.as_retriever()

prompt = PromptTemplate.from_template(
    """You are a forensic analyst specializing in detecting suspicious conversations and potential confidential information leaks.
Analyze the following conversation carefully, paying attention to:
1. Unusual or out-of-context terms (marked as SUSPICIOUS_TERM:)
2. Patterns that might indicate coded language or hidden meanings
3. Discussions about meetings, exchanges, or activities that seem secretive

Use the provided context and your analysis to answer the question.
If you detect suspicious activity or potential confidential information leaks, explain your reasoning and mention the filename where this information was found.
Consider the suspicion score provided, but use your own judgment as well.
Answer in Korean.

Question: {question}
Context: {context}
Answer: """
)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "폴더 내의 파일 중에서 기밀 거래나 의심스러운 대화가 있는 파일을 찾아주고, 왜 그렇게 판단했는지 설명해줘."
response = chain.invoke(question)
print(response)

# 의심 점수가 가장 높은 문서 출력
sorted_docs = sorted(preprocessed_docs, key=lambda x: x.metadata["suspicion_score"], reverse=True)
print("\n가장 의심스러운 파일:")
for doc in sorted_docs[:3]:  # 상위 3개 출력
    print(f"파일명: {doc.metadata['source']}, 의심 점수: {doc.metadata['suspicion_score']:.2f}")