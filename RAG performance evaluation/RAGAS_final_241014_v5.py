# 필요한 라이브러리 임포트
import nltk
nltk.download('punkt', quiet=True)

import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from sentence_transformers import SentenceTransformer
import re
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 환경 변수에서 OpenAI API 키 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAI 모델 초기화 (GPT-4 사용)
# 참고: 'gpt-4o-mini'는 실제 모델명이 아닙니다. 실제 사용 가능한 모델명으로 변경해야 합니다.
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# 모든 문서를 로드하는 함수
def load_all_documents():
    folder_path = f"D:\Python test\RAG\data_all_files"
    documents = []
    file_info = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                documents.append(Document(page_content=content, metadata={"source": filename}))
                file_info.append({"filename": filename, "type": "txt", "content_preview": content[:100]})
        elif filename.endswith(".xlsx"):
            xls = pd.ExcelFile(file_path)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                content = f"Sheet: {sheet_name}\n"
                for index, row in df.iterrows():
                    content += f"Row {index + 1}:\n"
                    for column, value in row.items():
                        content += f"  {column}: {value}\n"
                documents.append(Document(page_content=content, metadata={"source": f"{filename} - {sheet_name}"}))
                file_info.append({
                    "filename": filename,
                    "type": "xlsx",
                    "sheet": sheet_name,
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "content_preview": content[:200]  # 미리보기 길이를 늘림
                })
    return documents, file_info

# 선택된 파일만 로드하는 함수
def load_documents(selected_files=None):
    all_documents, _ = load_all_documents()
    if selected_files is None:
        return all_documents
    else:
        return [doc for doc in all_documents if doc.metadata["source"] in selected_files]

# 텍스트 전처리 함수
def preprocess_text(text):
    # 소문자 변환, 특수 문자 제거, 여러 공백을 하나로 치환
    text = re.sub(r'[^\w\s]', '', text.lower())
    return re.sub(r'\s+', ' ', text).strip()

# Precision@k 계산 함수
def improved_precision_at_k(relevant_docs, retrieved_docs, k):
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = [doc for doc in retrieved_k if doc in relevant_docs]
    return len(relevant_retrieved) / len(retrieved_k) if retrieved_k else 0

# Recall@k 계산 함수
def improved_recall_at_k(relevant_docs, retrieved_docs, k):
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = [doc for doc in retrieved_k if doc in relevant_docs]
    return len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0

# BLEU 점수 계산 함수
def improved_bleu(reference, candidate):
    reference = preprocess_text(reference).split()
    candidate = preprocess_text(candidate).split()
    smoothie = SmoothingFunction().method1
    return sentence_bleu([reference], candidate, weights=(0.5, 0.5), smoothing_function=smoothie)

# ROUGE 점수 계산 함수
def improved_rouge(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(preprocess_text(candidate), preprocess_text(reference))
    return scores[0]['rouge-l']['f']

# Sentence Transformer 모델 로드
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Sentence Transformer를 이용한 개선된 유사도 계산 함수
def improved_similarity(text1, text2):
    embedding1 = model.encode(text1)
    embedding2 = model.encode(text2)
    return cosine_similarity([embedding1], [embedding2])[0][0]

# 메인 실행 코드
# 전체 폴더의 문서 로드
documents, file_info = load_all_documents()

print("\n" + "="*50)
print("로드된 문서 정보")
print("="*50)
for info in file_info:
    print(f"\n파일명: {info['filename']}")
    print(f"파일 유형: {info['type']}")
    if info['type'] == 'xlsx':
        print(f"시트명: {info['sheet']}")
        print(f"데이터 형태: {info['shape']}")
        print(f"컬럼: {info['columns']}")
    print(f"내용 미리보기: {info['content_preview']}")
print("\n" + "="*50)

# 범죄사실 파일 경로
crime_fact_path = "D:/Python test/RAG/범죄사실2.txt"

# 범죄사실 파일 로드
with open(crime_fact_path, 'r', encoding='utf-8') as file:
    crime_fact = file.read()

# 벡터 저장소 생성
documents, _ = load_all_documents()  # 이 부분을 수정
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 상위 3개 문서 검색

# RAG 프롬프트 설정
prompt = PromptTemplate.from_template(
    """당신은 범죄 수사를 위한 전문 분석가입니다. 
    주어진 컨텍스트와 범죄사실을 바탕으로 관련 사건을 분석하고 답변하세요.

    범죄사실:
    {crime_fact}

    질문: {question} 
    컨텍스트: {context} 
    
    답변 작성 지침:
    1. 질문에 직접적으로 관련된 정보만 포함하세요.
    2. 컨텍스트에 명시적으로 언급된 사실만 사용하세요.
    3. 추측이나 가정을 피하고, 확실한 정보만 제공하세요.
    4. 질문에서 요구하는 정확한 형식(예: 날짜, 시간, 장소)으로 답변하세요.
    5. 불필요한 설명은 제외하고 간결하게 답변하세요.
    
    답변:"""
)

# RAG 체인 구성
rag_chain = (
    {
        "context": retriever, 
        "question": RunnablePassthrough(), 
        "crime_fact": lambda _: crime_fact
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 평가용 데이터셋 생성
eval_dataset = [
    {
        "question": "강백호는 마약을 거래한 것으로 의심받고 있습니다. 강백호는 마약을 어디에 두라고 했나요? 불필요한 답변은 제외하고 정확한 장소만 써주세요. (예시 : 공주대학교 식당 앞)",
        "ground_truths": ["SNU 도서관 앞 세번째 벤치", "SNU 도서관 앞 세 번째 벤치에"],
    },
    {
        "question": "이소연의 구글 계정 아이디는 무엇인가요?",
        "ground_truths": ["lee@gmail.com", "leegmail.com","lee"],
    },
    {
        "question": "Photos에 있는 IMG009.jpg 파일의 create time은 언제인가요?",
        "ground_truths": ["2024-01-09 18:00", "2024-01-09 6:00:00 PM","2024년 1월 9일 오후 6시"],
    },
    {
        "question": "강백호가 소연이랑 다투고 머리를 밀친 후 백사장에게 도움을 요청한 날짜와 시간은 언제인가요? 불필요한 답변은 제외하고 정확한 날짜와 시간만 써주세요. (예시 : 2023.1.1. 12:15)",
        "ground_truths": ["2024.1.4. 23:55", "2024년 1월 4일 23시 55분", "2024년 1월 4일 23:55"],
    },
]

# 결과를 저장할 리스트
results = []

# 각 질문에 대해 RAG 시스템 실행 및 결과 저장
for item in eval_dataset:
    question = item["question"]
    contexts = retriever.get_relevant_documents(question)
    answer = rag_chain.invoke(question)
    
    results.append({
        "question": question,
        "contexts": [doc.page_content for doc in contexts],
        "retrieved_docs": [doc.metadata["source"] for doc in contexts],
        "answer": answer,
        "ground_truths": item["ground_truths"]
    })

# 추가 평가 메트릭 계산
additional_metrics = {
    'precision@k': [],
    'recall@k': [],
    'bleu': [],
    'rouge': [],
    'similarity': []
}

for result in results:
    answer = result['answer']
    question = result['question']
    ground_truths = result['ground_truths']
    retrieved_docs = result['retrieved_docs']
    
    # 첫 번째 검색된 문서만 관련 있다고 간주
    relevant_docs = [retrieved_docs[0]] if retrieved_docs else []
    
    max_scores = {metric: 0 for metric in additional_metrics}
    
    for gt in ground_truths:
        scores = {
            'precision@k': improved_precision_at_k(relevant_docs, retrieved_docs, 1),
            'recall@k': improved_recall_at_k(relevant_docs, retrieved_docs, 1),
            'bleu': improved_bleu(gt, answer),
            'rouge': improved_rouge(gt, answer),
            'similarity': improved_similarity(gt, answer)
        }
        
        for metric, score in scores.items():
            max_scores[metric] = max(max_scores[metric], score)
    
    for metric, score in max_scores.items():
        additional_metrics[metric].append(score)

# RAGAS 평가 실행
ragas_results = evaluate(
    dataset=Dataset.from_dict({
        "question": [item["question"] for item in eval_dataset],
        "answer": [result["answer"] for result in results],
        "contexts": [result["contexts"] for result in results],
        "ground_truths": [item["ground_truths"] for item in eval_dataset]
    }),
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
    llm=llm
)

# 결과 출력
print("\n" + "="*50)
print("RAG 시스템 성능 평가 보고서")
print("="*50 + "\n")

# 각 질문에 대한 상세 결과 출력
for i, result in enumerate(results, 1):
    print(f"질문 {i}: {result['question']}")
    print(f"정답: {', '.join(result['ground_truths'])}")
    print(f"RAG 답변: {result['answer']}")
    print("\n검색된 문서:")
    for j, doc in enumerate(result['retrieved_docs'], 1):  # 모든 검색된 문서 출력
        print(f"  문서 {j}: {doc}")
    print("\n개별 메트릭 점수:")
    for metric in additional_metrics:
        print(f"  {metric}: {additional_metrics[metric][i-1]:.4f}")
    print("\n" + "-"*50 + "\n")

# RAGAS 평가 결과 출력
print("RAGAS 평가 결과:")
for metric, score in ragas_results.items():
    print(f"{metric}: {score:.4f}")

# 추가 평가 메트릭 평균 결과 출력
print("\n평균 추가 평가 메트릭 결과:")
for metric, scores in additional_metrics.items():
    print(f"{metric}: {np.mean(scores):.4f}")

# 문서 관련성 디버깅 정보 출력
print("\n문서 관련성 디버깅:")
for i, result in enumerate(results, 1):
    print(f"\n질문 {i}: {result['question']}")
    print(f"검색된 문서 수: {len(result['retrieved_docs'])}")
    relevant_docs = []
    print("검색된 문서:")
    for j, doc in enumerate(result['retrieved_docs'], 1):
        similarity_question = fuzz.partial_ratio(doc.lower(), preprocess_text(result['question']))
        similarity_truth = max([fuzz.partial_ratio(preprocess_text(truth), doc.lower()) for truth in result['ground_truths']])
        # 질문 유사도나 정답 유사도가 70% 이상이면 관련 있다고 판단
        is_relevant = similarity_question >= 70 or similarity_truth >= 70
        relevance = "관련" if is_relevant else "무관"
        if is_relevant:
            relevant_docs.append(doc)
        print(f"  문서 {j}: {relevance} (질문 유사도: {similarity_question}%, 정답 유사도: {similarity_truth}%) - {doc}")
    print(f"관련 있다고 판단된 문서 수: {len(relevant_docs)}")

# BLEU, ROUGE, similarity 점수 상세 분석
print("\nBLEU, ROUGE, similarity 점수 상세 분석:")
for i, result in enumerate(results, 1):
    print(f"\n질문 {i}:")
    answer = result['answer']
    ground_truths = result['ground_truths']
    
    print(f"RAG 답변: {answer}")
    print(f"정답: {', '.join(ground_truths)}")
    
    bleu_scores = [improved_bleu(gt, answer) for gt in ground_truths]
    rouge_scores = [improved_rouge(gt, answer) for gt in ground_truths]
    similarity_scores = [improved_similarity(gt, answer) for gt in ground_truths]
    
    print(f"BLEU 점수: {max(bleu_scores):.4f}")
    print(f"ROUGE 점수: {max(rouge_scores):.4f}")
    print(f"Similarity 점수: {max(similarity_scores):.4f}")
    
    print("단어 단위 비교:")
    gt_words = set(preprocess_text(ground_truths[0]).split())
    answer_words = set(preprocess_text(answer).split())
    common_words = gt_words.intersection(answer_words)
    print(f"정답에 있는 단어 수: {len(gt_words)}")
    print(f"RAG 답변에 있는 단어 수: {len(answer_words)}")
    print(f"공통 단어 수: {len(common_words)}")
    print(f"공통 단어: {', '.join(common_words)}")
    print(f"정답에만 있는 단어: {', '.join(gt_words - answer_words)}")
    print(f"RAG 답변에만 있는 단어: {', '.join(answer_words - gt_words)}")

print("\n평가 완료")

# 평가 메트릭 설명
print("\n평가 메트릭 설명:")
print("RAGAS 메트릭:")
print("1. Context Precision: 생성된 답변에 사용된 컨텍스트의 정확도")
print("2. Faithfulness: 생성된 답변이 제공된 컨텍스트에 충실한 정도")
print("3. Answer Relevancy: 생성된 답변이 질문에 관련된 정도")
print("4. Context Recall: 관련 있는 모든 정보가 검색된 컨텍스트에 포함된 정도")

print("\n추가 메트릭:")
print("5. Precision@k: 검색된 상위 k개 문서 중 관련 있는 문서의 비율")
print("6. Recall@k: 관련 있는 모든 문서 중 검색된 상위 k개 문서에 포함된 비율")
print("7. BLEU: 생성된 답변과 정답 간의 n-gram 유사도")
print("8. ROUGE: 생성된 답변과 정답 간의 단어 중복률")
print("9. Similarity: 생성된 답변과 정답 간의 의미적 유사도 (Sentence Transformer 사용)")

print("\n주의: 이 평가는 제한된 데이터셋과 간소화된 관련성 판단 기준을 사용하였으므로, 실제 성능은 다를 수 있습니다.")