import os
import pandas as pd
import json
from typing import List, Dict
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_teddynote import logging as teddynote_logging
import logging
from datetime import datetime

# 환경 변수 로드 및 OpenAI API 초기화
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다")

# 로그 설정
teddynote_logging.langsmith("KKT thesis_murder_v8")

# OpenAI 모델 초기화
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

def load_excel_data_with_types(file_path: str) -> List[Document]:
    """엑셀 파일을 로드하고 각 시트의 데이터를 Document 객체로 변환합니다.
    날짜와 숫자 데이터 타입을 보존합니다."""
    try:
        sheets_dict = pd.read_excel(file_path, sheet_name=None)
        documents = []
        for sheet_name, df in sheets_dict.items():
            # 데이터 타입 정보 저장
            column_types = df.dtypes.apply(lambda x: str(x)).to_dict()
            
            # 날짜 형식 데이터 처리
            for col in df.select_dtypes(include=['datetime64']).columns:
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # NaN 값을 적절히 처리
            df = df.fillna('')
            
            # 데이터를 문자열로 변환하되, 숫자 형식은 유지
            content = f"Sheet: {sheet_name}\n"
            content += df.to_string(index=False, float_format='%.15g')
            
            # 메타데이터에 열 이름과 데이터 타입 정보 추가
            metadata = {
                "source": f"{os.path.basename(file_path)}:{sheet_name}",
                "column_types": json.dumps(column_types)
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        return documents
    except Exception as e:
        logging.error(f"엑셀 파일 '{file_path}' 로드 중 오류 발생: {e}")
        return []

def load_text_files(folder_path: str) -> List[Document]:
    """지정된 폴더에서 모든 텍스트 파일을 읽어 Document 객체로 변환합니다."""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                documents.append(Document(page_content=content, metadata={"source": filename}))
    return documents

def create_vector_store(documents: List[Document]) -> FAISS:
    """문서로부터 벡터 저장소를 생성합니다."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    embedding_function = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_function)
    return vectorstore

# 파일 경로 설정
excel_path = "D:/Python test/RAG/mobile_forensic_data.xlsx"
audio_folder = "D:/Python test/RAG/audio_to_text_files"
image_folder = "D:/Python test/RAG/image_to_text_files"
crime_fact_path = "D:/Python test/RAG/범죄사실2.txt"

# 범죄사실 파일 로드
with open(crime_fact_path, 'r', encoding='utf-8') as file:
    crime_fact = file.read()

# 데이터 로드 및 벡터 저장소 생성
excel_documents = load_excel_data_with_types(excel_path)
audio_documents = load_text_files(audio_folder)
image_documents = load_text_files(image_folder)

all_documents = excel_documents + audio_documents + image_documents
vectorstore = create_vector_store(all_documents)
retriever = vectorstore.as_retriever()

# RAG 프롬프트 설정
prompt = PromptTemplate.from_template(
    """당신은 범죄 수사를 위한 전문 분석가입니다. 
    주어진 컨텍스트와 범죄사실을 바탕으로 관련 사건을 분석하고 답변하세요.
    데이터 타입 정보를 활용하여 날짜와 숫자를 정확히 해석하세요.

    범죄사실:
    {crime_fact}

    질문: {question} 
    컨텍스트: {context} 
    
    답변:"""
)

# RAG 체인 설정
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

# RAG 도구 정의
def rag_analysis_wrapper(*args, **kwargs):
    query = args[0] if args else kwargs.get('query', '')
    return rag_chain.invoke(query)

rag_tool = Tool(
    name="RAG_분석",
    func=rag_analysis_wrapper,
    description="RAG를 사용하여 실제 존재하는 텍스트 파일과 엑셀 파일을 분석하고 질문에 답합니다. 데이터 타입 정보를 활용하여 날짜와 숫자를 정확히 해석합니다."
)

# 에이전트 정의
범죄사실_요약_agent = Agent(
    role='범죄사실 요약 전문가',
    goal='범죄사실 파일을 간결하게 요약하고 보고서 작성',
    backstory='당신은 복잡한 범죄 정보를 명확하고 간결한 요약으로 정리하는 전문가입니다.',
    tools=[rag_tool],
    verbose=True,
    llm=llm
)

살인사건_은폐_탐지_agent = Agent(
    role='살인 은폐 탐정',
    goal='살인을 은폐하려는 시도를 탐지하고 보고서 작성',
    backstory='당신은 살인 은폐를 암시할 수 있는 미묘한 힌트와 행동을 식별하는 전문가입니다.',
    tools=[rag_tool],
    verbose=True,
    llm=llm
)

피해자_조사_agent = Agent(
    role='피해자 정보 분석가',
    goal='대상 파일에서 피해자와 관련된 모든 데이터를 수집하고 분석하여 보고서 작성',
    backstory='당신은 다양한 데이터 소스에서 피해자 관련 정보를 수집하고 분석하는 전문가입니다.',
    tools=[rag_tool],
    verbose=True,
    llm=llm
)

살인사건_시간_조사_agent = Agent(
    role='살인 시간 조사관',
    goal='가용한 데이터를 바탕으로 살인 시간을 추정하고 보고서 작성',
    backstory='당신은 타임라인을 분석하고 범죄 발생 가능 시간을 추정하는 데 능숙한 전문가입니다.',
    tools=[rag_tool],
    verbose=True,
    llm=llm
)

살인사건_장소_조사_agent = Agent(
    role='살인 장소 분석가',
    goal='은폐 시도 이전의 WiFi 정보를 바탕으로 살인 장소를 추정하고 보고서 작성',
    backstory='당신은 디지털 흔적, 특히 WiFi 연결 데이터를 통해 위치를 추론하는 전문가입니다.',
    tools=[rag_tool],
    verbose=True,
    llm=llm
)

# 각 에이전트의 작업 정의
범죄사실_요약_task = Task(
    description="""
    범죄사실 파일을 간결하게 요약하세요. 요약에는 다음 내용이 포함되어야 합니다:
    1. 범죄사실의 핵심 요약 (300단어 이내)
    2. 주요 관련자 목록
    3. 범죄 발생 추정 시간대
    4. 범죄 발생 추정 장소
    5. 기타 주목할 만한 정보

    데이터 분석 시 날짜와 숫자의 정확성에 특히 주의를 기울이세요.
    """,
    agent=범죄사실_요약_agent,
    expected_output="300단어 이내의 범죄사실 요약, 주요 관련자 목록, 추정 시간대와 장소, 주요 정보가 포함된 구조화된 보고서"
)

살인사건_은폐_탐지_task = Task(
    description="""
    모든 파일을 분석하여 살인을 은폐하려는 시도를 탐지하세요. 보고서에는 다음 내용이 포함되어야 합니다:
    1. 탐지된 은폐 시도 목록 (각 시도의 설명, 관련 증거 포함)
    2. 은폐 시도와 관련된 인물 분석
    3. 은폐 시도의 시간대 분석 (정확한 날짜와 시간 포함)
    4. 은폐 시도와 관련된 장소 정보
    5. 추가 조사가 필요한 사항

    날짜, 시간, 숫자 데이터의 정확성에 특별히 주의를 기울이세요.
    """,
    agent=살인사건_은폐_탐지_agent,
    expected_output="탐지된 은폐 시도 목록, 관련 인물 분석, 시간대 분석, 장소 정보, 추가 조사 사항이 포함된 상세한 보고서"
)

피해자_조사_task = Task(
    description="""
    대상 파일에서 피해자와 관련된 모든 데이터를 수집하고 분석하세요. 보고서에는 다음 내용이 포함되어야 합니다:
    1. 피해자의 개인정보 요약
    2. 피해자의 최근 활동 내역 (정확한 날짜와 시간 포함)
    3. 피해자와 관련된 주요 인물들과의 관계 분석
    4. 피해자의 마지막 알려진 위치와 시간
    5. 피해자와 관련된 주요 증거 목록

    날짜, 시간, 숫자 데이터의 정확성을 확인하고, 데이터 타입 정보를 활용하세요.
    """,
    agent=피해자_조사_agent,
    expected_output="피해자의 개인정보, 최근 활동 내역, 관계 분석, 마지막 위치와 시간, 주요 증거 목록이 포함된 종합적인 보고서"
)

살인사건_시간_조사_task = Task(
    description="""
    가용한 데이터를 바탕으로 살인 시간을 추정하세요. 보고서에는 다음 내용이 포함되어야 합니다:
    1. 추정된 살인 발생 시간 (가능한 한 정확한 날짜와 시간)
    2. 시간 추정의 근거가 되는 주요 증거들
    3. 시간대별 주요 사건 타임라인
    4. 알리바이 검증이 필요한 시간대
    5. 추가 조사가 필요한 시간대나 사건

    모든 시간 관련 데이터의 정확성을 철저히 확인하고, 데이터 타입 정보를 활용하세요.
    """,
    agent=살인사건_시간_조사_agent,
    expected_output="추정된 살인 시간, 근거 증거, 사건 타임라인, 알리바이 검증 필요 시간대, 추가 조사 필요 사항이 포함된 시간 분석 보고서"
)

살인사건_장소_조사_task = Task(
    description="""
    은폐 시도 이전에 기록된 WiFi 정보를 바탕으로 살인 장소를 추정하세요. 보고서에는 다음 내용이 포함되어야 합니다:
    1. 추정된 살인 발생 장소
    2. 장소 추정의 근거가 되는 WiFi 정보 분석 (연결 시간 포함)
    3. 관련된 위치들의 목록과 각 위치의 중요성
    4. 추정 장소 주변의 다른 관련 장소들
    5. 추가 조사가 필요한 장소나 지역

    WiFi 연결 시간과 관련된 모든 날짜와 시간 데이터의 정확성을 철저히 확인하고, 데이터 타입 정보를 활용하세요.
    """,
    agent=살인사건_장소_조사_agent,
    expected_output="추정된 살인 장소, WiFi 정보 분석, 관련 위치 목록, 주변 관련 장소, 추가 조사 필요 지역이 포함된 장소 분석 보고서"
)

# 크루 설정
crew = Crew(
    agents=[범죄사실_요약_agent, 살인사건_은폐_탐지_agent, 피해자_조사_agent, 살인사건_시간_조사_agent, 살인사건_장소_조사_agent],
    tasks=[범죄사실_요약_task, 살인사건_은폐_탐지_task, 피해자_조사_task, 살인사건_시간_조사_task, 살인사건_장소_조사_task],
    verbose=2,  # 더 자세한 로깅을 위해 verbose 레벨 유지
    process=Process.sequential,  
    max_loops=1,
    task_timeout=900  # 각 작업의 제한 시간을 15분으로 유지
)

# main 함수 수정 (결과 처리 부분)
def main():
    try:
        logging.info("벡터 저장소 생성 중...")
        logging.info(f"처리된 총 문서 수: {len(all_documents)}")
        logging.info(f"생성된 총 벡터 수: {vectorstore.index.ntotal}")
        
        results = crew.kickoff()
        
        print(results)
        print("작업 완료\n")

    except Exception as e:
        logging.error(f"main 함수 실행 중 오류 발생: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()