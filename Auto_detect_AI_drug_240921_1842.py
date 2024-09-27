import os
import pandas as pd
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
import logging
import tkinter as tk
from tkinter import filedialog

# 환경 변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다")

# OpenAI API 초기화
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

def load_excel_data(file_path: str) -> List[Document]:
    """엑셀 파일을 로드하고 각 시트의 데이터를 Document 객체로 변환합니다."""
    sheets_dict = pd.read_excel(file_path, sheet_name=None)
    documents = []
    for sheet_name, df in sheets_dict.items():
        content = f"Sheet: {sheet_name}\n{df.to_string(index=False)}"
        source = f"{os.path.basename(file_path)}:{sheet_name}"
        documents.append(Document(page_content=content, metadata={"source": source}))
    return documents

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

def select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()  # 메인 윈도우 숨기기
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return file_path

def select_folder(title):
    root = tk.Tk()
    root.withdraw()  # 메인 윈도우 숨기기
    folder_path = filedialog.askdirectory(title=title)
    return folder_path

# 파일 및 폴더 선택을 위한 GUI
excel_path = select_file("모바일 분석결과 엑셀파일을 선택해 주세요", [("Excel files", "*.xlsx")])
audio_folder = select_folder("오디오 파일이 들어있는 폴더를 선택해 주세요")
image_folder = select_folder("이미지 파일이 들어있는 폴더를 선택해 주세요")
crime_fact_path = select_file("범죄사실 파일을 선택해 주세요", [("Text files", "*.txt")])

# 범죄사실 파일 로드
with open(crime_fact_path, 'r', encoding='utf-8') as file:
    crime_fact = file.read()

# 데이터 로드 및 벡터 저장소 생성
excel_documents = load_excel_data(excel_path)
audio_documents = load_text_files(audio_folder)
image_documents = load_text_files(image_folder)

all_documents = excel_documents + audio_documents + image_documents
vectorstore = create_vector_store(all_documents)
retriever = vectorstore.as_retriever()

# RAG 설정
prompt = PromptTemplate.from_template(
    """당신은 범죄 수사를 위한 전문 분석가입니다. 
    주어진 컨텍스트와 범죄사실 파일의 내용을 바탕으로 관련 사건을 분석하고 보고서를 작성하세요.
    특히 마약거래, 약물복용, 마약은어 사용과 관련된 내용에 주목하세요.
    
    다음과 같은 키워드나 표현에 특별히 주의를 기울이세요:
    - 마약 거래 관련: "물건", "거래", "가져오다", "장소", "시간", "돈", "약"
    - 약물 복용 관련: "먹다", "주사", "효과", "취하다" 
    - 마약 은어 관련: slang_dictionary에 정의된 내용
    
    범죄사실 파일의 내용: {crime_fact}
    
    분석 지침:
    1. 범죄사실 파일의 내용과 주어진 컨텍스트를 비교 분석하세요.
    2. 마약 거래로 의심되는 대화나 정보를 식별하세요.
    3. 사용된 은어의 의미를 해석하고 실제 거래 내용을 추론하세요.
    4. 거래 장소, 시간, 금액, 수량 등 구체적인 정보를 추출하세요.
    5. 관련자들의 역할과 관계를 파악하세요.
    6. 각 정보의 출처(파일명)를 명확히 기록하세요.
    7. 종합적인 분석 결과를 바탕으로 간결한 보고서를 작성하세요.
    
    답을 모르면 모른다고 말하세요. 
    한국어로 답변하세요.

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

# 로깅 초기화
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

rag_tool = Tool(
    name="RAG_분석",
    func=lambda q: rag_chain.invoke(q),
    description="RAG를 사용하여 텍스트 파일을 분석하고 질문에 답합니다."
)

def get_slang_dictionary() -> Dict[str, str]:
    return {
        "떨이": "대마초",
        "케이": "케타민",
        "사탕": "엑스터시",
        "캔디": "엑스터시",
        "아이스": "필로폰",
        "허브": "합성대마",
        "스노우": "코카인",
        "찰리": "코카인",
        "엑기스": "엑스터시",
        "빠리": "필로폰",
        "물건": "마약",
        "약": "마약"
    }

def search_slang(query: str) -> str:
    slang_dict = get_slang_dictionary()
    found_slangs = [f"{slang}: {meaning}" for slang, meaning in slang_dict.items() if slang in query]
    return "발견된 은어:\n" + "\n".join(found_slangs) if found_slangs else "입력된 텍스트에서 알려진 은어를 찾을 수 없습니다."

slang_tool = Tool(
    name="은어_검색",
    func=search_slang,
    description="미리 정의된 은어 사전에서 은어를 검색합니다."
)

마약_탐지_분석가 = Agent(
    role='마약사건 탐지 분석가',
    goal='텍스트에서 마약거래 관련 내용을 찾아내서 관련 정보를 추출한다',
    backstory='당신은 텍스트에서 마약 거래와 관련된 내용을 탐지하는 전문가입니다. 은어를 사용한 마약거래 정황을 매우 세심하게 포착하여 마약거래 의심이 있는 내용을 찾아냅니다. 범죄사실 파일의 내용을 참고하여 관련 정보를 찾습니다.',
    tools=[rag_tool],
    verbose=True,
    llm=llm
)

은어_연구원 = Agent(
    role='은어 연구원',
    goal='마약 관련 은어에 대한 정보를 제공한다',
    backstory='당신은 마약 관련 용어와 은어를 이해하는 전문가입니다. 범죄사실 파일의 내용을 참고하여 관련 은어를 식별합니다.',
    tools=[slang_tool, rag_tool],
    verbose=True,
    llm=llm
)

마약_보고서_작성자 = Agent(
    role='보고서 작성자',
    goal='찾아낸 데이터의 내용과 사건 혐의를 바탕으로 종합적인 보고서를 작성한다',
    backstory='당신은 복잡한 정보를 종합하여 명확하고 통찰력 있는 한국어 보고서를 작성하는 전문가입니다.',
    tools=[rag_tool],
    verbose=True,
    llm=llm
)

마약사건_탐지 = Task(
    description=f"""
    RAG 도구를 사용하여 텍스트 파일을 분석하세요. 다음 사항에 집중하세요:
    1. 주어진 데이터에서 마약 거래와 관련된 내용을 식별
    2. 마약 거래로 의심되는 데이터를 확인한 후 관련 정보 추출
    3. 마약 거래로 추정되는 시간, 장소, 물품 등을 기록
    4. 사용된 특이한 표현이나 은어 목록 작성
    5. 각 정보의 출처(파일명) 기록
    6. 출처 파일의 작성시간을 파악하고 해당 작성 일시에 기록된 다른 정보와의 관계 파악
    7. 정보를 종합하여 마약 거래의 전체적인 맥락을 이해
    각 측면에 대해 RAG_분석 도구를 사용하세요.
    """,
    agent=마약_탐지_분석가,
    expected_output="마약거래와 관련한 내용파악, 마약거래로 추정되는 물품, 거래 시간, 장소 확인, 특이한 표현이나 은어 목록, 각 정보의 출처(파일명) 포함, 출처 파일의 작성시간을 파악하고 해당 작성 일시에 기록된 다른 정보와의 관계 파악, 정보를 종합하여 마약 거래의 전체적인 맥락을 정리한 결과 포함. 한글로 작성됨."
)

은어_연구_작업 = Task(
    description=f"""
    은어_검색 도구를 사용하여 텍스트에서 마약 관련 은어를 식별하세요.
    마약_탐지_분석가가 제공한 특이한 표현이나 단어 목록을 중점적으로 검사하세요.
    식별된 각 은어에 대해 그 의미를 설명하세요.
    추가로, RAG_분석 도구를 사용하여 은어가 사용된 구체적인 맥락을 파악하세요.
    각 은어의 출처(파일명)를 명확히 기록하세요.
    """,
    agent=은어_연구원,
    expected_output="분석된 대화에서 발견된 은어 목록, 각 은어의 의미 설명, 은어가 사용된 맥락, 각 은어의 출처(파일명) 포함. 한글로 작성됨."
)

마약거래_보고서_작업 = Task(
    description=f"""
    마약_탐지_분석가의 분석 결과, 은어 연구원의 연구 결과를 바탕으로 종합적인 보고서를 작성하세요. 보고서는 다음 내용을 포함해야 합니다:
    1. 마약_탐지_분석가의 분석 결과 주요 탐지된 내용의 전반적인 요약
    2. 마약_탐지_분석가의 분석 결과 대화에서 언급된 거래 물품, 시간, 장소에 대한 정보
    3. 은어_연구원의 분석 결과 대화에서 사용된 특이한 표현이나 은어 목록
    4. 은어_연구원의 분석결과 각 은어나 특이한 표현의 추정 의미와 그 근거
    5. 마약_탐지_분석가의 분석 결과 대화의 전반적인 맥락을 고려한 거래 물품의 실제 의미에 대한 추론
    6. 결론: 마약_탐지_분석가의 분석 결과 대화의 실제 목적과 내용에 대한 종합적 해석
    7. 마약_탐지_분석가의 분석 결과 대화가 들어있는 파일명과 출처 (예: 'Messages.db' 또는 '음성_001.txt')
    8. 마약_탐지_분석가의 분석 결과 대화가 작성된 일시에 기록된 다른 정보 (예: wifi 로그, 위치 정보)와의 관계 파악
    9. 1~8번까지의 정보와 범죄사실 파일 {crime_fact} 의 내용을 참고해서 종합적으로 사건을 재구성하여 범죄 추론한 내용기술:
    
    보고서를 명확하고 논리적인 구조로 작성하세요.
    모든 내용을 한글로 작성하세요.
    각 정보의 출처(파일명)를 명확히 밝히세요.
        
    마지막으로, RAG_분석 도구를 사용하여 작성된 보고서의 정확성을 재확인하고, 필요한 경우 수정하세요.
    """,
    agent=마약_보고서_작성자,
    expected_output="종합적인 보고서 (대화 요약, 거래 정보, 은어 분석, 추론 결과, 결론, 모든 정보의 출처(파일명) 명시, 대화가 작성된 일시에 기록된 다른 정보확인 및 관계파악, 종합 분석결과 포함. 한글로 작성됨."
)

crew = Crew(
    agents=[마약_탐지_분석가, 은어_연구원, 마약_보고서_작성자],
    tasks=[마약사건_탐지, 은어_연구_작업, 마약거래_보고서_작업],
    verbose=True,
    process=Process.sequential
)

def main():
    try:
        logging.info("벡터 저장소 생성 중...")
        logging.info(f"처리된 총 문서 수: {len(all_documents)}")
        logging.info(f"생성된 총 벡터 수: {vectorstore.index.ntotal}")
        
        result = crew.kickoff()
        print(result)
        logging.info("작업 완료")
        logging.info(result)
    except Exception as e:
        logging.error(f"오류 발생: {e}")

if __name__ == "__main__":
    main()