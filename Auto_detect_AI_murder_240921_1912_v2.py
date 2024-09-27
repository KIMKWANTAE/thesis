import os
import pandas as pd
import yaml
import logging
from typing import List
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

# 설정 로드 함수
def load_config():
    with open('config_murder.yaml', 'r', encoding='utf-8') as f:
        config_murder = yaml.safe_load(f)
    return config_murder

# 엑셀 데이터 로드 함수
def load_excel_data(file_path: str) -> List[Document]:
    try:
        sheets_dict = pd.read_excel(file_path, sheet_name=None)
        documents = []
        for sheet_name, df in sheets_dict.items():
            content = f"Sheet: {sheet_name}\n{df.to_string(index=False)}"
            source = f"{os.path.basename(file_path)}:{sheet_name}"
            documents.append(Document(page_content=content, metadata={"source": source}))
        logging.info(f"Successfully loaded {len(documents)} sheets from {file_path}")
        return documents
    except FileNotFoundError:
        logging.error(f"Excel file not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"Excel file is empty: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading Excel file {file_path}: {str(e)}")
        raise

# 텍스트 파일 로드 함수
def load_text_files(folder_path: str) -> List[Document]:
    documents = []
    try:
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                        documents.append(Document(page_content=content, metadata={"source": filename}))
                except UnicodeDecodeError:
                    logging.warning(f"Failed to decode file {filename} with UTF-8. Trying with another encoding.")
                    with open(file_path, "r", encoding="cp949") as file:
                        content = file.read()
                        documents.append(Document(page_content=content, metadata={"source": filename}))
        logging.info(f"Successfully loaded {len(documents)} text files from {folder_path}")
        return documents
    except FileNotFoundError:
        logging.error(f"Folder not found: {folder_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading text files from {folder_path}: {str(e)}")
        raise

# 벡터 저장소 생성 함수
def create_vector_store(documents: List[Document], chunk_size: int, chunk_overlap: int) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(documents)
    embedding_function = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_function)
    return vectorstore

# 설정 유효성 검사 함수
def validate_config(config_murder):
    required_keys = ['openai', 'paths', 'vectorstore', 'logging', 'rag', 'agents', 'tasks']
    for key in required_keys:
        if key not in config_murder:
            raise KeyError(f"Missing required section '{key}' in config_murder")
    
    if 'prompt_template' not in config_murder['rag']:
        raise KeyError("Missing 'prompt_template' in 'rag' section of config_murder")
    
    for agent in ['별건_살인죄_분석가', '살인_보고서_작성자']:
        if agent not in config_murder['agents'] or 'backstory' not in config_murder['agents'][agent]:
            raise KeyError(f"Missing '{agent}' or its 'backstory' in 'agents' section of config_murder")
    
    for task in ['별건_살인죄_탐지', '별건_살인죄_보고서_작업']:
        if task not in config_murder['tasks'] or 'description' not in config_murder['tasks'][task] or 'expected_output' not in config_murder['tasks'][task]:
            raise KeyError(f"Missing '{task}' or its 'description' or 'expected_output' in 'tasks' section of config_murder")

# 메인 함수
def main():
    try:
        # 환경 변수 로드
        load_dotenv()
        
        # 설정 로드
        config_murder = load_config()
        
        # OpenAI API 키 설정
        os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY is not set in environment variables")
        
        # 설정 검증
        validate_config(config_murder)
        
        # 로깅 설정
        logging_level = getattr(logging, config_murder['logging']['level'].upper())
        logging.basicConfig(level=logging_level, format=config_murder['logging']['format'])
        
        # OpenAI API 초기화
        llm = ChatOpenAI(model_name=config_murder['openai']['model_name'], temperature=0)
        
        # 데이터 로드
        excel_documents = load_excel_data(config_murder['paths']['excel_file'])
        audio_documents = load_text_files(config_murder['paths']['audio_folder'])
        image_documents = load_text_files(config_murder['paths']['image_folder'])
        all_documents = excel_documents + audio_documents + image_documents

        # 벡터 저장소 생성
        vectorstore = create_vector_store(all_documents, 
                                          config_murder['vectorstore']['chunk_size'], 
                                          config_murder['vectorstore']['chunk_overlap'])
        retriever = vectorstore.as_retriever()
        
        with open(config_murder['paths']['crime_fact_path'], 'r', encoding='utf-8') as file:
            crime_fact = file.read()

        # RAG 설정
        prompt = PromptTemplate.from_template(config_murder['rag']['prompt_template'])
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough(), "crime_fact": lambda _: crime_fact}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # RAG 도구 생성
        rag_tool = Tool(
            name="RAG_분석",
            func=lambda q: rag_chain.invoke(q),
            description="RAG를 사용하여 텍스트 파일을 분석하고 질문에 답합니다."
        )
        
        # 에이전트 생성
        별건_살인죄_분석가 = Agent(
            role='살인죄 사건 탐지 분석가',
            goal='텍스트에서 폭행, 살인으로 추정되는 내용을 찾아내서 관련 정보를 추출한다',
            backstory=config_murder['agents']['별건_살인죄_분석가']['backstory'],
            tools=[rag_tool],
            verbose=True,
            llm=llm
        )
        
        살인_보고서_작성자 = Agent(
            role='보고서 작성자',
            goal='찾아낸 데이터의 내용과 사건 혐의를 바탕으로 종합적인 보고서를 작성한다',
            backstory=config_murder['agents']['살인_보고서_작성자']['backstory'],
            tools=[rag_tool],
            verbose=True,
            llm=llm
        )
        
        # 태스크 생성
        별건_살인죄_탐지 = Task(
            description=config_murder['tasks']['별건_살인죄_탐지']['description'],
            agent=별건_살인죄_분석가,
            expected_output=config_murder['tasks']['별건_살인죄_탐지']['expected_output']
        )
        
        별건_살인죄_보고서_작업 = Task(
            description=config_murder['tasks']['별건_살인죄_보고서_작업']['description'],
            agent=살인_보고서_작성자,
            expected_output=config_murder['tasks']['별건_살인죄_보고서_작업']['expected_output']
        )
        
        # Crew 생성 및 실행
        crew = Crew(
            agents=[별건_살인죄_분석가, 살인_보고서_작성자],
            tasks=[별건_살인죄_탐지, 별건_살인죄_보고서_작업],
            verbose=True,
            process=Process.sequential
        )
        
        logging.info("Creating vector store...")
        logging.info(f"Total documents processed: {len(all_documents)}")
        logging.info(f"Total vectors created: {vectorstore.index.ntotal}")
        
        result = crew.kickoff()
        print(result)
        logging.info("Tasks completed")
        logging.info(result)
    
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        raise
    
if __name__ == "__main__":
    main()