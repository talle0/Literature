__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# API 키 저장을 위한 os 라이브러리 호출
import os
# Streamlit model - streamlit_chat
import streamlit as st
# OPENAI API 키 저장
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

import tiktoken
from loguru import logger

# Documents Loader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

# Embedding model
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# 벡터 저장소 모델
from langchain_community.vectorstores import Chroma

# Text splitter model
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Retrieval model
from langchain.chains import RetrievalQA

# ChatOpenAI
from langchain_openai import ChatOpenAI

openai = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = 0.4)
#openai = ChatOpenAI(model_name="gpt-4.0-turbo", temperature = 0)
embedding = OpenAIEmbeddings()

# main program
@st.cache_resource 
def main_f(files):
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    files_text = get_text(files)
    text_chunks = get_text_chunks(files_text)
    db = Chroma.from_documents(text_chunks, embedding_hf)
    return db

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()
        doc_list.extend(documents)
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceBgeEmbeddings(
                                        model_name="BAAI/bge-small-en",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = Chroma.from_documents(text_chunks, embeddings)
    return vectordb

def qa(db, query):
    qa1 = RetrievalQA.from_chain_type(
            llm=openai,
            chain_type="stuff",
            retriever=db.as_retriever(
                search_type = "mmr",
                search_kwargs={'k':4, 'fetch_k':10}),
            return_source_documents = True
            )
    result=qa1(query)
    return result

# 본 프로그램
st.set_page_config(
    page_title="Literature",
    page_icon=":newspaper:"
)

if 'query' not in st.session_state:
    st.session_state.query = ""
if 'db' not in st.session_state:
    st.session_state.db = None

st.title("Review Your :blue[Papers] :newspaper:")
 
# Sidebar 설정
with st.sidebar:
    files = st.file_uploader("Upload your files", type=['pdf'], accept_multiple_files=True)
    process = st.button("Process")
    if process:
        if not files:
            st.sidebar.warning ("파일을 넣어주세요..")
        else: 
            with st.spinner("Embedding..."):
                st.session_state.db = main_f(files)
            st.write ("임베딩 완료")

container1 = st.container(border=True)
container2 = st.container(border=True)

if st.session_state.db != None:
    with st.spinner("Thinking..."):
        query1 = "이 논문의 요약문을 만들어줘."
        response1 = qa(st.session_state.db, query1)
        query2 = "이 논문의 연구방법론을 설명해줘."
        response2 = qa(st.session_state.db, query2)    
        container1.markdown("__요약__")
        container1.markdown(response1['result'])
        container2.markdown("__연구방법__")
        container2.markdown(response2['result'])

    st.session_state.query = st.text_input("질문")   
    submit2=st.button("submit")     
    if submit2: 
        with st.spinner("Thinking..."):
            response = qa(st.session_state.db, st.session_state.query)
        source_documents = response['source_documents']
        st.write(st.session_state.query)
        st.markdown(response['result']) 
        with st.expander("참고 문서 확인"):
            st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
            st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
            st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content) 
    
 
