import os
import textwrap
from PyPDF2 import PdfReader
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


page_element="""
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://cdn.wallpapersafari.com/88/75/cLUQqJ.jpg");
background-size: cover;
}
[data-testid="stHeader"]{
background-color: rgba(0,0,0,0);
}
[data-testid="stSidebar"]> div:first-child{
background-image: url("https://mcdn.wallpapersafari.com/medium/89/87/X7GDE5.jpg");
background-size: cover;
}
</style>

"""

st.markdown(page_element, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: white';>Power of Laws 💬</h1>", unsafe_allow_html=True)
st.markdown("---")



llm = OpenAI(openai_api_key=st.secrets["LLM_API"])


def process_text(text):

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["LLM_API"])
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    
    return knowledgeBase

def wrap_text_preserve_newlines(text, width=110):

    lines = text.split('\n')
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def process_llm_response(llm_response):
    result_output = wrap_text_preserve_newlines(llm_response['result'])
    print(result_output)
    return result_output


flow_option = st.selectbox(
    'Choose an Option -',
    ('Power of Laws', 'Upload Another PDF'))

if flow_option == 'Power of Laws':
    query = st.text_input('Ask a question to the PDF')
    submit=st.button("Submit")

    model_name = "sentence-transformers/all-mpnet-base-v2"
    instructor_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
    )

    embedding = instructor_embeddings
    persist_directory='db'
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    if submit:
        query = f"""
        Do strictly follow the context and for not retrieved data, output as No data. The context is from the Book of Power of Laws - {query}
                """
        
        llm_response = qa_chain(query)
        result_ipc=process_llm_response(llm_response)
        st.write(result_ipc)



elif flow_option == 'Upload Another PDF':

    pdf = st.file_uploader('Upload your PDF Document', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Create the knowledge base object
        knowledgeBase = process_text(text)
        query = st.text_input('Ask a question to the PDF')
        cancel_button = st.button('Cancel')
        
        if cancel_button:
            st.stop()
        
        if query:
            docs = knowledgeBase.similarity_search(query)
            chain = load_qa_chain(llm, chain_type='stuff')
            
            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)
                
            st.write(response)
