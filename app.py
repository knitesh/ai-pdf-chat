import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from langchain.vectorstores  import FAISS
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from htmlTemplates import css, bot_template, user_template
# from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import Ollama
# from langchain_openai import ChatOpenAI

def get_pdf_text(file):
    text = ""
    for pdf in file:
        if pdf.type == "application/pdf":
            pdf_reader =PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()            
        else:
            text += pdf.read()
    return text


def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        chunk_size=1000,        
        chunk_overlap=200,
        separator="\n",
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks


def create_vector_store(text_chunks):
    # embeddings = OpenAIEmbeddings()
    # model = SentenceTransformer('hkunlp/instructor-xl')
    model_kwargs = {'device': 'cpu'} 
    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl',model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)

    vector_store = FAISS.from_texts(embedding=embeddings,texts=text_chunks)
    
    return vector_store

def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", model_kwargs={"temperature":0.5, "max_length":512})
    llm = Ollama(model="llama2")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    # check for NoneType' object is not callable

    # st.write(response)
    # enumerate in reverse order

    for i, message in reversed(list(enumerate(st.session_state.chat_history))):
        if i % 2 == 0:
            # st.write(message.content)
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            # st.write(message.content)
            # get messages after Helpful Answer:  in message.content
            extract = message.content.split("Helpful Answer: ")
            # get the last part
            # st.write(extract[-1])
            st.write(bot_template.replace(
                "{{MSG}}", extract[-1]), unsafe_allow_html=True)

def main():
    load_dotenv()    

    st.set_page_config(page_title="Streamlit App", page_icon=":shark:", layout="wide")
    st.write(css, unsafe_allow_html=True)

    # st.header("Ai ChatBot with embeddings")
    # st.text_input("Ask a question", key="message")
    if("conversation" not in st.session_state):
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Ai ChatBot with embedding")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Upload additional documents")
        pdf_docs = st.file_uploader("Upload file", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        if st.button("Process"):
           with st.spinner("Processing"):
                # get PDF text
               raw_text = get_pdf_text(pdf_docs)
            #    st.write(raw_text)

                #get the text chunks
               text_chunks = get_text_chunks(raw_text)
               st.write(text_chunks)

                #create vector store
               vectorstore = create_vector_store(text_chunks)

               # create conversion chain
               st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()