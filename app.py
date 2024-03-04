import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

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

def main():
    load_dotenv()
    st.set_page_config(page_title="Streamlit App", page_icon=":shark:", layout="wide")

    st.header("Ai ChatBot with embeddings")
    st.text_input("Ask a question", key="message")


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


if __name__ == "__main__":
    main()