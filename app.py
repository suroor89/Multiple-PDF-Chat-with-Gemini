import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversation_chain():
    prompt_template = """Answer the question as detailed as possible using the provided context, make sure to provide allthe details. 
    if the answer is not in the context, say "Answer is not available in the context", don provide the wrong information.
    Context: \n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain=load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db=FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)

    chain=get_conversation_chain()

    response=chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    print(response)
    st.write("Reply: ",response['output_text'])


def main():
    st.set_page_config(page_title="Chat with Multiple PDF", page_icon=":books:")
    st.header("Chat with Multiple PDF using Gemini :books:")

    # Initialize session state for processing status
    if "processed" not in st.session_state:
        st.session_state.processed = False

    with st.sidebar:
        st.title("PDF files: ")
        pdf_docs = st.file_uploader("Upload your PDF files here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.session_state.processed = True  # Set processed to True
                    st.success("PDF files processed successfully!")
                else:
                    st.warning("Please upload at least one PDF file.")
                    st.session_state.processed = False  # Ensure processed is False

    # Disable question input if not processed
    if st.session_state.processed:
        user_question = st.text_input("Ask a question about your PDF files:", key="question_input")
        if user_question:
            user_input(user_question)
    else:
        st.info("Please upload and process PDF files to enable asking questions.")

if __name__ == '__main__':
    main()

   