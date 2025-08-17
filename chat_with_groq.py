import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceEmbeddings # Replaced Google with HuggingFace
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq # Replaced Google with Groq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Set the Groq API key from the environment variable.
# Potential Error: Ensure your .env file has the GROQ_API_KEY and it is correct.
# The script will fail here if the key is not found.
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()

# --- Helper Functions ---

def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of uploaded PDF files.
    
    Args:
        pdf_docs (list): A list of uploaded PDF file objects.

    Returns:
        str: A single string containing all the extracted text from the PDFs.
    """
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            # Potential Error: A PDF might be corrupt or password-protected.
            st.error(f"Error reading {pdf.name}: {e}")
            continue
    return text

def get_text_chunks(text):
    """
    Splits a long string of text into smaller, manageable chunks.
    
    Args:
        text (str): The input text.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100000, chunk_overlap=10000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Creates and saves a FAISS vector store from text chunks.
    
    Args:
        text_chunks (list): A list of text chunks to be embedded.
    
    Potential Error: This step downloads the embedding model (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
    the first time it's run, so an internet connection is required. Subsequent runs will use the cached model.
    """
    try:
        # Use a popular, efficient open-source embedding model from HuggingFace
        # This runs on your local machine (CPU) and does not require an API key.
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create a FAISS vector store from the text chunks and embeddings
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
        # Save the vector store locally for later use
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        st.stop()

def get_conversational_chain():
    """
    Creates a LangChain question-answering chain with a custom prompt and a Groq LLM.
    
    Returns:
        A LangChain chain object.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    Make sure to provide all the details. If the answer is not in the provided context, 
    just say, "The answer is not available in the context." Do not provide a wrong answer.

    Context:
    {context}?

    Question: 
    {question}

    Answer:
    """
    
    # Initialize the Groq Chat model.
    # We use 'llama3-8b-8192' as it is fast and capable.
    # Potential Error: If the model name is incorrect or your Groq account has issues,
    # this initialization might fail.
    model = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192", temperature=0.3)
    
    # Create a prompt template
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Load the question-answering chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """
    Handles user input by performing a similarity search and running the QA chain.
    """
    try:
        # Load the local FAISS index and the same embedding model
        # Potential Error: Make sure "faiss_index" folder exists and was created with the same embedding model.
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) # Added allow_dangerous_deserialization for FAISS
        
        # Find documents similar to the user's question
        docs = new_db.similarity_search(user_question)
        
        # Get the conversational chain
        chain = get_conversational_chain()
        
        # Run the chain with the found documents and the user's question
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        
        # Display the response
        st.write("Reply: ", response["output_text"])
    except FileNotFoundError:
        st.error("FAISS index not found. Please upload and process PDF files first.")
    except Exception as e:
        # General error handler for unexpected issues during the process.
        st.error(f"An error occurred: {e}")

# --- Streamlit UI ---

def main():
    """
    The main function that sets up and runs the Streamlit web application.
    """
    st.set_page_config(page_title="Chat with PDFs (Groq Edition) 🚀", layout="wide")
    st.header("Chat with Multiple PDFs using GroQ and Llama3📑")

    # Input for the user's question
    user_question = st.text_input("Ask a Question from the PDF Files➡️")

    if user_question:
        user_input(user_question)

    # Sidebar for uploading files and processing
    with st.sidebar:
        st.title("Menu")
        st.markdown("---")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click 'Submit & Process'", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing... This may take a moment."):
                    # Step 1: Extract raw text from PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # Step 2: Split text into chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # Step 3: Create and save the vector store
                    get_vector_store(text_chunks)
                    
                    st.success("Done! You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()