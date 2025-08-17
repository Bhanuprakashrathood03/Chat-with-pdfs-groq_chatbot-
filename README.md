# Chat with Multiple PDFs using Groq and Llama 3 🚀

This is a Streamlit-based web application that allows users to upload multiple PDF documents and query them through a high-speed, conversational interface powered by the Groq API and Llama 3. Users can ask any question related to the uploaded documents and receive near-instant, context-aware answers.

## **This is how it looks initially in the Web Browser.**
<img width="1440" height="819" alt="Screenshot 2025-08-17 at 3 22 24 PM" src="https://github.com/user-attachments/assets/4983fb9c-e4c8-4acc-aa7f-e2236e364a3c" />

## **After a user uploads their files and submits a question, the application provides a detailed, context-aware response as shown below:**
<img width="1440" height="900" alt="Screenshot 2025-08-17 at 3 29 21 PM" src="https://github.com/user-attachments/assets/5b81cf22-4f93-4944-b255-96a2668d26b5" />
<img width="1440" height="900" alt="Screenshot 2025-08-17 at 3 30 35 PM" src="https://github.com/user-attachments/assets/8a26d938-22a3-429d-afda-1141bf7c800b" />

### Demo
[![Project Demo Video](https://img.youtube.com/vi/fXSFjh3ne3I.jpg)](https://www.youtube.com/watch?v=fXSFjh3ne3I)

# 👷🏻‍♂️Project Development Lifecycle.

## **High-Speed PDF Chatbot.**
This project's development centered on building a high-performance, private, and scalable Question-Answering application. The lifecycle focused on integrating best-in-class tools for each stage of the process, from data ingestion to user interaction.

## **Phase 1: System Architecture and Core Technologies.**

The application was architected around a modern, high-speed stack to ensure real-time user interaction. The primary technologies selected were:

**LLM Inference Engine:** The Groq API was chosen to power the core language model, leveraging its LPU Inference Engine with Llama 3 to achieve near-instantaneous response generation.

**Local Embedding Model:** To guarantee user privacy and eliminate external API dependencies for data processing, a Hugging Face Sentence Transformers model was implemented. This allows all document vectorization to occur on the local machine.

**Orchestration Framework:** LangChain serves as the backbone of the application, orchestrating the entire workflow from data processing to the final conversational chain.

## **Phase 2: Data Processing and Retrieval Pipeline.**

A robust pipeline was constructed to efficiently ingest, process, and index PDF documents for rapid information retrieval.

**Text Ingestion and Segmentation:** The PyPDF2 library is utilized to extract raw text from uploaded PDF files. This text is then segmented into optimized, semantically coherent chunks using LangChain's RecursiveCharacterTextSplitter.

**Vectorization and Indexing:** The text chunks are transformed into dense vector embeddings using the local Hugging Face model. These vectors are then indexed into a FAISS (Facebook AI Similarity Search) vector store, which enables high-speed similarity searches to find the most relevant context for a user's query.

## **Phase 3: Conversational AI Core.**

The application's question-answering logic was engineered for scalability and to handle extensive documents without technical limitations.

**Advanced Conversational Chain:** A map_reduce chain from LangChain was implemented. This architecture is specifically designed for large documents; it processes relevant document chunks in parallel (the "map" step) before synthesizing a final, comprehensive answer (the "reduce" step). This approach effectively manages API token limits and ensures thorough analysis of the source material.

## **Phase 4: User Interface (UI) Development.**

An intuitive and user-friendly front-end was developed using Streamlit. The UI provides a seamless experience, guiding the user through:
Uploading multiple PDF documents.
Initiating the processing and indexing pipeline with a single click.
Engaging in a real-time, interactive Q&A session with their documents.
