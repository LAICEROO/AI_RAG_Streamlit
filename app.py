import streamlit as st
from dotenv import load_dotenv
import os
import torch
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub
from mistralai import Mistral
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.schema import Document
from langchain.retrievers.document_compressors import LLMChainExtractor

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"Using {device} for embeddings")
    
    # Use multilingual-e5-large-instruct for embeddings with GPU if available
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large-instruct",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore, embeddings


def get_bm25_retriever(text_chunks):
    # Convert text chunks to documents for BM25
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5  # Number of documents to return
    return bm25_retriever


def create_mistral_contextual_compressor(embeddings):
    """Create a contextual compressor using Mistral API for more relevant document retrieval"""
    llm = MistralLLM()
    compressor = LLMChainExtractor.from_llm(llm)
    return compressor


class MistralLLM:
    """Custom LLM implementation for Mistral API to use with LangChain compressors"""
    
    def __init__(self):
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self.model = "mistral-small-latest"
        
    def __call__(self, prompt):
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error in MistralLLM: {str(e)}")
            return "Error retrieving context"


def get_conversation_chain(vectorstore, text_chunks, embeddings):
    # Set up Qwen model from HuggingFace
    llm = HuggingFaceHub(
        repo_id="Qwen/QwQ-32B",
        model_kwargs={"max_new_tokens": 32768}
    )
    
    # Set up BM25 retriever
    bm25_retriever = get_bm25_retriever(text_chunks)
    
    # Set up semantic retriever from vectorstore
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # Create the Mistral contextual compressor
    compressor = create_mistral_contextual_compressor(embeddings)
    
    # Apply contextual compression to the semantic retriever
    contextual_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=semantic_retriever
    )
    
    # Combine retrievers in the ensemble
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, contextual_retriever],
        weights=[0.3, 0.7]  # Give more weight to contextual retriever
    )
    
    # Set up memory
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    # Create conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=ensemble_retriever,
        memory=memory
    )
    
    return conversation_chain, ensemble_retriever


def handle_userinput(user_question):
    # Get documents from retriever
    docs = st.session_state.retriever.get_relevant_documents(user_question)
    
    # Get additional context from Mistral
    mistral_context = get_mistral_context(user_question, docs)
    
    # If we have mistral context, add it to the question
    if mistral_context:
        enhanced_question = f"""
        Question: {user_question}
        
        Additional context that might be helpful:
        {mistral_context}
        """
    else:
        enhanced_question = user_question
    
    # Get response from conversation chain
    response = st.session_state.conversation({'question': enhanced_question})
    st.session_state.chat_history = response['chat_history']

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def get_mistral_context(query, docs):
    """Get additional context from Mistral API."""
    if "MISTRAL_API_KEY" not in os.environ:
        st.warning("MISTRAL_API_KEY not found in environment variables. Skipping Mistral context.")
        return ""
    
    try:
        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        
        # Combine document content
        doc_content = "\n\n".join([doc.page_content for doc in docs[:3]])
        
        # Create prompt for Mistral
        prompt = f"""Given the following content from documents and a user query, 
        provide the most relevant information to answer the query. 
        
        DOCUMENTS:
        {doc_content}
        
        USER QUERY: {query}
        
        RELEVANT INFORMATION:"""
        
        # Get response from Mistral using the updated API
        chat_response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        return chat_response.choices[0].message.content
    except Exception as e:
        st.error(f"Error using Mistral API: {str(e)}")
        return ""


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore, embeddings = get_vectorstore(text_chunks)
                
                # create conversation chain and retriever
                conversation_chain, retriever = get_conversation_chain(vectorstore, text_chunks, embeddings)
                
                # Store in session state
                st.session_state.conversation = conversation_chain
                st.session_state.retriever = retriever


if __name__ == '__main__':
    main()