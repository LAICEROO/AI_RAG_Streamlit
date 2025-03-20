import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from langchain_mistralai import ChatMistralAI
from mistralai import Mistral
import requests
import json
import numpy as np

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    # First, break text into paragraphs
    paragraphs = text.split("\n\n")
    
    # Process each paragraph individually to ensure we don't exceed chunk size
    chunks = []
    current_chunk = ""
    current_size = 0
    
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
        
        # If paragraph is already too large, split it into sentences
        if len(paragraph) > 800:  # Using 800 to leave room for overlap
            sentences = paragraph.replace("\n", " ").split(". ")
            sentences = [s + "." if not s.endswith(".") else s for s in sentences if s.strip()]
            
            for sentence in sentences:
                # If adding sentence would exceed chunk size, start a new chunk
                if current_size + len(sentence) > 800:
                    if current_chunk:  # Only add if we have content
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    current_size = len(sentence)
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
                    current_size += len(sentence)
        else:
            # If adding paragraph would exceed chunk size, start a new chunk
            if current_size + len(paragraph) > 800:
                if current_chunk:  # Only add if we have content
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
                current_size = len(paragraph)
            else:
                # Add paragraph to current chunk
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                current_size += len(paragraph)
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Check if any chunks still exceed our limit
    for i, chunk in enumerate(chunks):
        if len(chunk) > 1000:
            st.warning(f"Chunk {i} has size {len(chunk)}, which exceeds the limit. Will be split further.")
            # Further split into smaller chunks
            text_splitter = CharacterTextSplitter(
                separator=" ",
                chunk_size=900,  # Lower than 1000 to ensure we don't exceed limit
                chunk_overlap=100,
                length_function=len
            )
            replacement_chunks = text_splitter.split_text(chunk)
            # Replace the oversized chunk with the smaller chunks
            chunks.pop(i)
            for j, replacement in enumerate(replacement_chunks):
                chunks.insert(i+j, replacement)
    
    return chunks


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class MultilangE5Embeddings(Embeddings):
    def __init__(self, model_name="intfloat/multilingual-e5-large-instruct", batch_size=8):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.batch_size = batch_size
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
    def embed_documents(self, texts):
        # Pre-process texts to add instruction format
        task = "Represent this document for retrieval:"
        processed_texts = [f"Instruct: {task}\nQuery: {text}" for text in texts]
        
        # Process in batches for better memory management
        embeddings_list = []
        for i in range(0, len(processed_texts), self.batch_size):
            batch_texts = processed_texts[i:i+self.batch_size]
            
            # Tokenize the input texts
            batch_dict = self.tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
            
            # Move tensors to device
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**batch_dict)
            
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Move back to CPU and convert to list
            embeddings_list.append(embeddings.cpu().numpy())
        
        # Concatenate all batch embeddings
        if len(embeddings_list) > 1:
            return np.vstack(embeddings_list).tolist()
        return embeddings_list[0].tolist()
    
    def embed_query(self, text):
        # Pre-process query to add instruction format
        task = "Represent this query for retrieval:"
        processed_text = f"Instruct: {task}\nQuery: {text}"
        
        # Tokenize the query
        batch_dict = self.tokenizer([processed_text], max_length=512, padding=True, truncation=True, return_tensors='pt')
        
        # Move tensors to device
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Move back to CPU and convert to list
        return embeddings.cpu().numpy().tolist()[0]
    
    # Implement the callable interface that LangChain expects
    def embed_text(self, text):
        return self.embed_query(text)
    
    # This makes the object callable directly
    def __call__(self, text):
        return self.embed_text(text)


def get_vectorstore(text_chunks):
    # Apply performance settings
    batch_size = st.session_state.embedding_batch_size if st.session_state.use_performance_mode else 4
    embeddings = MultilangE5Embeddings(batch_size=batch_size)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # Use Mistral AI directly
    api_key = os.environ["MISTRAL_API_KEY"]
    
    # Initialize the ChatMistralAI using LangChain's integration
    llm = ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.3,
        max_tokens=8192,
        top_p=0.9
    )

    # Use the updated memory API to avoid deprecation warnings
    from langchain.memory import ConversationBufferMemory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True, 
        output_key='answer'
    )
    
    # Create the chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        memory=memory,
        verbose=True,
        return_source_documents=True,
        chain_type="stuff"
    )
    
    return conversation_chain


def handle_userinput(user_question):
    web_context = ""
    web_sources = []
    
    # Check if we should perform a web search alongside the document search
    if st.session_state.web_search_enabled and user_question:
        with st.spinner("Searching the web for additional context..."):
            search_results = perform_tavily_search(
                query=user_question, 
                search_depth=st.session_state.web_search_depth,
                max_results=st.session_state.max_results,
                include_answer=st.session_state.include_answer,
                include_images=st.session_state.include_images,
                time_range=st.session_state.time_range
            )
            
            # If we got search results and they don't contain an error
            if search_results and "error" not in search_results:
                # Extract web search answer and sources
                if "answer" in search_results and search_results["answer"]:
                    web_context = search_results["answer"]
                
                # Extract web sources for citation
                if "results" in search_results and len(search_results["results"]) > 0:
                    for result in search_results["results"]:
                        web_sources.append({
                            "title": result.get("title", "No title"),
                            "url": result.get("url", "#"),
                            "content": result.get("content", "")[:150] + "..." if len(result.get("content", "")) > 150 else result.get("content", "")
                        })
                
                # Show the web search context to the user
                with st.expander("Web Search Results", expanded=True):
                    if web_context:
                        st.write(web_context)
                    
                    if web_sources:
                        st.markdown("### Sources:")
                        for i, source in enumerate(web_sources):
                            st.markdown(f"**Source {i+1}:** [{source['title']}]({source['url']})")
                            st.markdown(f"_Preview:_ {source['content']}")
                
                # Always add web results to the vectorstore 
                if "results" in search_results and st.session_state.conversation:
                    content_texts = [result.get("content", "") for result in search_results["results"] if "content" in result]
                    source_urls = [result.get("url", "") for result in search_results["results"] if "content" in result]
                    
                    if content_texts:
                        # Add to vectorstore
                        add_search_results_to_vectorstore(content_texts, source_urls)
    
    try:
        # Always use full integration mode for web results
        if web_context and web_sources:
            # Format web information to be explicitly used by the LLM
            formatted_web_info = f"""
Web search found the following information relevant to your question:

{web_context}

Sources:
"""
            for i, source in enumerate(web_sources):
                formatted_web_info += f"{i+1}. {source['title']} - {source['url']}\n"
            
            # Ensure the LLM uses this information
            enhanced_question = f"""
{user_question}

Use the following information from a recent web search to help with your answer:
{formatted_web_info}

Please incorporate this web information into your response and cite sources when appropriate.
"""
        else:
            enhanced_question = user_question
        
        # Get response from conversation chain
        response = st.session_state.conversation.invoke({'question': enhanced_question})
        
        # Update chat history in session state - but with the original question
        # Update the memory directly with original question
        if enhanced_question != user_question and hasattr(st.session_state.conversation, 'memory'):
            # Fix the memory to show the original question, not the enhanced one
            messages = st.session_state.conversation.memory.chat_memory.messages
            for i, msg in enumerate(messages):
                if msg.type == 'human' and msg.content == enhanced_question:
                    messages[i].content = user_question
            
        # Get updated chat history
        st.session_state.chat_history = st.session_state.conversation.memory.chat_memory.messages
        
        # Update the messages in session state for display
        st.session_state.messages = []
        for message in st.session_state.chat_history:
            if message.type == 'human':
                st.session_state.messages.append({"role": "user", "content": message.content})
            else:
                st.session_state.messages.append({"role": "assistant", "content": message.content})
                
        # Display source documents if available
        if 'source_documents' in response and response['source_documents']:
            with st.expander("Document Sources"):
                for i, doc in enumerate(response['source_documents']):
                    st.markdown(f"**Source {i+1}:**")
                    st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    if hasattr(doc.metadata, 'source') and doc.metadata.source:
                        st.write(f"From: {doc.metadata.source}")
                    st.divider()
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        import traceback
        st.error(traceback.format_exc())


def perform_tavily_search(query, search_depth="basic", max_results=5, include_answer=True, include_images=False, time_range=None):
    """
    Perform a search using the Tavily API.
    
    Args:
        query (str): The search query
        search_depth (str): Either "basic" or "advanced"
        max_results (int): Maximum number of results to return
        include_answer (bool): Whether to include an AI-generated answer
        include_images (bool): Whether to include images in the results
        time_range (str, optional): Time range for results (e.g., "day", "week", "month")
        
    Returns:
        dict: The search results from Tavily
    """
    url = "https://api.tavily.com/search"
    
    payload = {
        "query": query,
        "search_depth": search_depth,
        "include_answer": include_answer,
        "include_images": include_images,
        "max_results": max_results
    }
    
    # Add optional parameters if provided
    if time_range:
        payload["time_range"] = time_range
    
    headers = {
        "Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error during Tavily search: {e}")
        return {"error": str(e)}


def add_search_results_to_vectorstore(content_texts, source_urls):
    """Helper function to add search results to the vectorstore"""
    search_text_chunks = []
    metadata_list = []
    
    for i, (content, url) in enumerate(zip(content_texts, source_urls)):
        # Chunk each search result individually to avoid oversized chunks
        result_chunks = get_text_chunks(content)
        search_text_chunks.extend(result_chunks)
        
        # Create metadata for each chunk
        for _ in result_chunks:
            metadata_list.append({
                "source": f"Web search result: {url}",
                "type": "web_search"
            })
    
    if search_text_chunks:
        # Get the existing retriever's vectorstore
        existing_vectorstore = st.session_state.conversation.retriever.vectorstore
        
        # Add the new texts to the existing vectorstore with metadata
        existing_vectorstore.add_texts(
            texts=search_text_chunks,
            metadatas=metadata_list
        )
        
        # Recreate the conversation chain with the updated vectorstore
        st.session_state.conversation = get_conversation_chain(existing_vectorstore)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "web_search_enabled" not in st.session_state:
        st.session_state.web_search_enabled = False
    if "web_search_depth" not in st.session_state:
        st.session_state.web_search_depth = "basic"
    if "embedding_batch_size" not in st.session_state:
        st.session_state.embedding_batch_size = 8
    if "use_performance_mode" not in st.session_state:
        st.session_state.use_performance_mode = True
    
    # Set default values for removed UI elements
    st.session_state.auto_add_search_results = True
    st.session_state.search_integration_level = "full"

    st.title("Chat with multiple PDFs 📚")
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_question)
        
        # Generate and display assistant response
        if st.session_state.conversation:
            # Process the user input and get response
            handle_userinput(user_question)
            
            # Display the latest assistant message
            with st.chat_message("assistant"):
                for msg in reversed(st.session_state.messages):
                    if msg["role"] == "assistant":
                        st.write(msg["content"])
                        break
        else:
            with st.chat_message("assistant"):
                st.warning("Please upload and process documents first!")
                st.session_state.messages.append({"role": "assistant", "content": "Please upload and process documents first!"})

    with st.sidebar:
        # Add tabs to organize the sidebar
        tab1, tab2, tab3 = st.tabs(["Documents", "Web Search", "Advanced Settings"])
        
        with tab1:
            st.header("Your documents")
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
            
            if st.button("Process"):
                if pdf_docs:
                    with st.spinner("Processing documents..."):
                        # get pdf text
                        raw_text = get_pdf_text(pdf_docs)

                        # get the text chunks
                        text_chunks = get_text_chunks(raw_text)

                        # create vector store
                        vectorstore = get_vectorstore(text_chunks)

                        # create conversation chain
                        st.session_state.conversation = get_conversation_chain(vectorstore)
                    
                    st.success("Documents processed successfully!")
                    
                    # Clear messages when new documents are processed
                    st.session_state.messages = []
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "Documents processed! You can now ask questions about them."
                    })
                else:
                    st.error("Please upload at least one PDF file.")
        
        with tab2:
            st.header("Web Search")
            
            # Option to enable web search for all questions
            st.session_state.web_search_enabled = st.checkbox("Enable web search for all questions", 
                                                            value=st.session_state.web_search_enabled,
                                                            help="Searches the web for additional context when answering questions")
            
            # Web search depth selection (basic or advanced)
            st.session_state.web_search_depth = st.radio(
                "Search Depth", 
                ["basic", "advanced"], 
                index=0 if st.session_state.web_search_depth == "basic" else 1,
                help="Basic is faster but less thorough, Advanced is more comprehensive but slower"
            )
            
            # Advanced search options in an expander
            with st.expander("Advanced Search Options"):
                if "max_results" not in st.session_state:
                    st.session_state.max_results = 5
                st.session_state.max_results = st.slider(
                    "Max Results", 
                    min_value=1, 
                    max_value=20, 
                    value=st.session_state.max_results,
                    help="Maximum number of search results to retrieve"
                )
                
                if "include_answer" not in st.session_state:
                    st.session_state.include_answer = True
                st.session_state.include_answer = st.checkbox(
                    "Include AI Answer", 
                    value=st.session_state.include_answer,
                    help="Include an AI-generated summary of search results"
                )
                
                if "include_images" not in st.session_state:
                    st.session_state.include_images = False
                st.session_state.include_images = st.checkbox(
                    "Include Images", 
                    value=st.session_state.include_images,
                    help="Include images in search results (if available)"
                )
                
                if "time_range" not in st.session_state:
                    st.session_state.time_range = None
                time_range_options = [None, "day", "week", "month"]
                time_range_index = 0 if st.session_state.time_range is None else time_range_options.index(st.session_state.time_range)
                st.session_state.time_range = st.selectbox(
                    "Time Range", 
                    time_range_options, 
                    index=time_range_index,
                    format_func=lambda x: "Any time" if x is None else x.capitalize(),
                    help="Filter search results by recency"
                )
            
            # Only show search input if web search isn't auto-enabled
            if not st.session_state.web_search_enabled:
                tavily_query = st.text_input("Search the web:")
                
                if st.button("Search"):
                    if tavily_query:
                        with st.spinner("Searching the web..."):
                            search_results = perform_tavily_search(
                                query=tavily_query, 
                                search_depth=st.session_state.web_search_depth,
                                max_results=st.session_state.max_results,
                                include_answer=st.session_state.include_answer,
                                include_images=st.session_state.include_images,
                                time_range=st.session_state.time_range
                            )
                            st.session_state.search_results = search_results
                            
                            if "error" in search_results:
                                st.error(f"Search error: {search_results['error']}")
                            else:
                                st.success("Search completed!")
                    else:
                        st.warning("Please enter a search query.")
            else:
                st.info("Web search will be performed automatically for each question.")
            
            # Display search results if available
            if st.session_state.search_results and "error" not in st.session_state.search_results:
                st.subheader("Search Results")
                results = st.session_state.search_results
                
                if "answer" in results and results["answer"]:
                    st.markdown("### Answer")
                    st.write(results["answer"])
                
                if "results" in results:
                    for i, result in enumerate(results["results"]):
                        with st.expander(f"Result {i+1}: {result.get('title', 'No title')}"):
                            st.markdown(f"**Source:** [{result.get('url', 'No URL')}]({result.get('url', '#')})")
                            st.markdown(f"**Content:** {result.get('content', 'No content')}")
                            
                if st.button("Use these search results with RAG"):
                    if "results" in results:
                        content_texts = [result.get("content", "") for result in results["results"] if "content" in result]
                        source_urls = [result.get("url", "") for result in results["results"] if "content" in result]
                        
                        if content_texts:
                            if st.session_state.conversation:
                                # Use the helper function to add to vectorstore
                                add_search_results_to_vectorstore(content_texts, source_urls)
                                st.success("Search results added to your knowledge base!")
                            else:
                                # Create text chunks from search results
                                search_text_chunks = []
                                metadata_list = []
                                
                                for i, (content, url) in enumerate(zip(content_texts, source_urls)):
                                    # Chunk each search result individually
                                    result_chunks = get_text_chunks(content)
                                    search_text_chunks.extend(result_chunks)
                                    
                                    # Create metadata for each chunk
                                    for _ in result_chunks:
                                        metadata_list.append({
                                            "source": f"Web search result: {url}",
                                            "type": "web_search"
                                        })
                                
                                # Create a new vectorstore with just the search results
                                vectorstore = get_vectorstore(search_text_chunks)
                                
                                # Create conversation chain
                                st.session_state.conversation = get_conversation_chain(vectorstore)
                                
                                st.success("Created knowledge base from search results!")
        
        with tab3:
            st.header("Advanced Settings")
            
            # Performance optimization options
            st.session_state.use_performance_mode = st.checkbox(
                "Use performance mode", 
                value=st.session_state.use_performance_mode,
                help="Optimizes embedding computation for better performance, especially with large documents."
            )
            
            if st.session_state.use_performance_mode:
                st.session_state.embedding_batch_size = st.slider(
                    "Embedding Batch Size", 
                    min_value=1, 
                    max_value=32, 
                    value=st.session_state.embedding_batch_size,
                    help="Larger batch sizes can speed up embedding but use more memory."
                )
            
            # Add a button to clear conversation history
            if st.button("Clear Conversation History"):
                if st.session_state.conversation and hasattr(st.session_state.conversation, 'memory'):
                    st.session_state.conversation.memory.clear()
                st.session_state.messages = []
                st.session_state.chat_history = None
                st.success("Conversation history cleared!")
                
            # Add a button to clear search results
            if st.button("Clear Search Results"):
                st.session_state.search_results = None
                st.success("Search results cleared!")
            
            # Toggle developer mode to show debug information
            if "developer_mode" not in st.session_state:
                st.session_state.developer_mode = False
            
            st.session_state.developer_mode = st.checkbox(
                "Developer Mode", 
                value=st.session_state.developer_mode,
                help="Show additional debug information"
            )
            
            if st.session_state.developer_mode:
                with st.expander("Debug Information"):
                    st.subheader("Session State Variables")
                    for key, value in st.session_state.items():
                        if key not in ["conversation", "chat_history", "messages", "search_results"]:
                            st.write(f"**{key}:** {value}")
                    
                    if st.session_state.conversation:
                        st.subheader("Conversation Chain")
                        st.write("Conversation chain is initialized")
                        st.write(f"Chain type: {st.session_state.conversation.chain_type if hasattr(st.session_state.conversation, 'chain_type') else 'Unknown'}")
                        st.write(f"Has memory: {hasattr(st.session_state.conversation, 'memory')}")


if __name__ == '__main__':
    main()