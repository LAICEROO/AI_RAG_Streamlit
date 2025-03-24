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
from utils.embedding_utils import get_text_chunks, average_pool, MultilangE5Embeddings, get_vectorstore
from utils.hybrid_search import get_hybrid_retriever

def get_pdf_text(uploaded_files):
    """
    Extract text from various document formats.
    
    This function processes multiple document types:
    1. PDF: Uses PyPDF2 to extract text from each page
    2. TXT: Plain text files decoded with UTF-8
    3. DOCX: Microsoft Word documents using python-docx
    4. CSV: Tabular data using pandas
    5. JSON: Structured data pretty-printed
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        
    Returns:
        str: Combined text content from all files, with appropriate spacing
    """
    text = ""
    
    for file in uploaded_files:
        # Get file extension
        file_ext = file.name.split('.')[-1].lower()
        
        try:
            # Handle different file types
            if file_ext == 'pdf':
                # Handle PDF files using PyPDF2
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n\n"
                    
            elif file_ext == 'txt':
                # Handle text files with UTF-8 decoding
                text += file.getvalue().decode('utf-8') + "\n\n"
                
            elif file_ext == 'docx':
                # Handle DOCX files using python-docx (optional dependency)
                try:
                    from docx import Document
                    doc = Document(file)
                    for para in doc.paragraphs:
                        text += para.text + "\n"
                    text += "\n\n"
                except ImportError:
                    # If python-docx is not installed
                    st.error(f"Missing python-docx library. Install with 'pip install python-docx' to process {file.name}")
                    continue
                    
            elif file_ext == 'csv':
                # Handle CSV files using pandas (optional dependency)
                try:
                    import pandas as pd
                    df = pd.read_csv(file)
                    text += df.to_string() + "\n\n"
                except ImportError:
                    st.error(f"Missing pandas library. Install with 'pip install pandas' to process {file.name}")
                    continue
                    
            elif file_ext == 'json':
                # Handle JSON files with pretty-printing
                import json
                content = json.loads(file.getvalue().decode('utf-8'))
                text += json.dumps(content, indent=2) + "\n\n"
                
            else:
                # Skip unsupported file types with a warning
                st.warning(f"Unsupported file type: {file_ext} - {file.name} was skipped")
                
        except Exception as e:
            # Handle any errors during processing with detailed error reporting
            st.error(f"Error processing file {file.name}: {str(e)}")
            import traceback
            print(f"Error details for {file.name}: {traceback.format_exc()}")
            continue
            
    return text


def get_conversation_chain(vectorstore, text_chunks=None):
    """
    Create a conversation chain for RAG (Retrieval Augmented Generation) using Mistral AI.
    
    This function:
    1. Initializes the LLM using Mistral AI API
    2. Sets up conversation memory to maintain chat history
    3. Creates an appropriate retriever (standard or hybrid) based on settings
    4. Builds a conversational retrieval chain that combines all components
    
    Args:
        vectorstore: Vector database containing document embeddings
        text_chunks: Optional raw text chunks for hybrid search
        
    Returns:
        ConversationalRetrievalChain: Chain that handles conversational RAG
    """
    # Use Mistral AI directly
    api_key = os.environ["MISTRAL_API_KEY"]
    
    # Initialize the ChatMistralAI using LangChain's integration
    llm = ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.3,  # Lower temperature for more factual responses
        max_tokens=16384,  # Large context window
        top_p=0.9  # Slightly constrained sampling
    )

    # Use the updated memory API to avoid deprecation warnings
    from langchain.memory import ConversationBufferMemory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage

    # Setup conversation memory to track chat history
    memory = ConversationBufferMemory(
        memory_key='chat_history',  # Key used to access history in prompt
        return_messages=True,       # Return message objects, not strings
        output_key='answer'         # Store AI responses under this key
    )
    
    # Create hybrid retriever if text_chunks are provided and hybrid search is enabled
    if text_chunks and st.session_state.use_hybrid_search:
        try:
            # Setup hybrid retriever with semantic search, BM25, and optional reranking
            retriever = get_hybrid_retriever(
                vectorstore=vectorstore,
                text_chunks=text_chunks,
                k=st.session_state.retrieve_k,                    # Number of documents to retrieve
                semantic_weight=st.session_state.semantic_weight, # Balance between semantic and BM25
                use_reranking=st.session_state.use_contextual_reranking  # Whether to rerank results
            )
            
            # Check if we got a HybridRetriever or a fallback retriever
            if not hasattr(retriever, 'add_texts'):
                st.warning("Hybrid search creation failed. Using standard search instead.")
                st.session_state.use_hybrid_search = False
                
        except Exception as e:
            st.error(f"Error setting up hybrid search: {e}")
            st.warning("Falling back to standard vector search.")
            # Fall back to standard vector retrieval if hybrid setup fails
            retriever = vectorstore.as_retriever(search_kwargs={"k": st.session_state.retrieve_k})
            st.session_state.use_hybrid_search = False
    else:
        # Use standard vectorstore retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": st.session_state.retrieve_k})
    
    # Create the chain that combines LLM, retriever, and memory
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True,                  # Show debug info during execution
        return_source_documents=True,  # Return the source docs for citation
        chain_type="stuff"             # "Stuff" puts all docs in a single prompt
    )
    
    return conversation_chain


def handle_userinput(user_question):
    """
    Process user questions, optionally perform web search, and generate LLM responses.
    
    This complex function handles:
    1. Optional web search to supplement document knowledge
    2. Integration of web results with RAG system
    3. Invoking the conversation chain with appropriate context
    4. Displaying results and source documents to the user
    5. Managing chat history and memory
    
    Args:
        user_question: The user's input question
    """
    web_context = ""
    web_sources = []
    
    # Phase 1: Web Search Integration - Optionally perform web search to supplement document knowledge
    # Check if we should perform a web search alongside the document search
    if st.session_state.web_search_enabled and user_question:
        with st.spinner("Searching the web for additional context..."):
            # Call Tavily search API to get real-time web information
            # This extends our knowledge beyond the documents provided by the user
            search_results = perform_tavily_search(
                query=user_question, 
                search_depth=st.session_state.web_search_depth,
                max_results=st.session_state.max_results,
                include_answer=st.session_state.include_answer,
                include_images=st.session_state.include_images,
                time_range=st.session_state.time_range
            )
            
            # If we got valid search results (no errors), process them
            if search_results and "error" not in search_results:
                # Extract the AI-generated summary of web results if available
                if "answer" in search_results and search_results["answer"]:
                    web_context = search_results["answer"]
                
                # Extract individual web sources for citation and display
                if "results" in search_results and len(search_results["results"]) > 0:
                    for result in search_results["results"]:
                        web_sources.append({
                            "title": result.get("title", "No title"),
                            "url": result.get("url", "#"),
                            # Truncate long content for display purposes
                            "content": result.get("content", "")[:150] + "..." if len(result.get("content", "")) > 150 else result.get("content", "")
                        })
                
                if "results" in search_results and st.session_state.conversation:
                    # Extract content and URLs from search results
                    content_texts = [result.get("content", "") for result in search_results["results"] if "content" in result]
                    source_urls = [result.get("url", "") for result in search_results["results"] if "content" in result]
                    
                    if content_texts:
                        # Add web search results to vectorstore for retrieval
                        # This allows the hybrid search to incorporate web results
                        add_search_results_to_vectorstore(content_texts, source_urls)
    
    try:
        # Decision point: Check if we have web search results to incorporate into the response
        if web_context and web_sources:
            # Format web information to be explicitly used by the LLM
            # This creates a structured representation of web search results
            formatted_web_info = f"""
Web search found the following information relevant to your question:

{web_context}

Sources:
"""
            # Add all web sources with their URLs for proper citation
            for i, source in enumerate(web_sources):
                formatted_web_info += f"{i+1}. {source['title']} - {source['url']}\n"
            
            # PROMPT ENGINEERING: Enhanced instructions for the model when web search results are available
            # This specific prompt construction addresses several key issues:
            # 1. Establishes the assistant's role and behavior expectations
            # 2. Explicitly prevents "I don't know" default responses 
            # 3. Ensures the model prioritizes information from search results
            # 4. Structures the input with clear delineation between query and search content
            enhanced_question = f"""
You are a helpful assistant that always uses information from search results to provide thorough answers. 
You prioritize information from provided sources over your general knowledge.
When information is available, never reply with just "I don't know" or "Nie wiem".

User question: {user_question}

Use the following information from a recent web search to help with your answer:
{formatted_web_info}

IMPORTANT: Use this information to provide a complete answer. Never respond with just "I don't know" or "Nie wiem" if the information is available in these sources.
Provide a thorough answer using the information above and cite sources when appropriate.
"""
        else:
            # PROMPT ENGINEERING: Instructions for when only document search is available (no web results)
            # This alternative prompt handles the case where we rely solely on document retrieval:
            # 1. Sets assistant expectations for document-based queries
            # 2. Still prevents "I don't know" responses even without web search 
            # 3. Encourages providing partial information even when exact answers aren't found
            enhanced_question = f"""
You are a helpful assistant that always tries to provide valuable information based on available documents.
You should be resourceful and helpful rather than saying you don't know when asked a question.

User question: {user_question}

Please answer the question using information from the available documents.
If the exact answer isn't found, provide relevant information you have, but DO NOT respond with only "I don't know" or "Nie wiem".
"""
        
        # Send the enhanced question (with all the instructions and context) to the LLM
        # The conversation chain will use its retriever to find relevant documents
        # and incorporate them with the web search results we've provided
        response = st.session_state.conversation.invoke({'question': enhanced_question})
        
        # Update chat history to show the original question rather than our enhanced prompt
        # This keeps the conversation history clean and natural for the user
        if enhanced_question != user_question and hasattr(st.session_state.conversation, 'memory'):
            # Fix the memory to show the original question, not the enhanced one
            # This ensures chat history appears natural to the user
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
                
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        import traceback
        st.error(traceback.format_exc())


def perform_tavily_search(query, search_depth="basic", max_results=10, include_answer=True, include_images=False, time_range=None):
    """
    Perform a search using the Tavily API with improved error handling and options.
    
    This function integrates with Tavily's web search API to retrieve:
    1. Relevant web pages based on the query
    2. Optional AI-generated summaries of search results
    3. Sources with metadata like titles and URLs
    4. Configurable search options (depth, time range, etc.)
    
    Args:
        query (str): The search query
        search_depth (str): Either "basic" (faster) or "advanced" (more thorough)
        max_results (int): Maximum number of results to return (up to 30)
        include_answer (bool): Whether to include an AI-generated answer
        include_images (bool): Whether to include images in the results
        time_range (str, optional): Time range for results ("day", "week", "month")
        
    Returns:
        dict: The search results from Tavily containing answer and/or sources
    """
    url = "https://api.tavily.com/search"
    
    # API Key Validation - Ensure we have a valid API key from environment or session state
    # This is crucial for the web search functionality to work properly
    api_key = os.environ.get("TAVILY_API_KEY") or st.session_state.get("TAVILY_API_KEY")
    if not api_key:
        # Return error dictionary instead of raising exception to allow graceful fallback
        return {
            "error": "Tavily API key not found. Please set the TAVILY_API_KEY environment variable."
        }
    
    # Request Configuration - Prepare the search payload with all parameters
    # This builds the complete request with appropriate options for the Tavily API
    payload = {
        "query": query,
        "search_depth": search_depth,            # Controls search thoroughness vs. speed
        "include_answer": include_answer,        # Whether to include AI-generated summary
        "include_images": include_images,        # Whether to include image results
        "max_results": min(max_results, 30)      # Respect Tavily's maximum limit
    }
    
    # Add optional time range filter if specified
    # This allows restricting results to recent content
    if time_range:
        payload["time_range"] = time_range
    
    # Domain Filtering - Include only specified domains if configured
    # This allows the user to focus search on specific websites
    if st.session_state.get("include_domains"):
        payload["include_domains"] = st.session_state.include_domains
        
    # Domain Exclusion - Exclude specified domains if configured
    # This allows the user to avoid certain websites in results
    if st.session_state.get("exclude_domains"):
        payload["exclude_domains"] = st.session_state.exclude_domains
    
    # Set up API request headers with authentication
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        # Execute API Request - Make the actual call to Tavily search API
        # This sends our configured query and retrieves search results
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Explicitly raise exceptions for HTTP errors
        
        # Parse the JSON response from the API
        results = response.json()
        
        # Debugging - Log detailed search information in developer mode
        # This helps troubleshoot search-related issues
        if st.session_state.get("developer_mode"):
            print(f"Tavily search results - query: '{query}', results count: {len(results.get('results', []))}")
            if "answer" in results:
                print(f"Tavily answer: {results['answer'][:100]}...")
        
        return results
        
    except requests.exceptions.RequestException as e:
        # Error Handling - Handle HTTP and API-specific errors
        # This provides detailed error information for debugging
        error_message = f"Error during Tavily search: {e}"
        if hasattr(e, 'response') and e.response:
            try:
                # Try to extract API error details from response
                error_details = e.response.json()
                error_message += f" - {error_details.get('message', 'No details')}"
            except:
                # Fallback to HTTP status code if JSON parsing fails
                error_message += f" - Status code: {e.response.status_code}"
        
        # Display error to the user and return error information
        st.error(error_message)
        return {"error": error_message}
    except Exception as e:
        # Catch-All Error Handler - For unexpected errors
        # This ensures the function doesn't crash the application
        error_message = f"Unexpected error during Tavily search: {e}"
        st.error(error_message)
        return {"error": error_message}


def add_search_results_to_vectorstore(content_texts, source_urls):
    """
    Add web search results to the vectorstore for hybrid retrieval.
    
    This function:
    1. Chunks web content into appropriate sizes
    2. Creates metadata for each chunk with source tracking
    3. Adds chunks to existing vectorstore (and BM25 if using hybrid search)
    4. Updates the conversation chain with the new knowledge
    
    Args:
        content_texts: List of text contents from search results
        source_urls: List of URLs corresponding to each content text
    """
    # Initialization - Create containers for processed chunks and their metadata
    search_text_chunks = []
    metadata_list = []
    
    # Content Processing - Process each search result and create chunks with metadata
    for i, (content, url) in enumerate(zip(content_texts, source_urls)):
        # Text Chunking - Split content into manageable chunks for embedding
        # This is critical for proper semantic understanding and retrieval
        result_chunks = get_text_chunks(content)
        search_text_chunks.extend(result_chunks)
        
        # Metadata Creation - Add source tracking to each chunk
        # This ensures we can trace back to the original web sources
        for _ in result_chunks:
            metadata_list.append({
                "source": f"Web search result: {url}",
                "type": "web_search"
            })
    
    # Only proceed if we have valid chunks to add
    if search_text_chunks:
        try:
            # Vectorstore Integration - Two different approaches based on retriever type
            if st.session_state.use_hybrid_search and hasattr(st.session_state.conversation.retriever, 'add_texts'):
                # Hybrid Retriever Path - Update both vectorstore and BM25 in one operation
                # This keeps semantic and keyword search capabilities in sync
                st.session_state.conversation.retriever.add_texts(
                    texts=search_text_chunks,
                    metadatas=metadata_list
                )
            else:
                # Standard Retriever Path - Need to get the vectorstore and update it directly
                # Determine which type of retriever we're using to access its vectorstore
                if hasattr(st.session_state.conversation.retriever, '_vectorstore'):
                    # For hybrid retriever
                    existing_vectorstore = st.session_state.conversation.retriever._vectorstore
                elif hasattr(st.session_state.conversation.retriever, 'vectorstore'):
                    # For some standard retrievers
                    existing_vectorstore = st.session_state.conversation.retriever.vectorstore
                else:
                    # For FAISS vectorstore retriever
                    existing_vectorstore = st.session_state.conversation.retriever.vectorstore
                
                # Add the new texts to the existing vectorstore with metadata
                # This updates the vector database with new knowledge
                existing_vectorstore.add_texts(
                    texts=search_text_chunks,
                    metadatas=metadata_list
                )
                
                # Conversation Chain Update - Recreate with updated knowledge
                # This ensures the new information is available for retrieval
                if st.session_state.use_hybrid_search:
                    # For hybrid search, update all_text_chunks and recreate with both sources
                    # This maintains the BM25 search capabilities alongside vector search
                    st.session_state.all_text_chunks.extend(search_text_chunks)
                    
                    st.session_state.conversation = get_conversation_chain(
                        existing_vectorstore, 
                        search_text_chunks + st.session_state.all_text_chunks
                    )
                else:
                    # For standard search, just recreate with the updated vectorstore
                    st.session_state.conversation = get_conversation_chain(existing_vectorstore)
                    
            # Success notification - Let the user know we've incorporated web results
            st.success("Search results added to your knowledge base!")
        except Exception as e:
            # Error Handling - Report issues with adding to knowledge base
            # This helps troubleshoot when integration fails
            st.error(f"Error adding search results to knowledge base: {str(e)}")
            import traceback
            print(f"Error details: {traceback.format_exc()}")


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
    if "all_text_chunks" not in st.session_state:
        st.session_state.all_text_chunks = []
    if "use_hybrid_search" not in st.session_state:
        st.session_state.use_hybrid_search = True
    if "retrieve_k" not in st.session_state:
        st.session_state.retrieve_k = 10
    if "semantic_weight" not in st.session_state:
        st.session_state.semantic_weight = 0.7
    if "use_contextual_reranking" not in st.session_state:
        st.session_state.use_contextual_reranking = True
    
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
        tab1, tab2, tab3, tab4 = st.tabs(["Documents", "Web Search", "Retrieval Settings", "Advanced Settings"])
        
        with tab1:
            st.header("Your documents")
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process'", accept_multiple_files=True,
                type=["pdf", "txt", "docx", "csv", "json"]  # Support more file types
            )
            
            # Add info about unlimited file uploads
            st.info("You can upload as many documents as needed. The only limitation is your system memory.")
            
            # Add option to clear uploaded files
            if st.button("Clear Uploaded Files", key="clear_uploaded"):
                st.session_state.uploaded_files = None
                st.session_state.all_text_chunks = []
                st.rerun()
        
            if st.button("Process"):
                if pdf_docs:
                    with st.spinner("Processing documents..."):
                        # get pdf text
                        raw_text = get_pdf_text(pdf_docs)

                        # get the text chunks
                        text_chunks = get_text_chunks(raw_text)
                        
                        # Store all text chunks for hybrid search
                        st.session_state.all_text_chunks = text_chunks

                        # create vector store
                        vectorstore = get_vectorstore(
                            text_chunks, 
                            embedding_batch_size=st.session_state.embedding_batch_size, 
                            use_performance_mode=st.session_state.use_performance_mode
                        )

                        try:
                            # create conversation chain with appropriate retriever
                            st.session_state.conversation = get_conversation_chain(vectorstore, text_chunks)
                            st.success(f"✅ Successfully processed {len(pdf_docs)} documents with {len(text_chunks)} text chunks!")
                        except Exception as e:
                            st.error(f"Error creating conversation chain: {str(e)}")
                            import traceback
                            print(f"Error details: {traceback.format_exc()}")
                            # Fallback to standard retriever if hybrid fails
                            st.session_state.use_hybrid_search = False
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.success(f"✅ Documents processed! (using standard search) - {len(text_chunks)} text chunks created.")
                
                    # Clear messages when new documents are processed
                    st.session_state.messages = []
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"Documents processed! You can now ask questions about your {len(pdf_docs)} document(s)."
                    })
                else:
                    st.error("Please upload at least one document file.")
        
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
                    st.session_state.max_results = 10  # Default to 10 now, not 5
                st.session_state.max_results = st.slider(
                    "Max Results", 
                    min_value=1, 
                    max_value=30,  # Increased to Tavily's maximum of 30
                    value=st.session_state.max_results,
                    help="Maximum number of search results to retrieve (up to 30)"
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
                
                # Add domain inclusion/exclusion options
                st.subheader("Domain Filters")
                
                # Initialize domain lists if not already in session state
                if "include_domains" not in st.session_state:
                    st.session_state.include_domains = []
                if "exclude_domains" not in st.session_state:
                    st.session_state.exclude_domains = []
                    
                # Temporary variables for domain input
                col1, col2 = st.columns(2)
                
                with col1:
                    include_domain = st.text_input(
                        "Include Domain", 
                        placeholder="e.g., example.com",
                        help="Only include results from specific domains"
                    )
                    
                    if st.button("Add Include Domain"):
                        if include_domain and include_domain not in st.session_state.include_domains:
                            st.session_state.include_domains.append(include_domain)
                            st.rerun()
                            
                    if st.session_state.include_domains:
                        st.write("Included domains:")
                        for i, domain in enumerate(st.session_state.include_domains):
                            col1_1, col1_2 = st.columns([4, 1])
                            with col1_1:
                                st.text(domain)
                            with col1_2:
                                if st.button("❌", key=f"del_include_{i}"):
                                    st.session_state.include_domains.pop(i)
                                    st.rerun()
                
                with col2:
                    exclude_domain = st.text_input(
                        "Exclude Domain", 
                        placeholder="e.g., example.com",
                        help="Exclude results from specific domains"
                    )
                    
                    if st.button("Add Exclude Domain"):
                        if exclude_domain and exclude_domain not in st.session_state.exclude_domains:
                            st.session_state.exclude_domains.append(exclude_domain)
                            st.rerun()
                            
                    if st.session_state.exclude_domains:
                        st.write("Excluded domains:")
                        for i, domain in enumerate(st.session_state.exclude_domains):
                            col2_1, col2_2 = st.columns([4, 1])
                            with col2_1:
                                st.text(domain)
                            with col2_2:
                                if st.button("❌", key=f"del_exclude_{i}"):
                                    st.session_state.exclude_domains.pop(i)
                                    st.rerun()
            
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
                                
                                # Store these chunks for hybrid search
                                st.session_state.all_text_chunks = search_text_chunks
                                
                                # Create a new vectorstore with just the search results
                                vectorstore = get_vectorstore(
                                    search_text_chunks,
                                    embedding_batch_size=st.session_state.embedding_batch_size,
                                    use_performance_mode=st.session_state.use_performance_mode
                                )
                                
                                try:
                                    # Create conversation chain with appropriate retriever
                                    st.session_state.conversation = get_conversation_chain(vectorstore, search_text_chunks)
                                    st.success("Created knowledge base from search results!")
                                except Exception as e:
                                    st.error(f"Error creating conversation chain: {str(e)}")
                                    import traceback
                                    print(f"Error details: {traceback.format_exc()}")
                                    # Fallback to standard retriever if hybrid fails
                                    st.session_state.use_hybrid_search = False
                                    st.session_state.conversation = get_conversation_chain(vectorstore)
                                    st.success("Created knowledge base from search results (using standard search)!")
        
        with tab3:
            st.header("Retrieval Settings")
            
            # Hybrid search option
            st.session_state.use_hybrid_search = st.checkbox(
                "Use Hybrid Search", 
                value=st.session_state.use_hybrid_search,
                help="Combines semantic search (FAISS), keyword search (BM25), and contextual reranking"
            )
            
            # If hybrid search is enabled, show additional settings
            if st.session_state.use_hybrid_search:
                # Number of documents to retrieve
                st.session_state.retrieve_k = st.slider(
                    "Number of documents to retrieve", 
                    min_value=3, 
                    max_value=50,  # Increased from 20 to 50
                    value=st.session_state.retrieve_k,
                    help="More documents means more context for the LLM but may cause slower responses"
                )
                
                # Weight between semantic and keyword search
                st.session_state.semantic_weight = st.slider(
                    "Semantic Search Weight", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=st.session_state.semantic_weight,
                    help="Higher values favor semantic search (meaning), lower values favor BM25 (keywords)"
                )
                
                # Option to enable contextual reranking
                st.session_state.use_contextual_reranking = st.checkbox(
                    "Use GPT-4o mini Contextual Reranking", 
                    value=st.session_state.use_contextual_reranking,
                    help="Uses GPT 4o-mini model to rerank retrieved documents based on relevance to the query"
                )
                
                # If retrieval settings are changed, update the conversation
                if st.button("Apply Retrieval Settings"):
                    if st.session_state.conversation and hasattr(st.session_state.conversation, 'retriever'):
                        try:
                            # Get the current vectorstore
                            if hasattr(st.session_state.conversation.retriever, '_vectorstore'):
                                # For hybrid retriever
                                vectorstore = st.session_state.conversation.retriever._vectorstore
                            elif hasattr(st.session_state.conversation.retriever, 'vectorstore'):
                                # For some standard retrievers
                                vectorstore = st.session_state.conversation.retriever.vectorstore
                            else:
                                # For FAISS vectorstore retriever
                                vectorstore = st.session_state.conversation.retriever.vectorstore
                            
                            # Recreate conversation chain with current vectorstore and updated retrieval settings
                            st.session_state.conversation = get_conversation_chain(
                                vectorstore, 
                                st.session_state.all_text_chunks
                            )
                            
                            st.success("✅ Retrieval settings applied successfully!")
                        except Exception as e:
                            st.error(f"Error applying retrieval settings: {str(e)}")
                            import traceback
                            print(f"Error details: {traceback.format_exc()}")
                    else:
                        st.warning("No conversation chain to update. Please process documents first.")
            else:
                # Regular retrieval settings
                st.session_state.retrieve_k = st.slider(
                    "Number of documents to retrieve", 
                    min_value=3, 
                    max_value=50,  # Increased from 20 to 50
                    value=st.session_state.retrieve_k,
                    help="More documents means more context for the LLM but may cause slower responses"
                )
                
                # Button to apply changes
                if st.button("Apply Retrieval Settings"):
                    if st.session_state.conversation and hasattr(st.session_state.conversation.retriever, "search_kwargs"):
                        # Update the k parameter in the retriever
                        st.session_state.conversation.retriever.search_kwargs["k"] = st.session_state.retrieve_k
                        st.success("✅ Retrieval settings applied successfully!")
        
        with tab4:
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
                        if key not in ["conversation", "chat_history", "messages", "search_results", "all_text_chunks"]:
                            st.write(f"**{key}:** {value}")
                    
                    if st.session_state.conversation:
                        st.subheader("Conversation Chain")
                        st.write("Conversation chain is initialized")
                        st.write(f"Chain type: {st.session_state.conversation.chain_type if hasattr(st.session_state.conversation, 'chain_type') else 'Unknown'}")
                        st.write(f"Has memory: {hasattr(st.session_state.conversation, 'memory')}")
                        st.write(f"Retriever type: {type(st.session_state.conversation.retriever).__name__ if hasattr(st.session_state.conversation, 'retriever') else 'Unknown'}")
                        if st.session_state.use_hybrid_search:
                            st.write(f"Using hybrid retriever with semantic weight: {st.session_state.semantic_weight}")
                            st.write(f"GPT-4o mini reranking enabled: {st.session_state.use_contextual_reranking}")
                    
                    st.subheader("Text Chunks")
                    st.write(f"Number of text chunks: {len(st.session_state.all_text_chunks)}")


if __name__ == '__main__':
    main()