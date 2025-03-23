import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st


def get_text_chunks(text):
    """
    Split text into chunks of appropriate size for embedding.
    
    This function divides text into manageable chunks using these steps:
    1. Initially breaks text into paragraphs
    2. Processes each paragraph to ensure chunk size doesn't exceed limits (800-1000 chars)
    3. For large paragraphs, splits into sentences 
    4. Further splits oversized chunks using CharacterTextSplitter for maximum compatibility
    
    Args:
        text (str): The text to split into chunks.
        
    Returns:
        list: A list of text chunks ready for embedding.
    """
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
    """
    Perform average pooling on the last hidden states from the model.
    
    This creates a single vector representation from transformer outputs by:
    1. Using the attention mask to zero out padding tokens
    2. Computing a weighted average of all token embeddings
    3. Normalizing by the sum of attention weights
    
    Args:
        last_hidden_states (Tensor): The final layer hidden states from the transformer model
        attention_mask (Tensor): Binary mask indicating which tokens are real vs padding
        
    Returns:
        Tensor: The pooled representation vector for the entire sequence
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class MultilangE5Embeddings(Embeddings):
    """
    A class for generating embeddings using the multilingual E5 model.
    
    This class implements LangChain's Embeddings interface to provide:
    1. Multilingual embedding support via E5 (a powerful multilingual embedding model)
    2. Batch processing for performance optimization
    3. GPU acceleration when available
    4. Proper formatting of inputs with instruct prefixes
    5. Normalized embeddings for better similarity search
    """
    def __init__(self, model_name="intfloat/multilingual-e5-large-instruct", batch_size=8):
        """
        Initialize the E5 embeddings model with the specified parameters.
        
        Args:
            model_name (str): HuggingFace model identifier for the embedding model
            batch_size (int): Number of texts to process at once for better performance
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.batch_size = batch_size
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
    def embed_documents(self, texts):
        """
        Generate embeddings for a list of documents.
        
        This method:
        1. Formats texts with appropriate instruction prefix
        2. Processes documents in batches to optimize memory usage
        3. Handles tokenization, model inference, and pooling
        4. Normalizes the resulting embeddings for cosine similarity
        
        Args:
            texts (list): A list of document texts to embed
            
        Returns:
            list: A list of embedding vectors (one per input document)
        """
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
        """
        Generate an embedding for a query.
        
        Unlike document embedding, this method:
        1. Uses a query-specific instruction format
        2. Processes a single input (the query)
        3. Returns a single normalized vector
        
        Args:
            text (str): The query text to embed
            
        Returns:
            list: The query embedding vector
        """
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
        """
        Alias for embed_query to support LangChain's expected interface.
        
        Args:
            text (str): The text to embed
            
        Returns:
            list: The embedding vector
        """
        return self.embed_query(text)
    
    # This makes the object callable directly
    def __call__(self, text):
        """
        Make the class instance directly callable.
        
        Args:
            text (str): The text to embed
            
        Returns:
            list: The embedding vector
        """
        return self.embed_text(text)


def get_vectorstore(text_chunks, embedding_batch_size=8, use_performance_mode=True):
    """
    Create a vector store from text chunks.
    
    This function:
    1. Initializes the embedding model with appropriate batch size
    2. Creates a FAISS vector store with the specified text chunks
    3. Applies performance optimization settings if enabled
    
    Args:
        text_chunks (list): A list of text chunks to embed
        embedding_batch_size (int): The batch size for embedding generation
        use_performance_mode (bool): Whether to optimize for performance
        
    Returns:
        FAISS: A vector store containing the embeddings for efficient similarity search
    """
    from langchain_community.vectorstores import FAISS
    
    # Apply performance settings
    batch_size = embedding_batch_size if use_performance_mode else 4
    embeddings = MultilangE5Embeddings(batch_size=batch_size)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore 