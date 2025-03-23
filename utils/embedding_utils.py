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
    
    Args:
        text (str): The text to split into chunks.
        
    Returns:
        list: A list of text chunks.
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
    
    Args:
        last_hidden_states (Tensor): The last hidden states from the model.
        attention_mask (Tensor): The attention mask for the input.
        
    Returns:
        Tensor: The pooled representation.
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class MultilangE5Embeddings(Embeddings):
    """
    A class for generating embeddings using the multilingual E5 model.
    """
    def __init__(self, model_name="intfloat/multilingual-e5-large-instruct", batch_size=8):
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
        
        Args:
            texts (list): A list of document texts.
            
        Returns:
            list: A list of embeddings.
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
        
        Args:
            text (str): The query text.
            
        Returns:
            list: The query embedding.
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
        return self.embed_query(text)
    
    # This makes the object callable directly
    def __call__(self, text):
        return self.embed_text(text)


def get_vectorstore(text_chunks, embedding_batch_size=8, use_performance_mode=True):
    """
    Create a vector store from text chunks.
    
    Args:
        text_chunks (list): A list of text chunks to embed.
        embedding_batch_size (int, optional): The batch size for embedding. Defaults to 8.
        use_performance_mode (bool, optional): Whether to use performance mode. Defaults to True.
        
    Returns:
        FAISS: A vector store containing the embeddings.
    """
    from langchain_community.vectorstores import FAISS
    
    # Apply performance settings
    batch_size = embedding_batch_size if use_performance_mode else 4
    embeddings = MultilangE5Embeddings(batch_size=batch_size)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore 