import numpy as np
import requests
from typing import Dict, List, Any, Optional, Tuple
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
import streamlit as st
import os
from openai import OpenAI

class HybridRetriever(BaseRetriever):
    """
    A hybrid retriever that combines semantic search (FAISS), BM25, and contextual retrieval.
    
    This implements a two-stage retrieval process with reranking:
    1. First stage: Combined keyword (BM25) and semantic (FAISS) search
    2. Second stage: Contextual reranking using GPT-4o-mini model
    
    The hybrid approach helps balance the strengths of different retrieval methods:
    - BM25: Good for exact keyword matching
    - Semantic search: Better at understanding meaning and context
    - Reranking: Improves relevance by considering query-document relationships
    """
    
    def __init__(
        self, 
        vectorstore, 
        texts: List[str],
        k: int = 10,
        alpha: float = 0.5,
        use_reranking: bool = True,
        openai_api_key: Optional[str] = None,
        rerank_top_k: int = 20
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            vectorstore: The vector store for semantic search
            texts: The raw text documents for BM25 indexing
            k: Number of documents to retrieve
            alpha: Weight for combining results (0 = only BM25, 1 = only semantic)
            use_reranking: Whether to use GPT-4o-mini for contextual reranking
            openai_api_key: OpenAI API key for GPT-4o-mini model
            rerank_top_k: How many documents to rerank (higher = better but slower)
        """
        # Initialize the parent class
        super().__init__()
        
        # Store parameters as instance variables
        self._vectorstore = vectorstore
        self._k = k
        self._alpha = alpha
        self._use_reranking = use_reranking
        self._openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
        self._rerank_top_k = min(rerank_top_k, k * 2)
        
        # Initialize OpenAI client if reranking is enabled
        if self._use_reranking and self._openai_api_key:
            self._openai_client = OpenAI(api_key=self._openai_api_key)
        else:
            self._openai_client = None
        
        # Create BM25 retriever from texts
        try:
            # Convert Document objects to text if needed
            text_contents = []
            for text in texts:
                if isinstance(text, Document):
                    text_contents.append(text.page_content)
                elif isinstance(text, str):
                    text_contents.append(text)
                else:
                    raise ValueError(f"Unexpected text type: {type(text)}")
            
            # Filter out empty documents or those that are too long
            filtered_texts = []
            for text in text_contents:
                if not text or not text.strip():
                    continue
                    
                # Truncate very long documents for BM25 to work efficiently
                if len(text) > 10000:
                    filtered_texts.append(text[:10000])
                else:
                    filtered_texts.append(text)
            
            # Only create BM25 if we have valid texts
            if filtered_texts:
                self._bm25_retriever = BM25Retriever.from_texts(filtered_texts)
                self._bm25_retriever.k = k
                
                # Create ensemble retriever (combines both retrievers)
                self._ensemble_retriever = EnsembleRetriever(
                    retrievers=[self._bm25_retriever, self._vectorstore.as_retriever(search_kwargs={"k": k})],
                    weights=[1-alpha, alpha]
                )
            else:
                st.warning("No valid texts for BM25 indexing. Using only vector search.")
                self._bm25_retriever = None
                self._ensemble_retriever = None
        except Exception as e:
            st.error(f"Error initializing BM25: {str(e)}")
            print(f"Error initializing BM25: {str(e)}")
            # Fallback to just using vector search if BM25 initialization fails
            self._bm25_retriever = None
            self._ensemble_retriever = None
    
    @property
    def vectorstore(self):
        """
        Access the vector store used for semantic search.
        
        Returns:
            The vector store object
        """
        return self._vectorstore
        
    @property
    def k(self):
        """
        Access the number of documents to retrieve.
        
        Returns:
            int: Number of documents to retrieve
        """
        return self._k
        
    @property
    def alpha(self):
        """
        Access the weight assigned to semantic search.
        
        Returns:
            float: Weight between 0 and 1 (0 = only BM25, 1 = only semantic)
        """
        return self._alpha
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Get documents relevant to the query.
        
        This is the main retrieval method that:
        1. Performs first-stage retrieval using ensemble of BM25 and semantic search
        2. Optionally applies second-stage contextual reranking
        3. Returns the top-k most relevant documents
        
        Args:
            query: The query string
            run_manager: Callback manager for LangChain
            
        Returns:
            List of relevant documents ordered by relevance
        """
        try:
            # First-stage retrieval using ensemble of BM25 and semantic search
            if self._ensemble_retriever:
                docs = self._ensemble_retriever.get_relevant_documents(query)
            else:
                # Fallback to just vector search if ensemble retriever is not available
                docs = self._vectorstore.as_retriever(search_kwargs={"k": self._k}).get_relevant_documents(query)
            
            # If reranking is disabled, return the ensemble results
            if not self._use_reranking or not self._openai_api_key:
                return docs[:self._k]
            
            # For reranking, get more initial candidates to then filter down
            if len(docs) > self._rerank_top_k:
                docs = docs[:self._rerank_top_k]
            
            # Second-stage: perform contextual reranking with GPT-4o-mini
            reranked_docs = self._rerank_with_gpt4o(query, docs)
            
            # Return the specified number of top documents
            return reranked_docs[:self._k]
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            print(f"Error retrieving documents: {str(e)}")
            # Fallback to just vector search if there's an error
            return self._vectorstore.as_retriever(search_kwargs={"k": self._k}).get_relevant_documents(query)
    
    def _rerank_with_gpt4o(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Rerank documents using GPT-4o-mini model for contextual relevance.
        
        This second-stage ranking:
        1. Gets relevance scores from GPT-4o-mini model for each document
        2. Sorts documents by these relevance scores
        3. Returns documents in the new, reranked order
        
        Args:
            query: The query string
            docs: List of candidate documents from first-stage retrieval
            
        Returns:
            Reranked list of documents ordered by contextual relevance
        """
        try:
            # Prepare inputs for the GPT-4o-mini model
            doc_texts = [doc.page_content for doc in docs]
            doc_scores = self._get_gpt4o_scores(query, doc_texts)
            
            # Create document-score pairs and sort by score
            doc_score_pairs = list(zip(docs, doc_scores))
            reranked_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
            
            # Extract just the documents, now in reranked order
            reranked_docs = [doc for doc, _ in reranked_pairs]
            
            return reranked_docs
        except Exception as e:
            st.warning(f"Error during GPT-4o-mini reranking: {str(e)}. Using original ranking.")
            print(f"Error during GPT-4o-mini reranking: {str(e)}")
            return docs
    
    def _get_gpt4o_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        Get relevance scores from GPT-4o-mini model via OpenAI API.
        
        This uses a direct relevance scoring approach:
        1. Asks GPT-4o-mini to evaluate document relevance to query
        2. Returns normalized relevance scores for ranking
        
        Args:
            query: The query string
            documents: List of document texts
            
        Returns:
            List of relevance scores for each document
        """
        if not self._openai_client:
            # If no API client, use a simple keyword-based relevance score instead
            return self._get_keyword_similarity_scores(query, documents)
            
        scores = []
        
        for doc in documents:
            # Truncate long documents for API limits
            doc_truncated = doc[:2048] if len(doc) > 2048 else doc
            
            # Skip empty documents
            if not doc_truncated.strip():
                scores.append(0.0)
                continue
                
            try:
                # Format prompt for relevance scoring
                prompt = f"""Rate the relevance of this document to the query on a scale of 0-10.
                Query: {query}
                Document: {doc_truncated}
                Score (0-10):"""
                
                # Get relevance score from GPT-4o-mini
                completion = self._openai_client.chat.completions.create(
                    model="gpt-4o-mini-2024-07-18",
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    max_tokens=5,
                    temperature=0.0
                )
                
                # Extract the score from the response
                try:
                    score = float(completion.choices[0].message.content.strip())
                    # Normalize score to 0-1 range
                    score = max(0.0, min(10.0, score)) / 10.0
                except (ValueError, TypeError):
                    # If we can't parse the score, use keyword similarity as fallback
                    score = self._get_keyword_similarity_score(query, doc)
                
                scores.append(score)
                
            except Exception as e:
                # If API fails, use keyword similarity as fallback
                print(f"GPT-4o-mini API error: {str(e)}")
                fallback_score = self._get_keyword_similarity_score(query, doc)
                scores.append(fallback_score)
        
        # Normalize scores if possible
        if scores:
            max_score = max(scores)
            if max_score > 0:
                scores = [s/max_score for s in scores]
        
        return scores
        
    def _calculate_similarity_score(self, query: str, summary: str, doc: str) -> float:
        """
        Calculate a combined similarity score between query and document summary.
        
        This method:
        1. Calculates Jaccard similarity between query and summary
        2. Calculates keyword presence ratio from query to document
        3. Combines both scores with appropriate weights
        
        Args:
            query: Search query string
            summary: BART-generated summary
            doc: Original document text
            
        Returns:
            float: Combined similarity score (0-1)
        """
        # Get query words (excluding common words)
        query_words = set(w.lower() for w in query.split() 
                        if len(w) > 3 and w.lower() not in 
                        {'the', 'and', 'that', 'for', 'with', 'this', 'from'})
                        
        # Get summary words
        summary_words = set(w.lower() for w in summary.split())
        
        # Get document words
        doc_words = set(w.lower() for w in doc.split() if len(w) > 3)
        
        # If no meaningful query words, return neutral score
        if not query_words:
            return 0.5
            
        # Calculate Jaccard similarity between query and summary
        if summary_words:
            intersection = query_words.intersection(summary_words)
            union = query_words.union(summary_words)
            jaccard_score = len(intersection) / len(union) if union else 0
        else:
            jaccard_score = 0
            
        # Calculate keyword presence in document
        if doc_words:
            query_term_ratio = sum(1 for w in query_words if w in doc_words) / len(query_words)
        else:
            query_term_ratio = 0
            
        # Combined score (weighted sum)
        return (0.7 * jaccard_score) + (0.3 * query_term_ratio)
        
    def _get_keyword_similarity_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        Simple keyword-based fallback scoring method.
        
        This is used when the GPT-4o-mini API is not available or fails.
        
        Args:
            query: The search query string
            documents: List of document texts
            
        Returns:
            List of keyword similarity scores for each document
        """
        scores = []
        for doc in documents:
            scores.append(self._get_keyword_similarity_score(query, doc))
        return scores
        
    def _get_keyword_similarity_score(self, query: str, document: str) -> float:
        """
        Calculate simple keyword similarity between query and document.
        
        This method:
        1. Detects if query contains non-Latin characters (for multilingual support)
        2. Uses either character trigrams (for non-Latin scripts) or word-based matching
        3. Calculates overlap between query and document
        
        Args:
            query: Search query string
            document: Document text
            
        Returns:
            float: Similarity score (0-1)
        """
        # Try to detect query language to adjust tokenization approach
        try:
            import re
            
            # Check if query contains non-Latin characters (might be non-English)
            non_latin = bool(re.search(r'[^\x00-\x7F]', query))
            
            # For non-Latin queries (like Polish, Russian, etc.), use character-level matching
            if non_latin:
                # Use character trigrams for non-Latin languages
                def get_trigrams(text):
                    text = text.lower()
                    return set(text[i:i+3] for i in range(len(text)-2) if text[i:i+3].strip())
                
                query_grams = get_trigrams(query)
                doc_grams = get_trigrams(document)
                
                if not query_grams or not doc_grams:
                    return 0.5
                
                # Calculate trigram overlap
                intersection = query_grams.intersection(doc_grams)
                union = query_grams.union(doc_grams)
                score = len(intersection) / len(union) if union else 0
                
                # Boost score slightly to compensate for trigram approach
                return min(1.0, score * 1.5)
            
            # For English and Latin-script languages, use word-based matching
            else:
                # Standard word-based approach for English and similar languages
                query_words = set(w.lower() for w in query.split() 
                                if len(w) > 3 and w.lower() not in 
                                {'the', 'and', 'that', 'for', 'with', 'this', 'from'})
                doc_words = set(w.lower() for w in document.split() if len(w) > 3)
                
                if not query_words or not doc_words:
                    return 0.5
                    
                # How many query terms appear in the document
                matches = sum(1 for w in query_words if w in doc_words)
                return matches / len(query_words) if query_words else 0
                
        except Exception as e:
            # Fall back to simpler approach if detection fails
            print(f"Language detection failed: {str(e)}")
            
            # Simple fallback - look for query substrings in document
            query = query.lower()
            document = document.lower()
            
            # Count how many non-trivial parts of the query appear in the document
            query_parts = [q for q in query.split() if len(q) > 3]
            if not query_parts:
                return 0.5
                
            matches = sum(1 for part in query_parts if part in document)
            return matches / len(query_parts)

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Add new texts to both the vector store and BM25 retriever.
        
        This method:
        1. Adds texts to the vectorstore for semantic search
        2. Updates the BM25 retriever with the new texts
        3. Recreates the ensemble retriever with the updated components
        
        Args:
            texts: List of text strings to add
            metadatas: Optional metadata for each text
            
        Returns:
            List of IDs of the added texts from the vectorstore
        """
        try:
            # Add to vectorstore
            ids = self._vectorstore.add_texts(texts, metadatas=metadatas)
            
            # If BM25 is not available, stop here
            if not self._bm25_retriever:
                return ids
                
            # Rebuild BM25 with all texts
            # This is inefficient but BM25Retriever doesn't support incremental updates
            all_docs = self._bm25_retriever.docs + [
                Document(page_content=text, metadata=meta if meta else {})
                for text, meta in zip(texts, metadatas if metadatas else [{}] * len(texts))
            ]
            
            # Extract just the text from documents
            all_texts = [doc.page_content for doc in all_docs]
            
            # Recreate BM25 retriever
            self._bm25_retriever = BM25Retriever.from_texts(all_texts)
            self._bm25_retriever.k = self._k
            
            # Update the ensemble retriever
            self._ensemble_retriever = EnsembleRetriever(
                retrievers=[self._bm25_retriever, self._vectorstore.as_retriever(search_kwargs={"k": self._k})],
                weights=[1-self._alpha, self._alpha]
            )
            
            return ids
        except Exception as e:
            st.error(f"Error adding texts to hybrid retriever: {str(e)}")
            print(f"Error adding texts: {str(e)}")
            # Try to at least add to vectorstore
            return self._vectorstore.add_texts(texts, metadatas=metadatas)

def get_hybrid_retriever(
    vectorstore, 
    text_chunks: List[str], 
    k: int = 10, 
    semantic_weight: float = 0.7,
    use_reranking: bool = True
) -> HybridRetriever:
    """
    Create a hybrid retriever combining semantic search, BM25, and contextual reranking.
    
    This factory function:
    1. Creates a HybridRetriever with the specified parameters
    2. Handles errors and provides fallback to standard retrieval if needed
    3. Configures the retriever with appropriate API keys and settings
    
    Args:
        vectorstore: The FAISS vector store for semantic search
        text_chunks: The list of text chunks for BM25 indexing
        k: Number of documents to retrieve
        semantic_weight: Weight for semantic search (0-1)
        use_reranking: Whether to use GPT-4o-mini for reranking
        
    Returns:
        A HybridRetriever instance or fallback standard retriever
    """
    try:
        # Get OpenAI API key from environment or session state
        openai_api_key = os.environ.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
        
        # Create and return the hybrid retriever
        return HybridRetriever(
            vectorstore=vectorstore,
            texts=text_chunks,
            k=k,
            alpha=semantic_weight,
            use_reranking=use_reranking,
            openai_api_key=openai_api_key,
            rerank_top_k=min(20, k * 2)  # Rerank at most 20 docs or 2*k, whichever is smaller
        )
    except Exception as e:
        st.error(f"Error creating hybrid retriever: {str(e)}. Falling back to standard retriever.")
        print(f"Error creating hybrid retriever: {str(e)}")
        # Fallback to standard retriever
        return vectorstore.as_retriever(search_kwargs={"k": k}) 