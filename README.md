# AI_RAG with streamlit
A Streamlit-based application that allows users to chat with their PDF documents using advanced language models and semantic search.

## 🌟 Features

- **PDF Document Processing**: Upload and process multiple PDF files
- **Semantic Search**: Uses FAISS for efficient similarity search
- **Context-Aware Responses**: Generates answers based on document context
- **Chat History**: Maintains conversation history for better context
- **GPU Acceleration**: Supports CUDA for faster processing
- **User-Friendly Interface**: Clean and intuitive Streamlit UI

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Language Model**: Qwen2.5-3B-Instruct
- **Embeddings**: Sentence-Transformers (all-mpnet-base-v2)
- **Vector Search**: FAISS
- **PDF Processing**: PyPDF2
- **Deep Learning**: PyTorch
- **GPU Support**: CUDA

## 📋 Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 6GB+ GPU VRAM

## 💻 Usage

1. Start the application:
   ```
    streamlit run app.py
   ```
2. Open your browser and navigate to `http://localhost:8501`

3. Upload PDF documents using the sidebar

4. Start asking questions about your documents!

## 🔍 How It Works

1. **Document Processing**:
   - PDFs are uploaded and text is extracted
   - Text is split into manageable chunks
   - Chunks are converted to embeddings using Sentence-Transformers

2. **Question Answering**:
   - User question is converted to embedding
   - Most relevant document chunks are retrieved using FAISS
   - Context is provided to Qwen model for answer generation

3. **Response Generation**:
   - Model generates response based on context and question
   - Response is displayed in chat interface
   - Chat history is maintained for context

## 🎯 Features in Detail

- **Semantic Search**: Uses FAISS for efficient similarity search in high-dimensional space
- **Context Window**: Maintains optimal context size for better responses
- **History Management**: Keeps track of recent conversations
- **GPU Acceleration**: Utilizes CUDA for faster processing
- **Memory Management**: Efficient handling of large documents
- **Error Handling**: Robust error handling for various edge cases

## 🔧 Configuration

Key parameters can be adjusted in the code:
- `max_length`: Maximum context length (default: 2048)
- `top_k`: Number of relevant chunks to retrieve (default: 5)
- `temperature`: Response randomness (default: 0.7)
- `chunk_size`: Size of text chunks (default: 500)
