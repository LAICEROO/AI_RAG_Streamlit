# AI_RAG with streamlit
The AI RAG Assistant is a Streamlit-based application that allows users to interact with uploaded PDF documents through natural language queries. By leveraging embedding models and a question-answering model, the application retrieves context from the uploaded PDFs to provide accurate and context-aware answers.

![image](https://github.com/user-attachments/assets/1e88d87a-ff5c-4cfa-923c-be669f784b64)

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
- **GPU Support**: CUDA

## 📋 Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 6GB+ GPU VRAM

## 🛠️ Instalation 

- Clone the repository:
  
```
   git clone https://github.com/LAICEROO/AI_RAG_Streamlit.git
```
  
- Install dependencies
  
```
   pip install -r requirements.txt
```

- Run the application:

```
   streamlit run app.py
```

- Access the application: Open the displayed URL in your web browser (e.g., http://localhost:8501).




## 💻 Usage

1. **Upload PDFs**:
  - Use the sidebar to upload one or more PDF documents.
  - The application processes the documents and indexes their content for retrieval.

2. **Ask Questions**:
  - Type your question in the text input field in the main interface.
  - Click the Ask button to retrieve an answer based on the uploaded PDFs.

3. **Review Chat History**:
  - The interface displays a history of your queries and the assistant's responses.

4. **Manage Files and History**:
  - Use the sidebar to view uploaded files and clear chat history if needed.

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

## 📝 Note
This application runs completely locally:
- No API keys required
- No internet connection needed after initial model download
- All processing is done on your local machine
- Your documents never leave your computer
