# AI RAG with Streamlit

A powerful Retrieval Augmented Generation (RAG) application built with Streamlit that combines document processing, hybrid search, and web search capabilities. This application allows users to chat with their documents while leveraging both local knowledge and real-time web information.

## 🌟 Features

### Document Processing
- Support for multiple document formats:
  - PDF files
  - Text files (TXT)
  - Word documents (DOCX)
  - CSV files
  - JSON files
- Automatic text chunking and processing
- Efficient document embedding using multilingual E5 model

### Advanced Search Capabilities
- **Hybrid Search System**:
  - Semantic search using FAISS vector store
  - Keyword-based search using BM25
  - Contextual reranking with GPT-4o-mini
- **Web Search Integration**:
  - Real-time web search using Tavily API
  - Configurable search depth and time range
  - Domain filtering and exclusion
  - AI-generated summaries of web results

### Chat Interface
- Interactive chat interface with document context
- Conversation memory and history
- Source document citations
- Support for multiple languages

### Performance Optimization
- Batch processing for embeddings
- Performance mode for large documents
- Configurable batch sizes
- Efficient memory management

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-rag-streamlit.git
cd ai-rag-streamlit
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```env
OPENAI_API_KEY=your_openai_api_key
MISTRAL_API_KEY=your_mistral_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## 🛠️ Configuration

### Document Processing Settings
- Adjust chunk sizes and overlap in `utils/embedding_utils.py`
- Configure text splitting parameters for optimal processing

### Search Settings
- Toggle hybrid search in the sidebar
- Adjust semantic search weight (0.0-1.0)
- Enable/disable contextual reranking
- Configure number of documents to retrieve

### Web Search Settings
- Enable/disable automatic web search
- Set search depth (basic/advanced)
- Configure time range filters
- Add domain inclusion/exclusion rules

## 📚 Usage Guide

1. **Document Upload**:
   - Go to the "Documents" tab in the sidebar
   - Upload your documents using the file uploader
   - Click "Process" to start document processing

2. **Asking Questions**:
   - Type your question in the chat input
   - The system will search through your documents and the web (if enabled)
   - View the response with source citations

3. **Web Search**:
   - Enable web search in the sidebar
   - Configure search parameters as needed
   - Results will be automatically integrated into responses

4. **Advanced Features**:
   - Use the "Retrieval Settings" tab to fine-tune search behavior
   - Access "Advanced Settings" for performance optimization
   - Enable developer mode for debugging information

## 🔧 Technical Details

### Architecture
- Frontend: Streamlit
- Document Processing: LangChain
- Embeddings: Multilingual E5 model
- Vector Store: FAISS
- Search: Hybrid system (FAISS + BM25 + GPT-4o-mini)
- LLM: Mistral AI

### Key Components
- `app.py`: Main application file
- `utils/hybrid_search.py`: Hybrid search implementation
- `utils/embedding_utils.py`: Document processing utilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
