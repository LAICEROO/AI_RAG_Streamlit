import os
import streamlit as st
import PyPDF2
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device Name:", device)

# Initialize models and variables
def initialize():
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = SentenceTransformer('all-mpnet-base-v2', device=device)
    if 'tokenizer' not in st.session_state:
        # Use Qwen2.5-3B-Instruct model
        st.session_state.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
        st.session_state.qa_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            device_map="auto",
            trust_remote_code=True
        )
    if 'index' not in st.session_state:
        dimension = 768  # For all-mpnet-base-v2
        st.session_state.index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
    if 'text_chunks' not in st.session_state:
        st.session_state.text_chunks = []
    if 'ids' not in st.session_state:
        st.session_state.ids = []

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def split_text(text, max_length=500):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + '. '
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + '. '
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# Function to generate embeddings for text chunks
def embed_text(chunks):
    embeddings = st.session_state.embedding_model.encode(chunks, convert_to_tensor=True)
    embeddings = embeddings.cpu().detach().numpy()
    return embeddings

# Function to process uploaded PDFs
def process_pdfs(pdf_paths):
    current_id = len(st.session_state.text_chunks)
    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text(text)
        embeddings = embed_text(chunks)
        # Generate IDs for new chunks
        new_ids = [current_id + i for i in range(len(chunks))]
        st.session_state.ids.extend(new_ids)
        st.session_state.text_chunks.extend(chunks)
        # Convert embeddings to numpy array
        embeddings = np.array(embeddings).astype('float32')
        # Add embeddings and IDs to FAISS index
        st.session_state.index.add_with_ids(embeddings, np.array(new_ids))
        current_id += len(chunks)

# Function to answer user questions
def answer_question(question, max_length=2048):
    question_embedding = st.session_state.embedding_model.encode([question], convert_to_tensor=True)
    question_embedding = question_embedding.cpu().detach().numpy().astype('float32')
    # Start with top_k=5, adjust as needed
    top_k = 5
    while top_k > 0:
        D, I = st.session_state.index.search(np.array(question_embedding), top_k)
        retrieved_chunks = [st.session_state.text_chunks[i] for i in I[0] if i < len(st.session_state.text_chunks)]
        context = ' '.join(retrieved_chunks)
        
        # Format prompt for Qwen model
        prompt = f"""<|im_start|>system
You are a helpful AI assistant that answers questions based on the provided context. 
Your answers should be accurate, concise, and directly based on the context provided.
<|im_end|>
<|im_start|>user
Context: {context}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""
        
        inputs = st.session_state.tokenizer(prompt, return_tensors='pt').to(device)
        total_length = inputs.input_ids.shape[1]
        if total_length <= max_length:
            break
        else:
            top_k -= 1

    # Generate the answer with Qwen model
    with torch.no_grad():
        outputs = st.session_state.qa_model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=st.session_state.tokenizer.pad_token_id,
            eos_token_id=st.session_state.tokenizer.eos_token_id
        )
    
    answer = st.session_state.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return answer.strip()

# Main function to run the Streamlit app
def main():
    st.title("GPU-Accelerated PDF Question Answering Application")
    initialize()

    # File uploader
    uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)
    if uploaded_files:
        pdf_paths = []
        for uploaded_file in uploaded_files:
            # Save uploaded file to disk
            if not os.path.exists("uploads"):
                os.makedirs("uploads")
            file_path = os.path.join("uploads", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_paths.append(file_path)
        process_pdfs(pdf_paths)
        st.success("PDFs have been processed and indexed.")

    # Question input
    question = st.text_input("Enter your question:")
    if question:
        with st.spinner("Generating answer..."):
            answer = answer_question(question)
        st.write("**Answer:**")
        st.write(answer)

if __name__ == '__main__':
    main()