import os
import streamlit as st
import PyPDF2
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from datetime import datetime

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device Name:", device)

# Initialize models and variables
def initialize():
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = SentenceTransformer('all-mpnet-base-v2', device=device)
    if 'tokenizer' not in st.session_state:
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
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'uploaded_files_list' not in st.session_state:
        st.session_state.uploaded_files_list = []
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""

def extract_text_from_pdf(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

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

def embed_text(chunks):
    embeddings = st.session_state.embedding_model.encode(chunks, convert_to_tensor=True)
    embeddings = embeddings.cpu().detach().numpy()
    return embeddings

def process_pdfs(pdf_paths):
    current_id = len(st.session_state.text_chunks)
    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text(text)
        embeddings = embed_text(chunks)
        new_ids = [current_id + i for i in range(len(chunks))]
        st.session_state.ids.extend(new_ids)
        st.session_state.text_chunks.extend(chunks)
        embeddings = np.array(embeddings).astype('float32')
        st.session_state.index.add_with_ids(embeddings, np.array(new_ids))
        current_id += len(chunks)

def format_chat_message(role, content):
    if role == "user":
        return f"<div style='background-color: #282434; color: white; padding: 10px; border-radius: 5px; margin: 5px 0;'><strong>You:</strong> {content}</div>"
    else:
        return f"<div style='background-color: #282434; color: white; padding: 10px; border-radius: 5px; margin: 5px 0;'><strong>Assistant:</strong> {content}</div>"

def answer_question(question, max_length=2048):
    question_embedding = st.session_state.embedding_model.encode([question], convert_to_tensor=True)
    question_embedding = question_embedding.cpu().detach().numpy().astype('float32')
    
    top_k = 5
    while top_k > 0:
        D, I = st.session_state.index.search(np.array(question_embedding), top_k)
        retrieved_chunks = [st.session_state.text_chunks[i] for i in I[0] if i < len(st.session_state.text_chunks)]
        context = ' '.join(retrieved_chunks)
        
        # Include recent chat history in the prompt
        recent_history = st.session_state.chat_history[-3:] if st.session_state.chat_history else []
        chat_context = "\n".join([f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
                                for msg in recent_history])
        
        prompt = f"""<|im_start|>system
You are a helpful AI assistant that answers questions based on the provided context and chat history. 
Your answers should be accurate, concise, and directly based on the context provided.
<|im_end|>
<|im_start|>user
Previous conversation:
{chat_context}

Context from documents:
{context}

Current question: {question}
<|im_end|>
<|im_start|>assistant
"""
        
        inputs = st.session_state.tokenizer(prompt, return_tensors='pt').to(device)
        total_length = inputs.input_ids.shape[1]
        if total_length <= max_length:
            break
        else:
            top_k -= 1

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

def display_chat_history():
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            st.markdown(format_chat_message(message['role'], message['content']), unsafe_allow_html=True)

def handle_submit():
    if st.session_state.current_question:
        return True
    return False

def main():
    st.title("PDF Chat Assistant")
    initialize()

    # Sidebar for file management and settings
    with st.sidebar:
        st.header("Document Management")
        uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)
        
        if uploaded_files:
            new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_files_list]
            if new_files:
                pdf_paths = []
                for uploaded_file in new_files:
                    if not os.path.exists("uploads"):
                        os.makedirs("uploads")
                    file_path = os.path.join("uploads", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    pdf_paths.append(file_path)
                    st.session_state.uploaded_files_list.append(uploaded_file.name)
                process_pdfs(pdf_paths)
                st.success(f"Processed {len(new_files)} new PDF(s)")
        
        if st.session_state.uploaded_files_list:
            st.write("Uploaded Documents:")
            for file_name in st.session_state.uploaded_files_list:
                st.write(f"📄 {file_name}")
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")

    # Main chat interface
    display_chat_history()

    # Question input with submit button
    question = st.text_input("Ask a question about your documents:", key="question_input")
    if st.button("Ask"):
        if not st.session_state.text_chunks:
            st.warning("Please upload some PDF documents first!")
            return
            
        if question:
            with st.spinner("Thinking..."):
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": question,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Generate and add assistant's response
                answer = answer_question(question)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                st.rerun()

if __name__ == '__main__':
    main()