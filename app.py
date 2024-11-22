import streamlit as st
from document_processor import DocumentProcessor
from embedding_manager import EmbeddingManager
from model_manager import ModelManager

def initialize_session_state():
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    if 'embedding_manager' not in st.session_state:
        st.session_state.embedding_manager = EmbeddingManager()
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    if 'uploaded_files_processed' not in st.session_state:
        st.session_state.uploaded_files_processed = set()
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []  # Lista do przechowywania historii pytań i odpowiedzi

def process_files(files):
    if files is not None:
        for pdf_file in files:
            if pdf_file.name not in st.session_state.uploaded_files_processed:
                with st.spinner(f"Przetwarzanie {pdf_file.name}..."):
                    text = st.session_state.doc_processor.extract_text_from_pdf(pdf_file)
                    chunks = st.session_state.doc_processor.chunk_text(text)
                    st.session_state.embedding_manager.create_embeddings(chunks)
                    st.session_state.uploaded_files_processed.add(pdf_file.name)
                st.success(f"Przetworzono {pdf_file.name}")

def main():
    st.title("System Odpowiadania na Pytania z PDF 📄🤖")

    initialize_session_state()

    # Uploader plików
    uploaded_files = st.file_uploader(
        "Prześlij pliki PDF", 
        type=['pdf'],
        accept_multiple_files=True
    )

    # Przetwarzanie plików po ich przesłaniu
    if uploaded_files:
        process_files(uploaded_files)

    # Przycisk czyszczenia
    if st.button("Wyczyść Wszystko"):
        st.session_state.embedding_manager.clear()
        st.session_state.uploaded_files_processed.clear()
        st.session_state.qa_history.clear()
        st.success("Wyczyszczono wszystkie dokumenty i historię pytań!")

    # Wprowadzanie pytania
    question = st.text_input("💬 Zadaj pytanie dotyczące Twoich dokumentów:")

    if question:
        if len(st.session_state.embedding_manager.texts) == 0:
            st.warning("Najpierw prześlij pliki PDF!")
            return

        with st.spinner("Generowanie odpowiedzi..."):
            try:
                # Szukanie kontekstu
                context = st.session_state.embedding_manager.search(question, k=4)
                if context and isinstance(context, list):
                    combined_context = "\n\n".join(context)
                    answer = st.session_state.model_manager.generate_answer(question, [combined_context])
                else:
                    answer = "Nie znaleziono odpowiedniego kontekstu."

                # Dodanie pytania i odpowiedzi do historii
                st.session_state.qa_history.append((question, answer))

                # Wyświetlenie całej historii pytań i odpowiedzi
                for idx, (q, a) in enumerate(st.session_state.qa_history):
                    st.write(f"### 🧐 Pytanie {idx+1}:")
                    st.write(q)
                    st.write(f"### 💡 Odpowiedź {idx+1}:")
                    st.write(a)
                    st.write("---")  # Separator

            except Exception as e:
                st.error(f"Wystąpił błąd: {str(e)}")

    else:
        # Wyświetlenie historii, jeśli istnieje, nawet gdy nie zadano nowego pytania
        if st.session_state.qa_history:
            for idx, (q, a) in enumerate(st.session_state.qa_history):
                st.write(f"### 🧐 Pytanie {idx+1}:")
                st.write(q)
                st.write(f"### 💡 Odpowiedź {idx+1}:")
                st.write(a)
                st.write("---")  # Separator

if __name__ == "__main__":
    main()
