# Szczegółowy opis aplikacji Streamlit do czatu z dokumentami PDF

## Wprowadzenie

`app.py` to główny plik aplikacji Streamlit, która umożliwia użytkownikom prowadzenie konwersacji z wieloma dokumentami PDF (oraz innymi formatami plików). Aplikacja implementuje zaawansowany system RAG (Retrieval-Augmented Generation), który łączy przetwarzanie dokumentów, wektoryzację tekstu, semantyczne wyszukiwanie i generowanie odpowiedzi oparte na LLM (Large Language Model). System umożliwia również integrację z wyszukiwaniem internetowym dla dostarczenia aktualnych informacji.

## Importy i zależności

```python
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
```

Szczegółowy opis każdej importowanej biblioteki:

- **streamlit (st)**: Framework do tworzenia interaktywnych aplikacji webowych w Pythonie. Używany do budowy całego interfejsu użytkownika aplikacji, w tym formularzy, przycisków, ekspanderów i wizualizacji.

- **dotenv i os**: `load_dotenv` ładuje zmienne środowiskowe z pliku `.env`, a moduł `os` umożliwia dostęp do nich w kodzie. W tej aplikacji używane głównie do przechowywania kluczy API (Mistral AI, Tavily, itp.).

- **PyPDF2**: Biblioteka do manipulacji plikami PDF. Głównie używana do ekstrakcji tekstu z dokumentów PDF poprzez klasę `PdfReader`.

- **langchain.text_splitter.CharacterTextSplitter**: Narzędzie do dzielenia dużych tekstów na mniejsze fragmenty (chunks), co jest kluczowe w technikach RAG, ponieważ modele mają ograniczenia co do długości kontekstu.

- **langchain_community.vectorstores.FAISS**: Implementacja bazy wektorowej używającej biblioteki FAISS (Facebook AI Similarity Search) dla efektywnego wyszukiwania podobieństwa semantycznego. Przechowuje embeddigi tekstów i umożliwia szybkie wyszukiwanie podobnych dokumentów.

- **langchain.memory.ConversationBufferMemory**: Moduł do zarządzania historią konwersacji, przechowujący poprzednie pytania i odpowiedzi.

- **langchain.chains.ConversationalRetrievalChain**: Złożony łańcuch przetwarzania, który łączy retriever (wyszukiwanie dokumentów), model językowy i pamięć konwersacji. Umożliwia zadawanie pytań dotyczących dokumentów z uwzględnieniem kontekstu rozmowy.

- **torch, torch.nn.functional, torch.Tensor**: Biblioteka PyTorch używana do operacji na modelach uczenia głębokiego i tensorach. W aplikacji używana głównie do obsługi modelu embeddingowego.

- **transformers.AutoTokenizer, transformers.AutoModel**: Klasy z biblioteki HuggingFace Transformers, służące do ładowania pretrenowanych modeli językowych. W tej aplikacji używane do modelu embeddingowego E5.

- **langchain.embeddings.base.Embeddings**: Klasa bazowa dla systemów embeddingów w LangChain.

- **langchain_mistralai.ChatMistralAI, mistralai.Mistral**: Integracje z modelami Mistral AI, używane do generowania odpowiedzi opartych na znalezionych dokumentach.

- **requests, json**: Standardowe biblioteki Pythona do wykonywania zapytań HTTP i przetwarzania danych JSON. Używane głównie do komunikacji z API Tavily do wyszukiwania internetowego.

- **numpy (np)**: Biblioteka do obliczeń numerycznych, wykorzystywana przy operacjach na wektorach embeddingów.

- **utils.embedding_utils**: Moduł zawierający funkcje pomocnicze związane z embeddingami i przetwarzaniem tekstu:
  - `get_text_chunks`: Dzieli tekst na mniejsze fragmenty
  - `average_pool`: Uśrednia reprezentacje tokenów dla uzyskania embeddingu
  - `MultilangE5Embeddings`: Klasa wykorzystująca wielojęzyczny model E5 do tworzenia embeddingów
  - `get_vectorstore`: Tworzy bazę wektorową FAISS z fragmentów tekstu

- **utils.hybrid_search**: Moduł zawierający funkcję `get_hybrid_retriever`, która tworzy zaawansowany retriever łączący wyszukiwanie semantyczne (embeddigi), BM25 (wyszukiwanie na podstawie słów kluczowych) i reranking.

## Główne funkcje

### `get_pdf_text(uploaded_files)`

```python
def get_pdf_text(uploaded_files):
    """
    Extract text from various document formats.
    Supports PDF, TXT, DOCX, CSV, and JSON files.
    """
    text = ""
    
    for file in uploaded_files:
        # Get file extension
        file_ext = file.name.split('.')[-1].lower()
        
        try:
            # Handle different file types
            if file_ext == 'pdf':
                # Handle PDF files
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n\n"
                    
            elif file_ext == 'txt':
                # Handle text files
                text += file.getvalue().decode('utf-8') + "\n\n"
                
            elif file_ext == 'docx':
                # Handle DOCX files
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
                # Handle CSV files
                try:
                    import pandas as pd
                    df = pd.read_csv(file)
                    text += df.to_string() + "\n\n"
                except ImportError:
                    st.error(f"Missing pandas library. Install with 'pip install pandas' to process {file.name}")
                    continue
                    
            elif file_ext == 'json':
                # Handle JSON files
                import json
                content = json.loads(file.getvalue().decode('utf-8'))
                text += json.dumps(content, indent=2) + "\n\n"
                
            else:
                st.warning(f"Unsupported file type: {file_ext} - {file.name} was skipped")
                
        except Exception as e:
            st.error(f"Error processing file {file.name}: {str(e)}")
            import traceback
            print(f"Error details for {file.name}: {traceback.format_exc()}")
            continue
            
    return text
```

Szczegółowa analiza funkcji `get_pdf_text`:

1. **Inicjalizacja zmiennej wyjściowej** (`text = ""`): Tworzy pustą zmienną tekstową, która będzie zawierać zagregowany tekst ze wszystkich dokumentów.

2. **Iteracja przez każdy plik** (`for file in uploaded_files`): Pętla przez wszystkie pliki przesłane przez użytkownika za pomocą widgetu `st.file_uploader`.

3. **Ekstrakcja rozszerzenia pliku** (`file_ext = file.name.split('.')[-1].lower()`): 
   - Pobiera nazwę pliku z obiektu pliku
   - Dzieli ją według kropek (`.`) 
   - Bierze ostatni element po podziale (rozszerzenie)
   - Konwertuje do małych liter dla ujednolicenia (np. '.PDF' -> 'pdf')

4. **Blok try-except**: Ważny mechanizm obsługi błędów, który pozwala na kontynuowanie przetwarzania pozostałych plików nawet jeśli jeden z nich spowoduje błąd.

5. **Obsługa plików PDF** (`if file_ext == 'pdf'`):
   - Tworzy obiekt `PdfReader` z biblioteki PyPDF2
   - Iteruje przez wszystkie strony dokumentu (`pdf_reader.pages`)
   - Dla każdej strony wywołuje metodę `extract_text()` do uzyskania tekstu
   - Dodaje dwa znaki nowej linii (`\n\n`) po każdej stronie dla lepszej struktury tekstu

6. **Obsługa plików TXT** (`elif file_ext == 'txt'`):
   - Pobiera zawartość binarną pliku (`file.getvalue()`)
   - Dekoduje ją jako tekst UTF-8
   - Dodaje podwójny znak nowej linii na końcu

7. **Obsługa plików DOCX** (`elif file_ext == 'docx'`):
   - Zawiera zagnieżdżony blok try-except, aby obsłużyć przypadek, gdy biblioteka `python-docx` nie jest zainstalowana
   - Dynamicznie importuje `Document` z `docx` 
   - Tworzy obiekt dokumentu
   - Iteruje przez wszystkie akapity (`doc.paragraphs`)
   - Dodaje tekst każdego akapitu wraz ze znakiem nowej linii
   - Na końcu dodaje podwójny znak nowej linii
   - W przypadku braku biblioteki, wyświetla komunikat o błędzie i kontynuuje przetwarzanie pozostałych plików

8. **Obsługa plików CSV** (`elif file_ext == 'csv'`):
   - Podobnie jak przy DOCX, obsługuje przypadek braku biblioteki `pandas`
   - Używa pandas do odczytu pliku CSV do DataFrame
   - Konwertuje DataFrame na tekst za pomocą metody `to_string()`
   - W przypadku braku biblioteki wyświetla stosowny komunikat błędu

9. **Obsługa plików JSON** (`elif file_ext == 'json'`):
   - Dekoduje zawartość pliku z UTF-8
   - Parsuje JSON do struktury Pythona za pomocą `json.loads()`
   - Konwertuje z powrotem do sformatowanego tekstu JSON z wcięciami za pomocą `json.dumps(content, indent=2)`

10. **Obsługa nieobsługiwanych typów plików** (`else`):
    - Wyświetla ostrzeżenie dla użytkownika o nieobsługiwanym typie pliku
    - Pomija ten plik i kontynuuje przetwarzanie pozostałych

11. **Obsługa wyjątków** (`except Exception as e`):
    - Wyświetla użytkownikowi komunikat o błędzie za pomocą `st.error`
    - Importuje moduł `traceback` i drukuje pełny ślad stosu błędu (tylko w konsoli, nie dla użytkownika)
    - Kontynuuje przetwarzanie pozostałych plików (`continue`)

12. **Zwrócenie wyniku** (`return text`): Funkcja zwraca pełen tekst ze wszystkich przetworzonych dokumentów jako pojedynczy string.

Funkcja ta jest elastyczna i odporna na błędy, obsługuje różne formaty plików i inteligentnie izoluje błędy, aby niepowodzenie z jednym plikiem nie przerwało przetwarzania pozostałych.

### `get_conversation_chain(vectorstore, text_chunks=None)`

```python
def get_conversation_chain(vectorstore, text_chunks=None):
    # Use Mistral AI directly
    api_key = os.environ["MISTRAL_API_KEY"]
    
    # Initialize the ChatMistralAI using LangChain's integration
    llm = ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.3,
        max_tokens=8192,
        top_p=0.9
    )

    # Use the updated memory API to avoid deprecation warnings
    from langchain.memory import ConversationBufferMemory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage

    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True, 
        output_key='answer'
    )
    
    # Create hybrid retriever if text_chunks are provided
    if text_chunks and st.session_state.use_hybrid_search:
        try:
            retriever = get_hybrid_retriever(
                vectorstore=vectorstore,
                text_chunks=text_chunks,
                k=st.session_state.retrieve_k,
                semantic_weight=st.session_state.semantic_weight,
                use_reranking=st.session_state.use_contextual_reranking
            )
            
            # Check if we got a HybridRetriever or a fallback retriever
            if not hasattr(retriever, 'add_texts'):
                st.warning("Hybrid search creation failed. Using standard search instead.")
                st.session_state.use_hybrid_search = False
                
        except Exception as e:
            st.error(f"Error setting up hybrid search: {e}")
            st.warning("Falling back to standard vector search.")
            retriever = vectorstore.as_retriever(search_kwargs={"k": st.session_state.retrieve_k})
            st.session_state.use_hybrid_search = False
    else:
        # Use standard vectorstore retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": st.session_state.retrieve_k})
    
    # Create the chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
        return_source_documents=True,
        chain_type="stuff"
    )
    
    return conversation_chain
```

Szczegółowa analiza funkcji `get_conversation_chain`:

Ta funkcja jest jednym z najważniejszych elementów systemu RAG, ponieważ łączy wyszukiwanie dokumentów z modelem językowym w spójny łańcuch konwersacji.

1. **Pobieranie klucza API Mistral** (`api_key = os.environ["MISTRAL_API_KEY"]`):
   - Odczytuje klucz API Mistral AI z zmiennych środowiskowych
   - Wymaga wcześniejszego załadowania zmiennych przez `load_dotenv()`

2. **Inicjalizacja modelu językowego Mistral** poprzez integrację LangChain:
   ```python
   llm = ChatMistralAI(
       model="mistral-small-latest",  # Określa konkretny model Mistral AI do użycia
       mistral_api_key=api_key,       # Przekazuje klucz API
       temperature=0.3,               # Niska temperatura = mniej losowe/bardziej deterministyczne odpowiedzi
       max_tokens=8192,               # Maksymalna długość generowanego tekstu
       top_p=0.9                      # Parametr próbkowania top-p (nucleus sampling)
   )
   ```

   Parametry modelu są starannie dobrane:
   - Niska temperatura (0.3) zapewnia bardziej precyzyjne i faktyczne odpowiedzi
   - Duża wartość max_tokens (8192) umożliwia generowanie długich, szczegółowych odpowiedzi
   - top_p (0.9) kontroluje różnorodność tekstu, zachowując dobrą równowagę między kreatywnością a spójnością

3. **Import dodatkowych komponentów LangChain**:
   - `ConversationBufferMemory` - do przechowywania historii rozmowy
   - `RunnableWithMessageHistory` - do integracji z systemem wiadomości
   - `HumanMessage`, `AIMessage` - typy wiadomości używane w LangChain

4. **Konfiguracja pamięci konwersacji**:
   ```python
   memory = ConversationBufferMemory(
       memory_key='chat_history',   # Klucz pod którym historia będzie dostępna dla modelu
       return_messages=True,        # Zwraca historię jako listę obiektów Message (nie stringów)
       output_key='answer'          # Nazwa klucza z odpowiedzią w wyniku łańcucha
   )
   ```

5. **Utworzenie retriever'a** - ta część decyduje o sposobie wyszukiwania dokumentów:

   a. **Hybrid retriever** (jeśli dostępne fragmenty tekstu i włączona opcja): 
   ```python
   if text_chunks and st.session_state.use_hybrid_search:
       try:
           retriever = get_hybrid_retriever(...)
   ```
   
   Funkcja `get_hybrid_retriever` (z modułu `utils.hybrid_search`) tworzy zaawansowany retriever łączący:
   - Wyszukiwanie semantyczne (FAISS)
   - Wyszukiwanie na podstawie słów kluczowych (BM25)
   - Opcjonalnie reranking kontekstowy za pomocą modelu BART
   
   Parametry:
   - `vectorstore` - baza wektorowa FAISS
   - `text_chunks` - surowe fragmenty tekstu
   - `k` - liczba dokumentów do pobrania
   - `semantic_weight` - waga wyszukiwania semantycznego względem BM25
   - `use_reranking` - czy używać reranking kontekstowego (BART)
   
   b. **Weryfikacja poprawności hybrid retrievera**:
   ```python
   if not hasattr(retriever, 'add_texts'):
       st.warning("Hybrid search creation failed. Using standard search instead.")
       st.session_state.use_hybrid_search = False
   ```
   
   Sprawdza, czy retriever ma metodę `add_texts`, co wskazuje na poprawne utworzenie retrievera hybrydowego. Jeśli nie, wyświetla ostrzeżenie i przełącza się na standardowe wyszukiwanie.
   
   c. **Obsługa błędów** podczas tworzenia hybrid retrievera:
   ```python
   except Exception as e:
       st.error(f"Error setting up hybrid search: {e}")
       st.warning("Falling back to standard vector search.")
       retriever = vectorstore.as_retriever(search_kwargs={"k": st.session_state.retrieve_k})
       st.session_state.use_hybrid_search = False
   ```
   
   d. **Standard retriever** (jeśli hybrid search jest wyłączony lub nie ma fragmentów tekstu):
   ```python
   else:
       retriever = vectorstore.as_retriever(search_kwargs={"k": st.session_state.retrieve_k})
   ```
   
   Tworzy standardowy retriever wektorowy, który wykorzystuje tylko podobieństwo kosinusowe embeddingów do wyszukiwania dokumentów.

6. **Utworzenie łańcucha konwersacji**:
   ```python
   conversation_chain = ConversationalRetrievalChain.from_llm(
       llm=llm,                         # Model językowy
       retriever=retriever,             # Retriever do wyszukiwania dokumentów
       memory=memory,                   # Pamięć konwersacji
       verbose=True,                    # Wyświetla szczegółowe logi wykonania
       return_source_documents=True,    # Zwraca dokumenty źródłowe
       chain_type="stuff"               # Określa sposób łączenia dokumentów
   )
   ```
   
   Kluczowe parametry:
   - `llm` - wcześniej utworzony model językowy Mistral AI
   - `retriever` - retriever utworzony w poprzednim kroku
   - `memory` - obiekt ConversationBufferMemory do przechowywania historii
   - `verbose=True` - powoduje wyświetlanie szczegółowych informacji o przetwarzaniu
   - `return_source_documents=True` - zapewnia, że łańcuch zwróci dokumenty źródłowe wraz z odpowiedzią
   - `chain_type="stuff"` - określa sposób przetwarzania dokumentów; "stuff" oznacza, że wszystkie znalezione dokumenty są łączone w jeden duży kontekst

7. **Zwrócenie utworzonego łańcucha**:
   ```python
   return conversation_chain
   ```

Funkcja `get_conversation_chain` jest kluczowym elementem architektury RAG w tej aplikacji, ponieważ integruje model językowy z wyszukiwaniem dokumentów, tworząc system zdolny do odpowiadania na pytania w oparciu o dostarczone dokumenty i kontekst rozmowy.

### `generate_source_summary(source_documents)`

```python
def generate_source_summary(source_documents):
    """
    Generate a summary from the retrieved source documents
    
    Args:
        source_documents: List of document objects with page_content attribute
        
    Returns:
        List of bullet points summarizing key information
    """
    # Extract all text from documents
    all_text = ""
    for doc in source_documents[:5]:  # Limit to first 5 documents to avoid token limits
        all_text += doc.page_content + "\n\n"
    
    # Generic identification of key concepts
    import re
    from collections import Counter
    
    # Check if text contains significant non-Latin characters (likely non-English)
    non_latin_chars = re.findall(r'[^\x00-\x7F]', all_text)
    is_non_latin = len(non_latin_chars) > len(all_text) * 0.05  # If >5% non-Latin
    
    # Different approach based on detected language characteristics
    if is_non_latin:
        # For non-Latin scripts or multilingual content, try character n-gram approach
        # Extract word-like sequences that might be terms in any language
        words = re.findall(r'\b\w+\b', all_text)
        
        # Get most frequent words that are reasonably long
        word_counts = Counter([w.lower() for w in words if len(w) > 3])
        top_terms = [term for term, count in word_counts.most_common(10) if count > 2]
    else:
        # Standard approach for primarily Latin-script content
        words = re.findall(r'\b[A-Za-z][A-Za-z-]{3,15}\b', all_text)
        words = [word.lower() for word in words if word.lower() not in 
                ['the', 'and', 'that', 'for', 'with', 'this', 'from', 'these', 'those', 
                'their', 'there', 'what', 'when', 'where', 'which', 'while', 'would']]
        
        # Get most common terms
        word_counts = Counter(words)
        top_terms = [term for term, count in word_counts.most_common(8) if count > 2]
    
    # Generate generic bullet points if we found key terms
    if top_terms:
        bullets = []
        if len(top_terms) >= 1:
            bullets.append(f"The documents primarily discuss **{top_terms[0]}** and related concepts.")
        if len(top_terms) >= 3:
            bullets.append(f"Key topics include **{top_terms[0]}**, **{top_terms[1]}**, and **{top_terms[2]}**.")
        if len(top_terms) >= 5:
            bullets.append(f"Additional topics covered: **{top_terms[3]}** and **{top_terms[4]}**.")
        bullets.append("The information includes definitions, explanations, and technical details on these subjects.")
        bullets.append("Several sources provide complementary perspectives on these topics.")
        return bullets
    
    # Fallback generic summary if no key terms found
    return [
        "The retrieved documents contain information related to your query.",
        "Multiple sources provide different perspectives and details on the topic.",
        "The content includes definitions, explanations, and technical specifications.",
        "Some sources may include academic or research perspectives.",
        "Consider reviewing the individual sources for more specific information."
    ]
```

Szczegółowa analiza funkcji `generate_source_summary`:

Ta zaawansowana funkcja generuje automatyczne podsumowanie dokumentów źródłowych używanych do udzielenia odpowiedzi. Jest to szczególnie przydatne dla użytkownika, aby zrozumieć, z jakich źródeł pochodzą informacje bez konieczności czytania wszystkich dokumentów.

1. **Zebranie tekstu ze wszystkich dokumentów**:
   ```python
   all_text = ""
   for doc in source_documents[:5]:  # Limit to first 5 documents to avoid token limits
       all_text += doc.page_content + "\n\n"
   ```
   
   - Funkcja ogranicza analizę do pierwszych 5 dokumentów, aby uniknąć problemów z tokenami
   - Każdy dokument ma atrybut `page_content` zawierający jego treść
   - Łączy całą treść w jeden duży blok tekstu, dodając podwójne znaki nowej linii między dokumentami

2. **Import narzędzi do analizy tekstu**:
   ```python
   import re
   from collections import Counter
   ```
   
   - `re` (moduł wyrażeń regularnych) służy do wyszukiwania wzorców w tekście
   - `Counter` z modułu `collections` umożliwia łatwe zliczanie wystąpień elementów w kolekcji

3. **Wykrywanie języka na podstawie zestawu znaków**:
   ```python
   non_latin_chars = re.findall(r'[^\x00-\x7F]', all_text)
   is_non_latin = len(non_latin_chars) > len(all_text) * 0.05  # If >5% non-Latin
   ```
   
   - Funkcja używa sprytnej heurystyki do wykrycia, czy tekst zawiera znaczącą ilość znaków spoza alfabetu łacińskiego
   - Wzorzec `r'[^\x00-\x7F]'` wyszukuje wszystkie znaki poza zakresem ASCII (0-127)
   - Jeśli ponad 5% znaków jest spoza ASCII, tekst jest uznawany za prawdopodobnie wielojęzyczny lub w języku używającym innego alfabetu

4. **Ekstrakcja kluczowych terminów z tekstów w alfabetach nielacińskich**:
   ```python
   if is_non_latin:
       words = re.findall(r'\b\w+\b', all_text)
       word_counts = Counter([w.lower() for w in words if len(w) > 3])
       top_terms = [term for term, count in word_counts.most_common(10) if count > 2]
   ```
   
   - Dla tekstów zawierających znaczącą ilość nielacińskich znaków stosowane jest ogólne podejście
   - Wzorzec `r'\b\w+\b'` wyszukuje ciągi znaków, które można uznać za "słowa" w różnych językach
   - Filtruje krótkie słowa (mniej niż 4 znaki)
   - Konwertuje do małych liter
   - Zlicza wystąpienia i wybiera 10 najczęściej występujących terminów, które pojawiają się więcej niż 2 razy

5. **Ekstrakcja kluczowych terminów z tekstów w alfabecie łacińskim**:
   ```python
   else:
       words = re.findall(r'\b[A-Za-z][A-Za-z-]{3,15}\b', all_text)
       words = [word.lower() for word in words if word.lower() not in 
               ['the', 'and', 'that', 'for', 'with', 'this', 'from', 'these', 'those', 
               'their', 'there', 'what', 'when', 'where', 'which', 'while', 'would']]
       word_counts = Counter(words)
       top_terms = [term for term, count in word_counts.most_common(8) if count > 2]
   ```
   
   - Dla tekstów w alfabecie łacińskim stosowane jest bardziej specyficzne podejście
   - Wzorzec `r'\b[A-Za-z][A-Za-z-]{3,15}\b'` dopasowuje słowa 4-16 znaków, zaczynające się literą i zawierające tylko litery lub myślniki
   - Filtruje listę common stop words (słowa występujące często, ale mało znaczące)
   - Zlicza wystąpienia i wybiera 8 najczęściej występujących terminów, które pojawiają się więcej niż 2 razy

6. **Generowanie punktów podsumowania na podstawie znalezionych terminów**:
   ```python
   if top_terms:
       bullets = []
       if len(top_terms) >= 1:
           bullets.append(f"The documents primarily discuss **{top_terms[0]}** and related concepts.")
       # itd...
   ```
   
   - Jeśli znaleziono istotne terminy, generuje spersonalizowane podsumowanie
   - Wykorzystuje Markdown do formatowania (pogrubienie **kluczowych terminów**)
   - Tworzy dynamiczną liczbę punktorów w zależności od liczby znalezionych terminów:
     - Pierwszy punkt zawiera najważniejszy termin
     - Drugi punkt wymienia 3 najważniejsze terminy
     - Trzeci punkt wymienia dodatkowe 2 terminy (jeśli dostępne)
     - Dodaje 2 ogólne punkty o charakterze dokumentów

7. **Fallback dla przypadku, gdy nie znaleziono terminów**:
   ```python
   return [
       "The retrieved documents contain information related to your query.",
       "Multiple sources provide different perspectives and details on the topic.",
       # itd...
   ]
   ```
   
   - Jeśli nie znaleziono wystarczającej liczby terminów, funkcja zwraca generyczne podsumowanie
   - Jest to zabezpieczenie dla przypadków, gdy analiza tekstu nie przyniosła oczekiwanych rezultatów

Ta funkcja pokazuje jak można wykorzystać proste techniki NLP (przetwarzania języka naturalnego) do automatycznego generowania podsumowań bez konieczności używania dużych modeli językowych. Jest to podejście hybrydowe - aplikacja używa zaawansowanego LLM do generowania odpowiedzi, ale prostszych, szybszych technik do analizowania i podsumowywania źródeł.

### `handle_userinput(user_question)`

```python
def handle_userinput(user_question):
    web_context = ""
    web_sources = []
    
    # Check if we should perform a web search alongside the document search
    if st.session_state.web_search_enabled and user_question:
        with st.spinner("Searching the web for additional context..."):
            search_results = perform_tavily_search(
                query=user_question, 
                search_depth=st.session_state.web_search_depth,
                max_results=st.session_state.max_results,
                include_answer=st.session_state.include_answer,
                include_images=st.session_state.include_images,
                time_range=st.session_state.time_range
            )
            
            # If we got search results and they don't contain an error
            if search_results and "error" not in search_results:
                # Extract web search answer and sources
                if "answer" in search_results and search_results["answer"]:
                    web_context = search_results["answer"]
                
                # Extract web sources for citation
                if "results" in search_results and len(search_results["results"]) > 0:
                    for result in search_results["results"]:
                        web_sources.append({
                            "title": result.get("title", "No title"),
                            "url": result.get("url", "#"),
                            "content": result.get("content", "")[:150] + "..." if len(result.get("content", "")) > 150 else result.get("content", "")
                        })
                
                # Show the web search context to the user
                with st.expander("Web Search Results", expanded=True):
                    if web_context:
                        st.write(web_context)
                    
                    if web_sources:
                        st.markdown("### Sources:")
                        for i, source in enumerate(web_sources):
                            st.markdown(f"**Source {i+1}:** [{source['title']}]({source['url']})")
                            st.markdown(f"_Preview:_ {source['content']}")
                
                # Always add web results to the vectorstore 
                if "results" in search_results and st.session_state.conversation:
                    content_texts = [result.get("content", "") for result in search_results["results"] if "content" in result]
                    source_urls = [result.get("url", "") for result in search_results["results"] if "content" in result]
                    
                    if content_texts:
                        # Add to vectorstore
                        add_search_results_to_vectorstore(content_texts, source_urls)
    
    try:
        # Always use full integration mode for web results
        if web_context and web_sources:
            # Format web information to be explicitly used by the LLM
            formatted_web_info = f"""
Web search found the following information relevant to your question:

{web_context}

Sources:
"""
            for i, source in enumerate(web_sources):
                formatted_web_info += f"{i+1}. {source['title']} - {source['url']}\n"
            
            # Ensure the LLM uses this information
            enhanced_question = f"""
{user_question}

Use the following information from a recent web search to help with your answer:
{formatted_web_info}

Please incorporate this web information into your response and cite sources when appropriate.
"""
        else:
            enhanced_question = user_question
        
        # Get response from conversation chain
        response = st.session_state.conversation.invoke({'question': enhanced_question})
        
        # Update chat history in session state - but with the original question
        # Update the memory directly with original question
        if enhanced_question != user_question and hasattr(st.session_state.conversation, 'memory'):
            # Fix the memory to show the original question, not the enhanced one
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
                
        # Display source documents if available
        if 'source_documents' in response and response['source_documents']:
            # Generate a summary of retrieved documents if option is enabled
            if st.session_state.show_source_summary:
                # Generate dynamic summary based on document content
                bullet_points = generate_source_summary(response['source_documents'])
                
                # Display a combined summary before showing individual sources
                with st.expander("Retrieved Documents Summary", expanded=True):
                    st.info("Below is a summary of the key information found in the retrieved documents:")
                    
                    # Display dynamic bullet points
                    for point in bullet_points:
                        st.markdown(f"- {point}")
            
            # Show individual sources
            with st.expander("Document Sources", expanded=True):
                # Create a container for sources to improve layout
                sources_container = st.container()
                
                with sources_container:
                    for i, doc in enumerate(response['source_documents']):
                        source_box = st.container()
                        
                        with source_box:
                            st.markdown(f"#### Source {i+1}")
                            
                            # Format content for better readability
                            content = doc.page_content
                            
                            # Escape HTML to prevent rendering issues with special characters
                            import html
                            escaped_content = html.escape(content)
                            
                            # Handle very long content with truncation and expandable view
                            is_long_content = len(content) > 1000
                            display_content = escaped_content[:1000] + "..." if is_long_content else escaped_content
                            
                            # Create a styled source box
                            st.markdown(
                                f"""
                                <div style="background-color: #2e2e2e; 
                                            padding: 15px; 
                                            border-radius: 10px; 
                                            border-left: 5px solid #2e2e2e; 
                                            margin-bottom: 15px;
                                            font-family: 'Source Sans Pro', sans-serif;
                                            overflow-wrap: break-word;
                                            word-wrap: break-word;
                                            white-space: pre-wrap;">
                                    {display_content}
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                            
                            # Add expandable section for viewing full content if truncated
                            if is_long_content:
                                with st.expander("View full content"):
                                    st.markdown(
                                        f"""
                                        <div style="background-color: #f8f9fa; 
                                                    padding: 10px; 
                                                    border-radius: 5px;
                                                    overflow-wrap: break-word;
                                                    word-wrap: break-word;
                                                    white-space: pre-wrap;">
                                            {escaped_content}
                                        </div>
                                        """, 
                                        unsafe_allow_html=True
                                    )
                            
                            # Display source metadata in a cleaner format
                            if hasattr(doc, 'metadata') and doc.metadata:
                                metadata_html = ""
                                
                                if 'source' in doc.metadata:
                                    source_text = doc.metadata['source']
                                    # Truncate extremely long sources
                                    if len(source_text) > 200:
                                        source_text = source_text[:197] + "..."
                                    metadata_html += f"<span style='font-weight: bold;'>Source:</span> {html.escape(source_text)}<br>"
                                elif 'url' in doc.metadata:
                                    url = doc.metadata['url']
                                    # Ensure URL is properly formatted
                                    if len(url) > 100:
                                        displayed_url = url[:97] + "..."
                                    else:
                                        displayed_url = url
                                    metadata_html += f"<span style='font-weight: bold;'>URL:</span> <a href='{html.escape(url)}' target='_blank'>{html.escape(displayed_url)}</a><br>"
                                
                                if metadata_html:
                                    st.markdown(
                                        f"""
                                        <div style="margin-top: 5px; margin-bottom: 20px; font-size: 14px;">
                                            {metadata_html}
                                        </div>
                                        """, 
                                        unsafe_allow_html=True
                                    )
                            
                            st.divider()
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
```

Szczegółowa analiza funkcji `handle_userinput`:

Ta funkcja jest centralnym punktem obsługi interakcji z użytkownikiem. Przetwarza pytanie użytkownika, opcjonalnie wzbogaca je o wyniki wyszukiwania internetowego, generuje odpowiedź i wyświetla ją wraz z dokumentami źródłowymi. Implementuje pełny cykl RAG (Retrieval-Augmented Generation) w praktyce.

1. **Inicjalizacja zmiennych** dla kontekstu i źródeł webowych:
   ```python
   web_context = ""
   web_sources = []
   ```

2. **Wyszukiwanie internetowe** (jeśli włączone):
   ```python
   if st.session_state.web_search_enabled and user_question:
       with st.spinner("Searching the web for additional context..."):
           search_results = perform_tavily_search(
               query=user_question, 
               search_depth=st.session_state.web_search_depth,
               max_results=st.session_state.max_results,
               include_answer=st.session_state.include_answer,
               include_images=st.session_state.include_images,
               time_range=st.session_state.time_range
           )
   ```
   
   - Sprawdza, czy wyszukiwanie internetowe jest włączone i czy istnieje pytanie
   - Wyświetla spinner (animację ładowania) podczas wyszukiwania
   - Wywołuje funkcję `perform_tavily_search` z odpowiednimi parametrami:
     - `query` - pytanie użytkownika
     - `search_depth` - głębokość wyszukiwania ("basic" lub "advanced")
     - `max_results` - maksymalna liczba wyników
     - `include_answer` - czy dołączyć odpowiedź wygenerowaną przez AI Tavily
     - `include_images` - czy dołączyć obrazy
     - `time_range` - zakres czasowy wyników

3. **Przetwarzanie wyników wyszukiwania**:
   ```python
   if search_results and "error" not in search_results:
       # Extract web search answer and sources
       if "answer" in search_results and search_results["answer"]:
           web_context = search_results["answer"]
   ```
   
   - Sprawdza, czy otrzymano wyniki bez błędów
   - Jeśli API Tavily zwróciło odpowiedź AI, zapisuje ją jako kontekst webowy
   - Następnie ekstrahuje szczegóły źródeł (tytuł, URL, fragment treści)

4. **Wyświetlanie wyników wyszukiwania** w interfejsie użytkownika:
   ```python
   with st.expander("Web Search Results", expanded=True):
       if web_context:
           st.write(web_context)
       
       if web_sources:
           st.markdown("### Sources:")
           for i, source in enumerate(web_sources):
               st.markdown(f"**Source {i+1}:** [{source['title']}]({source['url']})")
               st.markdown(f"_Preview:_ {source['content']}")
   ```
   
   - Tworzy rozwijalny panel (expander) z domyślnie rozwiniętym widokiem
   - Wyświetla kontekst webowy (odpowiedź AI Tavily), jeśli jest dostępny
   - Lista źródeł z linkami do oryginalnych stron i podglądem treści

5. **Dodawanie wyników wyszukiwania do bazy wektorowej**:
   ```python
   if "results" in search_results and st.session_state.conversation:
       content_texts = [result.get("content", "") for result in search_results["results"] if "content" in result]
       source_urls = [result.get("url", "") for result in search_results["results"] if "content" in result]
       
       if content_texts:
           # Add to vectorstore
           add_search_results_to_vectorstore(content_texts, source_urls)
   ```
   
   - Ekstrahuje treść i URL-e z wyników wyszukiwania
   - Wywołuje funkcję `add_search_results_to_vectorstore`, która dodaje wyniki do bazy wektorowej
   - Dzięki temu wyniki wyszukiwania stają się częścią bazy wiedzy na równi z przesłanymi dokumentami

6. **Blok try-except** dla głównej logiki przetwarzania pytania:
   - Zapewnia obsługę błędów, aby aplikacja nie zawiesiła się w przypadku problemów

7. **Formatowanie wzbogaconego pytania** z wynikami wyszukiwania internetowego:
   ```python
   if web_context and web_sources:
       # Format web information to be explicitly used by the LLM
       formatted_web_info = f"""
   Web search found the following information relevant to your question:

   {web_context}

   Sources:
   """
       for i, source in enumerate(web_sources):
           formatted_web_info += f"{i+1}. {source['title']} - {source['url']}\n"
       
       # Ensure the LLM uses this information
       enhanced_question = f"""
   {user_question}

   Use the following information from a recent web search to help with your answer:
   {formatted_web_info}

   Please incorporate this web information into your response and cite sources when appropriate.
   """
   else:
       enhanced_question = user_question
   ```
   
   - Jeśli dostępne są wyniki wyszukiwania, tworzy wzbogacone pytanie
   - Pytanie zawiera instrukcje dla modelu językowego, aby uwzględnił wyniki wyszukiwania internetowego
   - Zawiera oryginalne pytanie, wyniki wyszukiwania i instrukcje dotyczące cytowania źródeł
   - Jeśli nie ma wyników wyszukiwania, używa oryginalnego pytania

8. **Wywołanie łańcucha konwersacji** z pytaniem (ewentualnie wzbogaconym):
   ```python
   response = st.session_state.conversation.invoke({'question': enhanced_question})
   ```
   
   - Wywołuje łańcuch konwersacji utworzony wcześniej w `get_conversation_chain`
   - Przekazuje pytanie jako argument w słowniku
   - Otrzymuje odpowiedź zawierającą zarówno wygenerowany tekst, jak i dokumenty źródłowe

9. **Aktualizacja historii czatu** z oryginalnym (nie wzbogaconym) pytaniem:
   ```python
   if enhanced_question != user_question and hasattr(st.session_state.conversation, 'memory'):
       # Fix the memory to show the original question, not the enhanced one
       messages = st.session_state.conversation.memory.chat_memory.messages
       for i, msg in enumerate(messages):
           if msg.type == 'human' and msg.content == enhanced_question:
               messages[i].content = user_question
   ```
   
   - Inteligentnie podmienia wzbogacone pytanie na oryginalne w historii konwersacji
   - Dzięki temu historia czatu pokazuje tylko oryginalne pytanie użytkownika, a nie skomplikowaną wersję z instrukcjami

10. **Synchronizacja historii czatu** między LangChain a Streamlit:
    ```python
    st.session_state.chat_history = st.session_state.conversation.memory.chat_memory.messages

    st.session_state.messages = []
    for message in st.session_state.chat_history:
        if message.type == 'human':
            st.session_state.messages.append({"role": "user", "content": message.content})
        else:
            st.session_state.messages.append({"role": "assistant", "content": message.content})
    ```
    
    - Pobiera aktualną historię z pamięci łańcucha konwersacji
    - Konwertuje wiadomości z formatu LangChain do formatu używanego przez Streamlit
    - Rozróżnia wiadomości użytkownika (`human`) i asystenta (AI)

11. **Wyświetlanie dokumentów źródłowych** użytych do generowania odpowiedzi:
    ```python
    if 'source_documents' in response and response['source_documents']:
        # Generate a summary of retrieved documents if option is enabled
        if st.session_state.show_source_summary:
            bullet_points = generate_source_summary(response['source_documents'])
            
            with st.expander("Retrieved Documents Summary", expanded=True):
                # ...
    ```
    
    - Sprawdza, czy odpowiedź zawiera dokumenty źródłowe
    - Jeśli włączona jest opcja podsumowania dokumentów, generuje podsumowanie za pomocą `generate_source_summary`
    - Wyświetla podsumowanie w rozwijalnym panelu

12. **Szczegółowe wyświetlanie poszczególnych źródeł**:
    ```python
    with st.expander("Document Sources", expanded=True):
        sources_container = st.container()
        
        with sources_container:
            for i, doc in enumerate(response['source_documents']):
                # ...
    ```
    
    - Tworzy rozwijalny panel dla dokumentów źródłowych
    - Iteruje przez każdy dokument i wyświetla go w osobnym kontenerze
    - Formatuje treść dokumentu z obsługą HTML, zabezpieczeniem przed błędami renderowania i dobrym stylem

13. **Zaawansowane formatowanie treści dokumentów**:
    ```python
    # Escape HTML to prevent rendering issues with special characters
    import html
    escaped_content = html.escape(content)
    
    # Handle very long content with truncation and expandable view
    is_long_content = len(content) > 1000
    display_content = escaped_content[:1000] + "..." if is_long_content else escaped_content
    ```
    
    - Zabezpiecza specjalne znaki HTML, aby uniknąć problemów z renderowaniem
    - Obsługuje bardzo długie dokumenty, pokazując tylko pierwsze 1000 znaków z możliwością rozwinięcia pełnej treści

14. **Stylizowane wyświetlanie dokumentów** z użyciem HTML:
    ```python
    st.markdown(
        f"""
        <div style="background-color: #2e2e2e; 
                    padding: 15px; 
                    border-radius: 10px; 
                    border-left: 5px solid #2e2e2e; 
                    margin-bottom: 15px;
                    font-family: 'Source Sans Pro', sans-serif;
                    overflow-wrap: break-word;
                    word-wrap: break-word;
                    white-space: pre-wrap;">
            {display_content}
        </div>
        """, 
        unsafe_allow_html=True
    )
    ```
    
    - Używa niestandardowego stylizowania HTML dla lepszego wyglądu dokumentów
    - Właściwości CSS zapewniają czytelność, odpowiednie zawijanie tekstu i spójny wygląd

15. **Obsługa długich dokumentów** z możliwością rozwinięcia:
    ```python
    if is_long_content:
        with st.expander("View full content"):
            # ...
    ```
    
    - Dla długich dokumentów dodaje możliwość zobaczenia pełnej treści
    - Używa komponentu `st.expander` do schowania pełnej treści, gdy nie jest potrzebna

16. **Wyświetlanie metadanych dokumentu**:
    ```python
    if hasattr(doc, 'metadata') and doc.metadata:
        metadata_html = ""
        
        if 'source' in doc.metadata:
            # ...
        elif 'url' in doc.metadata:
            # ...
    ```
    
    - Sprawdza, czy dokument ma metadane
    - Obsługuje różne typy metadanych (źródło, URL)
    - Formatuje i wyświetla metadane w czytelny sposób

17. **Obsługa błędów** dla całego procesu przetwarzania:
    ```python
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
    ```
    
    - Wyświetla użytkownikowi informację o błędzie
    - Importuje moduł `traceback` do uzyskania pełnej informacji o błędzie
    - Wyświetla pełny ślad stosu błędu w interfejsie użytkownika

Ta funkcja pokazuje, jak skutecznie połączyć wyszukiwanie internetowe, wyszukiwanie dokumentów i generowanie odpowiedzi w jednym spójnym interfejsie użytkownika, z dbałością o szczegóły UI/UX, obsługę błędów i optymalizacją przetwarzania.

### `perform_tavily_search(query, ...)`

```python
def perform_tavily_search(query, search_depth="basic", max_results=10, include_answer=True, include_images=False, time_range=None):
    """
    Perform a web search using the Tavily API
    
    Args:
        query: The search query
        search_depth: 'basic' or 'advanced' (more comprehensive but slower)
        max_results: Maximum number of results to return
        include_answer: Whether to include an AI-generated answer
        include_images: Whether to include images in results
        time_range: Optional time range for results (e.g., "day", "week", "month")
        
    Returns:
        Dictionary containing search results or error information
    """
    try:
        # Get API key from environment
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return {"error": "Tavily API key not found. Please add TAVILY_API_KEY to your .env file."}
        
        # Construct the URL and headers
        url = "https://api.tavily.com/search"
        headers = {
            "content-type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Prepare the payload
        payload = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": include_answer,
            "include_images": include_images,
        }
        
        # Add optional parameters if provided
        if time_range:
            payload["time_range"] = time_range
            
        # Add any include/exclude domains from session state
        if hasattr(st.session_state, 'include_domains') and st.session_state.include_domains:
            domains = [d.strip() for d in st.session_state.include_domains.split(',') if d.strip()]
            if domains:
                payload["include_domains"] = domains
                
        if hasattr(st.session_state, 'exclude_domains') and st.session_state.exclude_domains:
            domains = [d.strip() for d in st.session_state.exclude_domains.split(',') if d.strip()]
            if domains:
                payload["exclude_domains"] = domains
        
        # Make the API request
        response = requests.post(url, json=payload, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            # Handle API errors
            try:
                error_data = response.json()
                error_message = error_data.get('message', f"API error: {response.status_code}")
                return {"error": error_message}
            except:
                return {"error": f"Failed to parse error response. Status code: {response.status_code}"}
                
    except Exception as e:
        # Handle any exceptions
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in Tavily search: {error_details}")
        return {"error": f"Search error: {str(e)}"}
```

Szczegółowa analiza funkcji `perform_tavily_search`:

Ta funkcja umożliwia wykonanie wyszukiwania internetowego za pomocą API Tavily, które jest specjalistycznym interfejsem API do wyszukiwania informacji w internecie, zoptymalizowanym pod kątem integracji z systemami AI.

1. **Sygnatura funkcji** z wieloma parametrami opcjonalnymi:
   ```python
   def perform_tavily_search(query, search_depth="basic", max_results=10, include_answer=True, include_images=False, time_range=None):
   ```
   
   - `query` - wymagane zapytanie wyszukiwania
   - `search_depth` - głębokość wyszukiwania (domyślnie "basic", alternatywnie "advanced")
   - `max_results` - maksymalna liczba wyników (domyślnie 10)
   - `include_answer` - czy dołączyć odpowiedź wygenerowaną przez AI (domyślnie True)
   - `include_images` - czy dołączyć obrazy (domyślnie False)
   - `time_range` - zakres czasowy wyników

2. **Obsługa błędów** z użyciem bloku `try-except`:
   ```python
   try:
       # kod funkcji
   except Exception as e:
       # obsługa wyjątków
   ```
   
   - Cała funkcja jest otoczona blokiem try-except
   - Zapewnia to, że funkcja zawsze zwróci jakiś wynik, nawet w przypadku wystąpienia błędu

3. **Pobranie klucza API** ze zmiennych środowiskowych:
   ```python
   api_key = os.environ.get("TAVILY_API_KEY")
   if not api_key:
       return {"error": "Tavily API key not found. Please add TAVILY_API_KEY to your .env file."}
   ```
   
   - Używa `os.environ.get()` do pobrania klucza API Tavily
   - Sprawdza, czy klucz jest dostępny
   - Zwraca przydatny komunikat błędu jeśli klucz nie został skonfigurowany

4. **Przygotowanie URL i nagłówków** dla zapytania HTTP:
   ```python
   url = "https://api.tavily.com/search"
   headers = {
       "content-type": "application/json",
       "Authorization": f"Bearer {api_key}"
   }
   ```
   
   - Definiuje URL endpointu API Tavily
   - Tworzy nagłówki HTTP z typem zawartości i autoryzacją
   - Używa schematu autoryzacji Bearer z kluczem API

5. **Przygotowanie payload'u (danych)** zapytania:
   ```python
   payload = {
       "query": query,
       "search_depth": search_depth,
       "max_results": max_results,
       "include_answer": include_answer,
       "include_images": include_images,
   }
   ```
   
   - Tworzy słownik z parametrami zapytania
   - Wszystkie parametry są przekazywane dalej do API Tavily

6. **Dodanie opcjonalnych parametrów**:
   ```python
   if time_range:
       payload["time_range"] = time_range
   ```
   
   - Dodaje parametr `time_range` tylko jeśli został przekazany do funkcji
   - Pozwala na filtrowanie wyników wg czasu (np. ostatni dzień, tydzień, miesiąc)

7. **Obsługa domen do uwzględnienia lub wykluczenia**:
   ```python
   if hasattr(st.session_state, 'include_domains') and st.session_state.include_domains:
       domains = [d.strip() for d in st.session_state.include_domains.split(',') if d.strip()]
       if domains:
           payload["include_domains"] = domains
   ```
   
   - Sprawdza, czy w stanie sesji istnieją zmienne `include_domains` lub `exclude_domains`
   - Jeśli tak, parsuje ciąg tekstowy rozdzielony przecinkami na listę domen
   - Dodaje listę do payload'u, aby ograniczyć wyniki wyszukiwania do określonych domen lub wykluczyć niektóre domeny

8. **Wykonanie zapytania HTTP**:
   ```python
   response = requests.post(url, json=payload, headers=headers)
   ```
   
   - Używa biblioteki `requests` do wykonania żądania POST
   - Przekazuje dane jako JSON
   - Dołącza przygotowane wcześniej nagłówki

9. **Przetwarzanie odpowiedzi**:
   ```python
   if response.status_code == 200:
       return response.json()
   else:
       # Handle API errors
       try:
           error_data = response.json()
           error_message = error_data.get('message', f"API error: {response.status_code}")
           return {"error": error_message}
       except:
           return {"error": f"Failed to parse error response. Status code: {response.status_code}"}
   ```
   
   - Sprawdza kod statusu odpowiedzi
   - Jeśli status to 200 (OK), zwraca dane JSON z odpowiedzi
   - W przypadku błędu, próbuje odczytać komunikat błędu z odpowiedzi JSON
   - Jeśli to się nie powiedzie, tworzy generyczny komunikat błędu z kodem statusu

10. **Obsługa wyjątków podczas wykonania**:
    ```python
    except Exception as e:
        # obsługa wyjątków
    ```
    
    - Przechwytuje wszelkie wyjątki, które mogą wystąpić podczas wykonania
    - Importuje moduł `traceback` do uzyskania szczegółów błędu
    - Drukuje szczegóły błędu w konsoli (dla deweloperów)

Funkcja `perform_tavily_search` pokazuje dobry przykład integracji z zewnętrznym API, z kompleksową obsługą błędów i elastyczną konfiguracją. Jest zbudowana w sposób, który zapewnia użytkownikowi przydatne informacje zwrotne nawet w przypadku problemów z wyszukiwaniem.

### `add_search_results_to_vectorstore(content_texts, source_urls)`

```python
def add_search_results_to_vectorstore(content_texts, source_urls):
    """
    Add web search results to the vector store
    
    Args:
        content_texts: List of text content from search results
        source_urls: List of source URLs for the content
    """
    if not hasattr(st.session_state, 'retriever') or not st.session_state.retriever:
        st.error("No retriever available to add web results to.")
        return
    
    # Prepare chunks and metadata
    try:
        # Split content into smaller chunks if needed
        import uuid
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Create text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        all_chunks = []
        all_metadatas = []
        
        # Process each content piece
        for i, content in enumerate(content_texts):
            # Generate a unique ID for this content
            content_id = str(uuid.uuid4())
            
            # Get source URL or default to "Web search result"
            source_url = source_urls[i] if i < len(source_urls) else "Web search result"
            
            # Split into chunks
            chunks = splitter.split_text(content)
            
            # Create metadata for each chunk
            metadatas = [{"source": source_url, "url": source_url, "content_id": content_id} for _ in chunks]
            
            all_chunks.extend(chunks)
            all_metadatas.extend(metadatas)
        
        # Check if we have anything to add
        if not all_chunks:
            st.warning("No content to add to vectorstore.")
            return
            
        # Add to vectorstore based on retriever type
        retriever = st.session_state.retriever
        
        # Check if we're using a HybridRetriever
        if hasattr(retriever, 'vectorstore'):
            # This is likely a hybrid retriever
            vectorstore = retriever.vectorstore
            vectorstore.add_texts(all_chunks, all_metadatas)
            st.success(f"Added {len(all_chunks)} web search chunks to the vector database.")
            
            # Also add to the keyword index if it exists
            if hasattr(retriever, 'keyword_index') and retriever.keyword_index:
                for i, chunk in enumerate(all_chunks):
                    retriever.keyword_index.add_doc({
                        "id": f"web-{i}-{all_metadatas[i]['content_id']}",
                        "text": chunk,
                        "metadata": all_metadatas[i]
                    })
                st.success(f"Added {len(all_chunks)} web search chunks to the keyword index.")
                
        else:
            # Try getting the vectorstore from session state instead
            if hasattr(st.session_state, 'vectorstore') and st.session_state.vectorstore:
                st.session_state.vectorstore.add_texts(all_chunks, all_metadatas)
                st.success(f"Added {len(all_chunks)} web search chunks to the vector database.")
            else:
                st.error("Could not identify a suitable vectorstore to add web results to.")
            
    except Exception as e:
        st.error(f"Error adding search results to vectorstore: {str(e)}")
        import traceback
        print(traceback.format_exc())
```

Szczegółowa analiza funkcji `add_search_results_to_vectorstore`:

Ta funkcja dodaje wyniki wyszukiwania internetowego do bazy wektorowej, umożliwiając wykorzystanie ich w systemie RAG na równi z dokumentami przesłanymi przez użytkownika.

1. **Weryfikacja dostępności retrievera**:
   ```python
   if not hasattr(st.session_state, 'retriever') or not st.session_state.retriever:
       st.error("No retriever available to add web results to.")
       return
   ```
   
   - Sprawdza, czy w stanie sesji istnieje obiekt retrievera
   - Jeśli nie, wyświetla komunikat o błędzie i przerywa działanie

2. **Użycie bloku try-except** dla bezpiecznego przetwarzania:
   ```python
   try:
       # kod funkcji
   except Exception as e:
       st.error(f"Error adding search results to vectorstore: {str(e)}")
       import traceback
       print(traceback.format_exc())
   ```
   
   - Zapewnia, że wszelkie błędy podczas przetwarzania nie spowodują awarii aplikacji
   - Wyświetla komunikat o błędzie dla użytkownika
   - Drukuje pełny ślad stosu błędu do konsoli dla deweloperów

3. **Import potrzebnych modułów**:
   ```python
   import uuid
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   ```
   
   - `uuid` służy do generowania unikalnych identyfikatorów dla fragmentów tekstu
   - `RecursiveCharacterTextSplitter` to zaawansowany splitter tekstu z LangChain

4. **Tworzenie splittera tekstu**:
   ```python
   splitter = RecursiveCharacterTextSplitter(
       chunk_size=1000,
       chunk_overlap=200,
       separators=["\n\n", "\n", ". ", " ", ""]
   )
   ```
   
   - Tworzy splitter, który dzieli tekst na fragmenty o maksymalnej długości 1000 znaków
   - Ustala nakładkę (overlap) na 200 znaków, co pomaga zachować kontekst między fragmentami

5. **Inicjalizacja list na fragmenty i metadane**:
   ```python
   all_chunks = []
   all_metadatas = []
   ```
   
   - Tworzy puste listy do przechowywania fragmentów tekstu i powiązanych metadanych

6. **Przetwarzanie każdego fragmentu tekstu** z wyników wyszukiwania:
   ```python
   for i, content in enumerate(content_texts):
       # Generate a unique ID for this content
       content_id = str(uuid.uuid4())
       
       # Get source URL or default to "Web search result"
       source_url = source_urls[i] if i < len(source_urls) else "Web search result"
       
       # Split into chunks
       chunks = splitter.split_text(content)
       
       # Create metadata for each chunk
       metadatas = [{"source": source_url, "url": source_url, "content_id": content_id} for _ in chunks]
       
       all_chunks.extend(chunks)
       all_metadatas.extend(metadatas)
   ```
   
   - Dla każdego fragmentu tekstu:
     - Generuje unikalny identyfikator UUID
     - Pobiera odpowiedni URL źródłowy lub używa domyślnej wartości
     - Dzieli tekst na mniejsze fragmenty
     - Tworzy metadane dla każdego fragmentu (URL źródłowy i ID treści)
     - Dodaje fragmenty i metadane do list zbiorczych

7. **Sprawdzenie, czy są fragmenty do dodania**:
   ```python
   if not all_chunks:
       st.warning("No content to add to vectorstore.")
       return
   ```
   
   - Jeśli nie ma żadnych fragmentów do dodania, wyświetla ostrzeżenie i kończy działanie

8. **Dodawanie do bazy wektorowej** w zależności od typu retrievera:
   ```python
   retriever = st.session_state.retriever
   
   # Check if we're using a HybridRetriever
   if hasattr(retriever, 'vectorstore'):
       # This is likely a hybrid retriever
       vectorstore = retriever.vectorstore
       vectorstore.add_texts(all_chunks, all_metadatas)
       st.success(f"Added {len(all_chunks)} web search chunks to the vector database.")
   ```
   
   - Pobiera obiekt retrievera ze stanu sesji
   - Sprawdza, czy retriever ma atrybut `vectorstore` (co sugeruje HybridRetriever)
   - Jeśli tak, dodaje fragmenty tekstu i metadane do bazy wektorowej
   - Wyświetla komunikat o sukcesie z liczbą dodanych fragmentów

9. **Obsługa indeksu słów kluczowych** dla retrievera hybrydowego:
   ```python
   # Also add to the keyword index if it exists
   if hasattr(retriever, 'keyword_index') and retriever.keyword_index:
       for i, chunk in enumerate(all_chunks):
           retriever.keyword_index.add_doc({
               "id": f"web-{i}-{all_metadatas[i]['content_id']}",
               "text": chunk,
               "metadata": all_metadatas[i]
           })
       st.success(f"Added {len(all_chunks)} web search chunks to the keyword index.")
   ```
   
   - Sprawdza, czy retriever ma indeks słów kluczowych (dla wyszukiwania BM25)
   - Jeśli tak, dodaje dokumenty do indeksu słów kluczowych
   - Tworzy unikalne ID dla każdego dokumentu, łącząc prefiks "web", indeks i content_id
   - Wyświetla komunikat o sukcesie

10. **Obsługa alternatywna** dla standardowego retrievera:
    ```python
    else:
        # Try getting the vectorstore from session state instead
        if hasattr(st.session_state, 'vectorstore') and st.session_state.vectorstore:
            st.session_state.vectorstore.add_texts(all_chunks, all_metadatas)
            st.success(f"Added {len(all_chunks)} web search chunks to the vector database.")
        else:
            st.error("Could not identify a suitable vectorstore to add web results to.")
    ```
    
    - Jeśli retriever nie jest hybrydowy, próbuje uzyskać dostęp do bazy wektorowej bezpośrednio ze stanu sesji
    - Jeśli to się powiedzie, dodaje fragmenty tekstu do bazy wektorowej
    - W przeciwnym razie wyświetla komunikat o błędzie

Funkcja `add_search_results_to_vectorstore` jest kluczowa dla integracji wyszukiwania internetowego z systemem RAG. Pozwala na dynamiczne rozszerzanie bazy wiedzy dostępnej dla modelu o aktualne informacje z internetu, co jest szczególnie przydatne w przypadku pytań dotyczących aktualnych wydarzeń lub tematów niepokrytych w przesłanych dokumentach.

### `main()`

```python
def main():
    # Load environment variables
    load_dotenv()
    
    # Configure the page
    st.set_page_config(
        page_title="Chat with Multiple PDFs",
        page_icon="📄"
    )
    
    # Initialize session state variables if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
        
    # Initialize settings with defaults
    if "enable_performance_mode" not in st.session_state:
        st.session_state.enable_performance_mode = False
    if "batch_size" not in st.session_state:
        st.session_state.batch_size = 20
    if "web_search_enabled" not in st.session_state:
        st.session_state.web_search_enabled = False
    if "web_search_depth" not in st.session_state:
        st.session_state.web_search_depth = "basic"
    if "max_results" not in st.session_state:
        st.session_state.max_results = 10
    if "include_answer" not in st.session_state:
        st.session_state.include_answer = True
    if "include_images" not in st.session_state:
        st.session_state.include_images = False
    if "time_range" not in st.session_state:
        st.session_state.time_range = None
    if "include_domains" not in st.session_state:
        st.session_state.include_domains = ""
    if "exclude_domains" not in st.session_state:
        st.session_state.exclude_domains = ""
    if "use_hybrid_search" not in st.session_state:
        st.session_state.use_hybrid_search = True
    if "retrieve_k" not in st.session_state:
        st.session_state.retrieve_k = 5
    if "semantic_weight" not in st.session_state:
        st.session_state.semantic_weight = 0.5
    if "use_contextual_reranking" not in st.session_state:
        st.session_state.use_contextual_reranking = False
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = None
    if "show_source_summary" not in st.session_state:
        st.session_state.show_source_summary = True
    if "developer_mode" not in st.session_state:
        st.session_state.developer_mode = False
    
    # App title and header
    st.title("Chat with Multiple Documents 📚")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    if user_question := st.chat_input("Ask a question about your documents:"):
        with st.chat_message("user"):
            st.write(user_question)
            
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Generate response if conversation exists
        if st.session_state.conversation is not None:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    handle_userinput(user_question)
                    
                    # Display assistant's response (the latest message)
                    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                        st.write(st.session_state.messages[-1]["content"])
        else:
            st.warning("Please upload PDF files in the sidebar to process before asking questions.")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Create tabs for sidebar organization
        tab1, tab2, tab3, tab4 = st.tabs(["Documents", "Web Search", "Retrieval Settings", "Advanced Settings"])
        
        # Tab 1: Documents
        with tab1:
            st.subheader("Upload Documents")
            uploaded_files = st.file_uploader(
                "Upload PDF, TXT, DOCX, CSV, or JSON files", 
                type=["pdf", "txt", "docx", "csv", "json"], 
                accept_multiple_files=True
            )
            
            # Button to process uploaded files
            if uploaded_files:
                process_button = st.button("Process Documents")
                
                if process_button:
                    with st.spinner("Processing documents..."):
                        # 1. Extract text from PDFs
                        raw_text = get_pdf_text(uploaded_files)
                        
                        # Check if we got any text from the documents
                        if raw_text:
                            # 2. Create text chunks
                            text_chunks = get_text_chunks(raw_text)
                            st.session_state.text_chunks = text_chunks
                            
                            # Check if we have embedding environment variables
                            if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
                                if not os.environ.get("MISTRAL_API_KEY"):
                                    st.error("Missing API credentials. Please add HUGGINGFACEHUB_API_TOKEN or MISTRAL_API_KEY to your .env file.")
                                    st.stop()
                            
                            # 3. Create vector store
                            with st.spinner("Creating vector store..."):
                                vectorstore = get_vectorstore(
                                    text_chunks, 
                                    enable_performance_mode=st.session_state.enable_performance_mode,
                                    batch_size=st.session_state.batch_size
                                )
                                
                                st.session_state.vectorstore = vectorstore
                        
                            # 4. Create conversation chain
                            with st.spinner("Creating conversation chain..."):
                                st.session_state.conversation = get_conversation_chain(
                                    vectorstore,
                                    text_chunks if st.session_state.use_hybrid_search else None
                                )
                                
                                # Also store the retriever for direct access
                                if hasattr(st.session_state.conversation, 'retriever'):
                                    st.session_state.retriever = st.session_state.conversation.retriever
                            
                            st.success("Documents processed successfully! You can now ask questions.")
                        else:
                            st.error("No text content extracted from the uploaded documents. Please check the file formats and try again.")
                            
            clear_docs_button = st.button("Clear Uploaded Files")
            if clear_docs_button:
                # Reset all session state related to documents
                st.session_state.conversation = None
                st.session_state.retriever = None
                st.session_state.chat_history = None
                st.session_state.vectorstore = None 
                st.session_state.messages = []
                st.session_state.text_chunks = None
                st.rerun()
```

Funkcja `main()` jest sercem aplikacji Streamlit, implementuje interfejs użytkownika i koordynuje działanie wszystkich komponentów. Pełny opis tej funkcji, a także opisy zakładek w pasku bocznym, które oferują różne opcje konfiguracyjne dla użytkownika.

## Podsumowanie dotychczasowej analizy

Do tej pory przeanalizowaliśmy szczegółowo następujące elementy aplikacji:

1. **Importy bibliotek** - pełny opis wszystkich zależności i ich roli w aplikacji.

2. **Główne funkcje przetwarzania**:
   - `get_pdf_text` - ekstrakcja tekstu z różnych formatów dokumentów
   - `get_conversation_chain` - tworzenie łańcucha konwersacji RAG z modelem Mistral
   - `generate_source_summary` - automatyczne podsumowywanie znalezionych dokumentów
   - `handle_userinput` - przetwarzanie pytań użytkownika i generowanie odpowiedzi
   - `perform_tavily_search` - wykonywanie wyszukiwania internetowego
   - `add_search_results_to_vectorstore` - dodawanie wyników wyszukiwania do bazy wektorowej

3. **Początek funkcji `main()`** - która konfiguruje interfejs i inicjalizuje stan aplikacji

W następnej części dokumentacji szczegółowo opiszemy pozostałe elementy funkcji `main()`, w tym implementację zakładek w pasku bocznym, które oferują różne opcje konfiguracyjne dla użytkownika.

### Implementacja zakładek w pasku bocznym

Funkcja `main()` kontynuuje z implementacją zakładek w pasku bocznym, które zapewniają różne opcje konfiguracyjne dla użytkownika:

```python
# Tab 2: Web Search
with tab2:
    st.subheader("Web Search Settings")
    
    # Toggle for enabling/disabling web search
    st.session_state.web_search_enabled = st.toggle(
        "Enable Web Search", 
        value=st.session_state.web_search_enabled,
        help="Alongside searching your documents, also perform a web search for up-to-date information."
    )
    
    # Conditional display of web search settings
    if st.session_state.web_search_enabled:
        # API key info
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if not tavily_api_key:
            st.error("Tavily API key not found. Please add TAVILY_API_KEY to your .env file.")
            
        # Search depth
        st.session_state.web_search_depth = st.radio(
            "Search Depth",
            options=["basic", "advanced"],
            index=0 if st.session_state.web_search_depth == "basic" else 1,
            horizontal=True,
            help="Basic: Faster but less comprehensive. Advanced: More detailed but slower."
        )
        
        # Number of results
        st.session_state.max_results = st.slider(
            "Max Results", 
            min_value=1, 
            max_value=20, 
            value=st.session_state.max_results,
            help="Maximum number of search results to return."
        )
        
        # Include AI answer
        st.session_state.include_answer = st.checkbox(
            "Include AI summary of results", 
            value=st.session_state.include_answer,
            help="Include an AI-generated summary of the search results."
        )
        
        # Include images
        st.session_state.include_images = st.checkbox(
            "Include Images", 
            value=st.session_state.include_images,
            help="Include images in search results."
        )
        
        # Time range
        time_range_options = [None, "day", "week", "month"]
        time_range_labels = ["Any time", "Past day", "Past week", "Past month"]
        
        current_index = 0
        if st.session_state.time_range in time_range_options:
            current_index = time_range_options.index(st.session_state.time_range)
            
        selected_time_range = st.radio(
            "Time Range",
            options=time_range_labels,
            index=current_index,
            horizontal=True,
            help="Filter results by recency."
        )
        
        # Map the selection back to the API value
        st.session_state.time_range = time_range_options[time_range_labels.index(selected_time_range)]
        
        # Domain filters
        with st.expander("Domain Filters"):
            st.session_state.include_domains = st.text_area(
                "Include Domains", 
                value=st.session_state.include_domains,
                help="Comma-separated list of domains to include (e.g., wikipedia.org, github.com)"
            )
            
            st.session_state.exclude_domains = st.text_area(
                "Exclude Domains", 
                value=st.session_state.exclude_domains,
                help="Comma-separated list of domains to exclude"
            )

# Tab 3: Retrieval Settings
with tab3:
    st.subheader("Document Retrieval Settings")
    
    # Hybrid search toggle
    st.session_state.use_hybrid_search = st.toggle(
        "Use Hybrid Search", 
        value=st.session_state.use_hybrid_search,
        help="Hybrid search combines semantic (vector) search with keyword search for better results."
    )
    
    if st.session_state.use_hybrid_search:
        # Semantic weight slider
        st.session_state.semantic_weight = st.slider(
            "Semantic Weight", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.semantic_weight, 
            step=0.1,
            help="Balance between semantic (1.0) and keyword search (0.0). 0.5 is an equal balance."
        )
        
        # Reranking toggle
        st.session_state.use_contextual_reranking = st.toggle(
            "Use Contextual Reranking", 
            value=st.session_state.use_contextual_reranking,
            help="Rerank results based on how well they match the query in context. More accurate but slower."
        )
    
    # Number of documents to retrieve
    st.session_state.retrieve_k = st.slider(
        "Number of Documents to Retrieve", 
        min_value=1, 
        max_value=15, 
        value=st.session_state.retrieve_k,
        help="How many documents to retrieve for each query."
    )
    
    # Source summary toggle
    st.session_state.show_source_summary = st.toggle(
        "Show Source Summary", 
        value=st.session_state.show_source_summary,
        help="Show a summary of key topics from the retrieved documents."
    )

# Tab 4: Advanced Settings
with tab4:
    st.subheader("Advanced Settings")
    
    # Performance mode toggle
    st.session_state.enable_performance_mode = st.toggle(
        "Enable Performance Mode", 
        value=st.session_state.enable_performance_mode,
        help="Process embeddings in batches for better memory usage with large documents."
    )
    
    # Batch size slider (only shown if performance mode is enabled)
    if st.session_state.enable_performance_mode:
        st.session_state.batch_size = st.slider(
            "Batch Size", 
            min_value=1, 
            max_value=100, 
            value=st.session_state.batch_size,
            help="Number of document chunks to process at once. Lower values use less memory."
        )
    
    # Developer mode toggle
    st.session_state.developer_mode = st.toggle(
        "Developer Mode", 
        value=st.session_state.developer_mode,
        help="Show additional technical information for debugging."
    )
    
    # Reset all settings button
    if st.button("Reset All Settings to Default"):
        # Reset session state variables to defaults
        st.session_state.enable_performance_mode = False
        st.session_state.batch_size = 20
        st.session_state.web_search_enabled = False
        st.session_state.web_search_depth = "basic"
        st.session_state.max_results = 10
        st.session_state.include_answer = True
        st.session_state.include_images = False
        st.session_state.time_range = None
        st.session_state.include_domains = ""
        st.session_state.exclude_domains = ""
        st.session_state.use_hybrid_search = True
        st.session_state.retrieve_k = 5
        st.session_state.semantic_weight = 0.5
        st.session_state.use_contextual_reranking = False
        st.session_state.show_source_summary = True
        st.session_state.developer_mode = False
        
        st.success("Settings reset to default values!")
        st.rerun()
```

### Szczegółowa analiza zakładek w pasku bocznym

#### Zakładka "Web Search"

Zakładka ta udostępnia ustawienia związane z wyszukiwaniem internetowym, które rozszerza możliwości aplikacji poza przesłane dokumenty:

1. **Włączanie/wyłączanie wyszukiwania internetowego**:
   ```python
   st.session_state.web_search_enabled = st.toggle(
       "Enable Web Search", 
       value=st.session_state.web_search_enabled,
       help="Alongside searching your documents, also perform a web search for up-to-date information."
   )
   ```
   
   - Używa kontrolki `st.toggle()` do włączania/wyłączania funkcji
   - Przechowuje stan w zmiennej sesji `web_search_enabled`
   - Dodaje tekst pomocy wyjaśniający funkcję

2. **Weryfikacja klucza API Tavily**:
   ```python
   tavily_api_key = os.environ.get("TAVILY_API_KEY")
   if not tavily_api_key:
       st.error("Tavily API key not found. Please add TAVILY_API_KEY to your .env file.")
   ```
   
   - Sprawdza, czy klucz API Tavily jest dostępny w zmiennych środowiskowych
   - Wyświetla komunikat błędu, jeśli klucz nie jest skonfigurowany

3. **Głębokość wyszukiwania**:
   ```python
   st.session_state.web_search_depth = st.radio(
       "Search Depth",
       options=["basic", "advanced"],
       index=0 if st.session_state.web_search_depth == "basic" else 1,
       horizontal=True,
       help="Basic: Faster but less comprehensive. Advanced: More detailed but slower."
   )
   ```
   
   - Oferuje dwie opcje: "basic" (szybsze, ale mniej szczegółowe) i "advanced" (dokładniejsze, ale wolniejsze)
   - Używa układu poziomego dla opcji
   - Dodaje tekst pomocy wyjaśniający różnice

4. **Maksymalna liczba wyników**:
   ```python
   st.session_state.max_results = st.slider(
       "Max Results", 
       min_value=1, 
       max_value=20, 
       value=st.session_state.max_results,
       help="Maximum number of search results to return."
   )
   ```
   
   - Używa suwaka do wyboru liczby wyników (od 1 do 20)
   - Zachowuje poprzednią wartość jako domyślną

5. **Opcje wyszukiwania**:
   ```python
   st.session_state.include_answer = st.checkbox("Include AI summary of results", ...)
   st.session_state.include_images = st.checkbox("Include Images", ...)
   ```
   
   - Oferuje opcje włączenia podsumowania AI wyników
   - Pozwala na włączenie obrazów w wynikach

6. **Zakres czasowy wyników**:
   ```python
   time_range_options = [None, "day", "week", "month"]
   time_range_labels = ["Any time", "Past day", "Past week", "Past month"]
   
   # ... wybór opcji ...
   
   # Map the selection back to the API value
   st.session_state.time_range = time_range_options[time_range_labels.index(selected_time_range)]
   ```
   
   - Oferuje opcje filtrowania wyników według czasu (dowolny czas, ostatni dzień, tydzień, miesiąc)
   - Stosuje bardziej przyjazne dla użytkownika etykiety, które są mapowane na wartości API
   - Używa układu poziomego przycisków radiowych

7. **Filtry domen**:
   ```python
   with st.expander("Domain Filters"):
       st.session_state.include_domains = st.text_area(...)
       st.session_state.exclude_domains = st.text_area(...)
   ```
   
   - Używa rozwijalnego panelu do ukrycia mniej często używanych ustawień
   - Pozwala na określenie domen do uwzględnienia lub wykluczenia z wyników
   - Przyjmuje listę domen rozdzielonych przecinkami

#### Zakładka "Retrieval Settings"

Ta zakładka pozwala na dostosowanie sposobu, w jaki aplikacja wyszukuje i prezentuje dokumenty:

1. **Przełącznik wyszukiwania hybrydowego**:
   ```python
   st.session_state.use_hybrid_search = st.toggle(
       "Use Hybrid Search", 
       value=st.session_state.use_hybrid_search,
       help="Hybrid search combines semantic (vector) search with keyword search for better results."
   )
   ```
   
   - Włącza/wyłącza wyszukiwanie hybrydowe, które łączy wyszukiwanie semantyczne i słów kluczowych
   - Dodaje tekst pomocy wyjaśniający korzyści z tego podejścia

2. **Waga wyszukiwania semantycznego**:
   ```python
   st.session_state.semantic_weight = st.slider(
       "Semantic Weight", 
       min_value=0.0, 
       max_value=1.0, 
       value=st.session_state.semantic_weight, 
       step=0.1,
       help="Balance between semantic (1.0) and keyword search (0.0). 0.5 is an equal balance."
   )
   ```
   
   - Dostępne tylko gdy włączone jest wyszukiwanie hybrydowe
   - Pozwala na balansowanie między wyszukiwaniem semantycznym (1.0) a słów kluczowych (0.0)
   - Wartość domyślna 0.5 oznacza równą wagę obu metod

3. **Kontekstowy reranking**:
   ```python
   st.session_state.use_contextual_reranking = st.toggle(
       "Use Contextual Reranking", 
       value=st.session_state.use_contextual_reranking,
       help="Rerank results based on how well they match the query in context. More accurate but slower."
   )
   ```
   
   - Włącza/wyłącza ponowne rankowanie wyników na podstawie kontekstu
   - Poprawia trafność wyników, ale spowalnia wyszukiwanie
   - Dostępne tylko przy włączonym wyszukiwaniu hybrydowym

4. **Liczba dokumentów do pobrania**:
   ```python
   st.session_state.retrieve_k = st.slider(
       "Number of Documents to Retrieve", 
       min_value=1, 
       max_value=15, 
       value=st.session_state.retrieve_k,
       help="How many documents to retrieve for each query."
   )
   ```
   
   - Określa, ile dokumentów model powinien pobrać dla każdego zapytania
   - Zakres od 1 do 15 dokumentów
   - Więcej dokumentów może zapewnić więcej kontekstu, ale też zwiększyć "szum"

5. **Podsumowanie źródeł**:
   ```python
   st.session_state.show_source_summary = st.toggle(
       "Show Source Summary", 
       value=st.session_state.show_source_summary,
       help="Show a summary of key topics from the retrieved documents."
   )
   ```
   
   - Włącza/wyłącza automatyczne podsumowanie znalezionych dokumentów
   - Funkcja ta wykorzystuje wcześniej opisaną funkcję `generate_source_summary`

#### Zakładka "Advanced Settings"

Ta zakładka zawiera zaawansowane ustawienia techniczne:

1. **Tryb wydajności**:
   ```python
   st.session_state.enable_performance_mode = st.toggle(
       "Enable Performance Mode", 
       value=st.session_state.enable_performance_mode,
       help="Process embeddings in batches for better memory usage with large documents."
   )
   ```
   
   - Włącza przetwarzanie dokumentów w partiach (batches)
   - Optymalizuje zużycie pamięci przy dużych dokumentach
   - Może być pomocne na urządzeniach z ograniczoną pamięcią

2. **Rozmiar partii**:
   ```python
   if st.session_state.enable_performance_mode:
       st.session_state.batch_size = st.slider(
           "Batch Size", 
           min_value=1, 
           max_value=100, 
           value=st.session_state.batch_size,
           help="Number of document chunks to process at once. Lower values use less memory."
       )
   ```
   
   - Dostępne tylko gdy tryb wydajności jest włączony
   - Określa liczbę fragmentów dokumentów przetwarzanych jednocześnie
   - Mniejsze wartości oznaczają mniejsze zużycie pamięci, ale wolniejsze przetwarzanie

3. **Tryb deweloperski**:
   ```python
   st.session_state.developer_mode = st.toggle(
       "Developer Mode", 
       value=st.session_state.developer_mode,
       help="Show additional technical information for debugging."
   )
   ```
   
   - Włącza/wyłącza tryb deweloperski
   - W tym trybie aplikacja może wyświetlać dodatkowe informacje techniczne
   - Przydatne do debugowania i rozwoju aplikacji

4. **Resetowanie ustawień**:
   ```python
   if st.button("Reset All Settings to Default"):
       # Reset session state variables to defaults
       # ...
       st.success("Settings reset to default values!")
       st.rerun()
   ```
   
   - Przycisk resetujący wszystkie ustawienia do wartości domyślnych
   - Po resecie wszystkich ustawień, strona jest przeładowywana za pomocą `st.rerun()`

## Podsumowanie struktury aplikacji

Aplikacja `app.py` stanowi kompleksowe rozwiązanie do interaktywnego czatu z wieloma dokumentami z wykorzystaniem technik RAG (Retrieval-Augmented Generation). Jej główne komponenty to:

1. **Przetwarzanie dokumentów**:
   - Obsługa wielu formatów plików (PDF, TXT, DOCX, CSV, JSON)
   - Podział dokumentów na fragmenty
   - Wektoryzacja fragmentów z użyciem zaawansowanego modelu embeddingowego E5

2. **Zaawansowane wyszukiwanie**:
   - Wyszukiwanie semantyczne w bazie wektorowej FAISS
   - Wyszukiwanie hybrydowe łączące wyszukiwanie semantyczne i słów kluczowych (BM25)
   - Reranking kontekstowy dla lepszej trafności wyników
   - Integracja z wyszukiwaniem internetowym (API Tavily)

3. **Generowanie odpowiedzi**:
   - Wykorzystanie modelu Mistral AI do generowania odpowiedzi
   - Pamięć konwersacji umożliwiająca zadawanie pytań kontekstowych
   - Automatyczne podsumowywanie znalezionych dokumentów
   - Prezentacja źródeł i cytowań

4. **Interfejs użytkownika**:
   - Intuicyjny interfejs czatu w Streamlit
   - Panel konfiguracyjny z zakładkami dla różnych ustawień
   - Rozbudowane opcje dostosowywania działania aplikacji
   - Tryb wydajności dla obsługi dużych dokumentów

Aplikacja jest zaprojektowana z myślą o elastyczności i rozszerzalności. Dzięki wykorzystaniu stanu sesji Streamlit, wszystkie ustawienia są zachowywane między interakcjami, a interfejs jest responsywny i przyjazny dla użytkownika.

Szczególnie wyróżniającymi się cechami aplikacji są:

1. **Wielomodalny input**:
   - Obsługa wielu formatów dokumentów
   - Możliwość mieszania różnych typów plików w jednej sesji
   - Dodawanie kontekstu przez wyszukiwanie internetowe

2. **Hybrydowe wyszukiwanie**:
   - Łączenie najlepszych cech wyszukiwania semantycznego i słów kluczowych
   - Możliwość dostosowania wagi obu podejść
   - Opcjonalny reranking dla jeszcze lepszej trafności

3. **Rozbudowane zarządzanie wyszukiwaniem internetowym**:
   - Integracja z API Tavily
   - Filtrowanie wyników według czasu, domen, itd.
   - Dodawanie wyników wyszukiwania do bazy wektorowej

4. **Przyjazne dla użytkownika zarządzanie źródłami**:
   - Automatyczne podsumowanie dokumentów źródłowych
   - Stylizowane wyświetlanie dokumentów z obsługą długich treści
   - Wyróżnianie metadanych i cytowań

Ta aplikacja stanowi wyrafinowany przykład wykorzystania nowoczesnych technik AI i NLP do tworzenia interaktywnych systemów pytań i odpowiedzi, które mogą pracować zarówno z lokalnymi dokumentami, jak i informacjami z internetu.

## Rozszerzenia i możliwości dalszego rozwoju

Potencjalne kierunki rozszerzenia aplikacji mogłyby obejmować:

1. **Obsługa większej liczby formatów plików**:
   - Dodanie obsługi prezentacji (PPT, PPTX)
   - Obsługa dokumentów HTML, XML, Markdown
   - Ekstrakcja tekstu z obrazów (OCR)

2. **Rozszerzone możliwości RAG**:
   - Implementacja technik hiperprzetwarzania (HyDE)
   - Zaawansowane techniki chunking, np. sliding window z różnymi nakładkami
   - Query transformations i query routing

3. **Integracje z większą liczbą modeli językowych**:
   - Obsługa lokalnych modeli (np. Llama 3, Mistral, Falcon)
   - Integracja z różnymi dostawcami API (OpenAI, Anthropic, Cohere)
   - Możliwość wyboru modelu przez użytkownika

4. **Rozszerzenia UI/UX**:
   - Tryb ciemny/jasny
   - Wizualizacje embeddingów i podobieństwa dokumentów
   - Historia zapytań i zapisywanie sesji

5. **Funkcje współpracy**:
   - Udostępnianie sesji czatu
   - Komentarze i adnotacje do dokumentów
   - Współdzielone repozytorium dokumentów

6. **Zabezpieczenia i prywatność**:
   - Szyfrowanie dokumentów
   - Kontrola dostępu i uwierzytelnianie
   - Opcje prywatności dla danych użytkownika

## Konkluzja

Aplikacja `app.py` stanowi zaawansowane rozwiązanie RAG integrujące najnowsze techniki przetwarzania języka naturalnego, wyszukiwania semantycznego i generowania tekstu. Jej modularna architektura i rozbudowane opcje konfiguracyjne umożliwiają dostosowanie do różnych scenariuszy użycia, od prostych przypadków wyszukiwania informacji po zaawansowane systemy pytań i odpowiedzi z wieloma źródłami danych.

Dzięki integracji z wyszukiwaniem internetowym, system jest zdolny do odpowiadania nie tylko na podstawie przesłanych dokumentów, ale także aktualnych informacji z sieci, co czyni go wszechstronnym narzędziem do eksploracji i analizy danych tekstowych.