# Szczegółowa dokumentacja pliku app.py - rozszerzona

## Spis treści
1. [Architektura aplikacji](#architektura-aplikacji)
2. [Dokładny opis funkcji](#dokładny-opis-funkcji)
3. [Hybrydowy system wyszukiwania - szczegóły implementacji](#hybrydowy-system-wyszukiwania---szczegóły-implementacji)
4. [Przepływ danych - dogłębna analiza](#przepływ-danych---dogłębna-analiza)
5. [Szczegóły interfejsu użytkownika](#szczegóły-interfejsu-użytkownika)
6. [Obsługa wielojęzyczności](#obsługa-wielojęzyczności)
7. [Modyfikacja i rozszerzanie aplikacji](#modyfikacja-i-rozszerzanie-aplikacji)

## Architektura aplikacji

Aplikacja `app.py` jest zbudowana w architekturze wielowarstwowej:

1. **Warstwa prezentacji**: Wykorzystuje Streamlit do tworzenia interaktywnego interfejsu
2. **Warstwa logiki biznesowej**: Funkcje przetwarzające dokumenty, generujące odpowiedzi i wyszukujące informacje
3. **Warstwa dostępu do danych**: Wektorowe bazy danych (FAISS), systemy wyszukiwania (BM25) i API zewnętrzne (Tavily)

Kluczowe funkcje są zorganizowane modułowo:
- **Funkcje wejścia/wyjścia**: `get_pdf_text`
- **Funkcje przetwarzania**: `get_text_chunks`, `generate_source_summary`
- **Funkcje wyszukiwania**: `get_hybrid_retriever`, funkcje w module `hybrid_search`
- **Funkcje komunikacji z API**: `perform_tavily_search`
- **Funkcje zarządzania stanem aplikacji**: wszystko w funkcji `main`

## Dokładny opis funkcji

### get_pdf_text

```python
def get_pdf_text(uploaded_files)
```

Ta funkcja przetwarza przesłane pliki w celu wyodrębnienia tekstu. Implementacja zawiera:

1. **Detekcję typu pliku** - sprawdzanie rozszerzenia pliku w linii `file_ext = file.name.split('.')[-1].lower()`

2. **Specjalistyczne przetwarzanie dla każdego typu**:
   - **PDF**: 
     - Użycie `PdfReader` do przetwarzania stron
     - Pętla przez wszystkie strony: `for page in pdf_reader.pages:`
     - Ekstrakcja tekstu: `page.extract_text()` z dodaniem podwójnych znaków nowej linii dla separacji stron
   
   - **TXT**: 
     - Dekodowanie zawartości binarnej: `file.getvalue().decode('utf-8')`
     - Zachowanie oryginalnego formatowania z dodatkowymi znakami nowej linii
   
   - **DOCX**: 
     - Użycie biblioteki `docx` z obsługą importu: `from docx import Document`
     - Przetwarzanie paragrafów: `for para in doc.paragraphs:`
     - Zachowanie struktury z pojedynczymi znakami nowej linii między paragrafami
   
   - **CSV**: 
     - Konwersja do tekstu przy użyciu pandas: `df = pd.read_csv(file)` oraz `df.to_string()`
     - Zachowanie struktury tabeli w formacie tekstowym
   
   - **JSON**: 
     - Parsowanie JSON: `content = json.loads(file.getvalue().decode('utf-8'))`
     - Formatowanie z wcięciami dla czytelności: `json.dumps(content, indent=2)`

3. **Kompleksowa obsługa błędów**:
   - Użycie bloku `try-except` dla każdego typu pliku
   - Specyficzne komunikaty błędów dla braków bibliotek: `st.error(f"Missing python-docx library...")`
   - Szczegółowe informacje o błędach z traceback: `print(f"Error details for {file.name}: {traceback.format_exc()}")`
   - Kontynuacja przetwarzania przy błędach: `continue` w bloku `except`

### get_conversation_chain

```python
def get_conversation_chain(vectorstore, text_chunks=None)
```

Funkcja tworząca łańcuch konwersacyjny z wykorzystaniem RAG:

1. **Inicjalizacja modelu LLM**:
   - Użycie klucza API z zmiennych środowiskowych: `api_key = os.environ["MISTRAL_API_KEY"]`
   - Konfiguracja modelu MistralAI z parametrami:
     - Model: `"mistral-small-latest"`
     - Temperatura: `0.3` (niska dla bardziej deterministycznych odpowiedzi)
     - Maksymalna liczba tokenów: `8192` (długie odpowiedzi)
     - Top-p: `0.9` (sampling z 90% najbardziej prawdopodobnych tokenów)

2. **Tworzenie systemu pamięci**:
   - Użycie `ConversationBufferMemory` z parametrami:
     - `memory_key='chat_history'` - klucz używany w szablonach promptów
     - `return_messages=True` - przechowywanie historii jako obiektów wiadomości
     - `output_key='answer'` - klucz dla wyjścia z modelu

3. **Tworzenie retrievera z obsługą trybów**:
   - **Tryb hybrydowy** (gdy `text_chunks` istnieje i `use_hybrid_search=True`):
     - Tworzenie hybrydowego retrievera poprzez `get_hybrid_retriever`
     - Obsługa błędów z informacją o fallbacku: `st.warning("Hybrid search creation failed...")`
     - Przełączanie flagi: `st.session_state.use_hybrid_search = False` w przypadku problemów
   
   - **Tryb standardowy**:
     - Użycie standardowego retrievera: `vectorstore.as_retriever(search_kwargs={"k": st.session_state.retrieve_k})`
     - Parametr `k` kontroluje liczbę zwracanych dokumentów

4. **Tworzenie łańcucha konwersacyjnego**:
   - Użycie `ConversationalRetrievalChain.from_llm` z parametrami:
     - `llm` - model językowy
     - `retriever` - system wyszukiwania dokumentów
     - `memory` - system pamięci konwersacyjnej
     - `verbose=True` - logowanie działania łańcucha
     - `return_source_documents=True` - zwracanie źródłowych dokumentów
     - `chain_type="stuff"` - tryb działania łańcucha (wstrzykiwanie całych dokumentów)

### generate_source_summary

```python
def generate_source_summary(source_documents)
```

Funkcja ta generuje semantyczne podsumowanie odzyskanych dokumentów źródłowych używając następujących kroków:

1. **Ekstrakcja tekstu z dokumentów**:
   - Ograniczenie do pierwszych 5 dokumentów: `for doc in source_documents[:5]`
   - Concatenacja tekstu z separatorami: `all_text += doc.page_content + "\n\n"`

2. **Detekcja języka na podstawie znaków**:
   - Wyszukiwanie znaków spoza ASCII: `non_latin_chars = re.findall(r'[^\x00-\x7F]', all_text)`
   - Określenie progu 5%: `is_non_latin = len(non_latin_chars) > len(all_text) * 0.05`

3. **Różne strategie ekstrakcji terminów w zależności od języka**:
   - **Dla tekstów niełacińskich**:
     - Wyszukiwanie sekwencji znaków: `words = re.findall(r'\b\w+\b', all_text)`
     - Filtrowanie krótkich terminów: `[w.lower() for w in words if len(w) > 3]`
     - Zliczanie częstości: `word_counts = Counter([...])`
     - Wybór najpopularniejszych terminów: `top_terms = [term for term, count in word_counts.most_common(10) if count > 2]`
   
   - **Dla tekstów łacińskich**:
     - Bardziej restrictywne wyszukiwanie: `words = re.findall(r'\b[A-Za-z][A-Za-z-]{3,15}\b', all_text)`
     - Filtrowanie słów stop: `words = [word.lower() for word in words if word.lower() not in [...]]`
     - Zliczanie i wybieranie: `top_terms = [term for term, count in word_counts.most_common(8) if count > 2]`

4. **Generowanie punktów podsumowania**:
   - Dynamiczne tworzenie listy punktów opartych na znalezionych terminach
   - Różne szablony dla różnej liczby terminów
   - Formaty z pogrubieniem dla terminów: `f"The documents primarily discuss **{top_terms[0]}** and related concepts."`

5. **Obsługa przypadku braku terminów**:
   - Zwracanie generycznego podsumowania w formie listy punktów

### handle_userinput

```python
def handle_userinput(user_question)
```

Ta funkcja jest sercem interakcji użytkownika z aplikacją, przetwarzająca pytania i prezentująca odpowiedzi:

1. **Wyszukiwanie w sieci (opcjonalne)**:
   - Sprawdzenie czy wyszukiwanie jest włączone: `if st.session_state.web_search_enabled and user_question:`
   - Wykonanie wyszukiwania z odpowiednimi parametrami
   - Ekstrakcja odpowiedzi i źródeł
   - Wyświetlanie wyników w rozwijalnym elemencie UI
   - Automatyczne dodawanie wyników do wektorowej bazy danych

2. **Przetwarzanie pytania**:
   - Wzbogacanie pytania o wyniki z sieci (jeśli dostępne)
   - Wywołanie łańcucha konwersacyjnego: `response = st.session_state.conversation.invoke({'question': enhanced_question})`
   - Naprawianie historii konwersacyjnej (zamiana wzbogaconego pytania na oryginalne)
   - Aktualizacja historii czatu w stanie sesji

3. **Wyświetlanie odpowiedzi i źródeł**:
   - Aktualizacja historii wiadomości do wyświetlenia
   - Jeśli są dokumenty źródłowe:
     - Generowanie i wyświetlanie podsumowania (jeśli włączone)
     - Tworzenie rozwijalnego kontenera dla źródeł
     - Iteracja przez dokumenty z formatowaniem HTML

4. **Zaawansowane formatowanie dla dokumentów źródłowych**:
   - Obsługa długich dokumentów: `is_long_content = len(content) > 1000`
   - Ucinanie długich dokumentów: `display_content = escaped_content[:1000] + "..." if is_long_content else escaped_content`
   - Stylizowany kontener HTML z CSS dla każdego dokumentu
   - Dodatkowy rozwijany widok dla pełnej zawartości długich dokumentów
   - Formatowanie metadanych dokumentu (źródło, URL)

5. **Obsługa błędów**:
   - Pełna obsługa wyjątków z szczegółowymi komunikatami
   - Wyświetlanie traceback dla celów diagnostycznych

### perform_tavily_search

```python
def perform_tavily_search(query, search_depth="basic", max_results=10, include_answer=True, include_images=False, time_range=None)
```

Funkcja wykonująca wyszukiwanie internetowe poprzez API Tavily:

1. **Konfiguracja zapytania**:
   - Bazowy URL API: `url = "https://api.tavily.com/search"`
   - Pobieranie klucza API: `api_key = os.environ.get("TAVILY_API_KEY") or st.session_state.get("TAVILY_API_KEY")`
   - Przygotowanie payload z parametrami:
     - Zapytanie użytkownika: `"query": query`
     - Głębokość wyszukiwania: `"search_depth": search_depth` (basic/advanced)
     - Opcje zawartości: `"include_answer": include_answer`, `"include_images": include_images`
     - Limit wyników: `"max_results": min(max_results, 30)` (z ograniczeniem do 30)

2. **Opcje zaawansowane**:
   - Obsługa zakresu czasowego: `if time_range: payload["time_range"] = time_range`
   - Filtrowanie domen:
     - Domeny do uwzględnienia: `if st.session_state.get("include_domains"): payload["include_domains"] = ...`
     - Domeny do wykluczenia: `if st.session_state.get("exclude_domains"): payload["exclude_domains"] = ...`

3. **Wykonanie zapytania HTTP**:
   - Ustawienie nagłówków: `headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}`
   - POST request: `response = requests.post(url, json=payload, headers=headers)`
   - Sprawdzenie błędów HTTP: `response.raise_for_status()`
   - Parsowanie odpowiedzi JSON: `results = response.json()`

4. **Obsługa błędów i diagnostyka**:
   - Pełna obsługa wyjątków HTTP: `except requests.exceptions.RequestException as e:`
   - Ekstrakcja szczegółów błędów API: `error_details = e.response.json()`
   - Obsługa nieoczekiwanych błędów: `except Exception as e:`
   - Logowanie w trybie deweloperskim: `if st.session_state.get("developer_mode"): print(...)`

### add_search_results_to_vectorstore

```python
def add_search_results_to_vectorstore(content_texts, source_urls)
```

Funkcja dodająca wyniki wyszukiwania do bazy wiedzy:

1. **Przygotowanie fragmentów tekstu i metadanych**:
   - Inicjalizacja list: `search_text_chunks = []`, `metadata_list = []`
   - Iteracja przez pary (tekst, URL): `for i, (content, url) in enumerate(zip(content_texts, source_urls)):`
   - Dzielenie każdego wyniku na fragmenty: `result_chunks = get_text_chunks(content)`
   - Tworzenie metadanych dla każdego fragmentu z URL

2. **Dodawanie do retrievera z obsługą dwóch przypadków**:
   - **Dla hybrydowego retrievera**:
     - Sprawdzenie typu retrievera: `if st.session_state.use_hybrid_search and hasattr(st.session_state.conversation.retriever, 'add_texts'):`
     - Użycie metody `add_texts` retrievera hybrydowego (aktualizuje zarówno FAISS jak i BM25)
   
   - **Dla standardowego retrievera**:
     - Pobranie istniejącego vectorstore z retrievera poprzez sprawdzenie atrybutów: `_vectorstore`, `vectorstore`
     - Dodanie tekstów do vectorstore: `existing_vectorstore.add_texts(...)`
     - Odtworzenie łańcucha konwersacyjnego z zaktualizowanym vectorstore

3. **Aktualizacja stanu aplikacji**:
   - Dla hybrydowego wyszukiwania: `st.session_state.all_text_chunks.extend(search_text_chunks)`
   - Komunikat sukcesu: `st.success("Search results added to your knowledge base!")`

4. **Obsługa błędów**:
   - Pełna obsługa wyjątków z komunikatami dla użytkownika: `st.error(f"Error adding search results to knowledge base: {str(e)}")`
   - Diagnostyczny traceback: `print(f"Error details: {traceback.format_exc()}")`

### main

```python
def main()
```

Główna funkcja aplikacji definiująca interfejs użytkownika i inicjalizująca stan:

1. **Konfiguracja podstawowa**:
   - Ładowanie zmiennych środowiskowych: `load_dotenv()`
   - Ustawienia strony Streamlit: `st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")`

2. **Inicjalizacja zmiennych stanu sesji**:
   - Obsługa każdej zmiennej z wartością domyślną: 
     - `if "conversation" not in st.session_state: st.session_state.conversation = None`
     - Inicjalizacja wielu zmiennych konfiguracyjnych z wartościami domyślnymi

3. **Struktura interfejsu**:
   - Tytuł aplikacji: `st.title("Chat with multiple PDFs 📚")`
   - Wyświetlanie historii czatu: `for message in st.session_state.messages:`
   - Pole wejściowe czatu: `user_question = st.chat_input("Ask a question about your documents:")`
   - Obsługa wprowadzonego pytania
   - Panel boczny z zakładkami: `tab1, tab2, tab3, tab4 = st.tabs([...])`

4. **Zakładka Documents**:
   - Uploader plików: `pdf_docs = st.file_uploader(..., type=["pdf", "txt", "docx", "csv", "json"])`
   - Przycisk "Process" z logiką przetwarzania dokumentów
   - Przycisk do czyszczenia wgranych plików

5. **Zakładka Web Search**:
   - Opcje konfiguracyjne wyszukiwania w sieci
   - Zaawansowane opcje w rozwijanym panelu
   - Filtry domen (włączanie/wyłączanie)
   - Pole wyszukiwania i wyświetlanie wyników

6. **Zakładka Retrieval Settings**:
   - Przełącznik wyszukiwania hybrydowego
   - Suwak liczby dokumentów (3-50)
   - Suwak wagi semantycznej (0-1)
   - Przełącznik ponownego rankowania BART
   - Przycisk "Apply Retrieval Settings"

7. **Zakładka Advanced Settings**:
   - Opcje wydajności (tryb wydajności, rozmiar batcha)
   - Opcje wyświetlania dokumentów (podsumowanie)
   - Przyciski zarządzania (czyszczenie historii, wyników)
   - Tryb deweloperski z danymi diagnostycznymi

## Hybrydowy system wyszukiwania - szczegóły implementacji

Hybrydowy system wyszukiwania w aplikacji jest zaawansowanym rozwiązaniem łączącym kilka technik:

1. **Komponenty wyszukiwania**:
   - **FAISS** (wyszukiwanie semantyczne) - znajduje dokumenty na podstawie podobieństwa wektorowego
   - **BM25** (wyszukiwanie na podstawie słów kluczowych) - używa statystycznego algorytmu ważenia terminów
   - **BART** (kontekstowe ponowne rankowanie) - używa modelu języka do oceny dopasowania dokumentu do zapytania

2. **Proces wyszukiwania w HybridRetriever**:
   - **Pierwszy etap**: Ensemble retrievera łączy wyniki FAISS i BM25 z wagami określonymi przez parametr alfa
   - **Drugi etap** (opcjonalny): Ponowne rankowanie przez BART, gdzie:
     - Pobierane są większe zestawy dokumentów
     - Wykorzystywane jest API Hugging Face do generowania podsumowań
     - Obliczane są podobieństwa Jaccard między zapytaniem a podsumowaniem
     - Dokumenty są sortowane według końcowego wyniku

3. **Obsługa wielojęzyczności**:
   - Funkcja `_get_keyword_similarity_score` wykrywa zapytania zawierające znaki niełacińskie
   - Dla tekstów wielojęzycznych używane są trigramy znaków zamiast tokenizacji słów
   - Specjalne obliczanie nakładania się trigramów dla podobieństwa w językach niełacińskich

4. **Konfigurowalność systemu**:
   - Regulowana waga między wyszukiwaniem semantycznym a słowami kluczowymi (0-1)
   - Możliwość włączenia/wyłączenia ponownego rankowania BART
   - Dostosowywalna liczba dokumentów do pobrania (k)

## Przepływ danych - dogłębna analiza

1. **Przetwarzanie dokumentów**:
   - **Wejście**: Pliki przesłane przez użytkownika (PDF, TXT, DOCX, CSV, JSON)
   - **Ekstrakcja tekstu**: Funkcja `get_pdf_text` konwertuje pliki na tekst
   - **Chunking**: Funkcja `get_text_chunks` dzieli tekst na fragmenty (~800-1000 znaków)
   - **Wektoryzacja**: Model E5 generuje embeddingi dla każdego fragmentu
   - **Indeksowanie**: Fragmenty są dodawane do FAISS i BM25 (jeśli używane)
   - **Wyjście**: Zbudowany vectorstore i retriever

2. **Proces odpowiadania na pytania**:
   - **Wejście**: Pytanie użytkownika
   - **Opcjonalne wyszukiwanie w sieci**: API Tavily pobiera kontekst z internetu
   - **Wyszukiwanie dokumentów**: Retriever znajduje odpowiednie fragmenty z bazy wiedzy
   - **Generowanie odpowiedzi**: Model LLM łączy pytanie i znalezione dokumenty
   - **Podsumowanie dokumentów**: Funkcja `generate_source_summary` tworzy podsumowanie
   - **Wyjście**: Odpowiedź LLM i wyświetlone dokumenty źródłowe

3. **Przepływ danych w wyszukiwaniu hybrydowym**:
   - **Wejście**: Pytanie użytkownika
   - **Równoległe wyszukiwanie**: FAISS (semantyczne) i BM25 (słowa kluczowe)
   - **Łączenie wyników**: Wyniki są łączone z wagami określonymi przez parametr alfa
   - **Ponowne rankowanie**: Model BART ocenia dopasowanie dokumentów do zapytania
   - **Wyjście**: Posortowana lista dokumentów

4. **Przepływ danych dla wyszukiwania w sieci**:
   - **Wejście**: Pytanie użytkownika
   - **Zapytanie API**: Wysłanie zapytania do Tavily API
   - **Przetwarzanie wyników**: Ekstrakcja odpowiedzi i źródeł URL
   - **Chunking i indeksowanie**: Dodanie wyników do istniejącej bazy wiedzy
   - **Wyjście**: Rozszerzony kontekst dla LLM

## Szczegóły interfejsu użytkownika

Interfejs użytkownika aplikacji jest zbudowany w Streamlit i składa się z kilku głównych komponentów:

1. **Główny obszar czatu**:
   - Historia konwersacji wyświetlana z odpowiednimi ikonami dla użytkownika i asystenta
   - Pole wejściowe do wprowadzania pytań
   - Wyświetlanie odpowiedzi w czasie rzeczywistym

2. **Panel boczny z zakładkami**:
   - **Documents**:
     - Uploader plików z obsługą wielu formatów
     - Przyciski "Process" i "Clear Uploaded Files"
     - Informacje o liczbie przetworzonych dokumentów

   - **Web Search**:
     - Przełącznik włączania/wyłączania wyszukiwania w sieci
     - Wybór głębokości wyszukiwania (basic/advanced)
     - Sekcja zaawansowanych opcji:
       - Suwak liczby wyników (1-30)
       - Opcje zawartości (AI Answer, Images)
       - Zakres czasowy (day/week/month)
       - Filtry domen (include/exclude)

   - **Retrieval Settings**:
     - Przełącznik wyszukiwania hybrydowego
     - Suwak liczby dokumentów (3-50)
     - Suwak wagi semantycznej (0-1)
     - Przełącznik ponownego rankowania BART
     - Przycisk "Apply Retrieval Settings"

   - **Advanced Settings**:
     - Opcje wydajności (tryb wydajności, rozmiar batcha)
     - Opcje wyświetlania dokumentów (podsumowanie)
     - Przyciski zarządzania (czyszczenie historii, wyników)
     - Tryb deweloperski z danymi diagnostycznymi

3. **Ekspandery dla wyników**:
   - **Retrieved Documents Summary**: 
     - Rozwijany panel z punktami podsumowania
     - Dynamicznie generowane na podstawie znalezionych dokumentów
   
   - **Document Sources**:
     - Lista dokumentów źródłowych z formatowaniem HTML
     - Opcja "View full content" dla długich dokumentów
     - Metadane dokumentów (źródło/URL)

   - **Web Search Results**:
     - Podsumowanie wyszukiwania z AI
     - Lista źródeł internetowych z odnośnikami

## Obsługa wielojęzyczności

Aplikacja zawiera szereg mechanizmów wspierających wielojęzyczność:

1. **Model embedingowy**:
   - Wykorzystanie modelu `multilingual-e5-large-instruct` obsługującego 100+ języków
   - Instrukcje w formacie: `"Instruct: Represent this document for retrieval:\nQuery: {text}"`
   - Reprezentacje wektorowe działają międzyjęzykowo

2. **Detekcja języka**:
   - Heurystyki oparte na występowaniu znaków niełacińskich
   - Próg 5% znaków spoza ASCII do identyfikacji tekstów niełacińskich

3. **Dostosowane przetwarzanie**:
   - Różne metody tokenizacji zależnie od wykrytego języka
   - Trigramy znaków dla języków niełacińskich
   - Standardowa tokenizacja słów dla języków łacińskich

4. **Podobieństwo tekstów**:
   - Funkcja `_get_keyword_similarity_score` z logiką dla różnych języków
   - Obliczanie nakładania się n-gramów dla języków niełacińskich
   - Tradycyjne podobieństwo słów dla języków łacińskich

5. **Obsługa wielojęzycznych dokumentów**:
   - Prawidłowe kodowanie i dekodowanie UTF-8
   - Escapowanie HTML dla zachowania znaków specjalnych
   - Formatowanie CSS z myślą o wielojęzycznych treściach

## Modyfikacja i rozszerzanie aplikacji

Aplikacja została zaprojektowana modułowo, co ułatwia jej rozszerzanie:

1. **Dodawanie nowych formatów dokumentów**:
   - W funkcji `get_pdf_text` dodać obsługę nowego formatu
   - Zaimplementować logikę ekstrakcji tekstu
   - Zaktualizować listę obsługiwanych typów w `st.file_uploader`

2. **Zmiana modelu LLM**:
   - W funkcji `get_conversation_chain` podmienić inicjalizację modelu
   - Dostosować parametry i prompt template

3. **Dodanie nowych metod wyszukiwania**:
   - Rozszerzyć `HybridRetriever` lub stworzyć nowy typ retrievera
   - Zaimplementować logikę wyszukiwania w `_get_relevant_documents`
   - Zaktualizować UI o nowe opcje konfiguracji

4. **Rozbudowa interfejsu**:
   - Dodać nowe zakładki w funkcji `main`
   - Zaimplementować nowe sekcje UI
   - Dodać zmienne stanu sesji dla nowych funkcji

5. **Integracja z innymi API**:
   - Wzorować się na implementacji `perform_tavily_search`
   - Dodać obsługę kluczy API i konfiguracji
   - Zaktualizować UI o nowe opcje

Dzięki modułowej strukturze i czystemu kodowi, aplikacja może być łatwo rozszerzana o nowe funkcje i możliwości. 