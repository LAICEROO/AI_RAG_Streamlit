# Dokumentacja szczegółowa pliku app.py

## Spis treści
1. [Wprowadzenie](#wprowadzenie)
2. [Importy i zależności](#importy-i-zależności)
3. [Funkcje główne](#funkcje-główne)
   - [get_pdf_text](#get_pdf_text)
   - [get_conversation_chain](#get_conversation_chain)
   - [generate_source_summary](#generate_source_summary)
   - [handle_userinput](#handle_userinput)
   - [perform_tavily_search](#perform_tavily_search)
   - [add_search_results_to_vectorstore](#add_search_results_to_vectorstore)
   - [main](#main)
4. [Przepływ danych](#przepływ-danych)
5. [Interfejs użytkownika](#interfejs-użytkownika)
6. [Hybrydowy system wyszukiwania](#hybrydowy-system-wyszukiwania)
7. [Podsumowanie](#podsumowanie)

## Wprowadzenie

Plik `app.py` to główny skrypt aplikacji opartej na Streamlit, która umożliwia konwersację z dokumentami w różnych formatach (PDF, TXT, DOCX, CSV, JSON) przy użyciu modeli językowych i technik Retrieval-Augmented Generation (RAG). Aplikacja wykorzystuje zaawansowane metody wyszukiwania, w tym wyszukiwanie semantyczne (wykorzystujące FAISS), wyszukiwanie oparte na słowach kluczowych (BM25) oraz kontekstowe ponowne rankowanie dokumentów (BART). Dodatkowo, aplikacja oferuje możliwość wyszukiwania w internecie za pomocą API Tavily, co wzbogaca bazę wiedzy o aktualne informacje.

## Importy i zależności

Aplikacja wykorzystuje wiele zewnętrznych bibliotek i modułów:

- **Streamlit**: do budowy interfejsu użytkownika (`st`)
- **dotenv**: do ładowania zmiennych środowiskowych
- **PyPDF2**: do odczytu plików PDF
- **langchain**: do budowania łańcuchów konwersacyjnych i przechowywania wektorów
- **torch** i **transformers**: do obsługi modeli embedingowych 
- **requests** i **json**: do komunikacji z API zewnętrznymi
- **numpy**: do operacji na macierzach
- **Moduły własne**: `utils.embedding_utils` i `utils.hybrid_search` dla specjalistycznych funkcji

Najważniejsze importowane komponenty z własnych modułów to:
- `get_text_chunks`: funkcja dzieląca tekst na mniejsze fragmenty
- `MultilangE5Embeddings`: klasa tworząca reprezentacje wektorowe tekstu
- `get_vectorstore`: funkcja tworząca magazyn wektorów FAISS
- `get_hybrid_retriever`: funkcja tworząca hybrydowy system wyszukiwania

## Funkcje główne

### get_pdf_text

```python
def get_pdf_text(uploaded_files):
```

Ta funkcja przetwarza przesłane pliki i wyodrębnia z nich tekst. Obsługuje różne formaty:

1. **PDF**: używa PyPDF2 do wyodrębnienia tekstu z każdej strony
2. **TXT**: dekoduje zawartość jako UTF-8
3. **DOCX**: używa biblioteki `docx` do odczytu paragrafów
4. **CSV**: używa pandas do konwersji na tekst
5. **JSON**: formatuje zawartość JSON z wcięciami

Funkcja zawiera obsługę błędów dla każdego typu pliku i zwraca połączony tekst ze wszystkich dokumentów. Nieobsługiwane typy plików są pomijane z odpowiednim ostrzeżeniem.

### get_conversation_chain

```python
def get_conversation_chain(vectorstore, text_chunks=None):
```

Ta funkcja tworzy łańcuch konwersacyjny, który jest sercem funkcjonalności RAG (Retrieval-Augmented Generation):

1. Inicjalizuje model językowy Mistral AI z odpowiednimi parametrami (temperatura, max_tokens)
2. Tworzy system pamięci konwersacyjnej
3. Tworzy odpowiedni retriever (system wyszukiwania):
   - Jeśli `text_chunks` istnieje i włączone jest wyszukiwanie hybrydowe, tworzy hybrydowy retriever
   - W przeciwnym razie używa standardowego retrievera z vectorstore
4. Tworzy łańcuch konwersacyjny łączący model LLM, retriever i pamięć

Funkcja zawiera również obsługę błędów z mechanizmem fallback do standardowego wyszukiwania, gdy hybrydowe zawiedzie.

### generate_source_summary

```python
def generate_source_summary(source_documents):
```

Ta funkcja generuje podsumowanie znalezionych dokumentów źródłowych:

1. Wyodrębnia tekst z pierwszych 5 dokumentów (ograniczenie dla tokenu)
2. Sprawdza czy tekst zawiera znaczącą ilość znaków spoza łaciny (>5%)
3. Stosuje różne podejścia do ekstrakcji terminów zależnie od języka:
   - Dla tekstu zawierającego znaki niełacińskie: ekstrakcja sekwencji znaków za pomocą regex
   - Dla tekstu łacińskiego: standardowa ekstrakcja słów z filtrowaniem słów stop
4. Zlicza częstotliwości terminów i wybiera najczęstsze
5. Generuje punktory podsumowania na podstawie znalezionych terminów

Jeśli nie uda się znaleźć znaczących terminów, zwraca generyczne podsumowanie.

### handle_userinput

```python
def handle_userinput(user_question):
```

Ta funkcja obsługuje zapytanie użytkownika i jest kluczowym elementem interakcji:

1. Sprawdza czy włączone jest wyszukiwanie w sieci i w razie potrzeby wykonuje je
2. Formatuje informacje z wyszukiwania internetowego jako kontekst dla LLM
3. Wywołuje łańcuch konwersacyjny z pytaniem (zwykłym lub rozszerzonym o dane z sieci)
4. Aktualizuje historię konwersacji
5. Wyświetla dokumenty źródłowe z odpowiednim formatowaniem:
   - Jeśli włączone jest podsumowanie dokumentów, generuje i wyświetla je
   - Wyświetla poszczególne dokumenty z formatowaniem HTML
   - Obsługuje długie dokumenty, skracając je i dodając możliwość rozwinięcia
   - Wyświetla metadane dokumentów (źródło, URL)

Funkcja zawiera kompleksową obsługę błędów i formatowanie dla poprawnego wyświetlania.

### perform_tavily_search

```python
def perform_tavily_search(query, search_depth="basic", max_results=10, include_answer=True, include_images=False, time_range=None):
```

Ta funkcja wykonuje wyszukiwanie w sieci za pomocą API Tavily:

1. Przygotowuje URL i payload z parametrami wyszukiwania
2. Dodaje opcjonalne parametry (zakres czasowy, filtry domen)
3. Wykonuje zapytanie POST do API Tavily
4. Przetwarza wyniki i obsługuje błędy
5. Zwraca wyniki wyszukiwania

Obsługuje zaawansowane opcje takie jak filtrowanie po domenach, maksymalna liczba wyników i zakres czasowy.

### add_search_results_to_vectorstore

```python
def add_search_results_to_vectorstore(content_texts, source_urls):
```

Ta funkcja dodaje wyniki wyszukiwania internetowego do istniejącego magazynu wektorów:

1. Dzieli teksty wyników na mniejsze fragmenty
2. Tworzy metadane dla każdego fragmentu (URL, typ)
3. Dodaje fragmenty do retrievera:
   - Dla hybrydowego retrievera używa metody add_texts
   - Dla standardowego retrievera pobiera vectorstore i dodaje do niego
4. Odtwarza łańcuch konwersacyjny z zaktualizowanym magazynem wektorów

Funkcja obsługuje zarówno hybrydowe jak i standardowe retrievery, z pełną obsługą błędów.

### main

```python
def main():
```

Główna funkcja aplikacji:

1. Ładuje zmienne środowiskowe
2. Inicjalizuje konfigurację strony Streamlit
3. Inicjalizuje zmienne stanu sesji (conversation, chat_history, itp.)
4. Tworzy interfejs użytkownika z tytułem i obszarem czatu
5. Tworzy panel boczny z zakładkami:
   - Documents: zarządzanie dokumentami
   - Web Search: opcje wyszukiwania w sieci
   - Retrieval Settings: konfiguracja wyszukiwania
   - Advanced Settings: zaawansowane opcje

Każda zakładka zawiera odpowiednie elementy UI, takie jak pola, przyciski i suwaki.

## Przepływ danych

1. **Wczytywanie dokumentów**:
   - Użytkownik wgrywa dokumenty przez UI
   - Dokumenty są przetwarzane na tekst przez `get_pdf_text`
   - Tekst jest dzielony na fragmenty przez `get_text_chunks`
   - Fragmenty są wektoryzowane i zapisywane w FAISS przez `get_vectorstore`

2. **Zadawanie pytań**:
   - Użytkownik wpisuje pytanie
   - Jeśli włączone, wykonywane jest wyszukiwanie w sieci przez `perform_tavily_search`
   - Pytanie jest przetwarzane przez `handle_userinput`
   - Retriever znajduje odpowiednie fragmenty dokumentów
   - Model LLM generuje odpowiedź na podstawie fragmentów i pytania
   - Wyświetlane są odpowiedź i dokumenty źródłowe

3. **Dodawanie wyników wyszukiwania**:
   - Wyniki wyszukiwania są przetwarzane na fragmenty
   - Fragmenty są dodawane do magazynu wektorów przez `add_search_results_to_vectorstore`
   - Łańcuch konwersacyjny jest aktualizowany

## Interfejs użytkownika

Interfejs zawiera kilka głównych sekcji:

1. **Główny obszar czatu**: wyświetla historię konwersacji
2. **Panel boczny z zakładkami**:
   - **Documents**: wgrywanie i przetwarzanie dokumentów
   - **Web Search**: konfiguracja wyszukiwania w sieci
   - **Retrieval Settings**: konfiguracja systemu wyszukiwania (liczba dokumentów, balans semantyczny/słowa kluczowe)
   - **Advanced Settings**: ustawienia wydajności, opcje wyświetlania, tryb deweloperski

3. **Ekspandery dla wyników**:
   - Podsumowanie dokumentów
   - Dokumenty źródłowe
   - Wyniki wyszukiwania w sieci

## Hybrydowy system wyszukiwania

Aplikacja wykorzystuje zaawansowany hybrydowy system wyszukiwania, zdefiniowany w module `utils.hybrid_search`:

1. **Wyszukiwanie semantyczne (FAISS)**: znajduje dokumenty semantycznie podobne do zapytania
2. **Wyszukiwanie słów kluczowych (BM25)**: znajduje dokumenty zawierające słowa kluczowe z zapytania
3. **Ponowne rankowanie kontekstowe (BART)**: używa modelu BART do oceny, które dokumenty są najbardziej odpowiednie

System można konfigurować przez UI, ustawiając:
- Liczbę dokumentów do pobrania (3-50)
- Wagę wyszukiwania semantycznego (0-1)
- Włączenie/wyłączenie ponownego rankowania BART

## Podsumowanie

Plik `app.py` tworzy zaawansowaną aplikację do konwersacji z dokumentami, wykorzystującą najnowsze techniki RAG (Retrieval-Augmented Generation), hybrydowe wyszukiwanie i integrację z wyszukiwaniem internetowym. Aplikacja obsługuje różne formaty dokumentów, wielojęzyczne zapytania i oferuje liczne opcje konfiguracji.

Główne zalety aplikacji:
- Obsługa wielu formatów dokumentów
- Hybrydowy system wyszukiwania
- Integracja z wyszukiwaniem internetowym
- Zaawansowane formatowanie wyników
- Generowanie podsumowań dokumentów
- Opcje konfiguracji wydajności i wyszukiwania 