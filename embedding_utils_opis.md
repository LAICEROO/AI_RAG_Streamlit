# Opis modułu embedding_utils.py

## Wprowadzenie

Moduł `embedding_utils.py` jest kluczowym komponentem w systemie przetwarzania tekstu i wyszukiwania semantycznego w aplikacji RAG (Retrieval Augmented Generation). Zawiera on narzędzia do podziału tekstu na fragmenty, generowania embedingów (wektorowych reprezentacji tekstu) oraz tworzenia indeksu wektorowego dla efektywnego wyszukiwania podobnych dokumentów.

Embedingi są wielowymiarowymi reprezentacjami tekstu w przestrzeni wektorowej, gdzie semantycznie podobne teksty znajdują się blisko siebie. Ta właściwość pozwala na wyszukiwanie dokumentów nie tylko na podstawie dopasowania słów kluczowych, ale na podstawie ich znaczenia, co znacząco zwiększa efektywność wyszukiwania i jakość generowanych odpowiedzi.

## Szczegółowa analiza każdej funkcji i komponentu

### Funkcja `get_text_chunks(text)`

```python
def get_text_chunks(text):
    """
    Split text into chunks of appropriate size for embedding.
    
    Args:
        text (str): The text to split into chunks.
        
    Returns:
        list: A list of text chunks.
    """
```

Funkcja ta jest odpowiedzialna za inteligentny podział tekstu na mniejsze, semantycznie spójne fragmenty, które mogą być efektywnie przetwarzane przez modele embedingowe.

#### Działanie szczegółowe krok po kroku:

1. **Podział wstępny na akapity**:
   ```python
   paragraphs = text.split("\n\n")
   ```
   Ten krok dzieli tekst na naturalne jednostki semantyczne (akapity) używając podwójnego znaku nowej linii jako separatora.

2. **Inicjalizacja zmiennych do śledzenia fragmentów**:
   ```python
   chunks = []
   current_chunk = ""
   current_size = 0
   ```
   - `chunks` - lista przechowująca wszystkie wygenerowane fragmenty
   - `current_chunk` - aktualnie budowany fragment
   - `current_size` - bieżąca długość fragmentu w znakach

3. **Przetwarzanie każdego akapitu**:
   ```python
   for paragraph in paragraphs:
       # Skip empty paragraphs
       if not paragraph.strip():
           continue
   ```
   Puste akapity są pomijane, co pozwala na oczyszczenie tekstu z niepotrzebnych przerw.

4. **Sprawdzenie rozmiaru akapitu i ewentualny dalszy podział**:
   ```python
   # If paragraph is already too large, split it into sentences
   if len(paragraph) > 800:  # Using 800 to leave room for overlap
       sentences = paragraph.replace("\n", " ").split(". ")
       sentences = [s + "." if not s.endswith(".") else s for s in sentences if s.strip()]
   ```
   Jeśli akapit przekracza 800 znaków (ustalony próg), jest dzielony na zdania poprzez:
   - Zastąpienie pojedynczych znaków nowej linii spacjami
   - Podział tekstu na zdania przy użyciu kropki i spacji jako separatora
   - Dodanie kropki na końcu zdań, jeśli jej brakuje
   - Usunięcie pustych zdań

5. **Przetwarzanie zdań w dużych akapitach**:
   ```python
   for sentence in sentences:
       # If adding sentence would exceed chunk size, start a new chunk
       if current_size + len(sentence) > 800:
           if current_chunk:  # Only add if we have content
               chunks.append(current_chunk.strip())
           current_chunk = sentence
           current_size = len(sentence)
       else:
           current_chunk += " " + sentence if current_chunk else sentence
           current_size += len(sentence)
   ```
   Dla każdego zdania w dużym akapicie:
   - Sprawdza, czy dodanie zdania przekroczyłoby limit 800 znaków
   - Jeśli tak, zamyka bieżący fragment i dodaje go do listy fragmentów
   - Rozpoczyna nowy fragment od bieżącego zdania
   - Jeśli nie, dodaje zdanie do bieżącego fragmentu z odpowiednim separatorem

6. **Przetwarzanie mniejszych akapitów jako całości**:
   ```python
   else:
       # If adding paragraph would exceed chunk size, start a new chunk
       if current_size + len(paragraph) > 800:
           if current_chunk:  # Only add if we have content
               chunks.append(current_chunk.strip())
           current_chunk = paragraph
           current_size = len(paragraph)
       else:
           # Add paragraph to current chunk
           current_chunk += "\n\n" + paragraph if current_chunk else paragraph
           current_size += len(paragraph)
   ```
   Dla akapitów mniejszych niż 800 znaków:
   - Sprawdza, czy dodanie całego akapitu przekroczyłoby limit
   - Jeśli tak, zamyka bieżący fragment i dodaje go do listy fragmentów
   - Rozpoczyna nowy fragment od bieżącego akapitu
   - Jeśli nie, dodaje akapit do bieżącego fragmentu z zachowaniem formatowania (podwójny znak nowej linii)

7. **Dodanie ostatniego fragmentu**:
   ```python
   # Add the last chunk if it's not empty
   if current_chunk:
       chunks.append(current_chunk.strip())
   ```
   Po przetworzeniu wszystkich akapitów, ostatni fragment (który mógł nie zostać dodany w pętli) jest dodawany do listy, o ile zawiera treść.

8. **Sprawdzenie rozmiaru każdego fragmentu i dodatkowy podział w razie potrzeby**:
   ```python
   # Check if any chunks still exceed our limit
   for i, chunk in enumerate(chunks):
       if len(chunk) > 1000:
           st.warning(f"Chunk {i} has size {len(chunk)}, which exceeds the limit. Will be split further.")
           # Further split into smaller chunks
           text_splitter = CharacterTextSplitter(
               separator=" ",
               chunk_size=900,  # Lower than 1000 to ensure we don't exceed limit
               chunk_overlap=100,
               length_function=len
           )
           replacement_chunks = text_splitter.split_text(chunk)
           # Replace the oversized chunk with the smaller chunks
           chunks.pop(i)
           for j, replacement in enumerate(replacement_chunks):
               chunks.insert(i+j, replacement)
   ```
   Ten krok wprowadza dodatkowe zabezpieczenie:
   - Sprawdza, czy jakikolwiek fragment przekracza ostateczny limit 1000 znaków
   - Jeśli tak, wyświetla ostrzeżenie w interfejsie Streamlit
   - Tworzy obiekt `CharacterTextSplitter` z biblioteki LangChain
   - Konfiguruje splitter z odpowiednimi parametrami:
     - `separator=" "` - używa spacji jako separatora
     - `chunk_size=900` - ustawia maksymalny rozmiar fragmentu na 900 znaków
     - `chunk_overlap=100` - zapewnia 100 znaków nakładania się między fragmentami
     - `length_function=len` - używa standardowej funkcji długości
   - Dzieli zbyt duży fragment na mniejsze części
   - Zastępuje oryginalny fragment nowymi, mniejszymi fragmentami

9. **Zwrócenie listy fragmentów**:
   ```python
   return chunks
   ```
   Funkcja zwraca listę wszystkich utworzonych fragmentów tekstu.

Ta staranna metoda podziału tekstu zapewnia, że fragmenty mają odpowiedni rozmiar dla modeli embedingowych, zachowując jednocześnie semantyczną spójność tekstu. Nakładanie się fragmentów zapewnia, że kontekst nie jest tracony na granicach fragmentów.

### Funkcja `average_pool(last_hidden_states, attention_mask)`

```python
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Perform average pooling on the last hidden states from the model.
    
    Args:
        last_hidden_states (Tensor): The last hidden states from the model.
        attention_mask (Tensor): The attention mask for the input.
        
    Returns:
        Tensor: The pooled representation.
    """
```

Ta funkcja implementuje mechanizm uśredniania (average pooling) dla ukrytych stanów modelu transformerowego, co jest kluczowym krokiem w procesie generowania embedingów.

#### Działanie szczegółowe krok po kroku:

1. **Maskowanie nieistotnych tokenów**:
   ```python
   last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
   ```
   - `attention_mask` określa, które tokeny są rzeczywistymi tokenami wejściowymi (1), a które są tokenami wypełniającymi (0)
   - `~attention_mask[..., None].bool()` tworzy maskę boolowską, gdzie `True` oznacza tokeny wypełniające
   - `masked_fill` zastępuje wszystkie wartości odpowiadające tokenom wypełniającym (gdzie maska jest `True`) wartością 0.0
   - Zapobiega to uwzględnianiu tokenów wypełniających w obliczeniach

2. **Obliczanie średniej reprezentacji**:
   ```python
   return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
   ```
   - `last_hidden.sum(dim=1)` sumuje wartości ukrytych stanów wzdłuż wymiaru sekwencji (po tokenach)
   - `attention_mask.sum(dim=1)` oblicza liczbę rzeczywistych tokenów (sumę maski uwagi) dla każdego przykładu w partii
   - `[..., None]` rozszerza wymiar, aby umożliwić dzielenie wektorów
   - Dzielenie sumy stanów ukrytych przez liczbę rzeczywistych tokenów daje średnią reprezentację

Ta technika jest kluczowa dla generowania wysokiej jakości embedingów, ponieważ:
- Ignoruje tokeny wypełniające, które nie wnoszą żadnych informacji
- Zapewnia, że reprezentacja wektorowa ma spójną skalę, niezależnie od długości tekstu
- Uwzględnia wszystkie tokeny wejściowe z równą wagą, co pozwala na uchwycenie ogólnego znaczenia tekstu

### Klasa `MultilangE5Embeddings`

```python
class MultilangE5Embeddings(Embeddings):
    """
    A class for generating embeddings using the multilingual E5 model.
    """
```

Ta klasa implementuje interfejs `Embeddings` z biblioteki LangChain, zapewniając dostęp do zaawansowanego wielojęzycznego modelu E5. Model E5 (Embeddings from bidirectional Encoder representations) jest specjalnie zaprojektowany do generowania wysokiej jakości reprezentacji wektorowych tekstu.

#### Metoda `__init__`

```python
def __init__(self, model_name="intfloat/multilingual-e5-large-instruct", batch_size=8):
    super().__init__()
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModel.from_pretrained(model_name)
    self.batch_size = batch_size
    # Move model to GPU if available
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model = self.model.to(self.device)
```

Inicjalizacja obejmuje:
1. **Wczytanie tokenizera** - konwertuje tekst na tokeny, które model może przetwarzać:
   ```python
   self.tokenizer = AutoTokenizer.from_pretrained(model_name)
   ```
   - Korzysta z `AutoTokenizer` z biblioteki `transformers`
   - Ładuje predefiniowany tokenizer odpowiadający wybranemu modelowi

2. **Wczytanie modelu** - rzeczywisty model transformerowy do generowania embedingów:
   ```python
   self.model = AutoModel.from_pretrained(model_name)
   ```
   - Korzysta z `AutoModel` z biblioteki `transformers`
   - Domyślnie używa modelu `multilingual-e5-large-instruct`

3. **Konfiguracja rozmiaru partii** - określa, ile dokumentów będzie przetwarzanych jednocześnie:
   ```python
   self.batch_size = batch_size
   ```
   - Domyślna wartość to 8, co jest dobrym kompromisem między wydajnością a zużyciem pamięci

4. **Wykrywanie i konfiguracja urządzenia obliczeniowego** - przenosi model na GPU, jeśli jest dostępny:
   ```python
   self.device = "cuda" if torch.cuda.is_available() else "cpu"
   self.model = self.model.to(self.device)
   ```
   - Sprawdza dostępność CUDA (GPU NVIDIA)
   - Przenosi model na GPU, jeśli jest dostępny, co znacząco przyspiesza obliczenia
   - W przeciwnym razie używa CPU

#### Metoda `embed_documents`

```python
def embed_documents(self, texts):
    """
    Generate embeddings for a list of documents.
    
    Args:
        texts (list): A list of document texts.
        
    Returns:
        list: A list of embeddings.
    """
```

Ta metoda generuje embedingi dla listy dokumentów, obsługując wydajne przetwarzanie wsadowe:

1. **Przetwarzanie wstępne tekstów** - dodaje instrukcję do każdego dokumentu:
   ```python
   task = "Represent this document for retrieval:"
   processed_texts = [f"Instruct: {task}\nQuery: {text}" for text in texts]
   ```
   - Dodaje instrukcję "Represent this document for retrieval:" do każdego tekstu
   - Format "Instruct: [task]\nQuery: [text]" jest specyficzny dla modeli instrukcyjnych (instruct)
   - Pomaga modelowi zrozumieć, że tekst ma być reprezentowany jako dokument do wyszukiwania

2. **Przetwarzanie w partiach** - dzieli dokumenty na mniejsze partie:
   ```python
   embeddings_list = []
   for i in range(0, len(processed_texts), self.batch_size):
       batch_texts = processed_texts[i:i+self.batch_size]
   ```
   - Iteruje przez teksty w grupach o rozmiarze `self.batch_size`
   - Zapobiega przeciążeniu pamięci przy dużej liczbie dokumentów

3. **Tokenizacja** - konwersja tekstu na tokeny zrozumiałe dla modelu:
   ```python
   batch_dict = self.tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
   ```
   - `max_length=512` - ogranicza długość do 512 tokenów (standard dla wielu modeli transformerowych)
   - `padding=True` - wyrównuje długość wszystkich sekwencji w partii
   - `truncation=True` - przycina teksty przekraczające maksymalną długość
   - `return_tensors='pt'` - zwraca tensory PyTorch

4. **Przeniesienie tensorów na odpowiednie urządzenie**:
   ```python
   batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
   ```
   - Przenosi wszystkie tensory (dane wejściowe, maski) na GPU/CPU

5. **Generowanie embedingów**:
   ```python
   with torch.no_grad():
       outputs = self.model(**batch_dict)
   ```
   - `torch.no_grad()` - wyłącza śledzenie gradientu, co oszczędza pamięć (nie jest potrzebne przy inferencji)
   - Przekazuje słownik tokenów jako argumenty do modelu
   - Pobiera wyjście modelu, które zawiera ukryte stany

6. **Uśrednianie reprezentacji** - stosuje funkcję `average_pool`:
   ```python
   embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
   ```
   - Wykorzystuje wcześniej zdefiniowaną funkcję `average_pool`
   - Przekazuje ostatnie ukryte stany i maskę uwagi
   - Otrzymuje uśrednioną reprezentację dla każdego dokumentu

7. **Normalizacja** - standaryzuje długość wektorów:
   ```python
   embeddings = F.normalize(embeddings, p=2, dim=1)
   ```
   - Stosuje normalizację L2 (euklidesową)
   - `p=2` oznacza normę L2
   - `dim=1` oznacza normalizację wzdłuż wymiaru cech
   - Normalizacja zapewnia, że wszystkie wektory mają długość jednostkową
   - Ułatwia porównywanie podobieństwa kosinusowego

8. **Przeniesienie na CPU i konwersja do listy**:
   ```python
   embeddings_list.append(embeddings.cpu().numpy())
   ```
   - Przenosi tensor embedingów z powrotem na CPU
   - Konwertuje tensor PyTorch na tablicę NumPy
   - Dodaje embedingi do listy

9. **Łączenie i zwracanie wyników**:
   ```python
   if len(embeddings_list) > 1:
       return np.vstack(embeddings_list).tolist()
   return embeddings_list[0].tolist()
   ```
   - Jeśli było wiele partii, łączy wszystkie embedingi w jedną tablicę
   - Konwertuje tablicę NumPy na listę Python
   - Zwraca listę embedingów dla wszystkich dokumentów

#### Metoda `embed_query`

```python
def embed_query(self, text):
    """
    Generate an embedding for a query.
    
    Args:
        text (str): The query text.
        
    Returns:
        list: The query embedding.
    """
```

Ta metoda jest zoptymalizowana do generowania embedingu dla pojedynczego zapytania:

1. **Przetwarzanie wstępne zapytania**:
   ```python
   task = "Represent this query for retrieval:"
   processed_text = f"Instruct: {task}\nQuery: {text}"
   ```
   - Dodaje instrukcję "Represent this query for retrieval:"
   - Jest to inna instrukcja niż dla dokumentów, co pomaga modelowi rozróżnić zapytania od dokumentów

2. **Tokenizacja**:
   ```python
   batch_dict = self.tokenizer([processed_text], max_length=512, padding=True, truncation=True, return_tensors='pt')
   ```
   - Podobnie jak w `embed_documents`, ale dla pojedynczego tekstu w liście

3. **Przeniesienie na urządzenie, generowanie embedingu i normalizacja**:
   ```python
   batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
   
   with torch.no_grad():
       outputs = self.model(**batch_dict)
   
   embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
   embeddings = F.normalize(embeddings, p=2, dim=1)
   ```
   - Proces identyczny jak w metodzie `embed_documents`

4. **Zwrócenie pojedynczego embeddingu**:
   ```python
   return embeddings.cpu().numpy().tolist()[0]
   ```
   - Konwertuje tensor na CPU, następnie na tablicę NumPy i na listę
   - Zwraca pierwszy (i jedyny) element jako listę

#### Metody pomocnicze

```python
def embed_text(self, text):
    return self.embed_query(text)

def __call__(self, text):
    return self.embed_text(text)
```

Te metody zapewniają zgodność z różnymi interfejsami w ekosystemie LangChain:
- `embed_text` - prosta aliasa dla `embed_query`
- `__call__` - pozwala na używanie obiektu bezpośrednio jako funkcji, np. `embedding_model("zapytanie")`

### Funkcja `get_vectorstore`

```python
def get_vectorstore(text_chunks, embedding_batch_size=8, use_performance_mode=True):
    """
    Create a vector store from text chunks.
    
    Args:
        text_chunks (list): A list of text chunks to embed.
        embedding_batch_size (int, optional): The batch size for embedding. Defaults to 8.
        use_performance_mode (bool, optional): Whether to use performance mode. Defaults to True.
        
    Returns:
        FAISS: A vector store containing the embeddings.
    """
```

Ta funkcja tworzy indeks wektorowy używając biblioteki FAISS, który pozwala na efektywne wyszukiwanie podobnych dokumentów.

#### Działanie szczegółowe krok po kroku:

1. **Import FAISS**:
   ```python
   from langchain_community.vectorstores import FAISS
   ```
   - Importuje klasę `FAISS` z pakietu `langchain_community.vectorstores`
   - FAISS (Facebook AI Similarity Search) to wysoce zoptymalizowana biblioteka do wyszukiwania podobnych wektorów

2. **Konfiguracja trybu wydajności**:
   ```python
   batch_size = embedding_batch_size if use_performance_mode else 4
   ```
   - Jeśli `use_performance_mode` jest `True`, używa podanego rozmiaru partii
   - W przeciwnym razie używa mniejszego rozmiaru partii (4)
   - Mniejszy rozmiar partii może zapewnić lepszą jakość embeddingów, ale wolniejsze przetwarzanie

3. **Tworzenie modelu embeddingowego**:
   ```python
   embeddings = MultilangE5Embeddings(batch_size=batch_size)
   ```
   - Tworzy instancję wcześniej zdefiniowanej klasy `MultilangE5Embeddings`
   - Przekazuje skonfigurowany rozmiar partii

4. **Tworzenie indeksu FAISS**:
   ```python
   vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
   ```
   - Metoda `from_texts` automatycznie:
     - Generuje embedingi dla wszystkich fragmentów tekstu przy użyciu dostarczonego modelu
     - Buduje indeks FAISS dla tych embeddingów
     - Przechowuje oryginalne teksty wraz z odpowiadającymi im embeddingami

5. **Zwrócenie indeksu**:
   ```python
   return vectorstore
   ```
   - Zwraca skonfigurowany i gotowy do użycia indeks wektorowy

Indeks FAISS umożliwia:
- Szybkie wyszukiwanie najbardziej podobnych dokumentów do zapytania
- Efektywne przechowywanie dużych kolekcji embeddingów
- Skalowalne wyszukiwanie z wykorzystaniem algorytmów przybliżonego najbliższego sąsiada
- Integrację z systemem RAG poprzez interfejs LangChain

## Podsumowanie technicznych aspektów implementacji

### Zarządzanie zasobami i optymalizacja wydajności

- **Przetwarzanie wsadowe** - kluczowa technika optymalizacyjna:
  ```python
  for i in range(0, len(processed_texts), self.batch_size):
      batch_texts = processed_texts[i:i+self.batch_size]
  ```
  Pozwala na przetwarzanie wielu dokumentów jednocześnie, co znacząco zwiększa przepustowość.

- **Wykorzystanie GPU**:
  ```python
  self.device = "cuda" if torch.cuda.is_available() else "cpu"
  self.model = self.model.to(self.device)
  ```
  Automatyczna detekcja i wykorzystanie GPU drastycznie przyspiesza obliczenia wektorowe.

- **Wyłączenie śledzenia gradientu**:
  ```python
  with torch.no_grad():
      outputs = self.model(**batch_dict)
  ```
  Oszczędza pamięć i przyspiesza inferencję w porównaniu do trybu treningowego.

- **Konfigurowalny tryb wydajności**:
  ```python
  batch_size = embedding_batch_size if use_performance_mode else 4
  ```
  Pozwala użytkownikowi na dostosowanie kompromisu między szybkością a jakością.

### Techniki optymalizacji jakości embeddingów

- **Instrukcje specyficzne dla dokumentów i zapytań**:
  ```python
  task = "Represent this document for retrieval:"  # dla dokumentów
  task = "Represent this query for retrieval:"     # dla zapytań
  ```
  Różne instrukcje dla dokumentów i zapytań pomagają modelowi generować lepiej dopasowane reprezentacje.

- **Normalizacja L2**:
  ```python
  embeddings = F.normalize(embeddings, p=2, dim=1)
  ```
  Zapewnia, że wszystkie wektory mają długość jednostkową, co jest kluczowe dla dokładnego podobieństwa kosinusowego.

- **Maskowanie tokenów**:
  ```python
  last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
  ```
  Zapobiega wpływowi tokenów wypełniających na finalną reprezentację.

- **Uśrednianie z uwzględnieniem tylko rzeczywistych tokenów**:
  ```python
  return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
  ```
  Zapewnia spójną skalę reprezentacji niezależnie od długości tekstu.

### Integracja z ekosystemem LangChain

- **Implementacja interfejsu Embeddings**:
  ```python
  class MultilangE5Embeddings(Embeddings):
  ```
  Zapewnia zgodność z różnymi komponentami LangChain poprzez implementację wymaganego interfejsu.

- **Metody zgodności**:
  ```python
  def embed_text(self, text):
      return self.embed_query(text)
  
  def __call__(self, text):
      return self.embed_text(text)
  ```
  Dodatkowe metody zwiększają elastyczność i łatwość użycia w różnych kontekstach.

- **Integracja z FAISS**:
  ```python
  vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
  ```
  Bezproblemowa integracja z wydajnymi indeksami wektorowymi poprzez standardowy interfejs.

## Końcowe podsumowanie

Moduł `embedding_utils.py` stanowi zaawansowany komponent przetwarzania tekstu w aplikacji, oferujący:

1. Inteligentny podział tekstów na semantycznie spójne fragmenty z zachowaniem kontekstu
2. Generowanie wysokiej jakości wielojęzycznych embedingów przy użyciu modelu E5 z optymalizacjami dla wydajności i jakości
3. Efektywne indeksowanie wektorowe za pomocą FAISS dla szybkiego wyszukiwania semantycznego

Te funkcjonalności tworzą solidny fundament dla systemu RAG, pozwalając na precyzyjne wyszukiwanie odpowiednich fragmentów dokumentów na podstawie zapytań użytkownika. Implementacja uwzględnia zarówno wydajność, jak i jakość, z konfigurowalnymi parametrami pozwalającymi na dostosowanie do konkretnych potrzeb. 