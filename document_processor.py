import PyPDF2
from typing import List
import io

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    def extract_text_from_pdf(self, pdf_file) -> str:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text

    def chunk_text(self, text: str) -> List[str]:
        # Simple chunking by words - you might want to use a more sophisticated approach
        sentences = text.replace('\n', ' ').split('.')
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_words = len(sentence.split())
            if sentence_words > self.chunk_size:
                # Jeśli zdanie jest za długie, podziel je
                words = sentence.split()
                for i in range(0, len(words), self.chunk_size):
                    chunks.append(' '.join(words[i:i + self.chunk_size]))
                continue

            if current_size + sentence_words > self.chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_words
            else:
                current_chunk.append(sentence)
                current_size += sentence_words

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks