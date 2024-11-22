import re
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

class ModelManager:
    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"

        # Ensure CUDA is available
        if not torch.cuda.is_available():
            print("CUDA is not available. Please check your NVIDIA drivers and PyTorch installation.")
            return

        # Set environment variables for CUDA
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first GPU

        # Check CUDA availability
        self.device = torch.device("cuda")
        print(f"Using device: {self.device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Initialize model with optimizations for GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            use_cache=True,
            trust_remote_code=True
        )

        # Move model to GPU
        self.model.to(self.device)

        # Enable evaluation mode for better performance
        self.model.eval()

        # Display memory information
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    def generate_answer(self, question: str, context: List[str]) -> str:
        try:
            if not torch.cuda.is_available():
                return "Error: CUDA is not available"

            # Przygotowanie promptu
            prompt = f"""You are a helpful assistant. Please answer the following question clearly and concisely based on the provided context.

            Context:
            {context[0]}

            Question:
            {question}

            Answer:
                """

            # Tokenizacja promptu
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generowanie odpowiedzi
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.8,
                    top_k=50,
                    repetition_penalty=1.5,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Dekodowanie wygenerowanych tokenów
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Wyodrębnienie odpowiedzi
            answer_start = generated_text.find("Odpowiedź:")
            if answer_start != -1:
                answer = generated_text[answer_start + len("Odpowiedź:"):].strip()
            else:
                answer = generated_text[len(prompt):].strip()

            # Nie przycinaj odpowiedzi
            return answer

        except Exception as e:
            return f"Error generating response: {str(e)}"
