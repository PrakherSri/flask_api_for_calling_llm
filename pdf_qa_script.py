import PyPDF2
import requests
import json
from typing import Optional

class PDFQuestionAnswering:
    def __init__(self, hf_api_token: str, model_name: str = "deepset/roberta-base-squad2"):
        """
        Initialize the PDF Question Answering system.
        
        Args:
            hf_api_token (str): Your Hugging Face API token
            model_name (str): Hugging Face model name
        """
        self.hf_api_token = hf_api_token
        self.pdf_text = ""
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {hf_api_token}"}
        
        # Define working models
        self.working_models = {
            "qa": [
                "deepset/roberta-base-squad2",
                "distilbert-base-cased-distilled-squad", 
                "bert-large-uncased-whole-word-masking-finetuned-squad"
            ],
            "generation": [
                "gpt2",
                "microsoft/DialoGPT-medium",
                "google/flan-t5-base"
            ],
            "summarization": [
                "facebook/bart-large-cnn",
                "sshleifer/distilbart-cnn-12-6"
            ]
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Extract text from all pages
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                self.pdf_text = text.strip()
                print(f"Successfully extracted text from PDF: {len(self.pdf_text)} characters")
                return self.pdf_text
                
        except Exception as e:
            print(f"Error reading PDF: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, max_length: int = 1000) -> list:
        """
        Split text into chunks to handle long documents.
        
        Args:
            text (str): Text to chunk
            max_length (int): Maximum length of each chunk
            
        Returns:
            list: List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_length:
                current_chunk.append(word)
                current_length += len(word) + 1
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def change_model(self, model_name: str):
        """
        Change the model being used.
        
        Args:
            model_name (str): New model name
        """
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        print(f"Changed model to: {model_name}")
    
    def test_model_availability(self, model_name: str) -> bool:
        """
        Test if a model is available via the API.
        
        Args:
            model_name (str): Model name to test
            
        Returns:
            bool: True if model is available
        """
        test_url = f"https://api-inference.huggingface.co/models/{model_name}"
        test_payload = {"inputs": {"question": "test", "context": "test context"}}
        
        try:
            response = requests.post(test_url, headers=self.headers, json=test_payload)
            return response.status_code != 404
        except:
            return False
    
    def find_working_model(self, task_type: str = "qa") -> str:
        """
        Find a working model for the specified task.
        
        Args:
            task_type (str): Type of task ("qa", "generation", "summarization")
            
        Returns:
            str: Working model name
        """
        models_to_test = self.working_models.get(task_type, self.working_models["qa"])
        
        for model in models_to_test:
            print(f"Testing model: {model}")
            if self.test_model_availability(model):
                print(f"✓ Found working model: {model}")
                return model
            else:
                print(f"✗ Model not available: {model}")
        
        return "deepset/roberta-base-squad2"  # fallback
    
    def ask_question(self, question: str, context: Optional[str] = None) -> dict:
        """
        Ask a question about the PDF content using Hugging Face API.
        
        Args:
            question (str): Question to ask
            context (str): Optional context (uses PDF text if not provided)
            
        Returns:
            dict: Answer and confidence score
        """
        if not context:
            context = self.pdf_text
        
        if not context:
            return {"error": "No PDF text available. Please load a PDF first."}
        
        # Prepare the payload
        payload = {
            "inputs": {
                "question": question,
                "context": context
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "answer": result.get("answer", "No answer found"),
                    "score": result.get("score", 0),
                    "start": result.get("start", 0),  
                    "end": result.get("end", 0)
                }
            elif response.status_code == 404:
                return {"error": f"Model '{self.model_name}' not found. Try changing to a different model."}
            elif response.status_code == 503:
                return {"error": "Model is loading. Please wait a moment and try again."}
            else:
                return {"error": f"API request failed: {response.status_code} - {response.text}"}
                
        except Exception as e:
            return {"error": f"Error making API request: {str(e)}"}
    
    def ask_question_with_chunking(self, question: str) -> dict:
        """
        Ask a question using text chunking for better results with long documents.
        
        Args:
            question (str): Question to ask
            
        Returns:
            dict: Best answer found across all chunks
        """
        if not self.pdf_text:
            return {"error": "No PDF text available. Please load a PDF first."}
        
        chunks = self.chunk_text(self.pdf_text)
        best_answer = {"answer": "No answer found", "score": 0}
        
        print(f"Searching across {len(chunks)} text chunks...")
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            result = self.ask_question(question, chunk)
            
            if "error" not in result and result.get("score", 0) > best_answer["score"]:
                best_answer = result
                best_answer["chunk_index"] = i
        
        return best_answer
    
    def interactive_qa_session(self):
        """
        Start an interactive question-answering session.
        """
        if not self.pdf_text:
            print("No PDF loaded. Please load a PDF first using extract_text_from_pdf()")
            return
        
        print("\n=== Interactive PDF Q&A Session ===")
        print("Type 'quit' to exit")
        print("-" * 40)
        
        while True:
            question = input("\nEnter your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                print("Please enter a valid question.")
                continue
            
            print("Searching for answer...")
            result = self.ask_question_with_chunking(question)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"\nAnswer: {result['answer']}")
                print(f"Confidence: {result['score']:.3f}")
                if 'chunk_index' in result:
                    print(f"Found in chunk: {result['chunk_index'] + 1}")

# Example usage
def main():
    # Replace with your Hugging Face API token
    HF_API_TOKEN = "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"
    
    # Initialize with a working model
    pdf_qa = PDFQuestionAnswering(HF_API_TOKEN, "deepset/roberta-base-squad2")
    
    # Optional: Find and use a working model automatically
    print("Finding working model...")
    working_model = pdf_qa.find_working_model("summarization")
    pdf_qa.change_model(working_model)
    
    # Extract text from PDF
    pdf_path = "C:\\Users\\sriva\\Downloads\\1000666681_1063900905.pdf"  # Replace with your PDF path
    pdf_qa.extract_text_from_pdf(pdf_path)
    
    # Example questions
    #questions = [
    #    "What is the main topic of this document?",
    #    "Who are the authors mentioned?",
    #    "What are the key findings?",
    #    "What methodology was used?"
    #]

    questions = [
        "Summary of the document",
    ]
    
    # Ask questions programmatically
    print("=== Automated Q&A ===")
    for question in questions:
        print(f"\nQ: {question}")
        result = pdf_qa.ask_question_with_chunking(question)
        
        if "error" not in result:
            print(f"A: {result['answer']}")
            print(f"Confidence: {result['score']:.3f}")
        else:
            print(f"Error: {result['error']}")
    
    # Start interactive session
    pdf_qa.interactive_qa_session()

if __name__ == "__main__":
    main()