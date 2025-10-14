
import PyPDF2 as pdf2 # PDF handling
import numpy as np
import streamlit as st # FrontEnd 
import re # ReGex
import faiss # Embeddings Database
from sentence_transformers import SentenceTransformer


class RagSystem: 

    def __init__(self, path:str, embeddingmodel:str):
        self.path = path
        self.embeddingmodel = SentenceTransformer(embeddingmodel) #'all-MiniLM-L6-v2'
        self.text = None
        self.chunks = None
        self.embeddings = None
        self.index = None

    def collect_text(self) -> 'RagSystem':
        with open (self.path, 'rb') as pdftopic:
            reader = pdf2.PdfReader(pdftopic)
            self.text = ' '.join([page.extract_text() for page in reader.pages])
        return self
    
    def clean_text(self) -> 'RagSystem':
        pattern = r'RN-\d+\s+\|\s(.*?)(?=\s+ID:)'
        self.text = self.text.strip().replace('\n', ' ').replace('\t', ' ')
        self.chunks = re.findall(pattern, self.text)
        return self

    def create_embeddings(self) -> None:
            self.embeddings = self.embeddingmodel.encode(self.chunks)
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(np.array(self.embeddings))

    def encode_question(self, question:str) -> np.ndarray:
        question_encoded = self.embeddingmodel.encode([question])
        return question_encoded

    def search_response(self, question: str) -> 'RagSystem':
        q_emb = self.encode_question(question)
        D, I = self.index.search(np.array(q_emb), k=1)
        return self.chunks[I[0][0]]

if __name__ == '__main__':
    st.title("Neural Networks BoK")
    rag = RagSystem('./NeuralNetwork.pdf', 'all-MiniLM-L6-v2')
    rag.collect_text().clean_text().create_embeddings()
    user_question = st.text_input("Ask your question:")
    answer = rag.search_response(user_question)     
    st.write(f"Answer:{answer}")