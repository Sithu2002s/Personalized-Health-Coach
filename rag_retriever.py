# rag_retriever.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class RAGRetriever:
    def __init__(self, resources_path="C:/Users/Sithumi/src/data/resources"):
        self.resources_path = resources_path
        self.docs = []
        self.doc_names = []
        self.vectorizer = None
        self.doc_vectors = None
        self.load_resources()
        self.build_index()

    def load_resources(self):
        if not os.path.exists(self.resources_path):
            return
        for f in os.listdir(self.resources_path):
            if f.endswith(".txt"):
                path = os.path.join(self.resources_path, f)
                with open(path, "r", encoding="utf-8") as file:
                    self.docs.append(file.read())
                    self.doc_names.append(f)

    def build_index(self):
        if not self.docs:
            self.vectorizer = TfidfVectorizer()
            self.doc_vectors = None
            return
        
#TF-IDF gives higher scores to important words and lower scores to common words.
        
        self.vectorizer = TfidfVectorizer()    
        self.doc_vectors = self.vectorizer.fit_transform(self.docs)  #Convert every document into number vectors

    def retrieve(self, query, top_k=3): 
        if not self.docs or self.doc_vectors is None:
            return []
        
        q_vec = self.vectorizer.transform([query])  #user's query converted into numerical tf-idf vector
        sim = cosine_similarity(q_vec, self.doc_vectors)[0] #how close two texts are in meaning, 0-get first doc

        top_idx = sim.argsort()[::-1][:top_k]  #sort similarity scores from higest to lowest
        
        results = [{"filename": self.doc_names[i], "content": self.docs[i], "score": float(sim[i])} for i in top_idx]
        return results
