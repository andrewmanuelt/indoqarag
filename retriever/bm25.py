import os
from pathlib import Path

import bm25s
import Stemmer

from nltk.corpus import stopwords

class BM25Retriever():
    def __init__(self, k=30):
        self.bm25index = None
        self.bm25retriever = None
        self.bm25index_name = os.getenv("BM25_INDEX_PATH", "bm25_index")
        self.k = k
        
    def database_exists(self):
        return Path(self.bm25index_name).is_dir()
    
    def create_client(self, documents):
        documents = [str(document).lower() for document in documents]
        stopwords_id = stopwords.words("indonesian")
        stemmer = Stemmer.Stemmer("indonesian")
        corpus_tokens = bm25s.tokenize(documents, stopwords=stopwords_id, stemmer=stemmer)
        
        retriever = bm25s.BM25()
        self.bm25index = retriever.index(corpus_tokens)
        retriever.save(self.bm25index_name, corpus=documents)
    
    def load_retriever(self):
        self.bm25retriever = bm25s.BM25.load(self.bm25index_name, load_corpus=True)
        
    def search(self, query):
        stemmer = Stemmer.Stemmer("indonesian")
        query_tokens = bm25s.tokenize(str(query).lower(), stemmer=stemmer)
        return self.bm25retriever.retrieve(query_tokens, k=self.k)