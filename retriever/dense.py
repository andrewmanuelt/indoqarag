from pathlib import Path
from uuid import uuid4
from tqdm import tqdm

from langchain_chroma import Chroma
from langchain_core.documents import Document

from helper.embedding import load_embedding

class DenseRetriever():
    def __init__(self, k=5):
        self.model_embedding = load_embedding()
        self.vectorstore = None
        self.persist_directory = "database/indoqa"
        self.k = k

    def database_exists(self):
        return Path(self.persist_directory).is_dir()
        
    def client(self):
        self.vectorstore = Chroma(
            collection_name="indoqa",
            persist_directory=self.persist_directory,
            embedding_function=self.model_embedding
        )
    
    def create_collection(self, document):
        documents = []
        for doc in tqdm(document, total=len(document), desc="Storing documents..."):
            temp = Document(
                page_content=str(doc['context']).lower()
            )
            documents.append(temp)
            
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vectorstore.add_documents(documents=documents, ids=uuids)
        
    def search(self, query):
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 30,
                "lambda_mult": 0.7
            }
        ) 
        
        return retriever.invoke(str(query).lower())