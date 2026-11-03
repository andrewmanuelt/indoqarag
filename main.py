import os

from pathlib import Path
from dotenv import load_dotenv

from helper.dataset import Dataset
from helper.eval import (
    evaluate_dense_retriever, 
    # evaluate_bm25_retriever, evaluate_hybrid_retriever
)

from retriever.bm25 import BM25Retriever
from retriever.dense import DenseRetriever

load_dotenv(Path(__file__).with_name(".env"))
    
def main():
    prepare()
    
def prepare():
    bm25 = BM25Retriever()
    dense = DenseRetriever() 
    
    dataset = Dataset()
    train_dataset = dataset.load_indoqa_dataset('train')

    if not os.path.exists('./database/bm25/'):
        os.makedirs('./database/bm25/') 
    dataset.create_bm25_dataset()
    
    if bm25.database_exists():
        documents = [item["context"] for item in train_dataset]
        bm25.create_client(documents)
    bm25.load_retriever()

    if not dense.database_exists():
        dense.create_collection(train_dataset)
    dense.client()
    
    evaluate(dense, bm25, dataset)
    
    
def evaluate(dense, bm25, dataset):
    validation_dataset = dataset.load_indoqa_dataset('validation')
    
    dense_result = evaluate_dense_retriever(dense, validation_dataset)
    dataset.save_evaluation_result("dense", dense_result)
    
    # bm25_result = evaluate_bm25_retriever(bm25, validation_dataset)
    # dataset.save_evaluation_result("bm25", bm25_result)
    
    # hybrid_result = evaluate_hybrid_retriever(bm25, dense, validation_dataset)
    # dataset.save_evaluation_result("hybrid", hybrid_result)
    
if __name__ == '__main__':
    main()