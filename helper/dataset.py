import json

from pathlib import Path
from datasets import load_dataset

class Dataset():
    def load_indoqa_dataset(self, dtype: str):
        return load_dataset("jakartaresearch/indoqa")[dtype]
    
    def create_bm25_dataset(self):
        dataset = self.load_indoqa_dataset('train')
        with open('./database/bm25/bm25.txt', 'w', encoding='utf-8') as file:
            for item in dataset:
                file.write(f"{str(item['context']).lower()}\n")
                
    def save_evaluation_result(retriever_name, result):
        output_directory = Path(__file__).parent / "result"
        output_directory.mkdir(parents=True, exist_ok=True)
        
        output_path = output_directory / f"evaluation_{retriever_name}.json"
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(result, output_file, ensure_ascii=False, indent=2)
            
        print(f"Evaluation results saved to {output_path}.")