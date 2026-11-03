from langchain_huggingface import HuggingFaceEmbeddings

def load_embedding():
    return HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cuda"},             
        encode_kwargs={"normalize_embeddings": True} 
    )