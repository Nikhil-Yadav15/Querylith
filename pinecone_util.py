import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
load_dotenv()


PINECONE_API   = os.getenv("PINECONE_API_KEY")
INDEX_NAME     = os.getenv("PINECONE_INDEX")
MODEL_NAME     = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM  = 768

class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

def get_pinecone_store():
    pc = Pinecone(api_key=PINECONE_API)
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    
    index = pc.Index(INDEX_NAME)
    
    embeddings = SentenceTransformerEmbeddings(MODEL_NAME)
    
    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

def get_pinecone_client():
    return Pinecone(api_key=PINECONE_API)
