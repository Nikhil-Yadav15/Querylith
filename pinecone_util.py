import os, pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

PINECONE_ENV   = os.getenv("PINECONE_ENV")
PINECONE_API   = os.getenv("PINECONE_API_KEY")
INDEX_NAME     = os.getenv("PINECONE_INDEX" , "INDEX_NAME")
EMBEDDING_DIM  = 1536 #!dim

def get_pinecone_store():
    pinecone.init(api_key=PINECONE_API, environment=PINECONE_ENV)
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(INDEX_NAME, dimension=EMBEDDING_DIM, metric="cosine")
    index = pinecone.Index(INDEX_NAME)
    embeddings = OpenAIEmbeddings()          # !Embedding
    return Pinecone(index, embeddings.embed_query, "text")
