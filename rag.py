from langchain.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStoreRetriever
# from langchain.retrievers import MMRRetriever
from pinecone_util import get_pinecone_store

def ingest(pdf_url:str):
    # *Lazy-load
    loader = PyPDFLoader(pdf_url)
    docs   = list(loader.lazy_load())

    # * Semantic chunk
    splitter = SemanticChunker(embeddings=get_pinecone_store().embeddings)
    chunks   = splitter.split_documents(docs)

    # * Embed
    vectordb = get_pinecone_store()
    vectordb.add_documents(chunks)         
    return vectordb

def answer(vectordb, questions:list[str]) -> list[str]:
    # * MMR retriever k=4, lambda_mult=0.5
    retriever:VectorStoreRetriever = vectordb.as_retriever(search_type="mmr")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    answers=[]
    for q in questions:
        docs = retriever.get_relevant_documents(q)
        ans  = qa_chain.run(input_documents=docs, question=q)
        answers.append(ans.strip())
    return answers
