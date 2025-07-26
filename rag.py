from langchain_experimental.text_splitter import SemanticChunker
from pinecone_util import get_pinecone_store, get_pinecone_client
import os
from langchain_together import ChatTogether
from dotenv import load_dotenv
import fitz
import requests
from langchain.schema import Document
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


prompt = PromptTemplate.from_template(
    """Use the provided context to answer the question.
    Context:
    {context}

    Question: {question}

    Answer:"""
)
def load_pdf_from_url(pdf_url: str):
    response = requests.get(str(pdf_url))
    response.raise_for_status()
    logger.info("Loading PDF from URL")
    
    pdf_doc = fitz.open("pdf", response.content)
    logger.info("PDF loaded")
    
    docs = []
    for page_num in range(len(pdf_doc)):
        page = pdf_doc.load_page(page_num)
        text = page.get_text()
        
        if text.strip():
            doc = Document(
                page_content=text,
                metadata={
                    "source": str(pdf_url),
                    "page": page_num + 1
                }
            )
            docs.append(doc)
    
    pdf_doc.close()
    return docs

def ingest_and_answer_then_cleanup(pdf_url: str, questions: list[str]) -> list[str]:
    vectordb = None
    
    logger.info("Inside ingest_and_answer_then_cleanup")
    
    try:
        docs = load_pdf_from_url(pdf_url)
        
        if not docs:
            raise ValueError("No content could be extracted from the PDF")
        
        vectordb = get_pinecone_store()
        
        splitter = SemanticChunker(embeddings=vectordb.embeddings)
        chunks = splitter.split_documents(docs)
        logger.info(f"Number of chunks: {len(chunks)}")
        
        vectordb.add_documents(chunks)
        logger.info("Added chunks to vector store")
        
        retriever = vectordb.as_retriever(search_type="mmr")
        logger.info("Retiever created")
        llm = ChatTogether(
            together_api_key=os.getenv("TOGETHER_API_KEY"),
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            temperature=0.5
        )
        qa_chain = create_stuff_documents_chain(
                    llm=llm,
                    prompt=prompt,
                    output_parser=StrOutputParser() 
                )
        
        answers = []
        for num, q in enumerate(questions):
            relevant_docs = retriever.get_relevant_documents(q)
            logger.info(f"Relevant docs for question {num}: {len(relevant_docs)}")
            ans = qa_chain.invoke({
                            "context": relevant_docs,  
                            "question": q
                        })
            logger.info(f"Answer for question {num}")
            answers.append(ans.strip())
        
        return answers
        
    finally:
        if vectordb:
            cleanup_index()

def cleanup_index():
    try:
        logger.info("Cleaning up index")
        pc = get_pinecone_client()
        index_name = os.getenv("PINECONE_INDEX")
        
        logger.info(f"Deleting all vectors from index {index_name}")
        index = pc.Index(index_name)
        index.delete(delete_all=True)
        logger.info(f"Done!!")
        
    except Exception as e:
        print(f"Cleanup warning: {e}")

