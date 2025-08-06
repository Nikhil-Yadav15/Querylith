import asyncio
import concurrent.futures
import hashlib
import gc
import re
import numpy as np
from typing import List, Tuple, Dict, Any
from functools import lru_cache
import requests
import fitz
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from langchain_together import ChatTogether
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv
import json
from datetime import datetime
from io import BytesIO

load_dotenv()

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

qa_logger = logging.getLogger("QA_Details")
qa_handler = logging.FileHandler("qa_pipeline_log.json")
qa_handler.setFormatter(logging.Formatter("%(message)s"))
qa_logger.addHandler(qa_handler)
qa_logger.setLevel(logging.INFO)

def safe_json_serialize(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return {k: safe_json_serialize(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    else:
        return obj

def safe_json_dumps(obj: Any, **kwargs) -> str:
    try:
        return json.dumps(obj, **kwargs)
    except TypeError:
        safe_obj = safe_json_serialize(obj)
        return json.dumps(safe_obj, **kwargs)

_embedding_model = None
_together_model_1 = None
_together_model_2 = None

@lru_cache(maxsize=50)
def get_cached_document_hash(pdf_url: str) -> str:
    """Generate a hash for the PDF URL for caching purposes"""
    return hashlib.md5(pdf_url.encode()).hexdigest()

document_cache = {}
answer_cache = {}

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("embedding model...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", trust_remote_code=True)
        logger.info("Embedding model loaded")
    return _embedding_model

def get_together_model_1():
    global _together_model_1
    if _together_model_1 is None:
        logger.info("Together API model 1...")
        
        api_key = os.getenv("TOGETHER_API_KEY_1")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY_1 environment variable not set")
            
        _together_model_1 = ChatTogether(
            together_api_key=api_key,
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            temperature=0.5
        )
        logger.info("Together API model 1 loaded")
    return _together_model_1

def get_together_model_2():
    global _together_model_2
    if _together_model_2 is None:
        logger.info("Together API model 2...")
        
        api_key = os.getenv("TOGETHER_API_KEY_2")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY_2 environment variable not set")
            
        _together_model_2 = ChatTogether(
            together_api_key=api_key,
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            temperature=0.5
        )
        logger.info("Together API model 2 loaded")
    return _together_model_2

def get_llm_model(model_instance: str):
    if model_instance == "model_1":
        return get_together_model_1()
    elif model_instance == "model_2":
        return get_together_model_2()
    else:
        raise ValueError(f"Unknown model instance: {model_instance}")

def assign_models_to_questions(questions: List[str]) -> Dict[int, str]:
    """split 50/50"""
    total_questions = len(questions)
    mid_point = total_questions // 2
    
    assignments = {}
    for i in range(mid_point):
        assignments[i] = "model_1"
    for i in range(mid_point, total_questions):
        assignments[i] = "model_2"
    
    model_1_count = sum(1 for m in assignments.values() if m == "model_1")
    model_2_count = sum(1 for m in assignments.values() if m == "model_2")
    return assignments

def preprocess_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:()\-\'"]', '', text)
    text = text.replace('\n', ' ').strip()
    return text

async def download_pdf_async(pdf_url: str) -> bytes:
    loop = asyncio.get_event_loop()
    
    def download_pdf():
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        return response.content
    
    return await loop.run_in_executor(None, download_pdf)

def extract_text_from_pdf(pdf_content: bytes) -> List[Document]:
    logger.info("Extracting text from PDF...")
    try:
        import fitz
        if hasattr(fitz, 'open'):
            return extract_with_pymupdf(pdf_content)
    except (ImportError, AttributeError) as e:
        logger.warning(f"PyMuPDF not available: {e}")
    try:
        from PyPDF2 import PdfReader
        return extract_with_pypdf2(pdf_content)
    except ImportError:
        logger.warning("PyPDF2 not available")
    try:
        import pdfplumber
        return extract_with_pdfplumber(pdf_content)
    except ImportError:
        logger.error("No PDF processing library available")

def extract_with_pymupdf(pdf_content: bytes) -> List[Document]:
    import fitz
    pdf_doc = fitz.open("pdf", pdf_content)
    docs = []
    
    for page_num in range(len(pdf_doc)):
        page = pdf_doc.load_page(page_num)
        text = page.get_text()
        
        if text.strip():
            clean_text = preprocess_text(text)
            if len(clean_text) > 50:
                doc = Document(
                    page_content=clean_text,
                    metadata={"page": page_num + 1}
                )
                docs.append(doc)
    
    pdf_doc.close()
    return docs

def extract_with_pypdf2(pdf_content: bytes) -> List[Document]:
    from PyPDF2 import PdfReader
    
    pdf_file = BytesIO(pdf_content)
    pdf_reader = PdfReader(pdf_file)
    docs = []
    
    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        
        if text.strip():
            clean_text = preprocess_text(text)
            if len(clean_text) > 50:
                doc = Document(
                    page_content=clean_text,
                    metadata={"page": page_num + 1}
                )
                docs.append(doc)
    
    return docs

def extract_with_pdfplumber(pdf_content: bytes) -> List[Document]:
    import pdfplumber
    
    pdf_file = BytesIO(pdf_content)
    docs = []
    
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            
            if text and text.strip():
                clean_text = preprocess_text(text)
                if len(clean_text) > 50:
                    doc = Document(
                        page_content=clean_text,
                        metadata={"page": page_num + 1}
                    )
                    docs.append(doc)
    
    return docs

def fast_chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ";", ",", " "]
    )
    
    chunks = splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def vectorized_similarity_search(chunks: List[Document], questions: List[str]) -> Dict[int, List[Tuple[float, Document]]]:    
    model = get_embedding_model()
    chunk_texts = [chunk.page_content for chunk in chunks]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        chunk_emb_future = executor.submit(model.encode, chunk_texts, batch_size=32, show_progress_bar=False)
        question_emb_future = executor.submit(model.encode, questions, batch_size=len(questions), show_progress_bar=False)
        
        chunk_embeddings = chunk_emb_future.result()
        question_embeddings = question_emb_future.result()
    similarity_matrix = cosine_similarity(question_embeddings, chunk_embeddings)

    results = {}
    for q_idx, similarities in enumerate(similarity_matrix):
        top_k_indices = np.argsort(similarities)[-3:][::-1]
        top_similarities = similarities[top_k_indices]
        question = questions[q_idx]
        threshold = 0.3
        relevant_chunks = []
        
        chunk_selection_log = {
            "question_index": int(q_idx),
            "question": str(question),
            "threshold": float(threshold),
            "top_chunks_considered": []
        }
        
        for i, sim_score in zip(top_k_indices, top_similarities):
            chunk_info = {
                "chunk_index": int(i),
                "similarity_score": float(sim_score),
                "page": safe_json_serialize(chunks[i].metadata.get("page", "unknown")),
                "chunk_preview": str(chunks[i].page_content[:100] + "..." if len(chunks[i].page_content) > 100 else chunks[i].page_content),
                "selected": bool(sim_score > threshold)
            }
            chunk_selection_log["top_chunks_considered"].append(chunk_info)
            
            if sim_score > threshold:
                relevant_chunks.append((float(sim_score), chunks[i]))
        
        chunk_selection_log["selected_chunks_count"] = int(len(relevant_chunks))
        
        try:
            qa_logger.info(safe_json_dumps({
                "event": "chunk_selection",
                "data": chunk_selection_log
            }, indent=2))
        except Exception as e:
            logger.error(f"Failed to log chunk selection: {e}")
        
        results[q_idx] = relevant_chunks
    
    logger.info("Similarity search completed")
    return results

async def generate_answers_with_dual_together_models(questions: List[str], similarity_results: Dict[int, List[Tuple[float, Document]]]) -> List[str]:
    model_assignments = assign_models_to_questions(questions)
    
    prompt_template = PromptTemplate.from_template(
        """Based on the provided context, answer the question concisely and accurately in less than 100 words.
Context:
{context}

Question: {question}

Answer:"""
    )
    
    async def generate_single_answer(q_idx: int, question: str, model_instance: str) -> str:
        loop = asyncio.get_event_loop()
        
        def _generate():
            relevant_chunks = similarity_results.get(q_idx, [])
            model_info_map = {
                "model_1": "Together API Model 1 - Llama-3.3-70B-Instruct-Turbo-Free",
                "model_2": "Together API Model 2 - Llama-3.3-70B-Instruct-Turbo-Free"
            }
            
            qa_log_entry = {
                "question_index": int(q_idx),
                "question": str(question),
                "model_instance": model_instance,
                "model_info": model_info_map[model_instance],
                "api_key_used": f"TOGETHER_API_KEY_{model_instance.split('_')[1].upper()}",
                "extracted_chunks": [],
                "context_used": "",
                "prompt": "",
                "llm_response": "",
                "error": None,
                "debug_info": {}
            }
            
            if not relevant_chunks:
                qa_log_entry["llm_response"] = "I couldn't find relevant information to answer this question in the provided document."
                qa_log_entry["error"] = "No relevant chunks found"
                
                try:
                    qa_logger.info(safe_json_dumps({
                        "event": "qa_generation",
                        "data": qa_log_entry
                    }, indent=2))
                except Exception as e:
                    logger.error(f"Failed to log QA generation: {e}")
                
                return qa_log_entry["llm_response"]
            context_parts = []
            for i, (score, chunk) in enumerate(relevant_chunks[:3]):
                chunk_detail = {
                    "chunk_number": int(i + 1),
                    "similarity_score": float(score),
                    "page": safe_json_serialize(chunk.metadata.get("page", "unknown")),
                    "content": str(chunk.page_content),
                    "content_length": int(len(chunk.page_content))
                }
                qa_log_entry["extracted_chunks"].append(chunk_detail)
                context_parts.append(chunk.page_content)
            context = "\n\n".join(context_parts)

            qa_log_entry["debug_info"]["original_context_length"] = len(context)
            qa_log_entry["debug_info"]["context_empty"] = len(context.strip()) == 0
            
            if len(context.strip()) == 0:
                qa_log_entry["error"] = "Context is empty after chunk combination"
                qa_log_entry["llm_response"] = "Unable to generate answer due to empty context."
                return qa_log_entry["llm_response"]
            original_context_length = len(context)
            if len(context) > 3000:
                context = context[:3000] + "..."
                qa_log_entry["context_truncated"] = True
                qa_log_entry["original_context_length"] = int(original_context_length)
                qa_log_entry["truncated_context_length"] = int(len(context))
            
            qa_log_entry["context_used"] = str(context)
            
            try:
                formatted_prompt = prompt_template.format(context=context, question=question)
                qa_log_entry["prompt"] = str(formatted_prompt)
                qa_log_entry["debug_info"]["prompt_length"] = len(formatted_prompt)
                llm = get_llm_model(model_instance)
                
                response = llm.invoke([HumanMessage(content=formatted_prompt)])
                qa_log_entry["debug_info"]["response_type"] = str(type(response))
                qa_log_entry["debug_info"]["response_is_none"] = response is None
                
                if response is None:
                    raise ValueError(f"{model_instance.upper()} returned None response")
                
                logger.info(f"Q{q_idx}: Received response from {model_instance.upper()}")
                answer = None
                
                if hasattr(response, 'content') and response.content:
                    answer = str(response.content).strip()
                    qa_log_entry["debug_info"]["content_extraction_method"] = "response.content"
                elif hasattr(response, 'text') and response.text:
                    answer = str(response.text).strip()
                    qa_log_entry["debug_info"]["content_extraction_method"] = "response.text"
                else:
                    answer = str(response).strip()
                    qa_log_entry["debug_info"]["content_extraction_method"] = "str(response)"
                
                qa_log_entry["debug_info"]["extracted_answer_length"] = len(answer) if answer else 0
                qa_log_entry["debug_info"]["answer_is_empty"] = len(answer.strip()) == 0 if answer else True
                
                if not answer or len(answer.strip()) == 0:
                    logger.warning(f"Q{q_idx}: {model_instance.upper()} returned empty response")
                    qa_log_entry["error"] = f"{model_instance.upper()} returned empty response"
                    answer = f"The {model_instance.upper()} model returned an empty response. Please try rephrasing your question."
                
                qa_log_entry["llm_response"] = answer
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    qa_log_entry["token_usage"] = {
                        "input_tokens": safe_json_serialize(getattr(response.usage_metadata, 'input_tokens', None)),
                        "output_tokens": safe_json_serialize(getattr(response.usage_metadata, 'output_tokens', None)),
                        "total_tokens": safe_json_serialize(getattr(response.usage_metadata, 'total_tokens', None))
                    }
                elif hasattr(response, 'response_metadata') and response.response_metadata:
                    metadata = response.response_metadata
                    if 'token_usage' in metadata:
                        qa_log_entry["token_usage"] = safe_json_serialize(metadata['token_usage'])
                else:
                    qa_log_entry["debug_info"]["no_usage_metadata"] = True
                
                logger.info(f"Q{q_idx}: Successfully generated answer using {model_instance.upper()}")
                
            except Exception as e:
                error_msg = f"Error generating answer for question {q_idx} using {model_instance}: {e}"
                logger.error(error_msg)
                logger.error(f"Exception details: {type(e).__name__}: {str(e)}")
                qa_log_entry["error"] = str(e)
                qa_log_entry["debug_info"]["exception_type"] = type(e).__name__
                qa_log_entry["llm_response"] = "Sorry, I encountered an error while generating the answer."
                answer = qa_log_entry["llm_response"]
            try:
                qa_logger.info(safe_json_dumps({
                    "event": "qa_generation",
                    "data": qa_log_entry
                }, indent=2))
            except Exception as e:
                logger.error(f"Failed to log QA generation: {e}")
            
            return answer
        
        return await loop.run_in_executor(None, _generate)

    tasks = [
        generate_single_answer(i, question, model_assignments[i]) 
        for i, question in enumerate(questions)
    ]
    answers = await asyncio.gather(*tasks)
    
    return answers

async def pipeline(pdf_url: str, questions: List[str]) -> List[str]:

    model_assignments = assign_models_to_questions(questions)
    model_1_count = sum(1 for m in model_assignments.values() if m == "model_1")
    model_2_count = sum(1 for m in model_assignments.values() if m == "model_2")
    
    pipeline_log = {
        "event": "pipeline_start",
        "pdf_url": str(pdf_url),
        "questions_count": int(len(questions)),
        "questions": [str(q) for q in questions],
        "timestamp": datetime.now().isoformat(),
        "api_provider": "Dual Together API",
        "model_distribution": {
            "model_1": {
                "questions_assigned": model_1_count,
                "api_key": "TOGETHER_API_KEY_1",
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
            },
            "model_2": {
                "questions_assigned": model_2_count,
                "api_key": "TOGETHER_API_KEY_2", 
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
            }
        }
    }
    
    try:
        qa_logger.info(safe_json_dumps(pipeline_log, indent=2))
    except Exception as e:
        logger.error(f"Failed to log pipeline start: {e}")
    doc_hash = get_cached_document_hash(pdf_url)
    cache_key = (doc_hash, tuple(questions))
    
    if cache_key in answer_cache:
        logger.info("Returning cached answers")
        try:
            qa_logger.info(safe_json_dumps({
                "event": "cache_hit",
                "doc_hash": str(doc_hash),
                "questions_count": int(len(questions))
            }, indent=2))
        except Exception as e:
            logger.error(f"Failed to log cache hit: {e}")
        return answer_cache[cache_key]
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            pdf_future = executor.submit(asyncio.run, download_pdf_async(pdf_url))
            embedding_future = executor.submit(get_embedding_model)
            model_1_future = executor.submit(get_together_model_1)
            model_2_future = executor.submit(get_together_model_2)
            
            pdf_content = pdf_future.result()
            embedding_model = embedding_future.result()
            together_model_1 = model_1_future.result()
            together_model_2 = model_2_future.result()
        if doc_hash in document_cache:
            logger.info("Using cached document chunks")
            chunks = document_cache[doc_hash]
        else:
            docs = extract_text_from_pdf(pdf_content)
            if not docs:
                raise ValueError("No content could be extracted from the PDF")
            
            chunks = fast_chunk_documents(docs)
            document_cache[doc_hash] = chunks
            
            try:
                qa_logger.info(safe_json_dumps({
                    "event": "document_processing",
                    "data": {
                        "total_pages": int(len(docs)),
                        "total_chunks": int(len(chunks)),
                        "doc_hash": str(doc_hash),
                        "chunk_sizes": [int(len(chunk.page_content)) for chunk in chunks[:10]]
                    }
                }, indent=2))
            except Exception as e:
                logger.error(f"Failed to log document processing: {e}")

        similarity_results = vectorized_similarity_search(chunks, questions)

        answers = await generate_answers_with_dual_together_models(questions, similarity_results)

        answer_cache[cache_key] = answers

        try:
            qa_logger.info(safe_json_dumps({
                "event": "pipeline_complete",
                "doc_hash": str(doc_hash),
                "questions_count": int(len(questions)),
                "answers_generated": int(len(answers)),
                "timestamp": datetime.now().isoformat(),
                "api_provider": "Dual Together API",
                "model_usage": {
                    "model_1_questions": model_1_count,
                    "model_2_questions": model_2_count
                }
            }, indent=2))
        except Exception as e:
            logger.error(f"Failed to log pipeline completion: {e}")
        
        if len(answer_cache) > 100:
            keys_to_remove = list(answer_cache.keys())[:-50]
            for key in keys_to_remove:
                del answer_cache[key]
            gc.collect()
        
        return answers
        
    except Exception as e:
        error_log = {
            "event": "pipeline_error",
            "error": str(e),
            "pdf_url": str(pdf_url),
            "questions_count": int(len(questions)),
            "api_provider": "Dual Together API"
        }
        
        try:
            qa_logger.error(safe_json_dumps(error_log, indent=2))
        except Exception as log_error:
            logger.error(f"Failed to log pipeline error: {log_error}")
        
        logger.error(f"Error in pipeline: {e}")
        raise

