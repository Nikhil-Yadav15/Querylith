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

qa_handler = logging.FileHandler("qa_pipeline_log.json")
qa_handler.setFormatter(logging.Formatter("%(message)s"))

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
_together_model_3 = None
@lru_cache(maxsize=50)
def get_cached_document_hash(pdf_url: str) -> str:
    return hashlib.md5(pdf_url.encode()).hexdigest()
document_cache = {}
answer_cache = {}

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", trust_remote_code=True)
    return _embedding_model

def get_together_model_1():
    global _together_model_1
    if _together_model_1 is None:
        
        api_key = os.getenv("TOGETHER_API_KEY_1")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY_1 environment variable not set")
            
        _together_model_1 = ChatTogether(
            together_api_key=api_key,
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            temperature=0.5
        )
    return _together_model_1

def get_together_model_2():
    global _together_model_2
    if _together_model_2 is None:
        api_key = os.getenv("TOGETHER_API_KEY_2")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY_2 environment variable not set")
            
        _together_model_2 = ChatTogether(
            together_api_key=api_key,
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            temperature=0.5
        )
    return _together_model_2

def get_together_model_3():
    global _together_model_3
    if _together_model_3 is None:
        api_key = os.getenv("TOGETHER_API_KEY_3")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY_3 environment variable not set")
            
        _together_model_3 = ChatTogether(
            together_api_key=api_key,
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            temperature=0.5
        )
    return _together_model_3

def get_llm_model(model_instance: str):
    if model_instance == "model_1":
        return get_together_model_1()
    elif model_instance == "model_2":
        return get_together_model_2()
    elif model_instance == "model_3":
        return get_together_model_3()
    else:
        raise ValueError(f"Unknown model instance: {model_instance}")

def assign_questions_to_models(questions: List[str]) -> Dict[str, List[int]]:
    total_questions = len(questions)
    questions_per_model = total_questions // 3
    remainder = total_questions % 3
    
    current_index = 0
    model_assignments = {}
    model_1_count = questions_per_model + (1 if remainder > 0 else 0)
    model_assignments["model_1"] = list(range(current_index, current_index + model_1_count))
    current_index += model_1_count

    model_2_count = questions_per_model + (1 if remainder > 1 else 0)
    model_assignments["model_2"] = list(range(current_index, current_index + model_2_count))
    current_index += model_2_count

    model_assignments["model_3"] = list(range(current_index, total_questions))
    
    return model_assignments

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
        raise ImportError("Please install one of: PyMuPDF, PyPDF2, or pdfplumber")

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
        top_k_indices = np.argsort(similarities)[-2:][::-1]
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
            pass
        except Exception as e:
            logger.error(f"Failed to log chunk selection: {e}")
        
        results[q_idx] = relevant_chunks
    return results

def build_batched_prompt(question_indices: List[int], questions: List[str], similarity_results: Dict[int, List[Tuple[float, Document]]]) -> str:
    
    prompt_parts = [
        "You will receive multiple questions, each with its own relevant context from a PDF document. "
        "Answer each question based on its corresponding context. Provide concise answers (less than 100 words per question).\n\n"
    ]
    for i, q_idx in enumerate(question_indices, 1):
        question = questions[q_idx]
        relevant_chunks = similarity_results.get(q_idx, [])
        if relevant_chunks:
            context_parts = []
            for score, chunk in relevant_chunks[:2]: 
                context_parts.append(chunk.page_content)
            context = "\n\n".join(context_parts)

            if len(context) > 2000:
                context = context[:2000] + "..."
        else:
            context = "No relevant context found in the document."
        
        prompt_parts.append(f"Context for Question {i}:")
        prompt_parts.append(context)
        prompt_parts.append(f"\nQuestion {i}: {question}\n")
        prompt_parts.append("-" * 50 + "\n")

    json_format_instruction = f"""
Please provide your answers in the following JSON format:
{{
  "answers": [
    {{{", ".join([f'"question_{i}": "your answer to question {i}"' for i in range(1, len(question_indices) + 1)])}}}
  ]
}}

Make sure to:
1. Answer each question based on its corresponding context
2. Keep answers concise (less than 100 words each)
3. Use proper JSON formatting
4. Include all {len(question_indices)} answers
"""
    
    prompt_parts.append(json_format_instruction)
    
    return "\n".join(prompt_parts)

async def generate_batched_answers(questions: List[str], similarity_results: Dict[int, List[Tuple[float, Document]]]) -> List[str]:
    model_assignments = assign_questions_to_models(questions)
    
    async def process_model_batch(model_instance: str, question_indices: List[int]) -> Dict[str, str]:
        loop = asyncio.get_event_loop()
        
        def _generate_batch():
            batch_log = {
                "model_instance": model_instance,
                "question_indices": question_indices,
                "questions_count": len(question_indices),
                "api_key_used": f"TOGETHER_API_KEY_{model_instance.split('_')[1].upper()}",
                "prompt": "",
                "raw_response": "",
                "parsed_answers": {},
                "error": None
            }
            
            try:
                prompt = build_batched_prompt(question_indices, questions, similarity_results)
                batch_log["prompt"] = prompt
                batch_log["prompt_length"] = len(prompt)
                
                llm = get_llm_model(model_instance)
                response = llm.invoke([HumanMessage(content=prompt)])
                
                if hasattr(response, 'content') and response.content:
                    raw_response = str(response.content).strip()
                elif hasattr(response, 'text') and response.text:
                    raw_response = str(response.text).strip()
                else:
                    raw_response = str(response).strip()
                
                batch_log["raw_response"] = raw_response
                try:
                    json_start = raw_response.find('{')
                    json_end = raw_response.rfind('}') + 1
                    
                    if json_start != -1 and json_end != 0:
                        json_str = raw_response[json_start:json_end]
                        parsed_json = json.loads(json_str)
                        
                        if "answers" in parsed_json and isinstance(parsed_json["answers"], list) and len(parsed_json["answers"]) > 0:
                            answers_dict = parsed_json["answers"][0]
                            batch_log["parsed_answers"] = answers_dict
                        else:
                            raise ValueError("Invalid JSON structure: missing 'answers' key or empty list")
                    else:
                        raise ValueError("No valid JSON found in response")
                        
                except (json.JSONDecodeError, ValueError) as parse_error:
                    logger.warning(f"{model_instance.upper()}: JSON parsing failed: {parse_error}")
                    batch_log["error"] = f"JSON parsing failed: {parse_error}"
                    answers_dict = {}
                    for i, q_idx in enumerate(question_indices, 1):
                        question_key = f"question_{i}"
                        pattern = f"question {i}.*?answer.*?[:.]\\s*([^\\n]*(?:\\n[^\\n]*)*?)(?=question \\d+|$)"
                        import re
                        match = re.search(pattern, raw_response.lower(), re.DOTALL | re.IGNORECASE)
                        if match:
                            answers_dict[question_key] = match.group(1).strip()
                        else:
                            answers_dict[question_key] = f"Unable to parse answer for question {i}"
                    
                    batch_log["parsed_answers"] = answers_dict
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    batch_log["token_usage"] = {
                        "input_tokens": safe_json_serialize(getattr(response.usage_metadata, 'input_tokens', None)),
                        "output_tokens": safe_json_serialize(getattr(response.usage_metadata, 'output_tokens', None)),
                        "total_tokens": safe_json_serialize(getattr(response.usage_metadata, 'total_tokens', None))
                    }
                
                return batch_log["parsed_answers"]
                
            except Exception as e:
                error_msg = f"Error in {model_instance} batch processing: {e}"
                logger.error(error_msg)
                batch_log["error"] = str(e)
                error_answers = {}
                for i in range(1, len(question_indices) + 1):
                    error_answers[f"question_{i}"] = f"Error generating answer: {e}"
                
                return error_answers
            
            finally:
                try:
                    pass
                except Exception as log_error:
                    logger.error(f"Failed to log batch processing: {log_error}")
        
        return await loop.run_in_executor(None, _generate_batch)
    
    model_1_task = process_model_batch("model_1", model_assignments["model_1"])
    model_2_task = process_model_batch("model_2", model_assignments["model_2"])
    model_3_task = process_model_batch("model_3", model_assignments["model_3"])
    
    model_1_answers, model_2_answers, model_3_answers = await asyncio.gather(
        model_1_task, model_2_task, model_3_task)
    
    all_answers = [""] * len(questions)

    for i, q_idx in enumerate(model_assignments["model_1"]):
        question_key = f"question_{i + 1}"
        if question_key in model_1_answers:
            all_answers[q_idx] = model_1_answers[question_key]
        else:
            all_answers[q_idx] = "No answer provided by model 1"
    
    for i, q_idx in enumerate(model_assignments["model_2"]):
        question_key = f"question_{i + 1}"
        if question_key in model_2_answers:
            all_answers[q_idx] = model_2_answers[question_key]
        else:
            all_answers[q_idx] = "No answer provided by model 2"
    
    for i, q_idx in enumerate(model_assignments["model_3"]):
        question_key = f"question_{i + 1}"
        if question_key in model_3_answers:
            all_answers[q_idx] = model_3_answers[question_key]
        else:
            all_answers[q_idx] = "No answer provided by model 3"
    
    return all_answers

async def pipeline(pdf_url: str, questions: List[str]) -> List[str]:

    try:
        pass
    except Exception as e:
        logger.error(f"Failed to log pipeline start: {e}")
    doc_hash = get_cached_document_hash(pdf_url)
    cache_key = (doc_hash, tuple(questions))
    
    if cache_key in answer_cache:
        try:

            pass
        except Exception as e:
            logger.error(f"Failed to log cache hit: {e}")
        return answer_cache[cache_key]
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            pdf_future = executor.submit(asyncio.run, download_pdf_async(pdf_url))
            
            pdf_content = pdf_future.result()

        if doc_hash in document_cache:
            chunks = document_cache[doc_hash]
        else:
            docs = extract_text_from_pdf(pdf_content)
            if not docs:
                raise ValueError("No content could be extracted from the PDF")
            
            chunks = fast_chunk_documents(docs)
            document_cache[doc_hash] = chunks
            
            try:
                pass
            except Exception as e:
                logger.error(f"Failed to log document processing: {e}")

        similarity_results = vectorized_similarity_search(chunks, questions)

        answers = await generate_batched_answers(questions, similarity_results)
        
        answer_cache[cache_key] = answers

        try:
            pass
        except Exception as e:
            logger.error(f"Failed to log pipeline completion: {e}")

        if len(answer_cache) > 100:
            keys_to_remove = list(answer_cache.keys())[:-50]
            for key in keys_to_remove:
                del answer_cache[key]
            gc.collect()

        return answers
        
    except Exception as e:
        
        try:
            pass
        except Exception as log_error:
            logger.error(f"Failed to log pipeline error: {log_error}")
        
        logger.error(f"Error in pipeline: {e}")
        raise



