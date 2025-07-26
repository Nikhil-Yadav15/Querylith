import os
from fastapi import FastAPI, Header, HTTPException
from models import QuestionRequest, AnswerResponse
from rag import ingest_and_answer_then_cleanup
import uvicorn
from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

API_KEY_EXPECTED = os.getenv("HACKRX_API_KEY")

app = FastAPI(title="Querylith API", version="0.1.0")

@app.post("/hackrx/run", response_model=AnswerResponse)
def run_qa(payload:QuestionRequest, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Malformed token")
    if authorization.removeprefix("Bearer ").strip() != API_KEY_EXPECTED:
        raise HTTPException(status_code=401, detail="Invalid token")


    logger.info("Ingesting and Answering...")
    answers = ingest_and_answer_then_cleanup(payload.documents, payload.questions)
    logger.info("Answered")
    return {"answers": answers}
