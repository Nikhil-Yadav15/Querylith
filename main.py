import os
from fastapi import FastAPI, Header, HTTPException
from models import QuestionRequest, AnswerResponse
from rag import pipeline
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

app = FastAPI(title="Querylith API", version="0.2.0")

@app.post("/hackrx/run", response_model=AnswerResponse)
async def run_qa(payload: QuestionRequest, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Malformed token")
    if authorization.removeprefix("Bearer ").strip() != API_KEY_EXPECTED:
        raise HTTPException(status_code=401, detail="Invalid token")

    logger.info("Starting pipeline...")
    try:
        answers = await pipeline(str(payload.documents), payload.questions)
        logger.info("pipeline successfully")
        return {"answers": answers}
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
