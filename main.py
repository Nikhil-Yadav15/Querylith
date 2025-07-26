import os
from fastapi import FastAPI, Header, HTTPException
from models import QuestionRequest, AnswerResponse
from rag import ingest, answer

API_KEY_EXPECTED = os.getenv("HACKRX_API_KEY")

app = FastAPI(title="Querylith API", version="0.1.0")

@app.post("/hackrx/run", response_model=AnswerResponse)
def run_qa(payload:QuestionRequest, authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Malformed token")
    if authorization.removeprefix("Bearer ").strip() != API_KEY_EXPECTED:
        raise HTTPException(status_code=401, detail="Invalid token")

    vectordb = ingest(payload.documents)
    answers  = answer(vectordb, payload.questions)

    return {"answers": answers}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
