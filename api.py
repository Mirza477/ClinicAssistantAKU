# # api.py
# from typing import Any, List, Optional

# from fastapi.responses import StreamingResponse

# from fastapi import FastAPI

# from pydantic import BaseModel

# from fastapi.middleware.cors import CORSMiddleware

# import json

# from fastapi.responses import JSONResponse


# from src.chatbot import generate_response

# class ChatRequest(BaseModel):
#     messages: list
#     context: dict | None = None
#     session_state: str | None = None


# class ChatRequest(BaseModel):
#     messages: List[dict]
#     context: Optional[dict] = None
#     session_state: Any = None


# app = FastAPI()

# # 1) Enable CORS so frontend can talk to this API
# app.add_middleware(
#   CORSMiddleware,
#   allow_origins=["*"],            # or ["http://localhost:5173"]
#   allow_methods=["*"],
#   allow_headers=["*"],
#   allow_credentials=True,
# )

# @app.get("/health")
# async def health():
#     return {"status": "ok"}

# # @app.post("/chat/stream")
# # async def chat_stream(req: ChatRequest):
# #     async def event_generator():
# #         # result = generate_response(req.messages)
# #         # RIGHT: extract just the last user message string
# #         last_message = (
# #             req.messages[-1].get("content", "")
# #             if isinstance(req.messages, list) and req.messages
# #             else ""
# #         )
# #         result = generate_response(last_message)
# #         yield json.dumps(result) + "\n"
# #     return StreamingResponse(event_generator(), media_type="application/x-ndjson")

# # Old running code
# @app.post("/chat")
# async def chat(req: ChatRequest):
#     # extract the last user message
#     last = req.messages[-1].get("content", "") if req.messages else ""
#     # call your synchronous generate_response directly
#     result = generate_response(last)
#     return result

# api.py
from typing import Any, List, Optional, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from uuid import uuid4
from pydantic import BaseModel, Field


from src.feedback_service import FeedbackIn, add_feedback, get_all_feedback

from src.chatbot import generate_response

from src.conversation_service import add_record, ConversationRecord

from src.retriever import hybrid_retrieve


HistoryEntry = Dict[str, str]

History       = List[HistoryEntry]

class ChatRequest(BaseModel):
    # messages: List[Dict[str, Any]]
    messages: List[Dict[str, str]]

    history: List[Dict[str, str]] = Field(default_factory=list)

    # history: History = Field(default_factory=list)

    context: Optional[Dict[str, Any]] = None

    session_state: Optional[str] = None


    

app = FastAPI()

# Enable CORS so your Vite frontend can call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"]
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.post("/feedback")
async def post_feedback(fb: FeedbackIn):
    """
    Delegate to feedback_service.add_feedback to store one entry.
    """
    print(fb)
    feedback_id = add_feedback(fb)
    return {"ok": True, "feedback_id": feedback_id}

@app.get("/feedback")
async def list_feedback():
    """
    Delegate to feedback_service.get_all_feedback to retrieve all entries.
    """
    items = get_all_feedback()
    return items


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Returns a fully-shaped ChatAppResponse:
      - 'message' and 'delta' both hold the full assistant text
      - 'context' has placeholders for data_points, followups, thoughts
      - 'session_state' echoes back or is null
    """

    print(req.history)

    # Extract last user message
    last = ""
    if req.messages and isinstance(req.messages, list):

        last = req.messages[-1].get("content", "") 

      # 2) Trim history to only the last 3 entries
    MAX_TURNS = 5
    MAX_HISTORY_MESSAGES = MAX_TURNS * 2
    if len(req.history) > MAX_HISTORY_MESSAGES:
        req.history = req.history[-MAX_HISTORY_MESSAGES:]

    
    session_id = req.session_state or str(uuid4())

    docs = hybrid_retrieve(last)
    excerpts = [d.get("content", "")[:1000] for d in docs]


    # Call your generate_response, which may return a dict
    
    raw = generate_response(last,req.history)

    # Unwrap the actual text if needed
    if isinstance(raw, dict) and "response" in raw and isinstance(raw["response"], str):
        answer_text = raw["response"]
    else:
        answer_text = str(raw)


    # — save this Q/A turn to Cosmos —
    add_record(ConversationRecord(
       session_id=session_id,
       question=last,
      answer=answer_text
   ))
    


    # Shape into ChatAppResponse form
    response_payload = {
        "message": {
            "content": answer_text,
            "role": "assistant"
        },
        "delta": {
            "content": answer_text,
            "role": "assistant"
        },
        "context": {
            "data_points": excerpts,             # fill with citations if you have them
            "followup_questions": None,    # or a list of strings
            "thoughts": []                 # or intermediate reasoning steps
        },
        # "session_state": req.session_state or None
        "session_state": session_id
    }

    return JSONResponse(content=response_payload)

