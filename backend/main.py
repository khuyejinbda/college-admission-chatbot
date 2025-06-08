from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
import random

# Add the parent directory of 'adaptive_rag' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from adaptive_rag.utils.pipeline import initialize_graph_for_api, get_chatbot_response

app = FastAPI()

class ChatRequest(BaseModel):
    question: str
    user_id: str | None = None

@app.on_event("startup")
async def startup_event():
    print("Application startup: Initializing RAG graph...")
    initialize_graph_for_api()
    print("RAG graph initialized and ready.")

@app.post("/chat/")
async def chat(request: ChatRequest):
    user_id = request.user_id if request.user_id else str(random.randint(1, 1_000_000))
    response_data = get_chatbot_response(request.question, user_id)
    
    if "error" in response_data:
        return {"error": response_data["error"], "details": response_data.get("details", {})}
        
    # Assuming 'generation' contains the text response from the chatbot
    # and 'documents' might contain retrieved documents, if available.
    return {
        "user_id": user_id,
        "question": request.question,
        "answer": response_data.get("generation", "No answer generated."),
        "documents": response_data.get("documents", []) 
    }

if __name__ == "__main__":
    import uvicorn
    # This is for local development/testing of the FastAPI app directly.
    # Production deployments would use a Gunicorn or similar ASGI server.
    uvicorn.run(app, host="0.0.0.0", port=8000)
