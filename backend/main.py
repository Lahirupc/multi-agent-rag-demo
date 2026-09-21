import os
from dotenv import load_dotenv
load_dotenv() # Load environment variables for LangSmith

from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from auth import get_current_active_user, check_permission, User
from rate_limiter import rate_limiter
from logger import logger
from langchain_core.messages import HumanMessage
from agent_graph import graph
from security import SecurityGuardrails

app = FastAPI(title="Commercial Bank Agent API")

class ChatRequest(BaseModel):
    message: str

# Exception Handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred. The application is degrading gracefully."},
    )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("request_started", method=request.method, path=request.url.path)
    try:
        response = await call_next(request)
        logger.info("request_finished", method=request.method, path=request.url.path, status_code=response.status_code)
        return response
    except Exception as ex:
        logger.error("request_failed", method=request.method, path=request.url.path, error=str(ex))
        raise

@app.post("/chat", dependencies=[Depends(get_current_active_user)])
async def chat_endpoint(request: ChatRequest, user: User = Depends(get_current_active_user)):
    # 1. Rate Limiting Check
    await rate_limiter.check_rate_limit(user.username)
    
    # 2. RBAC check
    check_permission(user, "chat")
    
    # 3. Security Guardrails: Prompt Injection Check
    SecurityGuardrails.validate_input(request.message)
    
    logger.info("chat_request", user=user.username, message_length=len(request.message))
    
    # 4. Invoke Mult-Agent LangGraph
    # Note: A real checkpointer using AsyncSqliteSaver would inject thread_id to persist memory.
    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "recursion_depth": 0,
        "retrieved_context": []
    }
    
    # Run graph execution async
    final_state = await graph.ainvoke(initial_state)
    logger.info("chat_graph_completed", user=user.username)
    
    return {
        "status": "success", 
        "context_fetched": len(final_state.get("retrieved_context", [])),
        "response": "This is a strictly compliant response from the Commercial Bank bot (Simulated Orchestration)."
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
