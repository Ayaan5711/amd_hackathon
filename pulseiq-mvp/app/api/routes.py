"""FastAPI routes for PulseIQ."""

import json
import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.agent.graph import invoke_agent
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    SessionInfo,
    SessionsListResponse,
    UploadResponse,
)
from app.session.store import get_session_store
from app.utils.csv_loader import load_csv

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a survey CSV file.
    
    Creates a new session and returns the session ID for use in chat.
    """
    logger.info(f"Upload request: {file.filename}")
    
    # Validate file type
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )
    
    try:
        # Read file bytes
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(
                status_code=400,
                detail="File is empty"
            )
        
        # Load CSV
        df, schema = load_csv(contents, file.filename)
        
        # Create session
        session_store = get_session_store()
        session_id = session_store.create(
            df=df,
            schema=schema,
            filename=file.filename
        )
        
        logger.info(f"Upload successful: {session_id}")
        
        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            row_count=len(df),
            column_count=len(df.columns),
            data_schema=schema,
            message="File uploaded successfully. You can now start chatting about your survey data."
        )
        
    except ValueError as e:
        logger.warning(f"Upload validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process file: {str(e)}"
        )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a message to the agent (non-streaming).
    
    Returns the complete response after processing.
    """
    logger.info(f"Chat request: session={request.session_id}, message={request.message[:50]}...")
    
    # Get session
    session_store = get_session_store()
    session = session_store.get(request.session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Please upload your CSV again."
        )
    
    # Get history
    history = session_store.get_history(request.session_id)
    
    try:
        # Invoke agent
        result = invoke_agent(
            session_id=request.session_id,
            user_message=request.message,
            history=history,
            schema=session.schema
        )
        
        # Update history
        session_store.append_history(request.session_id, "user", request.message)
        session_store.append_history(
            request.session_id,
            "assistant",
            result.get("response_narrative", "")
        )
        
        return ChatResponse(
            session_id=request.session_id,
            response=result.get("response_narrative", ""),
            follow_up_suggestions=result.get("follow_up_suggestions", []),
            evidence=result.get("evidence", {}),
            tool_calls=[
                {"tool_name": tc["tool_name"], "arguments": tc["arguments"]}
                for tc in result.get("tool_calls", [])
            ]
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process message: {str(e)}"
        )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Send a message to the agent with streaming response.
    
    Returns Server-Sent Events (SSE) with response chunks.
    """
    logger.info(f"Chat stream request: session={request.session_id}")
    
    # Get session
    session_store = get_session_store()
    session = session_store.get(request.session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired"
        )
    
    # Get history
    history = session_store.get_history(request.session_id)
    
    async def event_generator():
        """Generate SSE events."""
        try:
            # Send initial event
            yield f"event: start\ndata: {json.dumps({'status': 'processing'})}\n\n"
            
            # Invoke agent
            result = invoke_agent(
                session_id=request.session_id,
                user_message=request.message,
                history=history,
                schema=session.schema
            )
            
            # Stream response in chunks (simulate streaming)
            response_text = result.get("response_narrative", "")
            words = response_text.split()
            
            for i in range(0, len(words), 3):  # Send 3 words at a time
                chunk = " ".join(words[i:i+3])
                yield f"event: chunk\ndata: {json.dumps({'text': chunk + ' '})}\n\n"
            
            # Send final event with complete data
            final_data = {
                "response": response_text,
                "follow_up_suggestions": result.get("follow_up_suggestions", []),
                "evidence": result.get("evidence", {}),
                "tool_calls": [
                    {"tool_name": tc["tool_name"], "arguments": tc["arguments"]}
                    for tc in result.get("tool_calls", [])
                ]
            }
            yield f"event: complete\ndata: {json.dumps(final_data)}\n\n"
            
            # Update history
            session_store.append_history(request.session_id, "user", request.message)
            session_store.append_history(request.session_id, "assistant", response_text)
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/sessions", response_model=SessionsListResponse)
async def list_sessions() -> SessionsListResponse:
    """List all active sessions (for admin/debugging)."""
    session_store = get_session_store()
    sessions_data = session_store.list_sessions()
    
    sessions = [
        SessionInfo(
            session_id=s["session_id"],
            filename=s["filename"],
            uploaded_at=s["uploaded_at"],
            last_accessed=s["last_accessed"],
            row_count=s["row_count"],
            column_count=s["column_count"],
            history_length=s["history_length"]
        )
        for s in sessions_data
    ]
    
    return SessionsListResponse(
        sessions=sessions,
        total=len(sessions)
    )


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    """Get details for a specific session."""
    session_store = get_session_store()
    session = session_store.get(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.to_dict()


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    """Delete a session."""
    session_store = get_session_store()
    
    if session_store.delete(session_id):
        return {"message": "Session deleted", "session_id": session_id}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    session_store = get_session_store()
    stats = session_store.get_stats()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        sessions_active=stats["total_sessions"]
    )
