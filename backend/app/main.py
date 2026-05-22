from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import asyncio
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
backend_dir = Path(__file__).parent.parent
load_dotenv(backend_dir / '.env')
print(f"🔧 Carregando .env de: {backend_dir / '.env'}")
print(f"🤖 LLM Provider: {os.getenv('LLM_PROVIDER', 'not set')}")
print(f"📦 Gemini Model: {os.getenv('GEMINI_MODEL', 'not set')}")

# Fix imports for direct execution
import sys
from pathlib import Path

# Add app directory to path for relative imports
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

from services.rag_service import get_rag_service
from utils.logging_config import setup_logging, get_error_tracker

import logging

# Setup logging before any code that uses logger
logger = setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", "logs/iabel_api.log")
)

# Import hybrid RAG+LoRA system (after logger is available)
try:
    sys.path.append(str(app_dir.parent / "fine_tuning"))
    from hybrid_rag_lora import HybridRAGService
    HYBRID_AVAILABLE = True
except ImportError as e:
    logger.warning("Hybrid RAG+LoRA not available: %s", e)
    HYBRID_AVAILABLE = False

app = FastAPI(
    title="IAbel API", 
    description="Enhanced AI Agent for Reservoir Engineering with Local RAG", 
    version="2.0.0"
)

app.add_middleware(
      CORSMiddleware,
      allow_origins=["*"],  # Permite todas as origens
      allow_credentials=True,
      allow_methods=["*"],
      allow_headers=["*"],
  )

# Initialize services
rag_service = get_rag_service()

# Initialize hybrid service if available
hybrid_service = None
if HYBRID_AVAILABLE:
    try:
        hybrid_service = HybridRAGService(rag_service=rag_service)
        logger.info("Hybrid RAG+LoRA service initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize hybrid service: {e}")
        hybrid_service = None

# Request/Response Models
class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    top_k: Optional[int] = 6
    include_sources: Optional[bool] = True
    mode: Optional[str] = "adaptive"  # New field for hybrid system

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    conversation_id: str
    total_sources: int
    sources: Optional[List[Dict[str, Any]]] = None
    timestamp: str

class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 8
    similarity_threshold: Optional[float] = 0.3

class ReindexRequest(BaseModel):
    force_reindex: Optional[bool] = False

# Enhanced RAG models
class EnhancedChatMessage(BaseModel):
    message: str
    mode: str  # 'rag_v1', 'rag_v2', 'rag_v3', 'lora_only', 'hybrid', 'adaptive'
    conversation_id: Optional[str] = None
    top_k: Optional[int] = 6
    include_sources: Optional[bool] = True

# Feedback models
class FeedbackRequest(BaseModel):
    response_id: str
    user_id: str
    question: str
    response_text: str
    rating: int  # 1-5 scale
    feedback_type: Optional[str] = "rating"
    feedback_text: Optional[str] = None
    aspects: Optional[Dict[str, int]] = None
    source_quality: Optional[int] = None
    citation_quality: Optional[int] = None

@app.get("/")
async def root():
    return {
        "message": "IAbel API - Enhanced Reservoir Engineering AI Agent",
        "version": "2.0.0",
        "features": [
            "Multilingual document search",
            "Context enhancement with acronym definitions",
            "Priority section boosting",
            "WebSocket streaming responses",
            "Local RAG system (no external APIs)"
        ]
    }

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file
        upload_dir = Path("data/pdfs")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process PDF with enhanced RAG system
        result = await rag_service.add_pdf_document(str(file_path), file.filename)
        
        if result['success']:
            return {
                "message": result['message'],
                "chunks_created": result['chunks_created'],
                "filename": result['filename'],
                "timestamp": result['timestamp']
            }
        else:
            raise HTTPException(status_code=500, detail=result['error'])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/chat/", response_model=ChatResponse)
async def chat_with_isabel(chat_message: ChatMessage):
    """Enhanced chat endpoint with multilingual support and hybrid RAG+LoRA"""
    try:
        # Use hybrid service if available and mode is specified
        if (hybrid_service and 
            chat_message.mode in ["adaptive", "hybrid", "lora_only", "rag_only", "rag_v1", "rag_v2"]):
            
            logger.info(f"Using hybrid service with mode: {chat_message.mode}")
            response = await hybrid_service.ask_question(
                question=chat_message.message,
                conversation_id=chat_message.conversation_id,
                top_k=chat_message.top_k or 6,
                include_sources=chat_message.include_sources or True,
                mode=chat_message.mode
            )
        else:
            # Fallback to regular RAG service
            logger.info("Using traditional RAG service")
            response = await rag_service.ask_question(
                question=chat_message.message,
                conversation_id=chat_message.conversation_id,
                top_k=chat_message.top_k or 6,
                include_sources=chat_message.include_sources or True
            )
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/search/")
async def search_documents(search_query: SearchQuery):
    """Enhanced document search with priority boosting"""
    try:
        results = await rag_service.search_documents(
            query=search_query.query,
            top_k=search_query.top_k or 8,
            similarity_threshold=search_query.similarity_threshold or 0.3
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.websocket("/chat/stream")
async def websocket_chat_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming chat responses"""
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            question = message_data.get('message', '')
            conversation_id = message_data.get('conversation_id')
            top_k = message_data.get('top_k', 6)
            
            if not question:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'error': 'Message is required'
                }))
                continue
            
            # Stream response
            async for chunk in rag_service.stream_response(
                question=question,
                conversation_id=conversation_id,
                top_k=top_k
            ):
                await websocket.send_text(json.dumps(chunk))
                
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket")
    except Exception as e:
        await websocket.send_text(json.dumps({
            'type': 'error',
            'error': str(e)
        }))

@app.post("/reindex/")
async def reindex_documents(reindex_request: ReindexRequest):
    """Reindex all PDF documents with progress updates"""
    
    async def generate_reindex_stream():
        async for update in rag_service.reindex_documents(
            force_reindex=reindex_request.force_reindex or False
        ):
            yield f"data: {json.dumps(update)}\n\n"
    
    return StreamingResponse(
        generate_reindex_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )

@app.get("/status/")
async def system_status():
    """Get comprehensive system status"""
    try:
        status = await rag_service.get_system_status()
        
        # Add error tracking information
        error_tracker = get_error_tracker()
        status['error_tracking'] = error_tracker.get_error_summary()
        
        return status
    except Exception as e:
        logger.error(f"Status endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")

@app.get("/errors/")
async def get_error_summary():
    """Get error tracking summary"""
    try:
        error_tracker = get_error_tracker()
        return {
            'summary': error_tracker.get_error_summary(),
            'recent_errors': error_tracker.get_recent_errors(20)
        }
    except Exception as e:
        logger.error(f"Error summary endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error summary error: {str(e)}")

@app.delete("/cache/")
async def clear_cache():
    """Clear all cached data"""
    try:
        success = rag_service.cache.clear_cache()
        if success:
            return {"message": "Cache cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear cache")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear error: {str(e)}")

@app.delete("/cache/{pattern}")
async def clear_cache_pattern(pattern: str):
    """Clear cached data matching pattern"""
    try:
        success = rag_service.cache.clear_cache(pattern)
        if success:
            return {"message": f"Cache entries matching '{pattern}' cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear cache")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear error: {str(e)}")

@app.get("/health/")
async def health_check():
    return {
        "status": "healthy", 
        "service": "IAbel Enhanced RAG API",
        "version": "2.0.0",
        "hybrid_available": HYBRID_AVAILABLE,
        "hybrid_initialized": hybrid_service is not None
    }

# New endpoints for hybrid system
@app.post("/train-lora/")
async def start_lora_training():
    """Start LoRA fine-tuning process"""
    if not HYBRID_AVAILABLE:
        raise HTTPException(status_code=503, detail="Hybrid system not available")
    
    try:
        # This would typically be run as a background task
        from fine_tuning.train_lora import train_lora_model
        
        _backend_root = Path(__file__).parent.parent
        pdf_directory = str(_backend_root / "data" / "pdfs")
        output_directory = str(_backend_root / "fine_tuning" / "outputs")
        
        # Note: In production, this should be a background task
        success = train_lora_model(pdf_directory, output_directory)
        
        return {
            "status": "completed" if success else "failed",
            "message": "LoRA training process completed" if success else "Training failed",
            "output_path": output_directory if success else None
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.get("/hybrid-status/")
async def get_hybrid_status():
    """Get hybrid system status and statistics"""
    if not HYBRID_AVAILABLE:
        return {
            "available": False,
            "message": "Hybrid RAG+LoRA system not available"
        }
    
    if hybrid_service:
        try:
            status = hybrid_service.get_system_status()
            return {
                "available": True,
                "initialized": True,
                **status
            }
        except Exception as e:
            return {
                "available": True,
                "initialized": False,
                "error": str(e)
            }
    else:
        return {
            "available": True,
            "initialized": False,
            "message": "Hybrid service not initialized"
        }

@app.post("/chat/hybrid/")
async def chat_hybrid_explicit(chat_message: ChatMessage):
    """Explicit hybrid chat endpoint with mode selection"""
    if not HYBRID_AVAILABLE or not hybrid_service:
        raise HTTPException(status_code=503, detail="Hybrid system not available")
    
    try:
        response = await hybrid_service.ask_question(
            question=chat_message.message,
            conversation_id=chat_message.conversation_id,
            top_k=chat_message.top_k or 6,
            include_sources=chat_message.include_sources or True,
            mode=chat_message.mode or "adaptive"
        )
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid chat error: {str(e)}")

@app.websocket("/chat/hybrid/stream")
async def websocket_hybrid_stream(websocket: WebSocket):
    """WebSocket endpoint for hybrid streaming responses"""
    if not HYBRID_AVAILABLE or not hybrid_service:
        await websocket.close(code=1011, reason="Hybrid system not available")
        return
    
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            question = message_data.get('message', '')
            conversation_id = message_data.get('conversation_id')
            mode = message_data.get('mode', 'adaptive')
            
            if not question:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'error': 'Message is required'
                }))
                continue
            
            # Stream hybrid response
            async for chunk in hybrid_service.stream_response(
                question=question,
                conversation_id=conversation_id,
                mode=mode
            ):
                await websocket.send_text(json.dumps(chunk))
                
    except WebSocketDisconnect:
        print("Client disconnected from hybrid WebSocket")
    except Exception as e:
        await websocket.send_text(json.dumps({
            'type': 'error',
            'error': str(e)
        }))

# Enhanced RAG endpoints
@app.post("/chat/enhanced/")
async def chat_enhanced(enhanced_message: EnhancedChatMessage):
    """Enhanced chat endpoint with mode selection"""
    if not HYBRID_AVAILABLE or not hybrid_service:
        raise HTTPException(status_code=503, detail="Hybrid system not available")
    
    try:
        logger.info(f"Enhanced chat request: {enhanced_message.message[:50]}... (mode: {enhanced_message.mode})")
        
        response = await hybrid_service.generate_response(
            question=enhanced_message.message,
            conversation_id=enhanced_message.conversation_id,
            mode=enhanced_message.mode
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Enhanced chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream/")
async def chat_stream(enhanced_message: EnhancedChatMessage):
    """Streaming chat endpoint with multi-mode support"""
    try:
        logger.info(f"Streaming chat request: {enhanced_message.message[:50]}... (mode: {enhanced_message.mode})")

        async def generate():
            try:
                # Route based on mode
                if enhanced_message.mode == 'rag_v3':
                    # RAG v3 mode - English with academic citations (real streaming)
                    logger.info("Using RAG v3 mode (streaming)")
                    async for chunk in rag_service.ask_question_v3_stream(
                        question=enhanced_message.message,
                        conversation_id=enhanced_message.conversation_id,
                        top_k=enhanced_message.top_k or 8,
                        include_sources=enhanced_message.include_sources
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"
                else:
                    # Use the direct RAG service for streaming (v1/v2)
                    async for chunk in rag_service.ask_question_stream(
                        question=enhanced_message.message,
                        conversation_id=enhanced_message.conversation_id,
                        top_k=enhanced_message.top_k or 4,
                        include_sources=enhanced_message.include_sources
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"

                # Send end marker
                yield f"data: {json.dumps({'type': 'end'})}\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            }
        )

    except Exception as e:
        logger.error(f"Stream setup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Feedback endpoints
@app.post("/feedback/")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback for continuous learning"""
    try:
        logger.info(f"Feedback submitted for response {feedback.response_id[:8]}... (rating: {feedback.rating})")
        
        # Get RAG service to access enhanced system
        rag_service = get_rag_service()
        
        if hasattr(rag_service, 'rag_system') and hasattr(rag_service.rag_system, 'record_user_feedback'):
            feedback_id = rag_service.rag_system.record_user_feedback(
                response_id=feedback.response_id,
                user_id=feedback.user_id,
                question=feedback.question,
                response_text=feedback.response_text,
                rating=feedback.rating,
                feedback_type=feedback.feedback_type,
                feedback_text=feedback.feedback_text,
                aspects=feedback.aspects
            )
            
            return {
                "feedback_id": feedback_id,
                "message": "Feedback recorded successfully"
            }
        else:
            raise HTTPException(status_code=503, detail="Feedback system not available")
            
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feedback/stats/")
async def get_feedback_stats():
    """Get feedback statistics and learning insights"""
    try:
        # Get RAG service to access enhanced system
        rag_service = get_rag_service()
        
        if hasattr(rag_service, 'rag_system') and hasattr(rag_service.rag_system, 'get_feedback_stats'):
            stats = rag_service.rag_system.get_feedback_stats()
            return stats
        else:
            return {"feedback_enabled": False, "message": "Feedback system not available"}
            
    except Exception as e:
        logger.error(f"Feedback stats error: {e}")
        return {"feedback_enabled": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)