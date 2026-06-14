"""FastAPI application entry point."""

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.governance_routes import router as governance_router
from app.api.routes import router as api_router
from app.api.survey_routes import router as survey_router
from app.config import HOST, LOG_LEVEL, PORT
from app.session.store import get_session_store

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pulseiq.log", mode="a")
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting PulseIQ MVP...")
    
    # Initialize session store
    session_store = get_session_store()
    logger.info(f"Session store ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down PulseIQ MVP...")
    session_store.shutdown()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="PulseIQ MVP",
    description="Agentic Survey Intelligence - Query your survey data in plain English",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")
app.include_router(governance_router, prefix="/api")
app.include_router(survey_router, prefix="/api")

# Mount static files (frontend)
try:
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
except RuntimeError:
    logger.warning("Frontend directory not found, serving API only")


@app.get("/health")
async def root_health():
    """Root health check for load balancers."""
    return {"status": "healthy", "service": "pulseiq-mvp"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL,
        reload=False
    )
