import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import os

from src import config
from src.api.endpoints import (
    stock, technical, fundamental, quantitative, market, ai, signals, accuracy
)

import logging



app = FastAPI(
    title="Stock Analysis API",
    description="Comprehensive Stock Analysis & AI Prediction API",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(stock.router, prefix="/api", tags=["Stock Data"])
app.include_router(technical.router, prefix="/api", tags=["Technical Analysis"])
app.include_router(fundamental.router, prefix="/api", tags=["Fundamental Analysis"])
app.include_router(quantitative.router, prefix="/api/quantitative", tags=["Quantitative Analysis"])
app.include_router(market.router, prefix="/api", tags=["Market Analysis"])
app.include_router(signals.router, prefix="/api", tags=["Signals"])
app.include_router(ai.router, prefix="/api", tags=["AI Analysis"]) 
# ai.router mounts /ai/analyze and /enhanced-prediction at /api -> /api/ai/analyze, /api/enhanced-prediction
app.include_router(accuracy.router, prefix="/api", tags=["Model Accuracy"]) # /api/model-accuracy, /api/price-targets


# Mount Static Files
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/error")
async def error_page():
    return FileResponse("static/error.html")

@app.get("/")
async def read_disk_root():
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.on_event("startup")
async def startup_event():
    logger = logging.getLogger("uvicorn.info")
    logger.info("System starting up...")
    logger.info("Stock Analysis System v1.0 Online")
    logger.info("Verifying API Key...")
    if not config.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not found in environment variables or config.py")
    else:
        logger.info("API Key check passed.")

@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    if full_path.startswith("api") or full_path.startswith("static"):
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
