"""
Stock Algorithms - Desktop Application
Runs as a standalone app with embedded browser window.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import os
import sys
import threading
import webbrowser
import logging
from contextlib import asynccontextmanager

from src import config
from src.api.endpoints import (
    stock, technical, fundamental, quantitative, market, ai, signals, accuracy
)


# ============================================
# PATH RESOLUTION FOR PYINSTALLER
# ============================================
def get_base_path():
    """Get the base path for bundled or development mode."""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return sys._MEIPASS
    else:
        # Running in development
        return os.path.dirname(os.path.abspath(__file__))

BASE_PATH = get_base_path()
STATIC_PATH = os.path.join(BASE_PATH, "static")


# ============================================
# FASTAPI APPLICATION
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler replacing deprecated on_event."""
    logger = logging.getLogger("uvicorn.info")
    logger.info("System starting up...")
    logger.info("Stock Analysis System v1.0 Online")
    logger.info(f"Static files path: {STATIC_PATH}")
    logger.info("Verifying API Key...")
    if not config.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not found in environment variables or config.py")
    else:
        logger.info("API Key check passed.")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Stock Analysis API",
    description="Comprehensive Stock Analysis & AI Prediction API",
    version="1.0.0",
    lifespan=lifespan
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
app.include_router(accuracy.router, prefix="/api", tags=["Model Accuracy"])


# Mount Static Files with correct path
if os.path.exists(STATIC_PATH):
    app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
    # Also mount at root for CSS/JS relative paths
    app.mount("/css", StaticFiles(directory=os.path.join(STATIC_PATH, "css")), name="css")
    app.mount("/js", StaticFiles(directory=os.path.join(STATIC_PATH, "js")), name="js")
    if os.path.exists(os.path.join(STATIC_PATH, "icons")):
        app.mount("/icons", StaticFiles(directory=os.path.join(STATIC_PATH, "icons")), name="icons")


@app.get("/error")
async def error_page():
    return FileResponse(os.path.join(STATIC_PATH, "error.html"))


@app.get("/")
async def read_root():
    return FileResponse(os.path.join(STATIC_PATH, "index.html"))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    if full_path.startswith("api") or full_path.startswith("static"):
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    return FileResponse(os.path.join(STATIC_PATH, "index.html"))


# ============================================
# DESKTOP APP LAUNCHER
# ============================================
def run_server():
    """Run the uvicorn server."""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


def open_browser():
    """Open the default browser after a short delay."""
    import time
    time.sleep(2)  # Wait for server to start
    webbrowser.open("http://127.0.0.1:8000")


def run_desktop_app():
    """
    Run as a desktop app with embedded browser.
    Uses pywebview if available, otherwise opens in default browser.
    """
    try:
        import webview
        
        # Start server in background thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        import time
        time.sleep(2)
        
        # Create native window with embedded browser
        webview.create_window(
            title="Antigravity Finance",
            url="http://127.0.0.1:8000",
            width=1400,
            height=900,
            resizable=True,
            min_size=(1000, 700)
        )
        webview.start()
        
    except ImportError:
        print("pywebview not installed. Opening in default browser...")
        print("To get native window, install: pip install pywebview")
        
        # Start server in background thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Open in default browser
        open_browser()
        
        # Keep main thread alive
        try:
            server_thread.join()
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    # Check if running as bundled app or in development
    if getattr(sys, 'frozen', False):
        # Running as .exe - launch desktop app
        run_desktop_app()
    else:
        # Development mode - just run server
        uvicorn.run(app, host=config.HOST, port=config.PORT)
