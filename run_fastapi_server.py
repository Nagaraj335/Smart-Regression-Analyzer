#!/usr/bin/env python
"""
Simple script to run the FastAPI server
Usage: python run_fastapi_server.py
"""

import uvicorn
from fastapi1 import app

if __name__ == "__main__":
    print("🚀 Starting FastAPI server...")
    print("🌐 Server will be available at: http://127.0.0.1:8000")
    print("📚 API Documentation at: http://127.0.0.1:8000/docs")
    print("📖 Alternative docs at: http://127.0.0.1:8000/redoc")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )