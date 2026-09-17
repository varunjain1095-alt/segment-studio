from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.utils.logging_config import setup_logging
from src.api.v1.endpoints import router as api_router
import logging
import os

setup_logging()
logger = logging.getLogger("main")

app = FastAPI(
    title="Customer Segmentation Studio API",
    description="Backend API for customer demographic and behavioral segmentation with predictive forecasting."
)

app.include_router(api_router)

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

@app.get("/")
async def serve_dashboard():
    return FileResponse("src/static/index.html")



