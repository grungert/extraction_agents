# example
# curl -X POST "http://localhost:8000/extract" -F "file=@input/8206743_COUPONS pour externe.2025.01.27 - copy.xlsx" -F "config_json=$(cat config/full_config.json)"


from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import tempfile
import json
import os

from src.extraction.extract_core import run_extraction

app = FastAPI()

@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    config_json: str = Form(...)
):
    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    # Parse config JSON string
    config_dict = json.loads(config_json)

    # Run extraction
    result = run_extraction(tmp_file_path, config_dict)

    # Clean up temp file
    try:
        os.remove(tmp_file_path)
    except Exception:
        pass

    return JSONResponse(content=result)
