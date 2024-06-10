import shutil
from pathlib import Path

import rootutils
from fastapi import FastAPI, File, UploadFile, HTTPException

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model_pipeline import run_model_pipeline

app = FastAPI()


@app.post("/predict/")
async def upload_zip_file(model_name: str, file: UploadFile = File(...)):
    temp_dir = Path("temp/")
    temp_dir.mkdir(exist_ok=True)
    zip_path = temp_dir / file.filename

    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        prediction = run_model_pipeline(measurement_file_path=zip_path,
                                        model_name=model_name,
                                        batch_size=32,
                                        verbose=True)
        return {"prediction": prediction}
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
    finally:
        zip_path.unlink()  # Cleanup the ZIP file after processing
