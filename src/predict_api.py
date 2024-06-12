import shutil
from pathlib import Path

import rootutils
from fastapi import FastAPI, File, UploadFile, HTTPException

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.model_pipeline import predict_file

app = FastAPI()


@app.post("/predict/")
async def upload_zip_file(model_name: str, wandb_artifact_path: str, batch_size: int, file: UploadFile = File(...)):
    temp_dir = Path("temp/")
    temp_dir.mkdir(exist_ok=True)
    zip_path = temp_dir / file.filename

    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        prediction = predict_file(measurement_file_path=zip_path,
                                  model_name=model_name,
                                  wandb_artifact_path=wandb_artifact_path,
                                  batch_size=batch_size,
                                  verbose=True)
        return {"prediction": prediction}
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
    finally:
        zip_path.unlink()  # Cleanup the ZIP file after processing
