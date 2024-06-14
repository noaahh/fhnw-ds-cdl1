from pathlib import Path
import tempfile

import joblib
import rootutils
import streamlit as st

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.model_pipeline import predict_file, LIGHTNING_MODULES

st.set_page_config(page_title="Model Prediction Dashboard", layout="wide")

st.title('Sensor Data Prediction Dashboard')
st.markdown("""Upload an exported ZIP file from [SensorLogger](https://github.com/tszheichoi/awesome-sensor-logger) 
to make predictions.""")

with st.sidebar:
    st.header("Settings")

    model = st.selectbox("Model", LIGHTNING_MODULES.keys(), help="Select the model to use for prediction.")

    wandb_artifact_path = st.text_input("Wandb Artifact Path", help="Wandb artifact path to download the model "
                                                                    "checkpoint. For example, "
                                                                    "'user/project/artifact:version'.")

    batch_size = st.number_input("Batch Size", min_value=1, value=128, step=1)

    scaler_file = st.file_uploader("Choose a Scaler file", type="pkl", help="Upload a Scaler file to scale the data "
                                                                            "properly. If not provided, results may be "
                                                                            "inaccurate and poor.")

uploaded_file = st.file_uploader("Choose a ZIP file", type="zip", help="Upload a ZIP file containing the data to be "
                                                                       "processed.")

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        zip_path = temp_dir_path / uploaded_file.name

        with open(zip_path, "wb") as buffer:
            buffer.write(uploaded_file.getvalue())

        scaler = None
        if scaler_file is not None:
            scaler_path = temp_dir_path / scaler_file.name
            with open(scaler_path, "wb") as buffer:
                buffer.write(scaler_file.getvalue())
            scaler = joblib.load(scaler_path)

        if st.button('Process File'):
            with st.spinner('Processing...'):
                try:
                    prediction = predict_file(measurement_file_path=zip_path,
                                              model_name=model,
                                              batch_size=batch_size,
                                              scaler=scaler,
                                              wandb_artifact_path=wandb_artifact_path,
                                              verbose=True)

                    st.success(f"Predicted label: {prediction}")
                except Exception as e:
                    st.error(f"Error processing the file: {e}")

with st.expander("Click here for more information about this tool"):
    st.write("""
    This tool processes uploaded ZIP files using a deep learning model to make predictions. Ensure that 
    your ZIP file is structured correctly according to the model's requirements. 

    Make sure the ZIP file is prefixed with the actual label of the activity. For example, if the activity is
    'walking', the ZIP file should be named 'walking-2022-01-01-12-00-00.zip'.

    The following sensors need to be activated in the SensorLogger app:
    - Accelerometer
    - Gravity
    - Orientation
    - Gyroscope

    The supported activities are:
    - Walking
    - Running
    - Climbing stairs (up and down)
    - Sitting
    - Standing
    """)
