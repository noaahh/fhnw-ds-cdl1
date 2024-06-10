from pathlib import Path
import rootutils
import streamlit as st

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.model_pipeline import run_model_pipeline

st.set_page_config(page_title="Model Prediction Dashboard", layout="wide")

st.title('Sensor Data Prediction Dashboard')
st.markdown("""Upload an exported ZIP file from [SensorLogger](https://github.com/tszheichoi/awesome-sensor-logger) 
to make predictions.""")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["MulticlassLogisticRegression", "DeepResBidirLSTM"])
    batch_size = st.number_input("Batch Size", min_value=1, value=32, step=1)
    verbose = st.checkbox("Verbose Mode", value=True)

uploaded_file = st.file_uploader("Choose a ZIP file", type="zip", help="Upload a ZIP file containing the data to be "
                                                                       "processed.")

if uploaded_file is not None:
    temp_dir = Path("temp/")
    temp_dir.mkdir(exist_ok=True)
    zip_path = temp_dir / uploaded_file.name

    with open(zip_path, "wb") as buffer:
        buffer.write(uploaded_file.getvalue())

    if st.button('Process File'):
        with st.spinner('Processing...'):
            try:
                prediction = run_model_pipeline(measurement_file_path=zip_path,
                                                model_name=model,
                                                batch_size=batch_size,
                                                verbose=verbose)

                st.success(f"Predicted label: {prediction}")
            except Exception as e:
                st.error(f"Error processing the file: {e}")
            finally:
                zip_path.unlink()  # Clean up the uploaded file

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

