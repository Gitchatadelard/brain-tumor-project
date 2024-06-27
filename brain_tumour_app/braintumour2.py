"""
Brain Tumour Classification
"""
from random import random
import time
import seaborn as sns
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from urllib.request import urlopen

#---------------------------------------------------------------------------------------------------------------#
#                                                   PAGE SETTINGS                                               #
#---------------------------------------------------------------------------------------------------------------#

# Setting page configuration
st.set_page_config(page_title = "Brain Tumour Detection and Classification", page_icon=":hospital:", layout="wide")
st.markdown("##")
st.markdown(
    """
    <style>
    .main {
        background-color: #f7ffff;
    }
    </style>
    """,
    unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------------------------#
#                                                 SIDEBAR SETTINGS                                              #
#---------------------------------------------------------------------------------------------------------------#

# Sidebar population
# sBox = st.sidebar.selectbox("Menu", ["Home", "Detection","The Team"])

#----------------------------------------- Sidebar options configuration ----------------------------------------

# Function to configure the Overview page specifications
def PageSpecifications():
    # ---------------------------------------------------- DETECTION PAGE------------------------------------------------------------
    # elif sBox == "Detection":
    # setting header
    # st.subheader("Detection")
    st.markdown("<h1 style='text-align: center; color: black;'> Brain Tumour Detection and Classification </h1>", unsafe_allow_html=True)

    def main():
        file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
        if file_uploaded is not None: 
            file_bytes = np.asarray(bytearray(file_uploaded.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)  
            image = Image.open(file_uploaded)
            col1, col2 = st.columns([1, 1])
            with st.spinner('Model working....'):
                with col1:  
                    st.image(opencv_image, channels='RGB', width=300, caption='Uploaded Image')
                    predictions, message = predict(image)
                with col2:
                    time.sleep(1)
                    st.success('Classified')
                    st.markdown("<h5 style='text-align: left; color: black;'> {} </h5>".format(predictions), unsafe_allow_html=True)
                    st.write(message)

    def predict(image):
        model_path = "./model/model.tflite"
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        image = np.array(image.resize((150, 150)), dtype=np.float32) 
        image = np.expand_dims(image, axis=0)
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        probabilities = np.array(output_data[0])
        result = probabilities.argmax()
        labels = {0: 'Glioma Tumour', 1: 'Meningioma Tumour', 2: 'No Tumour', 3: 'Pituitary Tumour'}
        pred = labels[result]
        message = ''
        recommendation = 'Proceed to the neurosurgery department for further consultation.'
        if result == 0:
            message = 'Abnormal cell growth in the glial cells. ' + recommendation
            result = f"{pred} Detected" 
        elif result == 1:
            message = 'Abnormal cell growth in the meninges. ' + recommendation
            result = f"{pred} Detected"
        elif result == 2:
            result = 'No tumour was detected.'
        elif result == 3:
            message = 'Abnormal cell growth in the pituitary gland. ' + recommendation
            result = f"{pred} Detected"

        return result, message

    if __name__ == "__main__":
        main()         

# ---------------------------------------------------- FUNCTION CALL ---------------------------------------------------------------

PageSpecifications()

# Disclaimer
st.markdown("<h5 style='text-align: center; color: red;'> Disclaimer! </h5>", unsafe_allow_html=True)
st.write('''The application should only be used for educational purposes, and not for self-diagnosis or any other personal use. 
For real life cases, consult your local medical authority for advice.''')
#------------------------------------------------------------------------------------------------------------------------------------------

