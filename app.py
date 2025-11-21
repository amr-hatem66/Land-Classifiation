import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import plotly.express as px

# Load your trained model
model = tf.keras.models.load_model('landClassification_vgg16_model.h5')
class_names = ['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial',
               'Pasture','PermanentCrop','Residential','River','SeaLake']

st.title("EuroSAT Land Type Classifier")
st.write("Upload a satellite image to classify its land type.")

# Image uploader
uploaded_file = st.file_uploader("Industrial_10.jpg", type=["jpg","png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((224,224))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    st.write(f"**Predicted Land Type:** {predicted_class}")
    
    # Probability chart
    prob_df = {"Class": class_names, "Probability": predictions[0]}
    fig = px.bar(prob_df, x="Class", y="Probability", text="Probability")
    st.plotly_chart(fig)
