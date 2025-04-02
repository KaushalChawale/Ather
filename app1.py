# Uses best_image_classifier_model.keras

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = 'best_image_classifier_model.keras'
IMG_SIZE = (224, 224)

# IMPORTANT: Verify these class names match your train_generator.class_indices
# Example: If train_generator.class_indices was {'Ather': 0, 'Other': 1}
CLASS_NAMES = {0: 'Ather Scooter', 1: 'Other Scooter'}
# If train_generator.class_indices was {'Other': 0, 'Ather': 1}, use:
# CLASS_NAMES = {0: 'Other Scooter', 1: 'Ather Scooter'}

# --- Model Loading ---
# Use caching to load the model only once
@st.cache_resource
def load_keras_model(model_path):
    """Loads the Keras model."""
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at {model_path}")
        st.stop()
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop() # Stop execution if model fails to load

# --- Image Preprocessing and Prediction ---
def preprocess_image(image):
    """Preprocesses the uploaded image for the model."""
    try:
        img = image.resize(IMG_SIZE)
        img_array = np.array(img)

        # Ensure image is RGB (handle potential grayscale or RGBA)
        if img_array.ndim == 2: # Grayscale
            img_array = np.stack((img_array,)*3, axis=-1)
        elif img_array.shape[2] == 4: # RGBA
            img_array = img_array[:, :, :3]

        # Check if shape is still correct after potential conversion
        if img_array.shape != (IMG_SIZE[0], IMG_SIZE[1], 3):
             st.warning(f"Image shape after conversion {img_array.shape} is not the expected {(IMG_SIZE[0], IMG_SIZE[1], 3)}. Trying to reshape.")
             # Attempt resize again just in case
             img = Image.fromarray(img_array).resize(IMG_SIZE)
             img_array = np.array(img)
             if img_array.shape != (IMG_SIZE[0], IMG_SIZE[1], 3):
                  raise ValueError(f"Cannot reshape image to {(IMG_SIZE[0], IMG_SIZE[1], 3)}")


        # Rescale and expand dimensions for batch
        img_array = img_array / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict(model, processed_image):
    """Makes a prediction using the loaded model."""
    try:
        prediction = model.predict(processed_image)
        return prediction[0][0] # Model output is likely [[probability]]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# --- Streamlit App Layout ---
st.set_page_config(page_title="Scooter Classifier", layout="centered", page_icon="🛵", initial_sidebar_state = "collapsed" )
st.title("🛴 Ather Scooter Classifier")
st.write("Upload an image of a scooter, and the model will predict if it's an Ather scooter or another type.")

# Load the model
model = load_keras_model(MODEL_PATH)

uploaded_file = st.file_uploader("Choose a scooter image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)
        st.write("") # Add some space

        # Button to trigger classification
        if st.button("Classify Vehicle"):
            with st.spinner("Classifying..."):
                # Preprocess the image
                processed_image = preprocess_image(image)

                if processed_image is not None:
                    # Make prediction
                    prediction_value = predict(model, processed_image)

                    if prediction_value is not None:
                        # Interpret prediction (assuming class 0 is Ather, class 1 is Other)
                        if prediction_value <= 0.5:
                            predicted_class_index = 0
                            confidence = 1 - prediction_value
                        else:
                            predicted_class_index = 1
                            confidence = prediction_value

                        predicted_class_name = CLASS_NAMES.get(predicted_class_index, "Unknown Class")

                        # Display the result
                        st.subheader("Prediction Result:")
                        if predicted_class_index == 0: # Ather
                             st.success(f"✅ Predicted: **{predicted_class_name}**")
                        else: # Other
                             st.info(f"ℹ️ Predicted: **{predicted_class_name}**")

                        st.write(f"Confidence: **{confidence:.2%}**")
                        # Optional: Show raw prediction value
                        # st.write(f"(Raw model output: {prediction_value:.4f})")

    except Exception as e:
        st.error(f"An error occurred processing the file: {e}")

else:
    st.info("Please upload an image file.")

st.sidebar.header("About")
st.sidebar.info(
    "This app uses a deep learning model (trained on MobileNetV2) "
    "to classify images of scooters as either 'Ather' or 'Other'."
    "\n\n**Note:** Accuracy depends heavily on the training data and the quality/angle of the uploaded image."
)